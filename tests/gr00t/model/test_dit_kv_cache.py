# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the DiT cross-attention K/V cache used during action sampling.

The VL features fed to the DiT are static within one `get_action` call, so the
cross-attention K/V projections of them are bitwise-identical across denoise
steps and can be computed once. These tests verify:

- bitwise parity of `action_pred` with the cache enabled vs disabled,
- that RNG consumption order is unchanged by the cache,
- that the cache is invalidated between `get_action` calls (new observation),
- that the K/V projections are actually computed only once per call,
- that the caching processors never outlive the sampling call, and
- that the training forward path is unaffected.

Unlike test_model_forward.py, the mocked backbone here is DETERMINISTIC: its
features are a pure function of a settable seed, so repeated calls can be
compared bit-for-bit.
"""

from unittest.mock import patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import CachedCrossAttnProcessor2_0
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _make_small_config(**overrides) -> Gr00tN1d7Config:
    """Return a minimal config for fast instantiation."""
    defaults = dict(
        model_name="nvidia/Cosmos-Reason2-2B",
        backbone_model_type="qwen",
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        select_layer=1,
        reproject_vision=False,
        use_flash_attention=False,
        load_bf16=False,
        tune_top_llm_layers=0,
        backbone_trainable_params_fp32=False,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            # 4 layers so AlternateVLDiT exercises both the text-token and the
            # image-token cross-attention blocks (cross at idx 0 and 2).
            "num_layers": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


class _DeterministicBackbone:
    """Backbone stub whose features are a pure function of `feature_seed`.

    The existing mock in test_model_forward.py draws fresh `torch.randn`
    features on every call, which would break bitwise parity comparisons.
    Setting a different `feature_seed` simulates a new observation.
    """

    def __init__(self, config, seq_len=8):
        self.config = config
        self.seq_len = seq_len
        self.feature_seed = 0

    def prepare_input(self, inputs):
        return BatchFeature(data=inputs)

    def __call__(self, vl_input):
        batch_size = 1
        for v in vl_input.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                batch_size = v.shape[0]
                break
        generator = torch.Generator().manual_seed(self.feature_seed)
        features = torch.randn(
            batch_size,
            self.seq_len,
            self.config.backbone_embedding_dim,
            generator=generator,
        )
        # Half text tokens, half image tokens so AlternateVLDiT's alternating
        # text/image cross-attention masks both attend to something.
        image_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)
        image_mask[:, self.seq_len // 2 :] = True
        return BatchFeature(
            data={
                "backbone_features": features,
                # bool (not long): SDPA rejects integer masks on the
                # AlternateVLDiT cross-attention path.
                "backbone_attention_mask": torch.ones(batch_size, self.seq_len, dtype=torch.bool),
                "image_mask": image_mask,
            }
        )


def _build_model(config):
    """Build a Gr00tN1d7 with the deterministic backbone (CPU, no download)."""
    with patch("gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls") as mock_get_cls:
        mock_get_cls.return_value = lambda **kwargs: _DeterministicBackbone(config)
        with patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"):
            from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

            model = Gr00tN1d7(config)

    model.eval()
    return model


def _make_inputs(config, batch_size, seed=123):
    generator = torch.Generator().manual_seed(seed)
    return {
        "state": torch.randn(
            batch_size, config.state_history_length, config.max_state_dim, generator=generator
        ),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
    }


def _cross_attn_key_projections(model):
    """Return the `to_k` linear of every cross-attention DiT block.

    With interleave_self_attention, even-indexed blocks are cross-attention.
    """
    dit = model.action_head.model
    return [block.attn1.to_k for idx, block in enumerate(dit.transformer_blocks) if idx % 2 == 0]


class _CallCounter:
    """Forward-hook based invocation counter for a list of modules."""

    def __init__(self, modules):
        self.counts = [0] * len(modules)
        self._handles = [
            module.register_forward_hook(self._make_hook(i)) for i, module in enumerate(modules)
        ]

    def _make_hook(self, index):
        def hook(module, args, output):
            self.counts[index] += 1

        return hook

    def reset(self):
        self.counts = [0] * len(self.counts)

    def remove(self):
        for handle in self._handles:
            handle.remove()


@pytest.mark.parametrize("use_alternate_vl_dit", [False, True])
@pytest.mark.parametrize("num_inference_timesteps", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_action_pred_bitwise_parity(use_alternate_vl_dit, num_inference_timesteps, batch_size):
    """Cache on vs off must produce bit-identical action_pred and leave the
    RNG stream in the same state (i.e. no change in RNG consumption order)."""
    torch.manual_seed(7)  # deterministic weight init
    config = _make_small_config(
        num_inference_timesteps=num_inference_timesteps,
        use_alternate_vl_dit=use_alternate_vl_dit,
    )
    model = _build_model(config)
    inputs = _make_inputs(config, batch_size)

    config.use_dit_kv_cache = False
    torch.manual_seed(0)
    reference = model.get_action(dict(inputs))["action_pred"]
    rng_state_after_reference = torch.get_rng_state()

    config.use_dit_kv_cache = True
    torch.manual_seed(0)
    cached = model.get_action(dict(inputs))["action_pred"]
    rng_state_after_cached = torch.get_rng_state()

    assert torch.equal(reference, cached)
    assert torch.equal(rng_state_after_reference, rng_state_after_cached)


@pytest.mark.parametrize("use_alternate_vl_dit", [False, True])
def test_kv_projection_computed_once_per_call(use_alternate_vl_dit):
    """The cache must actually be active: to_k of every cross-attention block
    runs exactly once per get_action call (vs once per denoise step)."""
    torch.manual_seed(7)
    config = _make_small_config(
        num_inference_timesteps=4, use_alternate_vl_dit=use_alternate_vl_dit
    )
    model = _build_model(config)
    inputs = _make_inputs(config, batch_size=1)

    counter = _CallCounter(_cross_attn_key_projections(model))
    try:
        config.use_dit_kv_cache = True
        model.get_action(dict(inputs))
        assert counter.counts == [1] * len(counter.counts), counter.counts

        counter.reset()
        config.use_dit_kv_cache = False
        model.get_action(dict(inputs))
        expected = [config.num_inference_timesteps] * len(counter.counts)
        assert counter.counts == expected, counter.counts
    finally:
        counter.remove()


def test_cache_invalidated_between_get_action_calls():
    """A new observation (different VL features) must never reuse stale K/V:
    the second cached call must match a cache-disabled reference bit-for-bit."""
    torch.manual_seed(7)
    config = _make_small_config()
    model = _build_model(config)
    inputs = _make_inputs(config, batch_size=2)

    config.use_dit_kv_cache = True
    model.backbone.feature_seed = 111
    torch.manual_seed(0)
    first = model.get_action(dict(inputs))["action_pred"]

    model.backbone.feature_seed = 222  # new observation -> new VL features
    torch.manual_seed(0)
    second = model.get_action(dict(inputs))["action_pred"]

    assert not torch.equal(first, second), "different VL features must change the prediction"

    config.use_dit_kv_cache = False
    torch.manual_seed(0)
    second_reference = model.get_action(dict(inputs))["action_pred"]
    assert torch.equal(second, second_reference)


def test_processors_restored_after_get_action():
    """The caching processors must not outlive the sampling call."""
    torch.manual_seed(7)
    config = _make_small_config()
    model = _build_model(config)
    inputs = _make_inputs(config, batch_size=1)
    dit = model.action_head.model

    processors_before = [block.attn1.processor for block in dit.transformer_blocks]
    config.use_dit_kv_cache = True
    model.get_action(dict(inputs))
    processors_after = [block.attn1.processor for block in dit.transformer_blocks]

    assert all(a is b for a, b in zip(processors_before, processors_after))
    assert not any(isinstance(p, CachedCrossAttnProcessor2_0) for p in processors_after)


def test_env_var_disables_cache(monkeypatch):
    """GR00T_DISABLE_DIT_KV_CACHE=1 must bypass the cache entirely."""
    torch.manual_seed(7)
    config = _make_small_config(num_inference_timesteps=4)
    config.use_dit_kv_cache = True  # opt in; the env var must still win
    model = _build_model(config)
    inputs = _make_inputs(config, batch_size=1)
    assert config.use_dit_kv_cache is True

    counter = _CallCounter(_cross_attn_key_projections(model))
    try:
        monkeypatch.setenv("GR00T_DISABLE_DIT_KV_CACHE", "1")
        model.get_action(dict(inputs))
        expected = [config.num_inference_timesteps] * len(counter.counts)
        assert counter.counts == expected, counter.counts
    finally:
        counter.remove()


def test_training_forward_unaffected():
    """The training forward path must keep the stock processors and gradients
    must still flow to the cross-attention K projection weights."""
    torch.manual_seed(7)
    config = _make_small_config()
    model = _build_model(config)
    model.train()
    dit = model.action_head.model
    processors_before = [block.attn1.processor for block in dit.transformer_blocks]

    batch_size = 2
    inputs = _make_inputs(config, batch_size)
    generator = torch.Generator().manual_seed(321)
    inputs["action"] = torch.randn(
        batch_size, config.action_horizon, config.max_action_dim, generator=generator
    )
    inputs["action_mask"] = torch.ones(batch_size, config.action_horizon, config.max_action_dim)

    output = model.forward(inputs)
    assert output["loss"].requires_grad
    output["loss"].backward()

    to_k = _cross_attn_key_projections(model)[0]
    assert to_k.weight.grad is not None
    assert torch.isfinite(to_k.weight.grad).all()

    processors_after = [block.attn1.processor for block in dit.transformer_blocks]
    assert all(a is b for a, b in zip(processors_before, processors_after))


def test_cached_kv_has_no_grad_and_processor_hit_counts():
    """The cache must not retain autograd graphs, and hit/miss counters must
    show one miss (first step) plus one hit per remaining step."""
    torch.manual_seed(7)
    config = _make_small_config(num_inference_timesteps=4)
    config.use_dit_kv_cache = True  # cache behavior under test; default is off
    model = _build_model(config)
    action_head = model.action_head
    inputs = _make_inputs(config, batch_size=1)

    backbone_inputs, action_inputs = model.prepare_input(dict(inputs))
    backbone_outputs = model.backbone(backbone_inputs)

    captured = {}
    original = type(action_head.model).cache_cross_attention_kv

    def capturing(self, enabled=True):
        context = original(self, enabled=enabled)

        class _Wrapper:
            def __enter__(self_inner):
                processors = context.__enter__()
                captured["processors"] = processors
                return processors

            def __exit__(self_inner, *exc):
                for processor in captured["processors"]:
                    if processor._cached_key is not None:
                        assert not processor._cached_key.requires_grad
                        assert not processor._cached_value.requires_grad
                        assert processor._cached_key.grad_fn is None
                        assert processor._cached_value.grad_fn is None
                return context.__exit__(*exc)

        return _Wrapper()

    with patch.object(type(action_head.model), "cache_cross_attention_kv", capturing):
        torch.manual_seed(0)
        action_head.get_action(backbone_outputs, action_inputs)

    processors = captured["processors"]
    assert len(processors) == len(action_head.model.transformer_blocks)
    num_cross_blocks = sum(
        1 for idx in range(len(processors)) if idx % 2 == 0
    )  # even blocks are cross-attention
    cross_processors = [p for p in processors if p.cache_misses or p.cache_hits]
    assert len(cross_processors) == num_cross_blocks
    for processor in cross_processors:
        assert processor.cache_misses == 1
        assert processor.cache_hits == config.num_inference_timesteps - 1
