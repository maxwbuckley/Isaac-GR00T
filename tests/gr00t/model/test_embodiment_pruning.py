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
Tests for the opt-in embodiment pruning memory optimization.

Gr00tN1d7.prune_embodiments(keep) slices the [max_num_embodiments, in, out]
category-specific action-head weights down to len(keep) rows while keeping
forwards addressable by the ORIGINAL embodiment ids via a remap table.

Covers:
- bitwise parity of get_action before/after pruning (same seed/inputs),
- parameter-count reduction by exactly len(keep)/max_num_embodiments,
- a clear error when an un-kept embodiment id arrives,
- the un-pruned fast path staying untouched (remap table is None),
- Gr00tPolicy(prune_to_embodiment=...) wiring.
"""

from unittest.mock import MagicMock, patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear
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
            "num_layers": 2,
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


def _make_deterministic_backbone(config, seq_len=8, seed=1234):
    """Mock backbone whose features depend only on a fixed seed and batch size.

    Unlike the mock in test_model_forward.py (fresh torch.randn per call, from
    the global RNG), this uses a dedicated torch.Generator re-seeded on every
    call, so identical inputs always yield identical backbone features -- a
    prerequisite for bitwise parity checks across two get_action calls.
    """
    backbone = MagicMock()

    def fake_forward(vl_input):
        B = 1
        for v in vl_input.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                B = v.shape[0]
                break
        device = next(
            (v.device for v in vl_input.values() if isinstance(v, torch.Tensor)),
            torch.device("cpu"),
        )
        dtype = next(
            (
                v.dtype
                for v in vl_input.values()
                if isinstance(v, torch.Tensor) and v.is_floating_point()
            ),
            torch.float32,
        )
        gen = torch.Generator(device="cpu").manual_seed(seed)
        features = torch.randn(
            B, seq_len, config.backbone_embedding_dim, generator=gen, dtype=torch.float32
        ).to(device=device, dtype=dtype)
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": torch.ones(B, seq_len, device=device, dtype=torch.long),
                "image_mask": torch.ones(B, seq_len, device=device, dtype=torch.bool),
            }
        )

    backbone.side_effect = fake_forward
    backbone.prepare_input = lambda x: BatchFeature(data=x)
    return backbone


@pytest.fixture
def small_model():
    """Build a Gr00tN1d7 with a deterministic mocked backbone (CPU only)."""
    config = _make_small_config()

    with patch("gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls") as mock_get_cls:
        mock_get_cls.return_value = lambda **kwargs: _make_deterministic_backbone(config)
        with patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"):
            from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

            model = Gr00tN1d7(config)

    model.eval()
    return model, config


def _make_get_action_inputs(config, embodiment_id, batch_size=2, seed=7):
    """Deterministic get_action inputs for a fixed embodiment id."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "state": torch.randn(
            batch_size, config.state_history_length, config.max_state_dim, generator=gen
        ),
        "embodiment_id": torch.full((batch_size,), embodiment_id, dtype=torch.long),
    }


def _category_specific_param_count(model) -> int:
    head = model.action_head
    return sum(
        p.numel()
        for m in (head.state_encoder, head.action_encoder, head.action_decoder)
        for p in m.parameters()
    )


class TestCategorySpecificLinearPruning:
    """Unit-level tests on the pruned linear layer itself."""

    def test_unpruned_has_no_remap_table(self):
        layer = CategorySpecificLinear(num_categories=4, input_dim=3, hidden_dim=5)
        assert layer.category_remap is None

    def test_prune_slices_and_remaps(self):
        torch.manual_seed(0)
        layer = CategorySpecificLinear(num_categories=4, input_dim=3, hidden_dim=5)
        x = torch.randn(2, 6, 3)
        cat_ids = torch.tensor([1, 3])
        expected = layer(x, cat_ids)

        layer.prune_categories([1, 3])
        assert layer.W.shape == (2, 3, 5)
        assert layer.b.shape == (2, 5)
        assert torch.equal(layer.category_remap, torch.tensor([-1, 0, -1, 1]))
        # Forward still consumes ORIGINAL ids and is bitwise identical.
        assert torch.equal(layer(x, cat_ids), expected)

    def test_prune_validates_keep(self):
        layer = CategorySpecificLinear(num_categories=4, input_dim=3, hidden_dim=5)
        with pytest.raises(ValueError, match="at least one"):
            layer.prune_categories([])
        with pytest.raises(ValueError, match="duplicate"):
            layer.prune_categories([1, 1])
        with pytest.raises(ValueError, match="out of range"):
            layer.prune_categories([4])

    def test_double_prune_raises(self):
        layer = CategorySpecificLinear(num_categories=4, input_dim=3, hidden_dim=5)
        layer.prune_categories([0])
        with pytest.raises(RuntimeError, match="already been pruned"):
            layer.prune_categories([0])

    def test_pruned_state_dict_has_no_remap_entry(self):
        layer = CategorySpecificLinear(num_categories=4, input_dim=3, hidden_dim=5)
        assert "category_remap" not in layer.state_dict()
        layer.prune_categories([2])
        assert "category_remap" not in layer.state_dict()


class TestModelPruningParity:
    """End-to-end parity/memory/error behavior through Gr00tN1d7.get_action."""

    def test_get_action_bitwise_parity_after_pruning(self, small_model):
        model, config = small_model
        embodiment = 2
        inputs = _make_get_action_inputs(config, embodiment)

        torch.manual_seed(42)
        before = model.get_action(dict(inputs))["action_pred"]

        model.prune_embodiments([embodiment])

        torch.manual_seed(42)
        after = model.get_action(dict(inputs))["action_pred"]

        assert torch.equal(before, after), "pruning must not change predictions bitwise"

    def test_parameter_count_drops_by_keep_fraction(self, small_model):
        model, config = small_model
        before = _category_specific_param_count(model)
        keep = [1]
        model.prune_embodiments(keep)
        after = _category_specific_param_count(model)
        assert after == before * len(keep) // config.max_num_embodiments

    def test_parameter_count_with_multiple_kept(self, small_model):
        model, config = small_model
        before = _category_specific_param_count(model)
        keep = [0, 3]
        model.prune_embodiments(keep)
        after = _category_specific_param_count(model)
        assert after == before * len(keep) // config.max_num_embodiments

    def test_unkept_embodiment_raises_clear_error(self, small_model):
        model, config = small_model
        model.prune_embodiments([2])
        inputs = _make_get_action_inputs(config, embodiment_id=1)
        with pytest.raises(ValueError, match=r"pruned to embodiment ids \[2\].*\[1\]"):
            model.get_action(inputs)

    def test_kept_ids_still_work_after_multi_prune(self, small_model):
        model, config = small_model
        keep = [0, 3]
        outputs_before = {}
        for e in keep:
            torch.manual_seed(42)
            outputs_before[e] = model.get_action(_make_get_action_inputs(config, e))["action_pred"]
        model.prune_embodiments(keep)
        for e in keep:
            torch.manual_seed(42)
            after = model.get_action(_make_get_action_inputs(config, e))["action_pred"]
            assert torch.equal(outputs_before[e], after)

    def test_unpruned_model_untouched(self, small_model):
        model, config = small_model
        head = model.action_head
        for module in (
            head.state_encoder.layer1,
            head.state_encoder.layer2,
            head.action_encoder.W1,
            head.action_encoder.W2,
            head.action_encoder.W3,
            head.action_decoder.layer1,
            head.action_decoder.layer2,
        ):
            assert module.category_remap is None
            assert module.W.shape[0] == config.max_num_embodiments
        # All embodiment ids remain serveable.
        for e in range(config.max_num_embodiments):
            out = model.get_action(_make_get_action_inputs(config, e, batch_size=1))
            assert out["action_pred"].shape == (1, config.action_horizon, config.max_action_dim)


def _make_policy_mocks(tag_value="new_embodiment", embodiment_id=10):
    """Build AutoModel/AutoProcessor mocks sufficient for Gr00tPolicy.__init__."""
    from gr00t.data.types import ModalityConfig

    model = MagicMock(name="model")
    processor = MagicMock(name="processor")
    processor.get_modality_configs.return_value = {
        tag_value: {
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        }
    }
    processor.embodiment_id_mapping = {tag_value: embodiment_id}
    return model, processor


class TestPolicyPruneWiring:
    """Gr00tPolicy(prune_to_embodiment=...) triggers model pruning correctly."""

    def _build_policy(self, model, processor, **kwargs):
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        with (
            patch("gr00t.policy.gr00t_policy.AutoModel") as auto_model,
            patch("gr00t.policy.gr00t_policy.AutoProcessor") as auto_processor,
        ):
            auto_model.from_pretrained.return_value = model
            auto_processor.from_pretrained.return_value = processor
            return Gr00tPolicy(
                embodiment_tag="new_embodiment",
                model_path="/nonexistent/checkpoint",
                device="cpu",
                **kwargs,
            )

    def test_prune_true_calls_model_with_mapped_id(self):
        model, processor = _make_policy_mocks(embodiment_id=10)
        self._build_policy(model, processor, prune_to_embodiment=True)
        model.prune_embodiments.assert_called_once_with([10])

    def test_default_false_does_not_prune(self):
        model, processor = _make_policy_mocks()
        self._build_policy(model, processor)
        model.prune_embodiments.assert_not_called()

    def test_missing_tag_in_mapping_raises(self):
        model, processor = _make_policy_mocks()
        processor.embodiment_id_mapping = {}
        with pytest.raises(ValueError, match="missing from the"):
            self._build_policy(model, processor, prune_to_embodiment=True)
        model.prune_embodiments.assert_not_called()

    def test_processor_without_mapping_raises(self):
        model, processor = _make_policy_mocks()
        processor.embodiment_id_mapping = None
        with pytest.raises(ValueError, match="embodiment_id_mapping"):
            self._build_policy(model, processor, prune_to_embodiment=True)
        model.prune_embodiments.assert_not_called()
