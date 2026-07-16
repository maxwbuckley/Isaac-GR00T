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

"""CPU-only bitwise parity tests for the ``Qwen3Backbone.forward`` dead-code fix.

``Qwen3Backbone.forward`` only consumes ``hidden_states[-1]`` (plus a token-id
derived image mask), yet historically called the full
``Qwen3VLForConditionalGeneration`` forward, which (a) projected EVERY token
through the lm_head to the full ~152k vocab (a ``[B, S, vocab]`` tensor computed
and discarded) and (b) allocated an unused KV ``DynamicCache`` because
``use_cache`` was never disabled. The fix passes ``logits_to_keep=1`` and
``use_cache=False``.

These tests build a tiny random Qwen3-VL checkpoint on disk (no download, no
GPU) and assert the new forward is BITWISE identical to a reimplementation of
the legacy forward, with and without image tokens. They also pin the semantics
landmine: ``hidden_states[-1]`` of the conditional-generation wrapper is the
PRE-final-RMSNorm output of the last kept decoder layer (the transformers
``check_model_inputs`` capture hooks record decoder-layer outputs, and
``Qwen3VLCausalLMOutputWithPast`` has no ``last_hidden_state`` field, so the
"tie last hidden state" replacement never fires). Downstream DiT weights were
trained against these pre-norm features; see the "No final norm!" note in
scripts/deployment/export_onnx_n1d7.py.
"""

import pytest
import torch


pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone  # noqa: E402
from transformers import Qwen3VLForConditionalGeneration  # noqa: E402
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig  # noqa: E402


VISION_START_TOKEN_ID = 5
VISION_END_TOKEN_ID = 6
IMAGE_TOKEN_ID = 7
VIDEO_TOKEN_ID = 8
VOCAB_SIZE = 128
TEXT_HIDDEN_SIZE = 32
# The checkpoint has 3 text layers; the backbone keeps the bottom SELECT_LAYER
# of them (popping the rest), mirroring production where select_layer < depth.
SELECT_LAYER = 2


def _tiny_config() -> Qwen3VLConfig:
    """Hand-built tiny Qwen3-VL config (2-block vision tower, 3 text layers)."""
    return Qwen3VLConfig(
        text_config=dict(
            vocab_size=VOCAB_SIZE,
            hidden_size=TEXT_HIDDEN_SIZE,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=256,
            rope_theta=10000.0,
            # Qwen3VLTextRotaryEmbedding requires rope_scaling with an mrope
            # section summing to head_dim // 2.
            rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]},
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            # Merged vision embeddings are scattered into the text stream, so
            # out_hidden_size must equal the text hidden size.
            out_hidden_size=TEXT_HIDDEN_SIZE,
            num_position_embeddings=16,
            deepstack_visual_indexes=[1],
        ),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
    )


@pytest.fixture(scope="module")
def backbone(tmp_path_factory, load_hf_model_weights) -> Qwen3Backbone:
    torch.manual_seed(0)
    checkpoint_dir = tmp_path_factory.mktemp("tiny_qwen3vl")
    model = Qwen3VLForConditionalGeneration(_tiny_config())
    model.save_pretrained(checkpoint_dir)
    # tests/conftest.py sets GROOT_SKIP_HF_MODEL_WEIGHTS=1, which stubs
    # from_pretrained to skip weight loading (all-zero weights) and would make
    # the bitwise-parity assertions vacuous. The checkpoint here is tiny, so
    # opt back into real weight loading for this fixture.
    with load_hf_model_weights():
        bb = Qwen3Backbone(
            model_name=str(checkpoint_dir),
            tune_llm=False,
            tune_visual=False,
            select_layer=SELECT_LAYER,
            use_flash_attention=False,
        )
    bb.eval()
    assert len(bb.model.language_model.layers) == SELECT_LAYER
    # Guard against vacuous parity: the random checkpoint weights must have
    # actually been loaded.
    assert bb.model.language_model.embed_tokens.weight.abs().sum().item() > 0
    return bb


def _text_only_inputs() -> dict:
    """Batch of 2 text-only sequences, one with left padding (no vision keys)."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, VISION_START_TOKEN_ID, (2, 7))
    attention_mask = torch.ones(2, 7, dtype=torch.long)
    attention_mask[1, :2] = 0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": None,
        "image_grid_thw": None,
    }


def _image_inputs() -> dict:
    """Batch of 2 sequences, each with one 4x4-patch image (4 merged tokens)."""
    torch.manual_seed(2)
    row = (
        [1, 2, VISION_START_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * 4  # t * (h/merge) * (w/merge) = 1 * 2 * 2
        + [VISION_END_TOKEN_ID, 3, 4]
    )
    input_ids = torch.tensor([row, row], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    # t*h*w = 16 patches per image; per-patch dim = in_channels * temporal_patch
    # * patch_size**2 = 3 * 2 * 2 * 2 = 24.
    pixel_values = torch.randn(2 * 16, 24)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def _legacy_forward(bb: Qwen3Backbone, vl_input: dict):
    """Reimplementation of the pre-fix ``Qwen3Backbone.forward`` (the baseline).

    Full wrapper call with ``output_hidden_states=True`` and default
    ``use_cache`` / ``logits_to_keep`` (i.e. full-vocab logits + KV cache),
    consuming ``hidden_states[-1]``.
    """
    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}
    outputs = bb.model(**vl_input, output_hidden_states=True)
    features = outputs.hidden_states[-1]
    image_mask = vl_input["input_ids"] == bb.model.config.image_token_id
    attention_mask = vl_input["attention_mask"] == 1
    result = BatchFeature(
        data={
            "backbone_features": features,
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,
        }
    )
    return result, outputs


def _assert_bitwise_equal(new: BatchFeature, legacy: BatchFeature) -> None:
    for key in ("backbone_features", "backbone_attention_mask", "image_mask"):
        assert torch.equal(new[key], legacy[key]), f"{key} is not bitwise identical"
        assert new[key].dtype == legacy[key].dtype
        assert new[key].shape == legacy[key].shape


@pytest.mark.parametrize("inputs_fn", [_text_only_inputs, _image_inputs], ids=["text", "image"])
def test_forward_bitwise_matches_legacy(backbone, inputs_fn):
    vl_input = inputs_fn()
    with torch.no_grad():
        legacy, _ = _legacy_forward(backbone, vl_input)
        new = backbone(BatchFeature(data=vl_input))
    _assert_bitwise_equal(new, legacy)
    assert new["backbone_features"].shape == (
        vl_input["input_ids"].shape[0],
        vl_input["input_ids"].shape[1],
        TEXT_HIDDEN_SIZE,
    )


def test_image_case_exercises_vision_path(backbone):
    vl_input = _image_inputs()
    with torch.no_grad():
        new = backbone(BatchFeature(data=vl_input))
    # 4 merged image tokens per sequence, batch of 2.
    assert new["image_mask"].sum().item() == 8


def test_legacy_forward_is_deterministic(backbone):
    """Guard: parity comparisons are only meaningful if repeat runs are bitwise
    stable on CPU (they are: eval mode, zero dropout, deterministic kernels)."""
    vl_input = _image_inputs()
    with torch.no_grad():
        first, _ = _legacy_forward(backbone, vl_input)
        second, _ = _legacy_forward(backbone, vl_input)
    _assert_bitwise_equal(first, second)


def test_legacy_path_wasted_full_vocab_logits_and_kv_cache(backbone):
    """Document the waste the fix removes (also guards the transformers
    behavior these tests assume: default use_cache=True, full-seq logits)."""
    vl_input = _text_only_inputs()
    batch, seq = vl_input["input_ids"].shape
    with torch.no_grad():
        _, outputs = _legacy_forward(backbone, vl_input)
    assert outputs.logits.shape == (batch, seq, VOCAB_SIZE)
    assert outputs.past_key_values is not None


def test_new_forward_skips_logits_and_kv_cache(backbone):
    """The fixed forward must reach the wrapper with use_cache=False and
    logits_to_keep=1, produce single-position logits, and allocate no cache."""
    vl_input = _text_only_inputs()
    batch, _ = vl_input["input_ids"].shape
    captured = {}
    original_forward = backbone.model.forward

    def spy(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        captured["kwargs"] = kwargs
        captured["outputs"] = outputs
        return outputs

    backbone.model.forward = spy
    try:
        with torch.no_grad():
            backbone(BatchFeature(data=vl_input))
    finally:
        del backbone.model.forward  # restore the class-level forward

    assert captured["kwargs"]["use_cache"] is False
    assert captured["kwargs"]["logits_to_keep"] == 1
    assert captured["outputs"].past_key_values is None
    assert captured["outputs"].logits.shape == (batch, 1, VOCAB_SIZE)


def test_features_are_pre_final_norm(backbone):
    """Pin the landmine: backbone features must be PRE-final-RMSNorm. If they
    ever came back post-norm (e.g. by switching to the base model's
    last_hidden_state), applying the final norm would be a no-op."""
    vl_input = _text_only_inputs()
    with torch.no_grad():
        new = backbone(BatchFeature(data=vl_input))
        normed = backbone.model.language_model.norm(new["backbone_features"])
    assert not torch.equal(normed, new["backbone_features"])
