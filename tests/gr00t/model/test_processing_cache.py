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
Test the inference-time memoization of constant text processing in
Gr00tN1d7Processor: language normalization, chat-template rendering, and the
text-side of VLM tokenization.

Uses the fixture configs in tests/fixtures/processor_config/ with a
deterministic stub VLM processor (no gated tokenizer download needed),
following the pattern of test_gr00t_processor.py. The stub counts calls so
tests can assert exactly which steps were skipped on a cache hit, while the
real cache-key logic in processing_gr00t_n1d7 is exercised end to end.
"""

import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
import numpy as np
from PIL import Image
import torch
from transformers.feature_extraction_utils import BatchFeature


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"
TAG = EmbodimentTag(EMBODIMENT)

VIDEO_KEYS = [
    "observation.images.rgb.head_256_256",
    "observation.images.rgb.left_wrist_256_256",
]
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANGUAGE_KEY = "annotation.human.action.task_description"


class _StubVLMProcessor:
    """Deterministic stand-in for Qwen3VLProcessor with call counters.

    - ``apply_chat_template`` renders one placeholder per image plus the text.
    - ``image_processor`` produces pixel_values that depend on image CONTENT
      (so tests can prove image features are recomputed on cache hits) and an
      image_grid_thw that depends on image SIZE.
    - ``__call__`` tokenizes each text deterministically (char codes, padded
      left) and embeds the image features, mirroring the fused structure of
      the real processor.
    """

    image_token = "<|image_pad|>"

    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="left")
        self.counters = {"chat_template": 0, "full_call": 0, "image_processor": 0}
        stub = self

        class _ImageProcessor:
            merge_size = 2

            def __call__(self, images=None, return_tensors="pt"):
                stub.counters["image_processor"] += 1
                pixel_values = torch.stack(
                    [
                        torch.from_numpy(np.asarray(img)[::32, ::32].astype(np.float32).reshape(-1))
                        for img in images
                    ]
                )
                grid = torch.tensor(
                    [[1, img.height // 32, img.width // 32] for img in images],
                    dtype=torch.long,
                )
                return BatchFeature(data={"pixel_values": pixel_values, "image_grid_thw": grid})

        self.image_processor = _ImageProcessor()

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        self.counters["chat_template"] += 1
        content = conversation[0]["content"]
        n_images = sum(1 for c in content if c["type"] == "image")
        text = next(c["text"] for c in content if c["type"] == "text")
        return self.image_token * n_images + text

    def __call__(self, text=None, images=None, return_tensors="pt", padding=True):
        self.counters["full_call"] += 1
        image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        sequences = [[ord(ch) % 251 + 3 for ch in t] for t in text]
        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.zeros(len(sequences), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for row, seq in enumerate(sequences):  # left padding
            input_ids[row, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, max_len - len(seq) :] = 1
        return BatchFeature(
            data={"input_ids": input_ids, "attention_mask": attention_mask, **image_inputs}
        )


def _make_processor():
    """Fresh eval-mode processor with an isolated stub VLM processor per component."""
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processor_module

    with patch.object(
        processor_module, "build_processor", side_effect=lambda *a, **k: _StubVLMProcessor()
    ):
        proc = processor_module.Gr00tN1d7Processor.from_pretrained(FIXTURE_DIR)
    proc.eval()
    return proc


def _state_dims():
    with open(FIXTURE_DIR / "statistics.json") as f:
        statistics = json.load(f)
    return {k: len(statistics[EMBODIMENT]["state"][k]["min"]) for k in STATE_KEYS}


def _make_observation(instruction: str, seed: int = 0) -> dict:
    """Batched (B=1) observation for process_observation, deterministic per seed."""
    rng = np.random.default_rng(seed)
    obs = {}
    for key in VIDEO_KEYS:
        obs[f"video.{key}"] = rng.integers(0, 256, (1, 1, 256, 256, 3), dtype=np.uint8)
    for key, dim in _state_dims().items():
        obs[f"state.{key}"] = rng.standard_normal((1, dim)).astype(np.float32)
    obs[LANGUAGE_KEY] = [instruction]
    return obs


def _make_messages(instruction: str, seed: int = 0) -> list[dict]:
    """EPISODE_STEP message with deterministic synthetic VLAStepData."""
    rng = np.random.default_rng(seed)
    with open(FIXTURE_DIR / "statistics.json") as f:
        statistics = json.load(f)
    images = {k: [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)] for k in VIDEO_KEYS}
    states = {
        k: rng.standard_normal((1, len(statistics[EMBODIMENT]["state"][k]["min"]))).astype(
            np.float32
        )
        for k in STATE_KEYS
    }
    actions = {
        k: rng.standard_normal((16, len(statistics[EMBODIMENT]["action"][k]["min"]))).astype(
            np.float32
        )
        for k in ACTION_KEYS
    }
    step_data = VLAStepData(
        images=images,
        states=states,
        actions=actions,
        text=instruction,
        embodiment=TAG,
    )
    return [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]


def _assert_tensor_dicts_equal(actual: dict, expected: dict):
    assert set(actual.keys()) == set(expected.keys())
    for key in expected:
        a, e = actual[key], expected[key]
        if isinstance(e, torch.Tensor):
            assert isinstance(a, torch.Tensor), key
            assert a.dtype == e.dtype, key
            assert torch.equal(a, e), f"Mismatch for key {key!r}"
        else:
            assert a == e, f"Mismatch for key {key!r}"


class TestProcessObservationCache:
    def test_second_identical_call_skips_text_work_and_matches_cold_reference(self):
        instruction = "Pick up the apple!"
        proc = _make_processor()

        out1 = proc.process_observation(_make_observation(instruction), TAG)
        counters = proc.processor.counters
        # Miss: template rendered once, full tokenization once, image features
        # computed twice (once for the cache key, once inside the full call).
        assert counters["chat_template"] == 1
        assert counters["full_call"] == 1
        assert counters["image_processor"] == 2

        out2 = proc.process_observation(_make_observation(instruction), TAG)
        # Hit: no new template render, no new tokenization — only image features.
        assert counters["chat_template"] == 1
        assert counters["full_call"] == 1
        assert counters["image_processor"] == 3

        # Cache-cold reference from a fresh processor: outputs must be bitwise equal.
        cold = _make_processor().process_observation(_make_observation(instruction), TAG)
        _assert_tensor_dicts_equal(dict(out1), dict(cold))
        _assert_tensor_dicts_equal(dict(out2), dict(cold))

    def test_different_instruction_misses_and_differs(self):
        proc = _make_processor()
        out_a = proc.process_observation(_make_observation("Pick up the apple!"), TAG)
        counters = proc.processor.counters
        out_b = proc.process_observation(_make_observation("Close the drawer."), TAG)
        assert counters["chat_template"] == 2
        assert counters["full_call"] == 2
        assert not torch.equal(out_a["input_ids"], out_b["input_ids"])

        # And the new instruction matches its own cold reference.
        cold = _make_processor().process_observation(_make_observation("Close the drawer."), TAG)
        _assert_tensor_dicts_equal(dict(out_b), dict(cold))

    def test_mutating_returned_tensors_does_not_corrupt_cache(self):
        instruction = "Pick up the apple!"
        cold = _make_processor().process_observation(_make_observation(instruction), TAG)

        proc = _make_processor()
        out1 = proc.process_observation(_make_observation(instruction), TAG)  # miss
        out1["input_ids"].add_(17)
        out1["attention_mask"].zero_()

        out2 = proc.process_observation(_make_observation(instruction), TAG)  # hit
        assert torch.equal(out2["input_ids"], cold["input_ids"])
        assert torch.equal(out2["attention_mask"], cold["attention_mask"])

        # Mutating a hit-path result must not corrupt subsequent hits either.
        out2["input_ids"].zero_()
        out3 = proc.process_observation(_make_observation(instruction), TAG)  # hit
        assert torch.equal(out3["input_ids"], cold["input_ids"])


class TestCallAndCollatorCache:
    """The Gr00tPolicy inference path: processor.__call__ then collator tokenization."""

    def test_collator_skips_tokenization_on_second_identical_step(self):
        proc = _make_processor()
        processed1 = proc(_make_messages("Pick up the apple!"))
        processed2 = proc(_make_messages("Pick up the apple!"))
        # Template rendered once (memoized inside __call__ / _apply_vlm_processing).
        assert proc.processor.counters["chat_template"] == 1

        collator = proc.collator
        batch1 = collator([processed1])["inputs"]
        assert collator.processor.counters["full_call"] == 1
        batch2 = collator([processed2])["inputs"]
        # Hit: tokenization skipped, image features recomputed.
        assert collator.processor.counters["full_call"] == 1
        assert collator.processor.counters["image_processor"] == 3

        # Cache-cold reference through a fresh processor + collator.
        cold_proc = _make_processor()
        cold = cold_proc.collator([cold_proc(_make_messages("Pick up the apple!"))])["inputs"]
        _assert_tensor_dicts_equal(dict(batch1), dict(cold))
        _assert_tensor_dicts_equal(dict(batch2), dict(cold))


class TestTrainingModeGuard:
    def test_training_mode_never_consults_caches(self):
        proc = _make_processor()
        proc.train()
        assert proc.collator.vlm_tokenize_cache is None

        # Full training-style processing, twice with identical inputs.
        proc(_make_messages("Pick up the apple!"))
        proc(_make_messages("Pick up the apple!"))
        # No memoization: the template is rendered on every call.
        assert proc.processor.counters["chat_template"] == 2
        # The caches were never consulted nor populated.
        for cache in proc._text_caches():
            assert len(cache) == 0
            assert cache.hits == 0
            assert cache.misses == 0

    def test_train_eval_toggles_collator_cache_and_clears_state(self):
        proc = _make_processor()
        assert proc.collator.vlm_tokenize_cache is proc._vlm_tokenize_cache

        # Populate the caches in eval mode.
        proc.process_observation(_make_observation("Pick up the apple!"), TAG)
        assert len(proc._language_cache) == 1
        assert len(proc._chat_template_cache) == 1
        assert len(proc._vlm_tokenize_cache) == 1

        proc.train()
        assert proc.collator.vlm_tokenize_cache is None
        for cache in proc._text_caches():
            assert len(cache) == 0

        proc.eval()
        assert proc.collator.vlm_tokenize_cache is proc._vlm_tokenize_cache


class TestFormalizeLanguageCache:
    def test_normalization_parity_and_hit(self):
        proc = _make_processor()
        raw = "Pick UP, the (red) apple!"
        expected = re.sub(r"[^\w\s]", "", raw.lower())
        assert proc._formalize_language(raw) == expected
        assert proc._formalize_language(raw) == expected
        assert proc._language_cache.hits == 1
        assert proc._language_cache.misses == 1

    def test_formalize_language_disabled_is_passthrough(self):
        proc = _make_processor()
        proc.formalize_language = False
        raw = "Pick UP, the (red) apple!"
        assert proc._formalize_language(raw) == raw
        assert len(proc._language_cache) == 0


class TestTokenizeVlmInputsFunction:
    """Function-level checks of the split tokenize/image-feature cache."""

    @staticmethod
    def _image(seed: int, size: int = 64) -> Image.Image:
        rng = np.random.default_rng(seed)
        return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))

    def test_hit_reuses_text_tensors_but_recomputes_image_features(self):
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import _LRUCache, _tokenize_vlm_inputs

        stub = _StubVLMProcessor()
        cache = _LRUCache()
        texts = ["<|image_pad|>pick up the apple"]

        out1 = _tokenize_vlm_inputs(stub, texts, [self._image(1)], cache)
        out2 = _tokenize_vlm_inputs(stub, texts, [self._image(2)], cache)
        assert stub.counters["full_call"] == 1  # second call hit the cache
        assert torch.equal(out1["input_ids"], out2["input_ids"])
        assert torch.equal(out1["attention_mask"], out2["attention_mask"])
        # Image features reflect the NEW image content, not the cached step's.
        assert not torch.equal(out1["pixel_values"], out2["pixel_values"])

        # A different image grid (resolution change) must be a miss: the real
        # processor expands a grid-dependent number of image-pad tokens.
        _tokenize_vlm_inputs(stub, texts, [self._image(3, size=128)], cache)
        assert stub.counters["full_call"] == 2

        # A different text must be a miss.
        _tokenize_vlm_inputs(stub, ["<|image_pad|>close the drawer"], [self._image(1)], cache)
        assert stub.counters["full_call"] == 3

    def test_disabled_cache_is_single_unmodified_call(self):
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import _tokenize_vlm_inputs

        stub = _StubVLMProcessor()
        out = _tokenize_vlm_inputs(stub, ["<|image_pad|>hello"], [self._image(0)], None)
        assert stub.counters["full_call"] == 1
        # No extra image_processor pre-pass when the cache is disabled.
        assert stub.counters["image_processor"] == 1
        assert "input_ids" in out and "pixel_values" in out

    def test_lru_eviction_is_bounded(self):
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import _LRUCache

        cache = _LRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert len(cache) == 2
        assert cache.get("a") is None  # evicted
        assert cache.get("b") == 2
        assert cache.get("c") == 3
