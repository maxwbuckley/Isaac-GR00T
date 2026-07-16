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
Test the run_gr00t_server.py latency flags (--denoising-steps, --warmup, --compile).

Uses mocked model and processor (in the style of tests/gr00t/policy/test_gr00t_policy.py)
to avoid downloading checkpoints or requiring a GPU.
"""

import logging
from unittest.mock import MagicMock, patch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.eval.run_gr00t_server import (
    ServerConfig,
    apply_server_optimizations,
    build_warmup_observation,
    main,
)
import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


EMBODIMENT = "libero_sim"

VIDEO_KEYS = ["observation.images.rgb.head_256_256", "observation.images.rgb.left_wrist_256_256"]
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANGUAGE_KEY = "annotation.human.action.task_description"

# Per-key state dims as they would appear in the checkpoint's dataset statistics.
STATE_DIMS = {k: 1 for k in STATE_KEYS[:-1]} | {"gripper": 2}


def _build_modality_configs():
    return {
        EMBODIMENT: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
            "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
            "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=ACTION_KEYS),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
        }
    }


def _build_norm_params():
    return {
        EMBODIMENT: {
            "state": {k: {"dim": np.array(d)} for k, d in STATE_DIMS.items()},
        }
    }


def _make_mock_policy():
    """Lightweight MagicMock policy exposing exactly what the server flags touch."""
    policy = MagicMock()
    policy.embodiment_tag = EmbodimentTag.resolve(EMBODIMENT)
    policy.get_modality_config.return_value = _build_modality_configs()[EMBODIMENT]
    policy.processor.state_action_processor.norm_params = _build_norm_params()
    policy.model.action_head.num_inference_timesteps = 4
    return policy


@pytest.fixture
def real_policy():
    """Real Gr00tPolicy with mocked AutoModel/AutoProcessor, so check_observation
    (strict validation) runs for real against the warmup observation."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")
    mock_model.dtype = torch.bfloat16
    mock_model.get_action = MagicMock(
        return_value=BatchFeature(data={"action_pred": torch.randn(1, 16, 7)})
    )
    mock_model.action_head.num_inference_timesteps = 4

    mock_processor = MagicMock()
    mock_processor.modality_configs = _build_modality_configs()
    mock_processor.get_modality_configs.return_value = _build_modality_configs()
    mock_processor.state_action_processor = MagicMock()
    mock_processor.state_action_processor.norm_params = _build_norm_params()
    mock_processor.eval = MagicMock()
    mock_processor.training = False
    mock_processor.collator = MagicMock()

    def fake_decode_action(action, embodiment_tag, state=None):
        return {k: np.zeros((1, 16, 1), dtype=np.float32) for k in ACTION_KEYS}

    mock_processor.decode_action = MagicMock(side_effect=fake_decode_action)

    with (
        patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
        patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.exists", return_value=True),
    ):
        MockAutoModel.from_pretrained.return_value = mock_model
        MockAutoProcessor.from_pretrained.return_value = mock_processor

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        p = Gr00tPolicy(
            embodiment_tag=EMBODIMENT,
            model_path="/fake/path",
            device="cpu",
        )
    return p


class TestServerConfigDefaults:
    def test_defaults_preserve_current_behavior(self):
        config = ServerConfig()
        assert config.denoising_steps is None
        assert config.warmup is True
        assert config.compile is False

    def test_defaults_do_not_touch_denoising_or_compile(self):
        policy = _make_mock_policy()
        with patch("torch.compile") as mock_compile:
            apply_server_optimizations(ServerConfig(warmup=False), policy)
        assert policy.model.action_head.num_inference_timesteps == 4
        mock_compile.assert_not_called()


class TestDenoisingSteps:
    def test_mutation_applied(self):
        policy = _make_mock_policy()
        apply_server_optimizations(ServerConfig(denoising_steps=2, warmup=False), policy)
        assert policy.model.action_head.num_inference_timesteps == 2

    def test_none_keeps_checkpoint_default(self):
        policy = _make_mock_policy()
        apply_server_optimizations(ServerConfig(denoising_steps=None, warmup=False), policy)
        assert policy.model.action_head.num_inference_timesteps == 4

    @pytest.mark.parametrize("steps", [0, -1])
    def test_invalid_steps_rejected(self, steps):
        policy = _make_mock_policy()
        with pytest.raises(ValueError, match="denoising-steps"):
            apply_server_optimizations(ServerConfig(denoising_steps=steps), policy)
        # The invalid value must not be written to the model.
        assert policy.model.action_head.num_inference_timesteps == 4

    def test_replay_policy_rejects_denoising_steps(self):
        with pytest.raises(ValueError, match="denoising-steps"):
            main(ServerConfig(dataset_path="/fake/dataset", denoising_steps=2))

    def test_replay_policy_rejects_compile(self):
        with pytest.raises(ValueError, match="compile"):
            main(ServerConfig(dataset_path="/fake/dataset", compile=True))


class TestWarmup:
    def test_warmup_invoked_exactly_once_when_enabled(self):
        policy = _make_mock_policy()
        apply_server_optimizations(ServerConfig(warmup=True), policy)
        assert policy.get_action.call_count == 1

    def test_warmup_not_invoked_when_disabled(self):
        policy = _make_mock_policy()
        apply_server_optimizations(ServerConfig(warmup=False), policy)
        policy.get_action.assert_not_called()

    def test_warmup_observation_is_well_formed(self):
        policy = _make_mock_policy()
        apply_server_optimizations(ServerConfig(warmup=True), policy)

        (observation,), _ = policy.get_action.call_args
        for key in VIDEO_KEYS:
            video = observation["video"][key]
            assert video.dtype == np.uint8
            assert video.ndim == 5  # (B, T, H, W, C)
            assert video.shape[0] == 1
            assert video.shape[1] == 1  # len(delta_indices)
            assert video.shape[-1] == 3
        for key in STATE_KEYS:
            state = observation["state"][key]
            assert state.dtype == np.float32
            assert state.shape == (1, 1, STATE_DIMS[key])
        assert observation["language"][LANGUAGE_KEY] == [["warmup"]]

    def test_warmup_observation_passes_real_strict_validation(self, real_policy):
        """The dummy observation must survive Gr00tPolicy.check_observation and the
        full (mocked-model) get_action pipeline, exactly like a real client request."""
        with patch.object(
            real_policy, "get_action", wraps=real_policy.get_action
        ) as spy_get_action:
            apply_server_optimizations(ServerConfig(warmup=True), real_policy)
        assert spy_get_action.call_count == 1

    def test_warmup_failure_does_not_crash_startup(self, caplog):
        policy = _make_mock_policy()
        policy.get_action.side_effect = RuntimeError("CUDA out of memory")
        with caplog.at_level(logging.WARNING):
            apply_server_optimizations(ServerConfig(warmup=True), policy)
        assert policy.get_action.call_count == 1
        assert "warmup" in caplog.text.lower()

    def test_state_dims_fall_back_to_one_without_metadata(self):
        policy = _make_mock_policy()
        policy.processor.state_action_processor.norm_params = {}
        observation = build_warmup_observation(policy)
        for key in STATE_KEYS:
            assert observation["state"][key].shape == (1, 1, 1)


class TestCompile:
    def test_compile_enabled_compiles_dit_forward_and_sets_cudnn_benchmark(self):
        policy = _make_mock_policy()
        original_forward = policy.model.action_head.model.forward
        compiled_forward = MagicMock(name="compiled_forward")
        original_benchmark = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = False
            with (
                patch("torch.compile", return_value=compiled_forward) as mock_compile,
                patch("torch.cuda.is_available", return_value=True),
            ):
                apply_server_optimizations(ServerConfig(compile=True, warmup=False), policy)
            mock_compile.assert_called_once_with(original_forward, mode="max-autotune")
            assert policy.model.action_head.model.forward is compiled_forward
            assert torch.backends.cudnn.benchmark is True
        finally:
            torch.backends.cudnn.benchmark = original_benchmark

    def test_compile_disabled_by_default(self):
        policy = _make_mock_policy()
        original_benchmark = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = False
            with patch("torch.compile") as mock_compile:
                apply_server_optimizations(ServerConfig(warmup=False), policy)
            mock_compile.assert_not_called()
            assert torch.backends.cudnn.benchmark is False
        finally:
            torch.backends.cudnn.benchmark = original_benchmark

    def test_compile_happens_before_warmup(self):
        """Warmup must run through the compiled forward so it absorbs JIT latency."""
        policy = _make_mock_policy()
        call_order = []
        policy.get_action.side_effect = lambda *a, **k: call_order.append("warmup") or (
            {},
            {},
        )
        with (
            patch(
                "torch.compile",
                side_effect=lambda *a, **k: call_order.append("compile") or MagicMock(),
            ),
            patch("torch.cuda.is_available", return_value=False),
        ):
            apply_server_optimizations(ServerConfig(compile=True, warmup=True), policy)
        assert call_order == ["compile", "warmup"]
