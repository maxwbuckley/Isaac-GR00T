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

from dataclasses import dataclass
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import numpy as np
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts.

    A dataset's ``meta/modality.json`` is a different (data-layout) schema and is
    not accepted here — point such users at a .py config instead of letting the
    ``ModalityConfig(**v)`` unpack raise a bare ``TypeError``.
    """
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig "
            f"fields (delta_indices, modality_keys, ...). A dataset's meta/modality.json uses a "
            f"different schema; pass a .py modality config (e.g. examples/SO100/so100_config.py) instead."
        ) from exc


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    # Latency / model-serving configs (Gr00tPolicy only; ignored for ReplayPolicy)
    denoising_steps: int | None = None
    """Number of flow-matching denoising steps for the action head (must be >= 1).
    None keeps the checkpoint default (4). WARNING: fewer steps directly increase
    flow-matching integration error and can significantly degrade task success,
    not just add noise — validate closed-loop success rate at the reduced step
    count before deploying (measured on RTX 5090: 4 -> 2 steps saves ~12% E2E
    latency; the quality cost has NOT been characterized)."""

    warmup: bool = True
    """Run one dummy inference through the policy before serving so the first real
    request doesn't pay CUDA context init / cuDNN autotune / lazy processor init.
    Disable with --no-warmup."""

    compile: bool = False
    """torch.compile the DiT forward (mode='max-autotune') and enable cuDNN autotune,
    mirroring scripts/deployment/benchmark_inference.py. Compilation happens before
    warmup so warmup absorbs the JIT latency. Enable with --compile."""


def _warmup_state_dims(policy: Gr00tPolicy) -> dict[str, int]:
    """Best-effort per-key state dims for a warmup observation.

    The modality config lists state keys but not their dimensions, so read them
    from the processor's state/action normalization metadata
    (``state_action_processor.norm_params[<tag>]["state"][<key>]["dim"]``), which
    is derived from the checkpoint's dataset statistics. Fall back to 1 for any
    key whose metadata is missing — warmup is best-effort by design.
    """
    state_keys = policy.get_modality_config()["state"].modality_keys
    try:
        norm_params = policy.processor.state_action_processor.norm_params[
            policy.embodiment_tag.value
        ]["state"]
    except (AttributeError, KeyError, TypeError):
        norm_params = {}

    dims = {}
    for key in state_keys:
        try:
            dims[key] = int(norm_params[key]["dim"])
        except (KeyError, TypeError, ValueError):
            dims[key] = 1
    return dims


def build_warmup_observation(policy: Gr00tPolicy) -> dict[str, Any]:
    """Build a dummy observation matching the policy's modality config.

    Shapes follow ``Gr00tPolicy.check_observation``: video values are uint8
    (B, T, H, W, 3) arrays, state values are float32 (B, T, D) arrays, and
    language values are (B, T) nested lists of strings, with T taken from each
    modality's delta_indices and B=1.
    """
    modality_configs = policy.get_modality_config()
    video_horizon = len(modality_configs["video"].delta_indices)
    state_horizon = len(modality_configs["state"].delta_indices)
    language_horizon = len(modality_configs["language"].delta_indices)
    state_dims = _warmup_state_dims(policy)

    return {
        "video": {
            key: np.zeros((1, video_horizon, 256, 256, 3), dtype=np.uint8)
            for key in modality_configs["video"].modality_keys
        },
        "state": {
            key: np.zeros((1, state_horizon, state_dims[key]), dtype=np.float32)
            for key in modality_configs["state"].modality_keys
        },
        "language": {
            key: [["warmup"] * language_horizon]
            for key in modality_configs["language"].modality_keys
        },
    }


def warmup_policy(policy: Gr00tPolicy) -> None:
    """Run one dummy inference so the first real request doesn't pay init costs.

    A warmup failure must never prevent serving: any exception is logged loudly
    and swallowed.
    """
    start = time.perf_counter()
    try:
        observation = build_warmup_observation(policy)
        policy.get_action(observation)
    except Exception:
        logging.warning(
            "Warmup inference failed after %.2fs; the server will still start, but the "
            "first real request will pay one-time initialization costs.",
            time.perf_counter() - start,
            exc_info=True,
        )
        return
    print(f"  Warmup inference completed in {time.perf_counter() - start:.2f}s")


def apply_server_optimizations(config: ServerConfig, policy: Gr00tPolicy) -> None:
    """Apply --denoising-steps / --compile / --warmup to a freshly built Gr00tPolicy.

    Order matters: denoising steps and compilation are applied first so the
    warmup inference runs the final configuration (and absorbs torch.compile's
    JIT latency).
    """
    import torch

    if config.denoising_steps is not None:
        if config.denoising_steps < 1:
            raise ValueError(f"--denoising-steps must be >= 1; got {config.denoising_steps}.")
        policy.model.action_head.num_inference_timesteps = config.denoising_steps
    effective_steps = getattr(policy.model.action_head, "num_inference_timesteps", None)
    source = "checkpoint default" if config.denoising_steps is None else "--denoising-steps"
    print(f"  Denoising steps: {effective_steps} ({source})")

    if config.compile:
        # Mirror scripts/deployment/benchmark_inference.py: compile the DiT
        # forward and enable cuDNN autotune.
        policy.model.action_head.model.forward = torch.compile(
            policy.model.action_head.model.forward, mode="max-autotune"
        )
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print("  torch.compile: enabled (mode='max-autotune', cuDNN benchmark on)")

    if config.warmup:
        warmup_policy(policy)


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
        apply_server_optimizations(config, policy)
    elif config.dataset_path is not None:
        if config.denoising_steps is not None or config.compile:
            raise ValueError(
                "--denoising-steps and --compile only apply to model serving "
                "(--model-path); they have no effect on a ReplayPolicy (--dataset-path)."
            )
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                modality_configs = _load_json_modality_configs(config_path)
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    with PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
