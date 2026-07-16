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

"""Calibration batches for post-training quantization of GR00T policies.

Batches are produced by running real dataset steps through the policy's own
processor + collator, so quantizer calibration sees exactly the activation
distributions of deployment inference. With ``with_actions=True`` the batches
also carry ground-truth action chunks, enabling the training-style forward
(``model(inputs) -> loss``) that gradient-based sensitivity scoring needs.
"""

from collections.abc import Iterator
from copy import deepcopy
import logging

import numpy as np
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import MessageType


logger = logging.getLogger(__name__)


def build_calibration_batches(
    policy,
    dataset_path: str,
    *,
    num_batches: int = 32,
    batch_size: int = 1,
    with_actions: bool = False,
    seed: int = 0,
) -> list[dict]:
    """Sample dataset steps and process them into device-resident model inputs.

    Args:
        policy: A ``Gr00tPolicy`` (provides processor, collator, modality configs).
        dataset_path: LeRobot-format dataset directory.
        num_batches: Number of batches to produce.
        batch_size: Samples per batch.
        with_actions: Include normalized GT action chunks (for loss-based scoring).
        seed: RNG seed for step sampling.

    Returns:
        List of input dicts accepted by ``Gr00tN1d7.get_action`` / ``forward``.
    """
    from gr00t.policy.gr00t_policy import _rec_to_device_dtype

    modality_configs = deepcopy(policy.modality_configs)
    if not with_actions:
        modality_configs.pop("action", None)

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path, modality_configs=policy.modality_configs
    )
    rng = np.random.default_rng(seed)

    # Sample (trajectory, step) pairs uniformly; keep clear of episode ends so the
    # forward-looking action window stays in range.
    action_horizon = (
        len(policy.modality_configs["action"].delta_indices)
        if "action" in policy.modality_configs
        else 0
    )
    pairs = []
    traj_lengths = [len(loader[i]) for i in range(len(loader))]
    for _ in range(num_batches * batch_size):
        traj_id = int(rng.integers(0, len(loader)))
        margin = action_horizon if with_actions else 1
        hi = max(1, traj_lengths[traj_id] - margin)
        pairs.append((traj_id, int(rng.integers(0, hi))))

    batches = []
    trajs = {}
    for b in range(num_batches):
        processed = []
        for s in range(batch_size):
            traj_id, step = pairs[b * batch_size + s]
            if traj_id not in trajs:
                trajs[traj_id] = loader[traj_id]
            vla_step = extract_step_data(
                trajs[traj_id],
                step,
                modality_configs,
                policy.embodiment_tag,
                allow_padding=True,
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
            processed.append(policy.processor(messages))
        collated = policy.collate_fn(processed)
        if "inputs" in collated:
            collated = collated["inputs"]
        batches.append(
            _rec_to_device_dtype(collated, device=policy.model.device, dtype=torch.bfloat16)
        )

    logger.info(
        "Built %d calibration batches (batch_size=%d, with_actions=%s) from %s",
        len(batches),
        batch_size,
        with_actions,
        dataset_path,
    )
    return batches


def forward_loop(model, batches: list[dict], *, use_get_action: bool = True) -> None:
    """Run calibration forwards over ``batches`` (modelopt ``forward_loop`` contract)."""
    with torch.no_grad():
        for inputs in batches:
            if use_get_action:
                # A GT "action" key would switch get_action into its RTC
                # inpainting mode (which requires options); calibration wants
                # the plain sampling path.
                inputs = {k: v for k, v in inputs.items() if k not in ("action", "action_mask")}
                model.get_action(inputs)
            else:
                model(dict(inputs))


def data_loader_iter(batches: list[dict]) -> Iterator[dict]:
    yield from batches
