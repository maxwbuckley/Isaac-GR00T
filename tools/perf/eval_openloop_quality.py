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

"""Seeded open-loop quality eval worker for quantization A/B comparisons.

Runs one policy variant (BF16 or a --quantize spec) over dataset trajectories
and reports unnormalized action MSE/MAE against ground truth. The flow-matching
initial noise is re-seeded before every ``get_action`` call with a fixed
per-step seed, so different variants of the same checkpoint see identical noise
draws: any difference in predictions is attributable to the quantization, not
sampling variance. Raw per-chunk predictions are saved for cross-variant deltas.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--embodiment-tag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantize", default=None, help="None|nvfp4|nvfp4-wo|fp8|recipe path")
    parser.add_argument("--traj-ids", type=int, nargs="+", default=None, help="default: all")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7777)
    args = parser.parse_args()

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    import torch

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=args.model_path,
        device="cuda:0",
        quantization=args.quantize,
    )
    if args.denoising_steps is not None:
        policy.model.action_head.num_inference_timesteps = args.denoising_steps

    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    traj_ids = args.traj_ids if args.traj_ids is not None else list(range(len(loader)))

    obs_configs = deepcopy(modality)
    obs_configs.pop("action")
    action_keys = modality["action"].modality_keys
    state_keys = modality["state"].modality_keys

    result = {
        "variant": args.variant_name,
        "quantize": args.quantize,
        "dataset": args.dataset_path,
        "embodiment_tag": tag.value,
        "seed": args.seed,
        "execution_horizon": args.execution_horizon,
        "trajectories": {},
    }
    if getattr(policy, "quantization_plan", None):
        counts: dict[str, int] = {}
        for fmt in policy.quantization_plan.values():
            counts[fmt] = counts.get(fmt, 0) + 1
        result["quantization_plan_counts"] = counts

    all_pred, all_gt = [], []
    for traj_id in traj_ids:
        traj = loader[traj_id]
        actual_steps = min(args.steps, len(traj))
        preds = []
        for step in range(0, actual_steps, args.execution_horizon):
            data_point = extract_step_data(traj, step, obs_configs, tag)
            obs = {f"state.{k}": v for k, v in data_point.states.items()}
            for k, v in data_point.images.items():
                obs[f"video.{k}"] = np.array(v)
            for language_key in modality["language"].modality_keys:
                obs[language_key] = data_point.text
            parsed = parse_observation_gr00t(obs, modality)
            # Identical noise draw across variants for this (traj, step).
            torch.manual_seed(args.seed + 100_003 * traj_id + step)
            chunk, _ = policy.get_action(parsed)
            for j in range(args.execution_horizon):
                preds.append(
                    np.concatenate(
                        [np.atleast_1d(np.atleast_1d(chunk[key][0])[j]) for key in action_keys]
                    )
                )
        gt = np.concatenate(
            [np.vstack([np.asarray(a) for a in traj[f"action.{key}"]]) for key in action_keys],
            axis=-1,
        )[:actual_steps]
        pred = np.asarray(preds)[:actual_steps]
        assert gt.shape == pred.shape, (gt.shape, pred.shape)
        mse = float(np.mean((gt - pred) ** 2))
        mae = float(np.mean(np.abs(gt - pred)))
        result["trajectories"][str(traj_id)] = {"mse": mse, "mae": mae, "steps": int(actual_steps)}
        all_pred.append(pred)
        all_gt.append(gt)
        print(f"[{args.variant_name}] traj {traj_id}: MSE {mse:.6f} MAE {mae:.6f}")

    pred_cat = np.concatenate(all_pred)
    gt_cat = np.concatenate(all_gt)
    result["overall_mse"] = float(np.mean((gt_cat - pred_cat) ** 2))
    result["overall_mae"] = float(np.mean(np.abs(gt_cat - pred_cat)))
    _ = state_keys  # kept for symmetry with open_loop_eval; states not scored

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    np.savez_compressed(str(out).replace(".json", "_preds.npz"), pred=pred_cat, gt=gt_cat)
    print(
        f"[{args.variant_name}] OVERALL MSE {result['overall_mse']:.6f} "
        f"MAE {result['overall_mae']:.6f} -> {out}"
    )


if __name__ == "__main__":
    main()
