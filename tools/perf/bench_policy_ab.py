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

"""Single-variant E2E Gr00tPolicy benchmark worker.

One process measures exactly one code-tree/flag combination and writes a JSON
result. The orchestrator (run_ab.py) launches interleaved worker pairs with
PYTHONPATH selecting the code tree, so no variant benefits from a warmer OS
state than its counterpart.

Measured per process:
- model load wall time, peak host RSS, CUDA memory after load
- N end-to-end policy.get_action latencies (CUDA-synchronized) after W warmups
- one RNG-seeded action output saved for cross-variant bitwise parity checks

Observations are synthetic (fixed-seed uint8 frames + normalized states built
from the checkpoint's statistics.json) so every variant sees byte-identical
inputs without needing a dataset on disk. Latency is content-independent for
this pipeline (fixed image geometry and token counts), so synthetic inputs are
valid for A/B latency comparison; closed-loop quality is NOT measured here.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np


def build_observation(policy, snapshot_dir: str, batch_size: int, instruction: str):
    """Deterministic observation matching the policy's modality config."""
    modality = policy.get_modality_config()
    stats = json.load(open(Path(snapshot_dir) / "statistics.json"))
    emb_stats = stats[policy.embodiment_tag.value]

    rng = np.random.default_rng(1234)
    video = {}
    for key in modality["video"].modality_keys:
        t = len(modality["video"].delta_indices)
        video[key] = rng.integers(0, 255, size=(batch_size, t, 256, 256, 3), dtype=np.uint8)

    state = {}
    for key in modality["state"].modality_keys:
        dim = len(np.atleast_1d(emb_stats["state"][key]["mean"]))
        t = len(modality["state"].delta_indices)
        # Values at the per-key mean: guaranteed in-distribution for normalization.
        mean = np.asarray(emb_stats["state"][key]["mean"], dtype=np.float32).reshape(1, 1, dim)
        state[key] = np.tile(mean, (batch_size, t, 1))

    language_key = policy.language_key
    language = {language_key: [[instruction]] * batch_size}
    return {"video": video, "state": state, "language": language}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="libero_sim")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--instruction", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--quantize", default=None, help="None|nvfp4|nvfp4-wo|fp8|recipe path (Blackwell only)"
    )
    args = parser.parse_args()

    import os

    if args.disable_kv_cache:
        os.environ["GR00T_DISABLE_DIT_KV_CACHE"] = "1"

    import gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    import torch

    result = {
        "variant": args.variant_name,
        "gr00t_file": gr00t.__file__,  # proves which tree ran
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": {
            k: getattr(args, k)
            for k in (
                "iters",
                "warmup",
                "batch_size",
                "denoising_steps",
                "disable_kv_cache",
                "quantize",
            )
        },
        "compile": args.compile,
    }

    t0 = time.perf_counter()
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        quantization=args.quantize,
    )
    result["load_seconds"] = time.perf_counter() - t0
    result["peak_rss_after_load_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if torch.cuda.is_available():
        result["cuda_allocated_after_load_mib"] = torch.cuda.memory_allocated() / 2**20

    if args.denoising_steps is not None:
        policy.model.action_head.num_inference_timesteps = args.denoising_steps

    if args.compile:
        # Mirrors scripts/deployment/benchmark_inference.py
        policy.model.action_head.model.forward = torch.compile(
            policy.model.action_head.model.forward, mode="max-autotune"
        )
        torch.backends.cudnn.benchmark = True

    snapshot_dir = args.model_path
    obs = build_observation(policy, snapshot_dir, args.batch_size, args.instruction)

    # Parity artifact: seeded action output (identical initial-noise draw across
    # variants -> bitwise-comparable outputs when the fixes are truly no-diff).
    torch.manual_seed(4242)
    parity_action, _info = policy.get_action(obs)
    np.savez(
        args.output.replace(".json", "_parity.npz"),
        **{k: np.asarray(v) for k, v in parity_action.items()},
    )

    for _ in range(args.warmup):
        policy.get_action(obs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    for _ in range(args.iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.get_action(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.asarray(latencies)
    result["latencies_ms"] = latencies
    result["median_ms"] = float(np.median(lat))
    result["mean_ms"] = float(lat.mean())
    result["stdev_ms"] = float(lat.std(ddof=1))
    result["hz_at_median"] = 1000.0 / float(np.median(lat))
    if torch.cuda.is_available():
        result["cuda_max_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[{args.variant_name}] median {result['median_ms']:.1f} ms "
        f"({result['hz_at_median']:.1f} Hz), mean {result['mean_ms']:.1f} "
        f"± {result['stdev_ms']:.1f} ms over {args.iters} iters"
    )


if __name__ == "__main__":
    main()
