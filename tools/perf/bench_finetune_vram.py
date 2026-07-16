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

"""Fine-tuning peak-VRAM matrix driver.

Runs launch_finetune.py once per memory recipe (a few optimizer steps each) and
records peak GPU memory by sampling ``nvidia-smi`` while the run is alive. Each
recipe is a fresh subprocess; a CUDA OOM is recorded as a result ("OOM"), not a
driver failure — on cards near the documented ~35 GB peak, the batch-32 default
OOMing IS the finding.

Requires an otherwise-idle GPU (total memory.used sampling).

Usage:
  python tools/perf/bench_finetune_vram.py \
    --repo /path/to/tree --model-path <ckpt> --dataset demo_data/libero_demo \
    --python /home/maxwb/venvs/isaac-gr00t/bin/python --out /tmp/ft_vram.json
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time


RECIPES = {
    "baseline_b32": ["--global_batch_size", "32"],
    "accum_b8x4": ["--global_batch_size", "8", "--gradient_accumulation_steps", "4"],
    "accum_bf16": [
        "--global_batch_size", "8", "--gradient_accumulation_steps", "4", "--load-bf16",
    ],
    "accum_bf16_adam8bit": [
        "--global_batch_size", "8", "--gradient_accumulation_steps", "4", "--load-bf16",
        "--optim", "paged_adamw_8bit",
    ],
}


def sample_gpu_mib(stop_event, peaks):
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            peaks.append(int(out.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="code tree to benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--embodiment-tag", default="libero_sim")
    parser.add_argument("--python", default="python")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--recipes", nargs="*", default=list(RECIPES.keys()))
    args = parser.parse_args()

    results = {}
    for name in args.recipes:
        flags = RECIPES[name]
        out_dir = Path(args.out).parent / f"ft_{name}"
        cmd = [
            args.python,
            str(Path(args.repo) / "gr00t/experiment/launch_finetune.py"),
            "--base_model_path", args.model_path,
            "--dataset_path", args.dataset,
            "--embodiment_tag", args.embodiment_tag,
            "--output_dir", str(out_dir),
            "--max_steps", str(args.max_steps),
            "--save_steps", "999999",
            "--dataloader_num_workers", "4",
            "--num_gpus", "1",
            *flags,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = args.repo
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

        peaks: list[int] = []
        stop = threading.Event()
        sampler = threading.Thread(target=sample_gpu_mib, args=(stop, peaks), daemon=True)
        sampler.start()
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=args.repo)
        elapsed = time.perf_counter() - t0
        stop.set()
        sampler.join(timeout=5)

        oom = "CUDA out of memory" in proc.stderr or "OutOfMemoryError" in proc.stderr
        results[name] = {
            "flags": flags,
            "returncode": proc.returncode,
            "oom": oom,
            "peak_gpu_mib": max(peaks) if peaks else None,
            "wall_seconds": elapsed,
            "stderr_tail": proc.stderr[-1500:] if proc.returncode != 0 else "",
        }
        status = "OOM" if oom else ("OK" if proc.returncode == 0 else f"rc={proc.returncode}")
        print(f"{name:22s} {status:8s} peak {results[name]['peak_gpu_mib']} MiB "
              f"({elapsed:.0f}s)")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
