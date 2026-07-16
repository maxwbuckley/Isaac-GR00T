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

"""A/B benchmark orchestrator for quantization variants of one code tree.

Same protocol as run_ab.py (one process per measurement, interleaved ABBA
rounds, pooled latencies, Welch's t-test), but variants differ by
bench_policy_ab.py flags (e.g. --quantize nvfp4) instead of PYTHONPATH.

Usage:
  python tools/perf/run_quant_ab.py \
    --model-path <ckpt> --embodiment-tag <tag> \
    --variant bf16 --variant nvfp4:--quantize=nvfp4 \
    --rounds 3 --iters 30 --batch-size 1 --out-dir ~/gr00t_bench/nvfp4/ab_bs1
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


def parse_variant(spec: str) -> tuple[str, list[str]]:
    """ "label" or "label:--flag=v:--flag2" -> (label, [flags])."""
    parts = spec.split(":")
    return parts[0], [p for p in parts[1:] if p]


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return float(t), float(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--tree", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        dest="variants",
        help="label[:flag[:flag...]], e.g. nvfp4:--quantize=nvfp4  (first = baseline)",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [parse_variant(v) for v in args.variants]

    pooled: dict[str, list[float]] = {label: [] for label, _ in variants}
    results = []
    for r in range(args.rounds):
        # Forward then reversed order per round cancels linear drift for any
        # variant count (ABBA when there are two).
        order = list(variants) + list(reversed(variants))
        for i, (label, flags) in enumerate(order):
            name = f"{label}_r{r}_{i}"
            out_json = str(out_dir / f"{name}.json")
            env = os.environ.copy()
            env["PYTHONPATH"] = args.tree
            cmd = [
                args.python,
                str(Path(__file__).parent / "bench_policy_ab.py"),
                "--variant-name",
                name,
                "--model-path",
                args.model_path,
                "--embodiment-tag",
                args.embodiment_tag,
                "--output",
                out_json,
                "--iters",
                str(args.iters),
                "--warmup",
                str(args.warmup),
                "--batch-size",
                str(args.batch_size),
                *(["--compile"] if args.compile else []),
                *flags,
            ]
            subprocess.run(cmd, env=env, check=True)
            res = json.load(open(out_json))
            pooled[label].extend(res["latencies_ms"])
            results.append(res)

    print("\n===== POOLED RESULTS =====")
    for label, lat in pooled.items():
        lat = np.asarray(lat)
        print(
            f"{label:20s} n={len(lat):3d}  median {np.median(lat):7.1f} ms  "
            f"mean {lat.mean():7.1f} ms  stdev {lat.std(ddof=1):5.1f} ms  "
            f"({1000.0 * args.batch_size / np.median(lat):.1f} obs/s at median)"
        )

    baseline = variants[0][0]
    stats = {}
    for label, _ in variants[1:]:
        t, df = welch(pooled[baseline], pooled[label])
        med_a, med_b = np.median(pooled[baseline]), np.median(pooled[label])
        stats[label] = {
            "welch_t": t,
            "welch_df": df,
            "median_change_pct": float(100 * (med_b - med_a) / med_a),
        }
        print(
            f"{baseline} vs {label}: median {med_a:.1f} -> {med_b:.1f} ms ({stats[label]['median_change_pct']:+.1f}%), Welch t={t:.2f}"
        )

    summary = {
        "batch_size": args.batch_size,
        "compile": args.compile,
        "pooled_median_ms": {k: float(np.median(v)) for k, v in pooled.items()},
        "pooled_mean_ms": {k: float(np.mean(v)) for k, v in pooled.items()},
        "pooled_stdev_ms": {k: float(np.std(v, ddof=1)) for k, v in pooled.items()},
        "vs_baseline": stats,
        "cuda_allocated_after_load_mib": {
            r["variant"]: r.get("cuda_allocated_after_load_mib") for r in results
        },
        "cuda_max_allocated_mib": {r["variant"]: r.get("cuda_max_allocated_mib") for r in results},
        "load_seconds": {r["variant"]: r["load_seconds"] for r in results},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
