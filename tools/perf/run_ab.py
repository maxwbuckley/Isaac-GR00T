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

"""A/B benchmark orchestrator for Gr00tPolicy code trees.

Launches bench_policy_ab.py workers as subprocesses with PYTHONPATH selecting
the code tree per variant, in interleaved order (ABBA per round) so neither
variant systematically benefits from warmer OS/page-cache state. Pools the
per-iteration latencies, reports median/mean/stdev per variant, Welch's t-test
between the two trees, and a bitwise parity comparison of the seeded action
outputs.

Usage:
  python tools/perf/run_ab.py \
    --tree-a /path/to/main --tree-b /path/to/perf-integration \
    --model-path <checkpoint snapshot dir> \
    --python /home/maxwb/venvs/isaac-gr00t/bin/python \
    --rounds 3 --iters 30 --out-dir /tmp/gr00t_bench
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


def run_worker(python, tree, name, args, out_dir, extra_flags=()):
    out_json = str(Path(out_dir) / f"{name}.json")
    env = os.environ.copy()
    env["PYTHONPATH"] = tree
    cmd = [
        python,
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
        *extra_flags,
    ]
    subprocess.run(cmd, env=env, check=True)
    return json.load(open(out_json))


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return t, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-a", required=True, help="baseline code tree (e.g. main)")
    parser.add_argument("--tree-b", required=True, help="candidate code tree")
    parser.add_argument("--label-a", default="main")
    parser.add_argument("--label-b", default="perf-integration")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="libero_sim")
    parser.add_argument("--python", default="python")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = {args.label_a: [], args.label_b: []}
    results = []
    for r in range(args.rounds):
        # ABBA within each round cancels linear drift.
        order = [
            (args.tree_a, args.label_a),
            (args.tree_b, args.label_b),
            (args.tree_b, args.label_b),
            (args.tree_a, args.label_a),
        ]
        for i, (tree, label) in enumerate(order):
            name = f"{label}_r{r}_{i}"
            res = run_worker(args.python, tree, name, args, out_dir)
            pooled[label].extend(res["latencies_ms"])
            results.append(res)

    print("\n===== POOLED RESULTS =====")
    for label, lat in pooled.items():
        lat = np.asarray(lat)
        print(
            f"{label:20s} n={len(lat):3d}  median {np.median(lat):7.1f} ms  "
            f"mean {lat.mean():7.1f} ms  stdev {lat.std(ddof=1):5.1f} ms  "
            f"({1000.0 / np.median(lat):.1f} Hz at median)"
        )
    t, df = welch(pooled[args.label_a], pooled[args.label_b])
    print(f"Welch t={t:.2f}, df={df:.0f}  (|t| > ~2 => significant at ~5% for these df)")
    med_a = np.median(pooled[args.label_a])
    med_b = np.median(pooled[args.label_b])
    print(f"Median change: {med_a:.1f} -> {med_b:.1f} ms ({100 * (med_a - med_b) / med_a:+.1f}%)")

    # Bitwise parity across trees (first worker of each label).
    pa = np.load(out_dir / f"{args.label_a}_r0_0_parity.npz")
    pb = np.load(out_dir / f"{args.label_b}_r0_1_parity.npz")
    keys_equal = set(pa.files) == set(pb.files)
    all_equal = keys_equal and all(np.array_equal(pa[k], pb[k]) for k in pa.files)
    print(
        f"Seeded action parity ({args.label_a} vs {args.label_b}): "
        f"{'BITWISE IDENTICAL' if all_equal else 'MISMATCH — investigate before trusting latency numbers'}"
    )

    summary = {
        "pooled_median_ms": {k: float(np.median(v)) for k, v in pooled.items()},
        "pooled_mean_ms": {k: float(np.mean(v)) for k, v in pooled.items()},
        "pooled_stdev_ms": {k: float(np.std(v, ddof=1)) for k, v in pooled.items()},
        "welch_t": t,
        "welch_df": df,
        "parity_bitwise_identical": bool(all_equal),
        "load_seconds": {r["variant"]: r["load_seconds"] for r in results},
        "peak_rss_mib": {r["variant"]: r["peak_rss_mib"] for r in results},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
