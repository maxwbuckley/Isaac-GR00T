# Benchmark results — RTX 5090 (WSL2), 2026-07-16

Raw results for the `perf-integration` branch benchmarks, produced by
`tools/perf/run_ab.py` / `bench_policy_ab.py` / `bench_finetune_vram.py`.

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32.6 GB, driver 596.36, **WSL2** (expect
  higher run-to-run variance than bare-metal Linux)
- Stack: torch 2.9.0+cu128, transformers 4.57.3, flash-attn 2.8.3,
  Python 3.12 (repo-pinned via `uv sync`)
- Model: `nvidia/GR00T-N1.7-3B` (real weights), embodiment
  `real_g1_relative_eef_relative_joints` (1 camera x 2 frames), batch 1,
  4 denoise steps unless stated, synthetic fixed-seed observations
- Protocol: one process per variant, interleaved ABBA rounds, warmup 10-15,
  CUDA-synchronized E2E `policy.get_action` timings

## Headline numbers (medians)

| Measurement | main | perf-integration |
|---|---|---|
| Seeded action parity | — | **bitwise identical** |
| E2E eager (n=180/side) | 145.0 ms | 138.1 ms (−4.8%, Welch t=2.08) |
| Peak host RSS at load | 15.98 GB | 11.50 GB (−28%) |
| Load time (n=6/side) | 18.4 s | 20.0 s (+9%, UNVERIFIED anomaly) |
| + `--compile` (n=30) | — | **90.7 ms / 11.0 Hz** (x1.60 vs main) |
| + denoise steps 4→2 (n=30) | — | 122.0 ms (quality unmeasured) |
| B7 K/V cache on vs off (n=60/side) | — | on 142.2 / off 136.2 ms (t=2.37) → **cache refuted here; default flipped off** (also breaks compile CUDA graphs) |

Fine-tune peak VRAM (`ft_vram.json`, demo LIBERO dataset, 8 steps, real
weights; the shipped checkpoint's trainable set is ~1.6B params):

| Recipe | Peak | Outcome |
|---|---|---|
| default (b32, fp32, fp32 AdamW) | 32.1 GB (ceiling) | FAILED (CUDA error at ceiling) |
| + b8 x 4 accum | 32.1 GB (ceiling) | FAILED |
| + `--load-bf16` | 32.1 GB (at ceiling) | completed, ~2.9x slower (WDDM spill) — marginal |
| + `--optim paged_adamw_8bit` | **22.2 GB** | comfortable |

## Caveats

- WSL2 stdev is ~15-20% of mean for eager runs; treat sub-5% deltas with the
  Welch statistics in `summary.json`, not the point estimates.
- The +9% load-time reading contradicts the direct-bf16-load prediction and
  has NOT been re-verified; the RSS reduction is unambiguous.
- Quality (loss curves / closed-loop success) was NOT validated for
  `--load-bf16` / 8-bit AdamW / 2-step denoising in these runs.
