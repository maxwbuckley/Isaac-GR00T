# NVFP4 / mixed-precision quantization for Blackwell — RTX 5090 (WSL2), 2026-07-17

Results and engineering notes for adding NVFP4 (FP4 E2M1 + FP8 block scales)
support to GR00T N1.7 inference, targeting Blackwell-class GPUs (RTX 50xx
sm_120, Jetson Thor sm_110). Produced by the overnight quantization session on
the `feat/nvfp4-blackwell` branch.

## What was added

| Piece | Where |
|---|---|
| Recipe format (per-layer nvfp4/fp8/bf16 plan, JSON) | `gr00t/quantization/recipe.py` |
| Calibration batches from real dataset steps | `gr00t/quantization/calib.py` |
| modelopt PTQ + `auto_quantize` sensitivity search | `gr00t/quantization/modelopt_ptq.py` |
| Real-kernel torchao NVFP4/FP8 inference | `gr00t/quantization/nvfp4_inference.py` |
| Policy opt-in: `Gr00tPolicy(quantization="nvfp4" \| recipe.json)` | `gr00t/policy/gr00t_policy.py` |
| Recipe-search CLI | `scripts/deployment/quantize_nvfp4.py` |
| NVFP4 DiT → ONNX (FP4 QDQ) for TensorRT | `scripts/deployment/export_dit_nvfp4_onnx.py` |
| Seeded open-loop quality eval worker | `tools/perf/eval_openloop_quality.py` |
| Quantization-variant A/B benchmark orchestrator | `tools/perf/run_quant_ab.py` (+ `--quantize` in `bench_policy_ab.py`) |
| Unit tests | `tests/gr00t/quantization/` |

## Environment

- GPU: RTX 5090 (sm_120, 32.6 GB), driver 596.36, **WSL2** (high run-to-run
  variance; treat sub-5% deltas via the Welch stats in the JSONs)
- Stack: torch 2.9.0+cu128, transformers 4.57.3, diffusers (repo pin),
  **nvidia-modelopt 0.45.0**, **torchao 0.15.0** (see "What didn't work" for
  why not 0.16/0.17), TensorRT 10.15.1
- Models: `nvidia/GR00T-N1.7-3B` (base) and `nvidia/GR00T-N1.7-LIBERO/libero_10`
- Datasets: `demo_data/droid_sample` (base ckpt, OXE_DROID tag),
  `demo_data/libero_demo` (LIBERO ckpt)
- Protocol: one process per measurement, interleaved ABBA rounds, CUDA-synced
  E2E `policy.get_action`, pooled n=60–120 per variant, Welch's t-test.
  Quality evals re-seed the flow-matching noise per (traj, step) so variants
  see identical noise draws — prediction deltas are attributable to
  quantization alone.

## Sensitivity analysis (the headline scientific result)

`mtq.auto_quantize` (gradient-based scoring of the flow-matching loss, 24 real
batches, candidates {NVFP4, FP8, skip}, effective-bits budget 6.0) on the base
checkpoint + droid_sample:

| Component | NVFP4 | FP8 | BF16 (skip) |
|---|---|---|---|
| DiT action head (228 quantizable Linears) | **220** | 7 | 1 |
| LLM (Qwen3, 12 layers, 112 Linears) | **0** | 111 | 1 |
| VL self-attention (24) | 16 | 8 | 0 |
| ViT (excluded by default; when scored) | 11 | 54 | 41 |

The split is clean and interpretable: **the DiT tolerates NVFP4 almost
everywhere, while the language model is uniformly sensitive** (auto_quantize
put every LLM layer in FP8, none in NVFP4), and the ViT is the most sensitive
(consistent with NVIDIA's own TRT pipeline keeping it FP32). Fake-quant
flow-matching loss: BF16 0.1189 → mixed recipe 0.1183 (−0.5%, i.e. no
degradation). Recipe: `droid_auto_eb6.0_recipe.json`.

## Quality (seeded open-loop MSE/MAE vs ground truth)

droid_sample, base checkpoint, 3 trajectories × 200 steps, identical noise
across variants (real torchao kernels, not fake-quant):

| Variant | MSE | MAE | Δ MSE vs BF16 |
|---|---|---|---|
| BF16 | 0.03924 | 0.11054 | — |
| NVFP4 uniform (LLM+DiT+VLSA, 360 layers) | 0.03951 | 0.11114 | +0.7% |
| FP8 uniform | 0.03939 | 0.10960 | +0.4% |
| **Mixed recipe (eb 6.0)** | **0.03848** | **0.10889** | **−1.9%** |

All within run-to-run noise on this out-of-the-box eval. The **LIBERO
finetuned checkpoint** (in-distribution, 30× lower absolute MSE) is far more
discriminative — and is where the sensitivity-aware recipe earns its keep
(libero_demo, 5 trajectories × 200 steps, execution horizon 8, LIBERO-searched
recipe: 266 nvfp4 / 153 fp8 / 54 bf16):

| Variant | MSE | MAE | Δ MSE vs BF16 |
|---|---|---|---|
| BF16 | 0.001112 | 0.012335 | — |
| NVFP4 uniform | 0.001450 | 0.013968 | **+30%** |
| FP8 uniform | 0.001023 | 0.012220 | −8% |
| **Mixed recipe (eb 6.0)** | **0.000851** | **0.012120** | **−23%** |

**Uniform NVFP4 measurably degrades the finetuned policy (+30% MSE); the
auto_quantize mixed recipe fully preserves quality** (nominally below BF16 —
read as parity within seed noise). This is the central quality result: at the
same ~6.0 effective bits you must put the 4-bit budget where the model
tolerates it.

TensorRT NVFP4 numerics were validated by running the NVFP4 DiT engine
end-to-end (`standalone_inference_script.py --inference-mode tensorrt`, droid,
unseeded): TRT BF16 engine MSE 0.03622 vs TRT NVFP4 engine 0.03671 (+1.3%, in
noise); see `trt_quality_dit_engines.txt`.

## Performance

### PyTorch (torchao real kernels via `torch._scaled_mm`)

E2E `policy.get_action` medians, base ckpt, REAL_G1 embodiment, 4 denoise steps:

| Config | BF16 | NVFP4 | Δ |
|---|---|---|---|
| bs=1 eager (n=120) | 119.8 ms | 332.0 ms | **+177%** |
| bs=1 + `--compile` (n=30) | 75.9 ms | 115.3 ms | **+52%** |
| bs=8 eager (n=60) | 228.9 ms | 447.8 ms | +96% |
| bs=16 eager (n=60) | 359.7 ms | 570.7 ms | +59% |
| bs=1 eager, FP8 uniform | 119.8 ms | 1905.5 ms | +1490% |
| bs=1 eager, mixed recipe | 119.8 ms | 700.2 ms | +485% |

**PyTorch NVFP4 is a memory optimization, not a latency optimization, on this
stack.** The regression shrinks with batch (dynamic-act-quant overhead
amortizes) but never crosses over ≤ bs16. Isolated GEMM microbenchmarks show
the crossover: at the backbone's bs16 MLP shape (M≈2400) NVFP4 is 1.5–1.6×
*faster* than BF16, but the DiT's tiny-M GEMMs (M≈17–41) dominate E2E and lose
badly. Two stack caveats inflate these numbers: torchao 0.15 skips its C++
extensions on torch 2.9 ("requires torch ≥ 2.11"), and the FP8-per-row path is
pathologically slow without them (which also poisons the mixed recipe's
latency — 180 of its layers are FP8).

Weights VRAM resident after load (bs=1): BF16 6015 MiB → **NVFP4 3181 MiB
(−47%)**, recipe 3226 MiB, FP8 4037 MiB.

**Streaming quantized load** (`Gr00tPolicy(quantization=...)` keeps the model
on the host and passes `device=` to torchao `quantize_`, which moves each
module onto the GPU as it packs it): peak load VRAM now tracks the quantized
footprint instead of a full-BF16 transient. The streamed and in-place orders
produce the same packed weights — verified **bitwise-identical** action output
on NVFP4 (max |Δ| = 0.0); FP8 and the recipe go through the same
`quantize_(..., device=)` mechanism.

| Peak `cuda_max_allocated` (bs=1) | old (load→quantize) | new (streaming) |
|---|---|---|
| NVFP4 | 6486 MiB | **3832 MiB (−41%)** |
| FP8 | 6183 MiB | 4275 MiB (−31%) |
| mixed recipe | 6075 MiB | 5671 MiB (−7%) |

The recipe gains least because it runs two `quantize_` passes (the second
format's layers sit in BF16 on the GPU between passes); a single per-FQN-config
pass would close that gap. Net effect: a uniform-NVFP4 policy now needs ~4 GB
of VRAM to *load*, not ~6.5 GB — the memory saving finally shows up at load
time, which is what matters for small/edge Blackwell parts.

### TensorRT 10.15 (DiT engine, `--inference-mode tensorrt`)

droid observation, action-horizon 40, 30 iters:

| | PyTorch eager action-head | TRT BF16 engine | TRT NVFP4 engine |
|---|---|---|---|
| Action-head median | 73.4 ms | **21.5 ms** | 31.6 ms |
| E2E median | 169.4 ms | 115.0 ms | 126.3 ms |
| Engine size | — | 2197 MB | **603 MB (−71%)** |

**Even in TensorRT, the NVFP4 DiT engine is ~47% slower than the BF16 engine
at bs=1** on RTX 5090. The DiT's GEMMs are M≈41 at this embodiment; FP4's
throughput advantage needs much larger M, and the per-layer dynamic activation
quantize (DQ chains in the QDQ graph) costs more than the 4× weight-bandwidth
saving returns at these sizes. NVFP4's real wins here are the 71% smaller
engine and ~half the weight memory — decisive on memory-constrained targets
(Jetson Thor/Orin-class), not on a 32 GB desktop GPU.

## What worked

1. **`auto_quantize` gradient-based sensitivity search on a VLA flow-matching
   policy.** The training-style forward returns the flow-matching loss, which
   backpropagates cleanly once two modelopt landmines are disarmed (below).
   The DiT-robust / LLM-sensitive split it found is actionable for any
   deployment.
2. **Recipe-level quality parity.** The searched mixed recipe costs nothing
   measurable on seeded open-loop MSE on either checkpoint, while uniform
   NVFP4 loses +30% MSE on the finetuned LIBERO policy — direct evidence that
   the sensitivity search (not just the format) is what preserves quality.
3. **The full NVFP4 artifact chain**: modelopt PTQ → torch ONNX export with
   FP4 QDQ symbolics → `fp4qdq_to_2dq` post-processing → TRT strongly-typed
   engine builds and runs with correct outputs.
4. **torchao real NVFP4 kernels on sm_120** work out of the box
   (`NVFP4DynamicActivationNVFP4WeightConfig`, torch native `_scaled_mm` FP4),
   compose with `torch.compile`, and halve weight VRAM.

## What didn't work (details for future sessions)

1. **torchao version maze.** 0.17 moved the NVFP4 activation-quant triton
   kernel to an external `mslk` package whose GitHub URL (pytorch/MSLK) 404s
   and whose PyPI name is an empty placeholder (0.0.0, no module). 0.16 has
   the kernel in-tree but crashes the repo-pinned diffusers at import
   (diffusers' torchao shim references removed `uint4_layout` and its except
   path hits an undefined `logger`). **0.15.0 is the working pin** for torch
   2.9 + diffusers here. All versions skip their C++ extensions on torch < 2.11.
2. **torchao FP8 under `torch.inference_mode`** fails ("Cannot set
   version_counter for inference tensor"); works under `no_grad`. The policy
   now selects `no_grad` when the plan contains FP8 layers.
3. **modelopt's diffusers plugin** silently converts every diffusers
   `Attention` and reroutes SDPA through an `FP8SDPA` autograd.Function with
   **no backward**, independent of quantizer enable flags. This breaks
   `auto_quantize`'s gradient scoring with an opaque error. Fix:
   `mtq.unregister(diffusers Attention)` before quantizing
   (`_unregister_diffusers_attention`).
4. **modelopt JIT CUDA extensions vs mixed CUDA toolkits.** With CUDA 13.2
   first on PATH and torch cu128, the extension compiled against 13.2 headers
   and crashed the process (`munmap_chunk(): invalid pointer`). Fix: build
   once with `CUDA_HOME=/usr/local/cuda-12.9` (and clear
   `~/.cache/torch_extensions` if poisoned).
5. **modelopt default configs quantize non-Linear inputs** (e.g. LayerNorm).
   Harmless for fake-quant, but in the ONNX export the LayerNorm input DQ
   chain comes out fp32 against bf16 LN scales and TensorRT refuses the graph
   (`INormalizationLayer input and scale must have identical types`). Fix:
   disable quantizers attached to non-Linear modules after calibration
   (`_disable_non_linear_quantizers`).
6. **ONNX opset.** The legacy exporter caps at opset 19; FLOAT4E2M1
   `DequantizeLinear` needs opset 23. Bumping the default-domain opset import
   to 23 post-hoc parses fine in TRT 10.15.
7. **`get_action` treats a GT "action" key as RTC inpainting** and asserts on
   missing options — calibration forwards must strip `action`/`action_mask`.
8. **PyPI `mslk` is a name-squat** (installs, provides no module). Removed.
9. **modelopt's PyTorch real-quant NVFP4 GEMM requires `tensorrt_llm`** (not
   viable on WSL2), so torchao is the PyTorch real-kernel path.
10. **`hf download` without `hf_transfer`** crawled at <1 MB/s for the LIBERO
    checkpoint; with `HF_HUB_ENABLE_HF_TRANSFER=1` it saturated the line.

## Recommendations

- **For RTX 5090 latency**: stay BF16 + `--compile` (75.9 ms / 13.2 Hz E2E —
  best config measured on this box). The TRT BF16 DiT engine's 21.5 ms
  action-head is the fastest DiT path; a full TRT pipeline should compound.
- **For memory-constrained Blackwell (Thor, Orin-class)**: NVFP4 the DiT via
  the searched recipe — 71% smaller engine, ~2× smaller weights, no measured
  quality loss. Keep the LLM at FP8/BF16 per the sensitivity result.
- **For batch/offline serving**: revisit PyTorch NVFP4 after upgrading to
  torch ≥ 2.11 + torchao ≥ 0.17 (+ real MSLK), where the C++/triton fast paths
  and fused fp8 kernels change the picture; the M≈2400 GEMM already wins 1.6×.
- **Quality methodology**: the seeded open-loop eval is cheap (~2 min/variant)
  and discriminative; run it before/after any precision change. Closed-loop
  task success (LIBERO sim) remains unvalidated — the next session should run
  `gr00t/eval/sim` with the NVFP4 recipe before claiming task-level parity.

## Files

- `ab_bs{1,8,16}_eager.json`, `ab_bs1_compile.json` — pooled latency stats + Welch tests
- `quality_droid_*.json` — seeded open-loop quality (droid, base ckpt)
- `quality_libero_*.json` — seeded open-loop quality (LIBERO ckpt)
- `droid_auto_eb6.0_recipe.json`, `libero_auto_eb6.0_recipe.json` — searched mixed-precision recipes
- `trt_bench_*.txt`, `trt_quality_*.txt` — TensorRT benchmark/quality logs
- Raw per-worker JSONs, parity npz files, ONNX + engines: `~/gr00t_bench/nvfp4/` (durable workdir)
