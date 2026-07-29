# NVFP4 on Jetson AGX Thor — 2026-07-29

Deployment-path results for GR00T N1.7 on a Jetson AGX Thor Developer Kit.

**Headline: the NVFP4 latency verdict inverts on Thor.** On RTX 5090 the NVFP4
DiT engine was 47% *slower* than BF16 (`../rtx5090-nvfp4-2026-07-17/`). On Thor
the same recipe is **39% faster**, and the full pipeline reaches **12.8 Hz** —
17% faster than NVIDIA's published Thor figure. The cause is memory bandwidth,
and the crossover was predicted from a roofline before it was measured.

## Environment

- Jetson AGX Thor Developer Kit: 14-core Arm v9 (Neoverse-V3AE), Blackwell GPU
  (**sm_110, 20 SMs**), 128 GB unified LPDDR5X, MAXN power mode
- JetPack 7.1 / L4T R38.4.0, CUDA 13.0, driver 580.00
- torch 2.10.0 (Jetson `sbsa/cu130` wheels), transformers 4.57.6, flash-attn
  2.8.4, TensorRT 10.15.1 (pip), nvidia-modelopt 0.45.0
- Model `nvidia/GR00T-N1.7-LIBERO/libero_10`, dataset `demo_data/libero_demo`,
  embodiment `libero_sim`, batch 1, action horizon 40, 4 denoising steps

### Measured memory bandwidth

| | |
|---|---|
| Device read (reduction over 512 MB) | **232.8 GB/s** |
| Device copy (read+write) | 230.4 GB/s |
| RTX 5090 for reference (GDDR7 spec) | ~1790 GB/s |

**~7.7x less bandwidth than the 5090.** This single number drives every result
below.

## ⚠️ The published NVIDIA table is 1 camera; these runs are 2

`scripts/deployment/README.md` reports *"4 denoising steps, 1 camera"*. The
`libero_sim` embodiment uses **two** cameras (`image`, `wrist_image`). Export
metadata for these runs:

```
num_patches: 512    num_merged_patches: 128
num_vis_tokens: 128 vl_seq_len: 156
```

At 256x256 with patch-16 that is 256 patches per image, 512 total. A 1-camera
run is 256 patches -> 64 vision tokens -> ~92-token VL sequence.

`benchmark_inference.py --num-cameras N` (added here) restricts inference to the
first N views, and the active count is now printed on every run.

**Camera count matters far less than the 2x token ratio suggests.** Measured
both ways rather than argued:

| Path | 2 cameras | 1 camera | delta |
|---|---|---|---|
| Eager backbone | 56.14 ms | 51.73 ms | **-7.9%** |
| TRT backbone | 35 ms | 33.59 ms | **-4.0%** |
| Eager data processing | 5.67 ms | 4.99 ms | -12% |
| Eager E2E | 135.6 ms | 133.9 ms | -1.3% |

The backbone reads ~3.2 GB of ViT+LLM weights once per forward regardless of
sequence length (~14 ms at 232 GB/s), and the token-dependent compute at 156 vs
90 tokens is small beside it. Engine sizes confirm the same thing: halving the
tokens shrinks the ViT engine only 1625.9 -> 1548.7 MiB (-5%) and leaves the DiT
engine byte-identical, because engine size is dominated by weights, not
activations. **This pipeline is weight-bound end to end.**

An earlier draft of this file claimed the 2-camera backbone was "~2x the vision
work for 19% more time" and therefore faster than published per camera. That was
wrong — asserted from a single 2-camera run without varying the camera count.
The 1-camera measurements above refute it.

### Engines are camera-count-locked

The ViT input is a **static** `[num_patches, dim]` dimension, so camera count is
baked into the engine exactly like batch size:

```
Static dimension mismatch while setting input shape for pixel_values.
Set dimensions are [256,1536]. Expected dimensions are [512,1536].
```

Export with the same `--num-cameras` you intend to run (`export_onnx_n1d7.py`
now takes the flag). The DiT is the exception: its `vl_embs` axis is dynamic, so
one DiT engine serves any camera count.

## Roofline: why FP4 wins here and loses on a 5090

The DiT is 1,091,722,240 params, re-read every denoising step (14 MiB L2 holds
none of it), so 4 steps means 4 full passes over its weights:

| | Thor | RTX 5090 |
|---|---|---|
| Bandwidth | 232 GB/s | ~1790 GB/s |
| BF16 DiT weight traffic (4 steps) | 8.75 GB | 8.75 GB |
| Time at roofline | **37.7 ms** | 4.9 ms |
| Measured BF16 action head | 55.4 ms | 21.5 ms |
| **Bandwidth share** | **68%** | **23%** |

Dequantisation overhead is roughly fixed. On the 5090 it swamps a 4.9 ms
bandwidth component; on Thor it is small against 37.7 ms. Predicted NVFP4
action head from this model: ~33 ms. **Measured: 33.77 ms.**

The GR00T backbone is prefill-only (`use_cache=False, logits_to_keep=1`), so
the usual autoregressive-decode bandwidth argument does not apply to the LLM.
It applies to the **DiT**, whose 4 sequential steps at M~41 have the same low
arithmetic intensity.

## Speed

`benchmark_inference.py`, 30 iterations after 10 warmup, medians.

| Config | Data | Backbone | Action head | E2E | Rate |
|---|---|---|---|---|---|
| PyTorch eager | 5.6 | 60.2 | 80.5 | 146.5 ms | 6.8 Hz |
| TRT full pipeline BF16 | 9 | 35 | 60 | 105 ms | 9.5 Hz |
| *NVIDIA published Thor (1 camera)* | *8.2* | *28.9* | *56.6* | *93.8 ms* | *10.7 Hz* |
| TRT dit_only + NVFP4 | 6.2 | 57.9 | 33.8 | 101.0 ms | 9.9 Hz |
| **TRT full pipeline + NVFP4 DiT** | **5.6** | **34.3** | **37.3** | **77.8 ms** | **12.8 Hz** |

Full-pipeline NVFP4: median 77.8 ms, mean 79.3 +/- 3.4, min 76.2, max 89.4.
**1.88x over eager.**

### Apples-to-apples: 1 camera, matching the published table

Rebuilt engines at `--num-cameras 1` (`gr00t_trt_1cam/`), 30 iterations:

| Config | Data | Backbone | Action head | E2E | Rate |
|---|---|---|---|---|---|
| *NVIDIA published Thor* | *8.21* | ***28.89*** | *56.64* | *93.8 ms* | *10.7 Hz* |
| Ours, TRT BF16 | **4.38** | 33.59 | 59.46 | 97.9 ms | 10.2 Hz |
| Ours, TRT + NVFP4 DiT | **3.95** | 32.91 | **37.95** | **75.9 ms** | **13.2 Hz** |

At matched camera count: our data processing is **47% faster** than published
(B6/B4 from `perf-integration`, present in this tree), the action head is 5%
slower in BF16, and NVFP4 is **19% faster end-to-end** than the published row.

**The backbone remains 16% slower than published (33.59 vs 28.89 ms) and this is
UNEXPLAINED.** Camera count accounts for ~4% of it on the TRT path, not the gap.
Untested candidates, listed as hypotheses only: TensorRT version (this box used
pip `tensorrt` 10.15.1 alongside a system TRT 10.13.3), tactic selection during
our build, ViT precision handling, thermal state during the build. Anyone
picking this up should vary one at a time and measure rather than reason about
it — that mistake has already been made twice in this file's history.

### DiT engine size

| | |
|---|---|
| BF16 | 2084.6 MiB |
| Recipe-mixed NVFP4 | **605.8 MiB (-71%)** |

Matches the 5090's 603 MB. The memory win transfers; the latency win does not
(it inverts).

## Quality — parity, not improvement

`standalone_inference_script.py`, 5 trajectories x 200 steps, execution horizon
8, seed 42 fixed so all variants see identical noise draws. Unnormalized action
MSE/MAE vs ground truth.

| Variant | MSE | MAE | dMSE | dMAE |
|---|---|---|---|---|
| PyTorch BF16 (reference) | 0.001390 | 0.013069 | — | — |
| TRT BF16 engine | 0.001394 | 0.013093 | +0.3% | +0.2% |
| TRT NVFP4 (dit_only) | 0.001360 | 0.013193 | -2.2% | +0.9% |
| TRT NVFP4 (full pipeline) | 0.001282 | 0.013214 | -7.8% | +1.1% |

All three NVFP4 configurations land with MSE marginally *below* reference and
MAE marginally *above*. Opposite-signed sub-3% deltas are the signature of
numerical dither, **not** a quality gain. Read this as parity. TRT conversion
alone costs 0.3%, so quantisation adds roughly 8x that — still negligible.

TRT full-pipeline accuracy verification (BF16): cosine **0.999932** PASS.

## Recipe application

The sensitivity-searched `libero_auto_eb6.0_recipe.json` resolves cleanly on
this checkpoint: **473/473 recipe keys match module names**, and PTQ reports
`{'nvfp4': 266, 'fp8': 153, 'bf16': 50}` over 469 Linears.

Within **DiT scope** it is 246 nvfp4 / 6 fp8 / 1 bf16 — the FP8-heavy part of
the recipe is almost entirely the LLM, which is not in this engine. So this DiT
is ~97% FP4 by layer count, which is why its size matches uniform NVFP4's.

**TensorRT accepted the three-precision graph** (54.3 s build, no QDQ type
errors). Sensitivity-searched recipes are TRT-consumable, not torchao-only.

### GROOT_SKIP_PATTERNS costs ~nothing — keep them

| Pattern | Tensors | Params | Share of DiT |
|---|---|---|---|
| `*proj_out_1*` | 2 | 4.722 M | 0.432% |
| `*proj_out_2*` | 2 | 1.574 M | 0.144% |
| `*timestep_encoder*` | 4 | 2.756 M | 0.252% |
| **Total** | 8 | **9.05 M** | **0.83%** |

Quantising them would add at most ~0.2 ms to a ~27 ms saving, while
`proj_out_2` emits the velocity field itself and `timestep_encoder` conditions
every AdaLN in all 32 layers. Worst available risk/benefit ratio.

## torch.compile is unusable on this GPU

`torch/_inductor/utils.py:1676`:

```python
min_sms = 16 if device.type == "xpu" else 68  # 3080
```

**Thor has 20 SMs against a threshold of 68**, so Inductor emits zero Triton
GEMM templates (`num_triton_choices: 0`) and `max-autotune` degenerates to eager
GEMMs plus compile overhead. Measured 0.87-0.91x — a regression, on both `main`
and `perf-integration`. Thor passes the separate arch gate (`major == 11`); SM
count is the sole cause.

Unexplained: the backbone inflates 57 -> 81 ms under compile. Losing an
optimisation should not cost 24 ms. Untested hypothesis: Inductor decomposition
displacing the SDPA/FA2 fast path. `mode='default'` instead of `max-autotune`
is the cheap next experiment.

## PR #727 (image-path preprocessing) verified on aarch64

Isolated ABBA, 2 rounds/side, 50 iterations after 10 warmup, PR head `2198bf9`
vs base `main` `9c7e746`:

| data-processing stage | base | PR |
|---|---|---|
| mean | 9.33 +/- 2.40 ms | **6.76 +/- 1.50 ms** |
| median | 8.36 ms | **6.75 ms** |

-27.5% mean, -19.3% median, n=100/side pooled, **Welch t = 9.07 (df=166)**.
E2E 144.3 -> 141.5 ms (-2.0%) — directionally consistent both rounds but within
run-to-run spread, so only the stage-level result is established.

### Why the saving is smaller here than the 5090's -85%

Both runs use identical image volume (2 x 256x256x3 = 0.39 MB;
`bench_policy_ab.py:54` synthesises `(batch, t, 256, 256, 3)`), so image size
does not explain it. The cost is not copy bandwidth but the **strided CHW->HWC
gather**, measured on Thor at 720x720:

| Operation | Time |
|---|---|
| Plain memcpy of the same bytes | 0.054 ms (~28 GB/s) |
| `np.transpose` as a view | 0.0004 ms |
| Transpose materialised | **2.25 ms (41x memcpy)** |
| `Image.fromarray(contiguous)` | 0.313 ms |
| `Image.fromarray(transposed)` | **5.099 ms** |

It scales near-linearly at ~3.7 ms/MB. Thor's copies are fast; the transposed
materialisation costs 41x a flat copy of identical bytes because it reads with
stride H*W — one useful byte per cache line.

**Open question:** the 5090 README's 35.6 ms "process_cpu" figure is ~20x
Thor's for identical data. `process_cpu` appears nowhere in the committed
harness (`git grep` finds it only in that README's prose), so it cannot be
confirmed to scope the same stage as `benchmark_inference.py`'s "Data
Processing". Either WSL2's memory path punishes strided gathers severely, or
the two numbers measure different things. Re-running the microbenchmark above on
the 5090 box would settle it.

## perf-integration on Thor

Single run per side (not pooled ABBA), same benchmark script (byte-identical
between branches):

| Stage | `main` | `perf-integration` |
|---|---|---|
| Data processing | 9.0 ms | **6.08 ms (-32%)** |
| Backbone | 59 ms | 56.82 ms |
| Action head | 82 ms | 80.21 ms |
| **E2E eager** | **152 ms** | **144.5 ms (-5.0%)** |

Backbone/action-head deltas are within the +/-8.7 ms spread; only the
preprocessing improvement is robust at this sample size.

## Caveats

- **Open-loop MSE only.** Closed-loop LIBERO task success was not run. These
  results do not license deployment on their own.
- **No uniform-NVFP4 arm on Thor.** The recipe works, but at DiT scope it is
  246/253 NVFP4, so uniform may well perform identically here. The 5090's +30%
  MSE for uniform was model-wide, including the FP8-sensitive LLM. Without this
  arm we cannot show the recipe was *necessary* at DiT scope.
- Speed figures other than the PR #727 ABBA are single runs per configuration.
- torchao PyTorch-path quantisation was not attempted: torchao 0.17 is the only
  aarch64 resolution here and the pinned working version is 0.15.0; all versions
  skip C++ extensions on torch < 2.11 (this box is 2.10.0). The TensorRT path
  avoids torchao entirely.

## Reproducing

```bash
# Recipe-driven NVFP4 DiT export (needs nvidia-modelopt[onnx])
python scripts/deployment/export_dit_nvfp4_onnx.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo --embodiment-tag libero_sim \
  --recipe tools/perf/results/rtx5090-nvfp4-2026-07-17/libero_auto_eb6.0_recipe.json \
  --output-dir gr00t_trt_nvfp4/onnx_recipe

python scripts/deployment/build_tensorrt_engine.py --mode single \
  --onnx gr00t_trt_nvfp4/onnx_recipe/dit_nvfp4.onnx \
  --engine gr00t_trt_nvfp4/engines/dit_nvfp4.engine --precision bf16

# Full pipeline: symlink the BF16 engines, swap the DiT
# (trt_model_forward.py expects the filename dit_bf16.engine)
```

Note `--embodiment-tag` takes the enum **value** (`libero_sim`) in
`benchmark_inference.py` but the enum **name** (`LIBERO_PANDA`) in
`standalone_inference_script.py`.
