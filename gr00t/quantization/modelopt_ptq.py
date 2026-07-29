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

"""TensorRT Model Optimizer (modelopt) PTQ and sensitivity search for GR00T N1.7.

This is the recipe-search / quality-evaluation half of NVFP4 support: modelopt
fake-quantization simulates NVFP4/FP8 numerics bit-accurately without needing
FP4 kernels, so searched recipes transfer to any Blackwell deployment (TensorRT
engines on RTX 50xx / Jetson Thor, or the torchao runtime path in
``gr00t.quantization.nvfp4_inference``).
"""

from copy import deepcopy
import logging

import torch

from gr00t.quantization.calib import forward_loop
from gr00t.quantization.recipe import QuantRecipe


logger = logging.getLogger(__name__)

# Quantization format names -> modelopt config attribute names.
FORMAT_CFG_NAMES = {
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "nvfp4_awq": "NVFP4_AWQ_LITE_CFG",
    "nvfp4_svdquant": "NVFP4_SVDQUANT_DEFAULT_CFG",
    "fp8": "FP8_DEFAULT_CFG",
    "mxfp4": "MXFP4_DEFAULT_CFG",
    "w4a16_nvfp4": "W4A16_NVFP4_CFG",
    "w4a8_nvfp4_fp8": "W4A8_NVFP4_FP8_CFG",
}

# GR00T-specific exclusions, applied on top of modelopt's defaults (which already
# skip vision towers, lm_head, embeddings, and BatchNorms):
# - proj_out_1/proj_out_2: final DiT AdaLN-and-action projection; tiny and the
#   single most error-amplifying layer (its output *is* the velocity field).
# - timestep_encoder: 2-layer MLP on a 256-dim sinusoid; negligible compute,
#   conditions every AdaLN in the DiT.
# CategorySpecificLinear (embodiment tables) is bmm on raw Parameters and is
# never wrapped by modelopt, so it stays BF16 without an explicit rule.
# The *_bmm/softmax quantizers come from modelopt's diffusers plugin, which
# swaps the DiT's SDPA for a quantized fp8_sdpa autograd.Function; that
# Function has no backward (breaking auto_quantize's gradient scoring) and has
# no counterpart in the torchao/TRT Linear-only deployment path, so attention
# stays BF16 everywhere.
GROOT_SKIP_PATTERNS = (
    "*proj_out_1*",
    "*proj_out_2*",
    "*timestep_encoder*",
    "*lm_head*",
    "*bmm_quantizer*",
    "*softmax_quantizer*",
)


def _unregister_diffusers_attention() -> None:
    """Keep modelopt from converting diffusers ``Attention`` modules.

    The conversion unconditionally reroutes SDPA through an ``FP8SDPA``
    autograd.Function (no backward defined), independent of quantizer enable
    flags — it would break gradient-based sensitivity scoring and diverge from
    the Linear-only torchao/TRT deployment path. Unregistering leaves the
    attention math untouched; the inner to_q/to_k/to_v/to_out Linears are
    still quantized individually. Idempotent.
    """
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import QuantModuleRegistry

    try:
        from diffusers.models.attention import Attention
    except ImportError:
        return
    if QuantModuleRegistry.get(Attention) is not None:
        mtq.unregister(Attention)


def _get_format_cfg(fmt: str) -> dict:
    import modelopt.torch.quantization as mtq

    if fmt not in FORMAT_CFG_NAMES:
        raise ValueError(f"Unknown format '{fmt}'; choose from {sorted(FORMAT_CFG_NAMES)}")
    cfg = deepcopy(getattr(mtq, FORMAT_CFG_NAMES[fmt]))
    quant_cfg = cfg["quant_cfg"]
    if isinstance(quant_cfg, dict):
        for pattern in GROOT_SKIP_PATTERNS:
            quant_cfg[pattern] = {"enable": False}
    else:  # list-of-rules style
        quant_cfg.extend({"quantizer_name": p, "enable": False} for p in GROOT_SKIP_PATTERNS)
    return cfg


def _disable_non_linear_quantizers(model) -> int:
    """Disable act/weight quantizers modelopt attached to non-Linear modules.

    modelopt's default configs insert input quantizers on many module types
    (LayerNorm, activations, ...). Only Linear GEMMs have fast low-precision
    kernels in any GR00T deployment path, and a quantizer on e.g. a LayerNorm
    input emits an fp32 DQ chain that breaks TensorRT's type checking at the
    LayerNormalization node. Returns the number of quantizers disabled.
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    disabled = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            continue
        # Only touch quantizers attached directly to this non-Linear module.
        for child_name, child in module.named_children():
            if isinstance(child, TensorQuantizer) and "quantizer" in child_name:
                if child.is_enabled:
                    child.disable()
                    disabled += 1
    return disabled


def _quantizer_cfgs(fmt: str) -> dict:
    """Extract the {weight,input}_quantizer cfg bodies from a format's default config."""
    cfg = _get_format_cfg(fmt)
    out = {}
    for rule in cfg["quant_cfg"]:
        name = rule.get("quantizer_name")
        if name in ("*weight_quantizer", "*input_quantizer") and "cfg" in rule:
            out[name.lstrip("*")] = rule["cfg"]
    missing = {"weight_quantizer", "input_quantizer"} - out.keys()
    if missing:
        raise ValueError(f"format '{fmt}' has no default rule for {sorted(missing)}")
    return out


def _recipe_rules(recipe: QuantRecipe, model) -> list[dict]:
    """Translate a per-layer recipe into modelopt list-style quant_cfg rules.

    Starts from everything disabled, then enables each Linear at the format the
    recipe assigns it (bf16 layers are simply left disabled). Emitting one rule
    per (layer, quantizer) keeps the mapping explicit rather than relying on
    glob precedence between overlapping patterns.
    """
    bodies = {f: _quantizer_cfgs(f) for f in ("nvfp4", "fp8")}
    rules: list[dict] = [{"quantizer_name": "*", "enable": False}]
    counts: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        fmt = recipe.format_for(name)
        counts[fmt] = counts.get(fmt, 0) + 1
        if fmt == "bf16":
            continue
        for quantizer in ("weight_quantizer", "input_quantizer"):
            rules.append(
                {
                    "quantizer_name": f"*{name}.{quantizer}",
                    "cfg": deepcopy(bodies[fmt][quantizer]),
                }
            )
    logger.info("Recipe resolved over %d Linears: %s", sum(counts.values()), counts)
    return rules


def ptq_with_recipe(model, calib_batches: list[dict], recipe: QuantRecipe):
    """Mixed-precision PTQ driven by a per-layer recipe (fake quant).

    Unlike ptq(), which applies one format everywhere, this honours the
    sensitivity-searched plan: layers the search found tolerant go to NVFP4,
    sensitive ones to FP8, and the most sensitive stay BF16. GROOT_SKIP_PATTERNS
    are appended last so they win over the recipe -- the recipe was searched on
    the same architecture, but those exclusions are deployment-path invariants
    (no low-precision kernel exists for them in TRT/torchao), not tuning choices.
    """
    import modelopt.torch.quantization as mtq

    _unregister_diffusers_attention()
    cfg = deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["quant_cfg"] = _recipe_rules(recipe, model) + [
        {"quantizer_name": p, "enable": False} for p in GROOT_SKIP_PATTERNS
    ]
    model = mtq.quantize(model, cfg, lambda m: forward_loop(m, calib_batches))
    n = _disable_non_linear_quantizers(model)
    logger.info(
        "PTQ (recipe: %s) done (%d non-Linear quantizers disabled):\n%s",
        recipe.description or "unnamed",
        n,
        summarize_quantization(model),
    )
    return model


def ptq(model, calib_batches: list[dict], fmt: str = "nvfp4"):
    """Uniform-format PTQ with calibration on real policy inputs (fake quant)."""
    import modelopt.torch.quantization as mtq

    _unregister_diffusers_attention()
    cfg = _get_format_cfg(fmt)
    model = mtq.quantize(model, cfg, lambda m: forward_loop(m, calib_batches))
    n = _disable_non_linear_quantizers(model)
    logger.info(
        "PTQ (%s) done (%d non-Linear quantizers disabled):\n%s",
        fmt,
        n,
        summarize_quantization(model),
    )
    return model


def sensitivity_search(
    model,
    score_batches: list[dict],
    *,
    effective_bits: float = 6.0,
    formats: tuple[str, ...] = ("nvfp4", "fp8"),
    num_score_steps: int = 32,
    verbose: bool = True,
):
    """Sensitivity-aware mixed-precision search via modelopt ``auto_quantize``.

    Gradient-based scoring backpropagates the policy's flow-matching loss, so
    ``score_batches`` must be built ``with_actions=True``. All parameters get
    ``requires_grad`` temporarily enabled: the backbone is frozen for inference,
    but scoring needs gradients to flow to every quantized layer.

    Returns ``(model, search_state)`` with the chosen per-layer formats applied.
    """
    import modelopt.torch.quantization as mtq

    _unregister_diffusers_attention()
    quant_cfgs = [_get_format_cfg(f) for f in formats]

    def forward_step(m, batch):
        return m(dict(batch))["loss"]

    requires_grad_snapshot = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = True
    was_training = model.training
    model.train()  # autograd needs training-mode forward; dropout modules are p=0 or frozen-eval
    try:
        model, state = mtq.auto_quantize(
            model,
            constraints={"effective_bits": effective_bits},
            quantization_formats=quant_cfgs,
            data_loader=score_batches,
            forward_step=forward_step,
            loss_func=lambda output, data: output,
            num_calib_steps=len(score_batches),
            num_score_steps=min(num_score_steps, len(score_batches)),
            verbose=verbose,
            method="gradient",
        )
    finally:
        for n, p in model.named_parameters():
            if n in requires_grad_snapshot:
                p.requires_grad = requires_grad_snapshot[n]
        if not was_training:
            model.eval()
    _disable_non_linear_quantizers(model)
    logger.info(
        "auto_quantize (effective_bits=%.2f) done:\n%s",
        effective_bits,
        summarize_quantization(model),
    )
    return model, state


def _classify_quantizer(q) -> str | None:
    if q is None or not getattr(q, "is_enabled", False):
        return None
    num_bits = getattr(q, "num_bits", None)
    if num_bits == (2, 1):
        return "nvfp4"
    if num_bits == (4, 3):
        return "fp8"
    return f"other{num_bits}"


def extract_recipe(model, description: str = "", metadata: dict | None = None) -> QuantRecipe:
    """Read the per-layer formats off a (auto_)quantized model into a QuantRecipe."""
    layers = {}
    for name, module in model.named_modules():
        wq = getattr(module, "weight_quantizer", None)
        if wq is None or not hasattr(module, "weight"):
            continue
        fmt = _classify_quantizer(wq) or "bf16"
        layers[name] = fmt
    return QuantRecipe(
        default_format="bf16",
        layers=layers,
        description=description,
        metadata=metadata or {},
    )


def summarize_quantization(model) -> str:
    """Human-readable per-format layer counts and weight-byte totals."""
    counts: dict[str, int] = {}
    weight_bytes: dict[str, float] = {}
    bits = {"nvfp4": 4.5, "fp8": 8.0, "bf16": 16.0}  # nvfp4: 4b data + 8b scale per 16
    for name, module in model.named_modules():
        wq = getattr(module, "weight_quantizer", None)
        if wq is None or not hasattr(module, "weight"):
            continue
        fmt = _classify_quantizer(wq) or "bf16"
        counts[fmt] = counts.get(fmt, 0) + 1
        numel = module.weight.numel()
        weight_bytes[fmt] = weight_bytes.get(fmt, 0.0) + numel * bits.get(fmt, 16.0) / 8
    lines = [
        f"  {fmt}: {counts[fmt]} layers, {weight_bytes[fmt] / 2**20:.0f} MiB"
        for fmt in sorted(counts)
    ]
    return "\n".join(lines) if lines else "  (no quantizable layers found)"


@torch.no_grad()
def measure_loss(model, batches: list[dict]) -> float:
    """Mean flow-matching loss over ``batches`` (built with actions)."""
    total = 0.0
    for batch in batches:
        total += float(model(dict(batch))["loss"])
    return total / max(1, len(batches))
