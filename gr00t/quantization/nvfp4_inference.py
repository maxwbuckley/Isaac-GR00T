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

"""Real NVFP4/FP8 inference for GR00T N1.7 on Blackwell GPUs via torchao.

Weights are stored packed (FP4 + FP8 block scales) and matmuls execute on the
GPU's FP4/FP8 tensor cores through ``torch._scaled_mm``. Requires compute
capability >= 10.0 (Blackwell: sm_100 Thor/Spark, sm_120 RTX 50xx) and torchao
with NVFP4 support (torchao 0.16.x verified; 0.17 needs the external ``mslk``
kernel package).

Use ``Gr00tPolicy(..., quantization="nvfp4")`` or pass a recipe JSON path
produced by ``scripts/deployment/quantize_nvfp4.py`` for sensitivity-aware
mixed precision (NVFP4 for robust layers, FP8/BF16 for sensitive ones).
"""

import fnmatch
import logging

import torch
from torch import nn

from gr00t.quantization.recipe import QuantRecipe, load_recipe


logger = logging.getLogger(__name__)

# Module scopes eligible for quantization, by short name; matched as FQN
# substrings because HF wrappers nest an extra ``.model`` level (the LLM lives
# at ``backbone.model.model.language_model.``). The ViT is excluded from every
# default: NVIDIA's own TRT pipeline keeps it FP32 for accuracy, and modelopt's
# NVFP4 defaults skip vision towers too.
SCOPES = {
    "llm": ".language_model.",
    "vit": ".visual.",
    "dit": "action_head.model.",
    "vlsa": "action_head.vl_self_attention.",
}
DEFAULT_SCOPES = ("llm", "dit", "vlsa")

# Mirrors modelopt_ptq.GROOT_SKIP_PATTERNS (relative to the Gr00tN1d7 root).
SKIP_PATTERNS = (
    "*proj_out_1*",
    "*proj_out_2*",
    "*timestep_encoder*",
    "*lm_head*",
)


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def eligible_linears(
    model: nn.Module, scopes: tuple[str, ...] = DEFAULT_SCOPES
) -> dict[str, nn.Linear]:
    """FQN -> module map of quantizable Linear layers within the given scopes."""
    fragments = tuple(SCOPES[s] for s in scopes)
    out = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        qualified = f".{name}"  # so root-level scope fragments also match as substrings
        if not any(frag in qualified for frag in fragments):
            continue
        if any(fnmatch.fnmatch(name, p) for p in SKIP_PATTERNS):
            continue
        # NVFP4 packs pairs along K and scales blocks of 16; FP8 rowwise needs
        # 16-aligned N. One conservative rule for both formats.
        if module.in_features % 32 != 0 or module.out_features % 16 != 0:
            logger.debug(
                "Skipping %s: dims %dx%d not FP4-alignable",
                name,
                module.in_features,
                module.out_features,
            )
            continue
        out[name] = module
    return out


def _torchao_config(fmt: str):
    if fmt == "nvfp4":
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

        return NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True)
    if fmt == "nvfp4-wo":
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig

        return NVFP4WeightOnlyConfig()
    if fmt == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

        return Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    raise ValueError(f"Unknown torchao format '{fmt}'")


def quantize_policy_model(
    model: nn.Module,
    spec: str,
    *,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    device: torch.device | str | int | None = None,
) -> dict[str, str]:
    """Apply real quantization to a loaded ``Gr00tN1d7`` in place.

    Args:
        model: The policy model, in bf16. May be on the host (CPU): pass
            ``device`` to stream it onto the accelerator during quantization.
        spec: "nvfp4" | "nvfp4-wo" | "fp8" (uniform over default scopes), or a
            path to a recipe JSON for per-layer mixed precision.
        scopes: Which module scopes to consider (uniform specs only; recipes
            carry their own layer lists).
        device: If given, torchao moves each module onto this device as it is
            quantized (``quantize_(..., device=...)``), so the full bf16 model
            is never resident on the device at once — peak memory tracks the
            quantized footprint. If None, quantize in place on the model's
            current device. Quantization requires a Blackwell-class device; the
            check runs against the current CUDA device regardless.

    Returns:
        Mapping of layer FQN -> applied format (excluding untouched bf16 layers).
    """
    from torchao.quantization import quantize_

    if not _is_blackwell():
        raise RuntimeError(
            "NVFP4/FP8 real quantization requires a Blackwell-class GPU "
            f"(compute capability >= 10.0); found {torch.cuda.get_device_capability() if torch.cuda.is_available() else 'no CUDA'}."
        )

    if spec in ("nvfp4", "nvfp4-wo", "fp8"):
        recipe = QuantRecipe(default_format="nvfp4" if spec != "fp8" else "fp8")
        uniform_fmt = spec
        eligible = eligible_linears(model, scopes)
        plan = {name: uniform_fmt for name in eligible}
    else:
        recipe = load_recipe(spec)
        # Recipes may reference layers outside the default scopes; honor them.
        eligible = eligible_linears(model, tuple(SCOPES))
        plan = {}
        for name in eligible:
            fmt = recipe.format_for(name)
            if fmt != "bf16":
                plan[name] = fmt

    by_fmt: dict[str, set[str]] = {}
    for name, fmt in plan.items():
        by_fmt.setdefault(fmt, set()).add(name)

    for fmt, names in by_fmt.items():
        cfg = _torchao_config(fmt)
        quantize_(
            model,
            cfg,
            filter_fn=lambda module, fqn, names=names: fqn in names,
            device=device,
        )
        logger.info("Applied %s to %d layers", fmt, len(names))

    return plan


def summarize_plan(plan: dict[str, str]) -> str:
    counts: dict[str, int] = {}
    for fmt in plan.values():
        counts[fmt] = counts.get(fmt, 0) + 1
    return ", ".join(f"{fmt}: {n} layers" for fmt, n in sorted(counts.items())) or "(none)"
