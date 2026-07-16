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

"""Sensitivity-aware NVFP4/FP8 quantization recipe search for GR00T N1.7.

Uses NVIDIA TensorRT Model Optimizer (modelopt) to calibrate on real dataset
steps and (optionally) run ``auto_quantize``: gradient-based per-layer
sensitivity scoring followed by a mixed-precision format search under an
effective-bits budget. The result is a recipe JSON consumable by
``Gr00tPolicy(quantization=<recipe path>)`` (real Blackwell kernels via
torchao) and, in fake-quant form, by TensorRT deployment flows.

Examples:
    # Uniform NVFP4 PTQ, report calibrated loss delta
    python scripts/deployment/quantize_nvfp4.py --model-path <ckpt> \
        --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA \
        --mode ptq --format nvfp4

    # Sensitivity-aware mixed NVFP4/FP8 recipe at 6.0 effective bits
    python scripts/deployment/quantize_nvfp4.py --model-path <ckpt> \
        --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA \
        --mode auto --effective-bits 6.0 --output-recipe nvfp4_recipe.json
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.quantization.calib import build_calibration_batches
from gr00t.quantization.modelopt_ptq import (
    FORMAT_CFG_NAMES,
    extract_recipe,
    measure_loss,
    ptq,
    sensitivity_search,
    summarize_quantization,
)
import torch
import tyro


@dataclass
class Args:
    model_path: str
    """Path to the pretrained model checkpoint directory."""

    dataset_path: str = "demo_data/libero_demo"
    """LeRobot-format dataset used for calibration and sensitivity scoring."""

    embodiment_tag: str = "LIBERO_PANDA"
    """Embodiment tag (name or value, case-insensitive)."""

    mode: str = "auto"
    """'ptq' (uniform format) or 'auto' (sensitivity-aware mixed precision)."""

    format: str = "nvfp4"
    """PTQ format; one of the keys in FORMAT_CFG_NAMES (ptq mode only)."""

    formats: tuple[str, ...] = ("nvfp4", "fp8")
    """Candidate formats for the auto mode search."""

    effective_bits: float = 6.0
    """Weight-size budget for auto mode (16.0 = no quantization, 4.5 = all NVFP4)."""

    calib_batches: int = 32
    """Calibration batches (batch size 1, real dataset steps)."""

    score_batches: int = 24
    """Scoring batches for gradient-based sensitivity (auto mode)."""

    output_recipe: str | None = None
    """Where to write the recipe JSON (auto mode). Defaults next to the checkpoint."""

    device: str = "cuda:0"

    seed: int = 1234


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=args.model_path, device=args.device)
    model = policy.model

    calib = build_calibration_batches(
        policy,
        args.dataset_path,
        num_batches=args.calib_batches,
        with_actions=False,
        seed=args.seed,
    )
    scored = build_calibration_batches(
        policy,
        args.dataset_path,
        num_batches=args.score_batches,
        with_actions=True,
        seed=args.seed + 1,
    )

    torch.manual_seed(args.seed)
    loss_before = measure_loss(model, scored)
    logging.info("BF16 flow-matching loss on %d batches: %.6f", len(scored), loss_before)

    if args.mode == "ptq":
        if args.format not in FORMAT_CFG_NAMES:
            raise SystemExit(f"--format must be one of {sorted(FORMAT_CFG_NAMES)}")
        model = ptq(model, calib, fmt=args.format)
        recipe_desc = f"uniform {args.format} PTQ"
    elif args.mode == "auto":
        model, _state = sensitivity_search(
            model,
            scored,
            effective_bits=args.effective_bits,
            formats=args.formats,
            num_score_steps=args.score_batches,
        )
        recipe_desc = (
            f"auto_quantize effective_bits={args.effective_bits} formats={list(args.formats)}"
        )
    else:
        raise SystemExit("--mode must be 'ptq' or 'auto'")

    torch.manual_seed(args.seed)
    loss_after = measure_loss(model, scored)
    logging.info(
        "Quantized loss: %.6f (BF16 %.6f, +%.2f%%)",
        loss_after,
        loss_before,
        100 * (loss_after - loss_before) / max(abs(loss_before), 1e-9),
    )
    print(summarize_quantization(model))

    recipe = extract_recipe(
        model,
        description=recipe_desc,
        metadata={
            "model_path": str(args.model_path),
            "dataset_path": str(args.dataset_path),
            "embodiment_tag": tag.value,
            "loss_bf16": loss_before,
            "loss_quantized": loss_after,
            "calib_batches": args.calib_batches,
            "score_batches": args.score_batches,
        },
    )
    out = args.output_recipe or str(Path(args.model_path) / "nvfp4_recipe.json")
    recipe.save(out)
    counts: dict[str, int] = {}
    for fmt in recipe.layers.values():
        counts[fmt] = counts.get(fmt, 0) + 1
    logging.info("Recipe written to %s: %s", out, json.dumps(counts))


if __name__ == "__main__":
    main(tyro.cli(Args))
