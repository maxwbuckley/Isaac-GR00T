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

"""Per-layer quantization recipes for GR00T models.

A recipe maps fully-qualified ``nn.Linear`` names (relative to the ``Gr00tN1d7``
root) to a format in {"nvfp4", "fp8", "bf16"}. Recipes are produced by the
sensitivity-aware search in ``scripts/deployment/quantize_nvfp4.py`` and consumed
by both the modelopt fake-quant evaluation path and the torchao real-quant
inference path, so a single searched recipe drives every deployment flavor.
"""

from dataclasses import dataclass, field
import fnmatch
import json
from pathlib import Path


FORMATS = ("nvfp4", "fp8", "bf16")


@dataclass
class QuantRecipe:
    """Mixed-precision quantization plan: per-layer formats plus a default."""

    default_format: str = "nvfp4"
    layers: dict[str, str] = field(default_factory=dict)
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.default_format not in FORMATS:
            raise ValueError(f"default_format must be one of {FORMATS}, got {self.default_format}")
        bad = {v for v in self.layers.values() if v not in FORMATS}
        if bad:
            raise ValueError(f"Unknown layer formats {bad}; must be in {FORMATS}")

    def format_for(self, layer_name: str) -> str:
        """Format for a layer: exact match first, then glob patterns, then default."""
        if layer_name in self.layers:
            return self.layers[layer_name]
        for pattern, fmt in self.layers.items():
            if fnmatch.fnmatch(layer_name, pattern):
                return fmt
        return self.default_format

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "default_format": self.default_format,
            "description": self.description,
            "metadata": self.metadata,
            "layers": self.layers,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "QuantRecipe":
        payload = json.loads(Path(path).read_text())
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported recipe version {payload.get('version')} in {path}")
        return cls(
            default_format=payload["default_format"],
            layers=payload.get("layers", {}),
            description=payload.get("description", ""),
            metadata=payload.get("metadata", {}),
        )


def load_recipe(path: str | Path) -> QuantRecipe:
    return QuantRecipe.load(path)
