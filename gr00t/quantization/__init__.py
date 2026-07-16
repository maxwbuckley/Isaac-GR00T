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

"""Post-training quantization for GR00T N1.7 on Blackwell-class GPUs (NVFP4/FP8).

Two complementary paths:

- :mod:`gr00t.quantization.modelopt_ptq` — TensorRT Model Optimizer (modelopt)
  fake-quantization: calibration, sensitivity-aware ``auto_quantize`` mixed-precision
  recipe search, and quality evaluation. Hardware-independent; the resulting recipe
  transfers to TensorRT deployment on any Blackwell device (RTX 50xx, Jetson Thor).
- :mod:`gr00t.quantization.nvfp4_inference` — real NVFP4 weight quantization with
  torchao, executing on Blackwell's FP4 tensor cores through ``torch._scaled_mm``.
  This is the PyTorch serving path used by ``Gr00tPolicy(quantization=...)``.
"""

from gr00t.quantization.recipe import QuantRecipe, load_recipe


__all__ = ["QuantRecipe", "load_recipe"]
