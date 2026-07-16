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

"""
Parity tests for the fused host-to-device transfer in Gr00tPolicy._get_action.

The old flow was two full tree traversals:
  1. policy-side ``_rec_to_dtype(x, bf16)``: CPU cast of float tensors only;
  2. ``Gr00tN1d7.prepare_input``'s ``to_device_with_dtype``: ``.to(device,
     dtype)`` for floats, ``.to(device)`` for everything else.

The new ``_rec_to_device_dtype`` does both in a single traversal with one
``.to`` per tensor. These tests reimplement the old two-step path and assert
bitwise-equal values plus identical dtype/device per leaf on CPU.
"""

from typing import Any

from gr00t.policy.gr00t_policy import _rec_to_device_dtype
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _old_rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Reimplementation of the removed policy-side CPU pre-cast (_rec_to_dtype)."""
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _old_rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_old_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


def _old_to_device_with_dtype(x: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Reimplementation of Gr00tN1d7.prepare_input's to_device_with_dtype traversal."""
    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x):
            return x.to(device, dtype=dtype)
        return x.to(device)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _old_to_device_with_dtype(v, device, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_old_to_device_with_dtype(v, device, dtype) for v in x]
    else:
        return x


def _make_inputs() -> dict:
    """Nested inputs structure with mixed float/int/bool tensors and non-tensor leaves."""
    g = torch.Generator().manual_seed(0)
    return {
        "inputs": BatchFeature(
            data={
                "state": torch.randn(2, 1, 64, generator=g, dtype=torch.float32),
                "pixel_values": torch.randn(4, 3, 8, 8, generator=g, dtype=torch.float32),
                "input_ids": torch.randint(0, 1000, (2, 12), generator=g, dtype=torch.int64),
                "attention_mask": torch.ones(2, 12, dtype=torch.bool),
                "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.int32),
                "half_precision": torch.randn(3, 3, generator=g).to(torch.float16),
                "double_precision": torch.randn(3, 3, generator=g, dtype=torch.float64),
                "embodiment_id": torch.tensor([2], dtype=torch.int32),
            }
        ),
        "tensor_list": [
            torch.randn(2, 2, generator=g, dtype=torch.float32),
            torch.arange(5, dtype=torch.uint8),
        ],
        "scalar": 7,
        "string": "unchanged",
        "none": None,
    }


def _assert_trees_equal(new: Any, old: Any, path: str = "root"):
    assert type(new) is type(old), f"{path}: {type(new)} != {type(old)}"
    if isinstance(new, torch.Tensor):
        assert new.dtype == old.dtype, f"{path}: dtype {new.dtype} != {old.dtype}"
        assert new.device == old.device, f"{path}: device {new.device} != {old.device}"
        assert torch.equal(new, old), f"{path}: values differ"
    elif isinstance(new, dict):
        assert new.keys() == old.keys(), path
        for k in new:
            _assert_trees_equal(new[k], old[k], f"{path}.{k}")
    elif isinstance(new, list):
        assert len(new) == len(old), path
        for i, (n, o) in enumerate(zip(new, old)):
            _assert_trees_equal(n, o, f"{path}[{i}]")
    else:
        assert new == old, path


class TestFusedTransferParity:
    def test_matches_old_two_step_path_on_cpu(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16
        inputs = _make_inputs()

        old = _old_to_device_with_dtype(_old_rec_to_dtype(inputs, dtype), device, dtype)
        new = _rec_to_device_dtype(inputs, device=device, dtype=dtype)

        _assert_trees_equal(new, old)

    def test_float_leaves_cast_and_non_float_preserved(self):
        new = _rec_to_device_dtype(_make_inputs(), device="cpu", dtype=torch.bfloat16)
        batch = new["inputs"]
        # BatchFeature (dict-like) is converted to a plain dict, matching the
        # old _rec_to_dtype behavior relied on by model.get_action(**inputs).
        assert type(batch) is dict
        assert batch["state"].dtype == torch.bfloat16
        assert batch["pixel_values"].dtype == torch.bfloat16
        assert batch["half_precision"].dtype == torch.bfloat16
        assert batch["double_precision"].dtype == torch.bfloat16
        assert batch["input_ids"].dtype == torch.int64
        assert batch["attention_mask"].dtype == torch.bool
        assert batch["image_grid_thw"].dtype == torch.int32
        assert new["tensor_list"][1].dtype == torch.uint8
        assert new["scalar"] == 7
        assert new["string"] == "unchanged"
        assert new["none"] is None

    def test_fp32_to_bf16_rounding_matches_cpu_pre_cast(self):
        """The dtype conversion in the fused .to call is bit-identical to the
        old standalone CPU cast (round-to-nearest-even in both)."""
        x = torch.linspace(-10, 10, steps=4096, dtype=torch.float32)
        # Include values that exercise rounding ties and subnormals.
        x = torch.cat([x, torch.tensor([1.00390625, -1.00390625, 3.0e-39, float("inf")])])
        fused = x.to("cpu", dtype=torch.bfloat16, non_blocking=True)
        pre_cast = x.to(dtype=torch.bfloat16)
        assert torch.equal(fused, pre_cast)

    def test_idempotent_second_pass_is_passthrough(self):
        """Tensors already on the target device/dtype pass through .to as no-ops,
        so prepare_input's later traversal returns the same tensor objects."""
        once = _rec_to_device_dtype(_make_inputs(), device="cpu", dtype=torch.bfloat16)
        twice = _rec_to_device_dtype(once, device="cpu", dtype=torch.bfloat16)
        _assert_trees_equal(twice, once)
        # .to returns self when nothing changes: no copies on the second pass.
        assert twice["inputs"]["state"] is once["inputs"]["state"]
        assert twice["inputs"]["input_ids"] is once["inputs"]["input_ids"]

    @pytest.mark.gpu
    def test_matches_old_two_step_path_on_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        inputs = _make_inputs()

        old = _old_to_device_with_dtype(_old_rec_to_dtype(inputs, dtype), device, dtype)
        new = _rec_to_device_dtype(inputs, device=device, dtype=dtype)
        torch.cuda.synchronize()

        _assert_trees_equal(new, old)
