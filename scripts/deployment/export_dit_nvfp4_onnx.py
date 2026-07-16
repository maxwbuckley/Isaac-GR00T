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

"""Export the GR00T N1.7 DiT as an NVFP4-quantized ONNX model for TensorRT.

The DiT is the NVFP4-robust component per gradient-based sensitivity analysis
(scripts/deployment/quantize_nvfp4.py); this script produces the TensorRT-side
artifact: calibrate NVFP4 fake-quant on real dataset steps, export ONNX with
modelopt's FP4 QDQ symbolics, then post-process TRT_FP4QDQ nodes into the
double-DequantizeLinear form TensorRT parses natively.

Build the engine with scripts/deployment/build_tensorrt_engine.py and run it
via standalone_inference_script.py --inference-mode tensorrt.
"""

from dataclasses import dataclass
import logging
import os
import sys

import torch
import tyro


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


@dataclass
class Args:
    model_path: str
    """Path to the pretrained model checkpoint directory."""

    dataset_path: str = "demo_data/droid_sample"
    """LeRobot-format dataset used for calibration and input capture."""

    embodiment_tag: str = "oxe_droid_relative_eef_relative_joint"
    """Embodiment tag (name or value, case-insensitive)."""

    output_dir: str = "onnx_nvfp4"
    """Directory for dit_nvfp4.onnx."""

    calib_batches: int = 32
    """Calibration batches (batch size 1, real dataset steps)."""

    batch_size: int = 1
    """Batch dimension baked into the exported graph."""

    seed: int = 1234


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    from export_onnx_n1d7 import DiTInputCapture, _consolidate_external_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.quantization.calib import build_calibration_batches, forward_loop
    from gr00t.quantization.modelopt_ptq import ptq, summarize_quantization

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=args.model_path, device="cuda")

    calib = build_calibration_batches(
        policy,
        args.dataset_path,
        num_batches=args.calib_batches,
        with_actions=False,
        seed=args.seed,
    )

    # Capture real DiT input shapes during a calibrated forward.
    dit_capture = DiTInputCapture()
    hook = policy.model.action_head.model.register_forward_pre_hook(
        dit_capture.hook_fn, with_kwargs=True
    )
    forward_loop(policy.model, calib[:1])
    hook.remove()
    assert dit_capture.captured, "failed to capture DiT inputs"

    # NVFP4 PTQ over the full policy model (recipe scope); only the DiT is exported.
    model = ptq(policy.model, calib, fmt="nvfp4")
    print(summarize_quantization(model))

    dit = model.action_head.model
    dit.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "dit_nvfp4.onnx")

    dtype = torch.bfloat16
    sa_shape = (args.batch_size,) + dit_capture.sa_embs.shape[1:]
    vl_shape = (args.batch_size,) + dit_capture.vl_embs.shape[1:]
    sa_embs = torch.randn(sa_shape, dtype=dtype, device="cuda")
    vl_embs = torch.randn(vl_shape, dtype=dtype, device="cuda")
    timestep = torch.ones((args.batch_size,), dtype=torch.int64, device="cuda")
    export_inputs = [sa_embs, vl_embs, timestep]
    input_names = ["sa_embs", "vl_embs", "timestep"]
    dynamic_axes = {"vl_embs": {1: "vl_seq_len"}}
    if dit_capture.image_mask is not None:
        im_shape = (args.batch_size,) + dit_capture.image_mask.shape[1:]
        export_inputs.append(torch.ones(im_shape, dtype=torch.bool, device="cuda"))
        input_names.append("image_mask")
        dynamic_axes["image_mask"] = {1: "vl_seq_len"}
    if dit_capture.backbone_attention_mask is not None:
        bm_shape = (args.batch_size,) + dit_capture.backbone_attention_mask.shape[1:]
        export_inputs.append(torch.ones(bm_shape, dtype=torch.bool, device="cuda"))
        input_names.append("backbone_attention_mask")
        dynamic_axes["backbone_attention_mask"] = {1: "vl_seq_len"}

    class DiTWrapper(torch.nn.Module):
        def __init__(self, dit_module, use_image_mask, use_backbone_mask):
            super().__init__()
            self.dit = dit_module
            self.use_image_mask = use_image_mask
            self.use_backbone_mask = use_backbone_mask

        def forward(
            self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None
        ):
            kwargs = {}
            if self.use_image_mask and image_mask is not None:
                kwargs["image_mask"] = image_mask
            if self.use_backbone_mask and backbone_attention_mask is not None:
                kwargs["backbone_attention_mask"] = backbone_attention_mask
            return self.dit(sa_embs, vl_embs, timestep, **kwargs)

    wrapped = DiTWrapper(
        dit,
        dit_capture.image_mask is not None,
        dit_capture.backbone_attention_mask is not None,
    ).eval()

    from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers

    logger.info("Exporting NVFP4 DiT to %s (legacy exporter, FP4 QDQ symbolics)...", output_path)
    with torch.inference_mode(), configure_linear_module_onnx_quantizers(wrapped):
        torch.onnx.export(
            wrapped,
            tuple(export_inputs),
            output_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    # Replace trt::TRT_FP4QDQ custom nodes with the double-DequantizeLinear
    # pattern TensorRT parses natively.
    from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
    import onnx

    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx_model = fp4qdq_to_2dq(onnx_model)
    # FLOAT4E2M1 DequantizeLinear requires opset >= 23; the base export is
    # opset 19 (legacy exporter ceiling), so bump the default-domain import.
    for opset in onnx_model.opset_import:
        if opset.domain == "" and opset.version < 23:
            opset.version = 23
    onnx_model.ir_version = max(onnx_model.ir_version, 10)
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        location=os.path.basename(output_path) + ".data",
    )
    _consolidate_external_data(output_path)
    logger.info("NVFP4 DiT ONNX written to %s", output_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
