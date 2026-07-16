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

from gr00t.quantization.recipe import QuantRecipe, load_recipe
import pytest
import torch
from torch import nn


class TestQuantRecipe:
    def test_roundtrip(self, tmp_path):
        recipe = QuantRecipe(
            default_format="nvfp4",
            layers={"backbone.model.language_model.layers.0.self_attn.q_proj": "fp8"},
            description="test",
            metadata={"loss_bf16": 0.5},
        )
        path = tmp_path / "recipe.json"
        recipe.save(path)
        loaded = load_recipe(path)
        assert loaded == recipe

    def test_format_for_exact_glob_default(self):
        recipe = QuantRecipe(
            default_format="nvfp4",
            layers={
                "action_head.model.transformer_blocks.0.attn1.to_q": "fp8",
                "*proj_out*": "bf16",
            },
        )
        assert recipe.format_for("action_head.model.transformer_blocks.0.attn1.to_q") == "fp8"
        assert recipe.format_for("action_head.model.proj_out_2") == "bf16"
        assert recipe.format_for("action_head.model.transformer_blocks.1.ff.net.2") == "nvfp4"

    def test_rejects_unknown_formats(self):
        with pytest.raises(ValueError):
            QuantRecipe(default_format="int3")
        with pytest.raises(ValueError):
            QuantRecipe(layers={"x": "fp42"})

    def test_rejects_unknown_version(self, tmp_path):
        path = tmp_path / "recipe.json"
        path.write_text('{"version": 2, "default_format": "nvfp4", "layers": {}}')
        with pytest.raises(ValueError, match="version"):
            load_recipe(path)


class TestEligibleLinears:
    def _toy_model(self):
        # Mimics the Gr00tN1d7 module paths that scoping keys on (note the
        # doubled .model: Qwen3VLForConditionalGeneration nests a Qwen3VLModel).
        model = nn.Module()
        model.backbone = nn.Module()
        model.backbone.model = nn.Module()
        model.backbone.model.model = nn.Module()
        model.backbone.model.model.language_model = nn.Sequential(nn.Linear(64, 64))
        model.backbone.model.model.visual = nn.Sequential(nn.Linear(64, 64))
        model.backbone.model.lm_head = nn.Linear(64, 640)
        model.action_head = nn.Module()
        model.action_head.model = nn.Module()
        model.action_head.model.blocks = nn.Sequential(nn.Linear(64, 128))
        model.action_head.model.proj_out_2 = nn.Linear(64, 29)
        model.action_head.model.odd = nn.Linear(30, 64)  # in_features % 32 != 0
        model.action_head.vl_self_attention = nn.Sequential(nn.Linear(64, 64))
        return model

    def test_scoping_and_skips(self):
        from gr00t.quantization.nvfp4_inference import eligible_linears

        model = self._toy_model()
        names = set(eligible_linears(model))
        assert "backbone.model.model.language_model.0" in names
        assert "action_head.model.blocks.0" in names
        assert "action_head.vl_self_attention.0" in names
        # Not in default scopes / explicitly skipped / misaligned dims:
        assert "backbone.model.model.visual.0" not in names
        assert "backbone.model.lm_head" not in names
        assert "action_head.model.proj_out_2" not in names
        assert "action_head.model.odd" not in names

    def test_vit_scope_opt_in(self):
        from gr00t.quantization.nvfp4_inference import eligible_linears

        model = self._toy_model()
        names = set(eligible_linears(model, scopes=("llm", "vit")))
        assert "backbone.model.model.visual.0" in names


@pytest.mark.gpu
class TestRealQuantization:
    def test_quantize_toy_model_nvfp4(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("requires Blackwell-class GPU")
        from gr00t.quantization.nvfp4_inference import quantize_policy_model

        model = TestEligibleLinears()._toy_model().to("cuda", torch.bfloat16)
        plan = quantize_policy_model(model, "nvfp4")
        assert plan  # something got quantized
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out = model.backbone.model.language_model(x)
        assert out.shape == (2, 8, 64)
        assert not out.isnan().any()
