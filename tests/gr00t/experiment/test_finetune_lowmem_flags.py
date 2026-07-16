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

"""CPU-only guards for the fine-tuning low-memory flags:

* ``FinetuneConfig.load_bf16`` / ``FinetuneConfig.optim`` defaults are unchanged
  (fp32 backbone, plain AdamW), invalid optimizers are rejected, and 8-bit
  optimizers fail fast at config-construction time when bitsandbytes is missing.
* ``launch_finetune.build_experiment_config`` propagates both flags onto the
  experiment config while keeping ``backbone_trainable_params_fp32`` forced True.
* ``examples/finetune.sh`` only forwards ``--load-bf16`` / ``--optim`` when the
  LOAD_BF16 / OPTIM env vars are set (defaults leave the flags absent so the
  config defaults apply).
"""

from __future__ import annotations

import builtins
import os
from pathlib import Path
import stat
import subprocess
import sys
import types

from gr00t.configs.finetune_config import ALLOWED_OPTIMS, BITSANDBYTES_OPTIMS, FinetuneConfig
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
FINETUNE_SH = REPO_ROOT / "examples" / "finetune.sh"


def make_ft_config(**overrides) -> FinetuneConfig:
    kwargs = dict(
        base_model_path="dummy/base",
        dataset_path="dummy/dataset",
        embodiment_tag="new_embodiment",
    )
    kwargs.update(overrides)
    return FinetuneConfig(**kwargs)


# ---------------------------------------------------------------------------
# FinetuneConfig field validation
# ---------------------------------------------------------------------------


def test_defaults_unchanged():
    ft_config = make_ft_config()
    assert ft_config.load_bf16 is False
    assert ft_config.optim == "adamw_torch"


def test_invalid_optim_rejected():
    with pytest.raises(ValueError, match="optim must be one of"):
        make_ft_config(optim="sgd")


@pytest.mark.parametrize("optim", BITSANDBYTES_OPTIMS)
def test_8bit_optim_without_bitsandbytes_raises_actionable_error(monkeypatch, optim):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("No module named 'bitsandbytes'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError, match="pip install bitsandbytes"):
        make_ft_config(optim=optim)


@pytest.mark.parametrize("optim", BITSANDBYTES_OPTIMS)
def test_8bit_optim_with_bitsandbytes_accepted(monkeypatch, optim):
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    ft_config = make_ft_config(optim=optim)
    assert ft_config.optim == optim


@pytest.mark.parametrize("optim", [o for o in ALLOWED_OPTIMS if o not in BITSANDBYTES_OPTIMS])
def test_non_8bit_optims_do_not_require_bitsandbytes(monkeypatch, optim):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("No module named 'bitsandbytes'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    ft_config = make_ft_config(optim=optim)
    assert ft_config.optim == optim


# ---------------------------------------------------------------------------
# launch_finetune plumbing
# ---------------------------------------------------------------------------


def test_build_experiment_config_defaults():
    from gr00t.experiment.launch_finetune import build_experiment_config

    config = build_experiment_config(make_ft_config())
    assert config.model.load_bf16 is False
    assert config.training.optim == "adamw_torch"
    assert config.model.backbone_trainable_params_fp32 is True


def test_build_experiment_config_propagates_lowmem_flags():
    from gr00t.experiment.launch_finetune import build_experiment_config

    config = build_experiment_config(make_ft_config(load_bf16=True, optim="adamw_torch_fused"))
    assert config.model.load_bf16 is True
    assert config.training.optim == "adamw_torch_fused"
    # The fp32-trainable-params invariant must stay forced regardless of load_bf16.
    assert config.model.backbone_trainable_params_fp32 is True


# ---------------------------------------------------------------------------
# examples/finetune.sh plumbing (stub python captures argv)
# ---------------------------------------------------------------------------


def run_finetune_sh(tmp_path: Path, extra_env: dict[str, str]) -> list[str]:
    """Run examples/finetune.sh with a stub ``python`` and return captured argv."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    argv_file = tmp_path / "argv.txt"
    stub = stub_dir / "python"
    stub.write_text(f'#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "{argv_file}"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{stub_dir}:{env['PATH']}",
            "NUM_GPUS": "1",
            "USE_WANDB": "0",
        }
    )
    env.pop("LOAD_BF16", None)
    env.pop("OPTIM", None)
    env.update(extra_env)

    result = subprocess.run(
        [
            "bash",
            str(FINETUNE_SH),
            "--base-model-path",
            "dummy/base",
            "--dataset-path",
            "dummy/dataset",
            "--embodiment-tag",
            "new_embodiment",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"finetune.sh failed:\n{result.stderr}"
    return argv_file.read_text().splitlines()


def test_finetune_sh_omits_lowmem_flags_by_default(tmp_path):
    argv = run_finetune_sh(tmp_path, extra_env={})
    assert "--load-bf16" not in argv
    assert "--no-load-bf16" not in argv
    assert "--optim" not in argv


def test_finetune_sh_forwards_lowmem_flags_when_env_set(tmp_path):
    argv = run_finetune_sh(tmp_path, extra_env={"LOAD_BF16": "true", "OPTIM": "paged_adamw_8bit"})
    assert "--load-bf16" in argv
    optim_idx = argv.index("--optim")
    assert argv[optim_idx + 1] == "paged_adamw_8bit"


def test_finetune_sh_load_bf16_false_forwards_no_flag(tmp_path):
    argv = run_finetune_sh(tmp_path, extra_env={"LOAD_BF16": "false"})
    assert "--no-load-bf16" in argv
    assert "--load-bf16" not in argv
