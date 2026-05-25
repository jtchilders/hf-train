"""End-to-end smoke test for image classification via the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

pytestmark = pytest.mark.network


def test_image_classification_2_steps_completes(tmp_path: Path) -> None:
    from hf_train.cli import main
    cfg = {
        "task": "image_classification",
        "model": {
            "source": "hub",
            "name": "hf-internal-testing/tiny-random-ViTForImageClassification",
            "num_labels": 10,
        },
        "data": {
            "source": "hub", "name": "uoft-cs/cifar10",
            "image_column": "img", "label_column": "label",
            "train_split": "train[:16]", "eval_split": None,
            "image_size": 32,
        },
        "training": {
            "backend": "cpu", "precision": "fp32",
            "max_steps": 2, "per_device_train_batch_size": 2,
            "eval_strategy": "no", "save_strategy": "no", "logging_steps": 1,
        },
        "output": {"output_dir": str(tmp_path / "run")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    out = tmp_path / "run"
    assert json.loads((out / "results.json").read_text())["status"] == "completed"
