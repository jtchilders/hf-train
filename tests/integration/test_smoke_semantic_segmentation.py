"""End-to-end smoke test for semantic segmentation via the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

pytestmark = pytest.mark.network


def test_semantic_segmentation_2_steps_completes(tmp_path: Path) -> None:
    from hf_train.cli import main
    cfg = {
        "task": "semantic_segmentation",
        "model": {
            "source": "hub",
            "name": "hf-internal-testing/tiny-random-SegformerForSemanticSegmentation",
            # FoodSeg103 mask values cover [0, 103]; tiny Segformer is initialized
            # with ignore_mismatched_sizes=True so the head is reinitialized for
            # the larger class count.
            "num_labels": 104,
        },
        "data": {
            "source": "hub", "name": "EduardoPacheco/FoodSeg103",
            "image_column": "image", "mask_column": "label",
            "train_split": "train[:4]", "eval_split": None,
            "image_size": 64,
        },
        "training": {
            "backend": "cpu", "precision": "fp32",
            "max_steps": 2, "per_device_train_batch_size": 1,
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
