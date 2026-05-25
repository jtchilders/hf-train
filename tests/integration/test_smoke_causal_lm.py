"""End-to-end smoke test for causal LM via the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

pytestmark = pytest.mark.network


def test_causal_lm_2_steps_completes(tmp_path: Path) -> None:
    from hf_train.cli import main
    cfg = {
        "task": "causal_lm",
        "model": {"source": "hub", "name": "sshleifer/tiny-gpt2"},
        "data": {
            "source": "hub", "name": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
            "text_column": "text", "train_split": "train[:32]", "eval_split": None,
            "max_seq_length": 32,
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
    assert (out / "config.resolved.yaml").exists()
    assert (out / "results.json").exists()
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "completed"
    assert results["stopped_by"] == "max_steps"

    metrics = (out / "metrics.jsonl").read_text().splitlines()
    assert len(metrics) >= 2

    assert (out / "train.log").stat().st_size > 0
