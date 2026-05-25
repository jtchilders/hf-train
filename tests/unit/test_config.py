from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


def _load(yaml_path: Path):
    from hf_train.config import load_config
    return load_config(yaml_path)


def _load_str(yaml_text: str, tmp_path: Path):
    from hf_train.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    return load_config(p)


def test_causal_lm_minimal_loads(fixtures_dir: Path) -> None:
    cfg = _load(fixtures_dir / "configs" / "causal_lm_minimal.yaml")
    assert cfg.task == "causal_lm"
    assert cfg.model.name == "sshleifer/tiny-gpt2"
    assert cfg.training.max_steps == 10


def test_image_classification_minimal_loads(fixtures_dir: Path) -> None:
    cfg = _load(fixtures_dir / "configs" / "image_classification_minimal.yaml")
    assert cfg.task == "image_classification"
    assert cfg.model.num_labels == 10


def test_semantic_segmentation_minimal_loads(fixtures_dir: Path) -> None:
    cfg = _load(fixtures_dir / "configs" / "semantic_segmentation_minimal.yaml")
    assert cfg.task == "semantic_segmentation"
    assert cfg.data.mask_column == "annotation"


def test_both_max_epochs_and_max_steps_set_is_error(tmp_path: Path) -> None:
    bad = """
task: causal_lm
model: {source: hub, name: x}
data: {source: hub, name: y, text_column: text}
training:
  max_epochs: 3
  max_steps: 100
output: {output_dir: /tmp/x}
"""
    with pytest.raises(ValidationError, match="max_epochs.*max_steps"):
        _load_str(bad, tmp_path)


def test_neither_max_epochs_nor_max_steps_set_is_error(tmp_path: Path) -> None:
    bad = """
task: causal_lm
model: {source: hub, name: x}
data: {source: hub, name: y, text_column: text}
training: {}
output: {output_dir: /tmp/x}
"""
    with pytest.raises(ValidationError, match="max_epochs.*max_steps"):
        _load_str(bad, tmp_path)


def test_image_classification_without_num_labels_is_error(tmp_path: Path) -> None:
    bad = """
task: image_classification
model: {source: hub, name: x}
data: {source: hub, name: y, image_column: img, label_column: label}
training: {max_steps: 1}
output: {output_dir: /tmp/x}
"""
    with pytest.raises(ValidationError, match="num_labels"):
        _load_str(bad, tmp_path)


def test_semantic_segmentation_without_mask_column_is_error(tmp_path: Path) -> None:
    bad = """
task: semantic_segmentation
model: {source: hub, name: x, num_labels: 3}
data: {source: hub, name: y, image_column: image}
training: {max_steps: 1}
output: {output_dir: /tmp/x}
"""
    with pytest.raises(ValidationError, match="mask_column"):
        _load_str(bad, tmp_path)


def test_causal_lm_without_text_column_is_error(tmp_path: Path) -> None:
    bad = """
task: causal_lm
model: {source: hub, name: x}
data: {source: hub, name: y}
training: {max_steps: 1}
output: {output_dir: /tmp/x}
"""
    with pytest.raises(ValidationError, match="text_column"):
        _load_str(bad, tmp_path)


def test_eval_strategy_steps_requires_eval_steps(tmp_path: Path) -> None:
    bad = """
task: causal_lm
model: {source: hub, name: x}
data: {source: hub, name: y, text_column: text}
training: {max_steps: 100, eval_strategy: steps}
output: {output_dir: /tmp/x}
"""
    with pytest.raises(ValidationError, match="eval_steps"):
        _load_str(bad, tmp_path)


def test_defaults_applied_causal_lm(fixtures_dir: Path) -> None:
    """max_seq_length should be filled by causal-LM defaults."""
    cfg = _load(fixtures_dir / "configs" / "causal_lm_minimal.yaml")
    assert cfg.data.max_seq_length == 1024


def test_defaults_applied_image_classification(fixtures_dir: Path) -> None:
    """image_size should be filled by image-task defaults."""
    cfg = _load(fixtures_dir / "configs" / "image_classification_minimal.yaml")
    assert cfg.data.image_size == 224


def test_resolved_yaml_roundtrips(fixtures_dir: Path, tmp_path: Path) -> None:
    """Loading, resolving, dumping, and re-loading produces the same config."""
    from hf_train.config import dump_config
    cfg = _load(fixtures_dir / "configs" / "causal_lm_minimal.yaml")
    out_path = tmp_path / "resolved.yaml"
    dump_config(cfg, out_path)
    cfg2 = _load(out_path)
    assert cfg == cfg2


def test_trainer_kwargs_passthrough_preserves_arbitrary_keys(tmp_path: Path) -> None:
    yaml_text = """
task: causal_lm
model: {source: hub, name: x}
data: {source: hub, name: y, text_column: text}
training:
  max_steps: 10
  trainer_kwargs:
    push_to_hub: false
    report_to: ["tensorboard"]
    weird_custom_key: 42
output: {output_dir: /tmp/x}
"""
    cfg = _load_str(yaml_text, tmp_path)
    assert cfg.training.trainer_kwargs == {
        "push_to_hub": False,
        "report_to": ["tensorboard"],
        "weird_custom_key": 42,
    }
