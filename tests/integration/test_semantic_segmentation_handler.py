"""Semantic segmentation handler tests using a tiny Segformer."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.network

_TINY_SEG = "hf-internal-testing/tiny-random-SegformerForSemanticSegmentation"

# Verified in Step 1: 'scene_parse_150' (and 'zhoubolei/scene_parse_150') ship
# loader scripts which datasets 4.x refuses to run ("Dataset scripts are no
# longer supported"). EduardoPacheco/FoodSeg103 is a script-free Parquet
# dataset with an 'image' (JPEG) column and 'label' (PNG mask) column.
_SEG_NAME = "EduardoPacheco/FoodSeg103"
_IMAGE_COL = "image"
_MASK_COL = "label"
_NUM_LABELS = 3   # tiny Segformer test fixture has 3 classes by default


@pytest.fixture
def handler():
    from hf_train.tasks import get_handler
    return get_handler("semantic_segmentation")


def test_prepare_preprocessor_returns_image_processor(handler) -> None:
    proc = handler.prepare_preprocessor(_TINY_SEG)
    assert hasattr(proc, "__call__")


def test_prepare_model_hub(handler) -> None:
    from transformers import SegformerForSemanticSegmentation
    from hf_train.config.schema import ModelConfig
    cfg = ModelConfig(source="hub", name=_TINY_SEG, num_labels=_NUM_LABELS)
    model = handler.prepare_model(cfg, num_labels=_NUM_LABELS)
    assert isinstance(model, SegformerForSemanticSegmentation)
    assert model.config.num_labels == _NUM_LABELS


def test_prepare_dataset_pairs_image_and_mask(handler) -> None:
    from hf_train.config.schema import DataConfig
    cfg = DataConfig(
        source="hub", name=_SEG_NAME,
        image_column=_IMAGE_COL, mask_column=_MASK_COL, image_size=64,
    )
    proc = handler.prepare_preprocessor(_TINY_SEG)
    ds = handler.prepare_dataset(cfg, proc, split="train[:4]")
    assert len(ds) == 4
    assert "pixel_values" in ds[0]
    assert "labels" in ds[0]


def test_collator_pads_to_common_size(handler) -> None:
    import torch
    from hf_train.config.schema import DataConfig
    cfg = DataConfig(
        source="hub", name=_SEG_NAME,
        image_column=_IMAGE_COL, mask_column=_MASK_COL, image_size=64,
    )
    proc = handler.prepare_preprocessor(_TINY_SEG)
    ds = handler.prepare_dataset(cfg, proc, split="train[:2]")
    collator = handler.make_collator(proc)
    batch = collator([ds[i] for i in range(2)])
    assert isinstance(batch["pixel_values"], torch.Tensor)
    assert batch["pixel_values"].shape[0] == 2
    assert batch["labels"].shape[0] == 2


def test_compute_metrics_returns_mean_iou(handler) -> None:
    import numpy as np
    fn = handler.make_compute_metrics()
    assert fn is not None
    # logits: (batch=1, num_classes=3, H=4, W=4); labels: (batch=1, H=4, W=4)
    logits = np.random.randn(1, 3, 4, 4).astype(np.float32)
    labels = np.zeros((1, 4, 4), dtype=np.int64)
    result = fn(type("EvalPred", (), {"predictions": logits, "label_ids": labels})())
    assert "mean_iou" in result
