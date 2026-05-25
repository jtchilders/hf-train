"""Image classification handler tests using a tiny ViT test fixture."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.network

_TINY_VIT = "hf-internal-testing/tiny-random-ViTForImageClassification"

# Verified in Step 1: 'cifar10' raises HfUriError under datasets 4.x; the
# canonical namespaced id 'uoft-cs/cifar10' resolves and exposes columns
# 'img' and 'label'.
_CIFAR_NAME = "uoft-cs/cifar10"
_IMAGE_COL = "img"
_LABEL_COL = "label"


@pytest.fixture
def handler():
    from hf_train.tasks import get_handler
    return get_handler("image_classification")


def test_prepare_preprocessor_returns_image_processor(handler) -> None:
    proc = handler.prepare_preprocessor(_TINY_VIT)
    assert hasattr(proc, "__call__")


def test_prepare_model_hub(handler) -> None:
    from transformers import ViTForImageClassification
    from hf_train.config.schema import ModelConfig
    cfg = ModelConfig(source="hub", name=_TINY_VIT, num_labels=10)
    model = handler.prepare_model(cfg, num_labels=10)
    assert isinstance(model, ViTForImageClassification)
    assert model.config.num_labels == 10


def test_prepare_dataset_returns_processed(handler) -> None:
    from hf_train.config.schema import DataConfig
    cfg = DataConfig(
        source="hub", name=_CIFAR_NAME,
        image_column=_IMAGE_COL, label_column=_LABEL_COL, image_size=32,
    )
    proc = handler.prepare_preprocessor(_TINY_VIT)
    ds = handler.prepare_dataset(cfg, proc, split="train[:8]")
    assert len(ds) == 8
    assert "pixel_values" in ds[0]
    assert "labels" in ds[0]


def test_collator_stacks_pixel_values(handler) -> None:
    import torch
    from hf_train.config.schema import DataConfig
    cfg = DataConfig(
        source="hub", name=_CIFAR_NAME,
        image_column=_IMAGE_COL, label_column=_LABEL_COL, image_size=32,
    )
    proc = handler.prepare_preprocessor(_TINY_VIT)
    ds = handler.prepare_dataset(cfg, proc, split="train[:4]")
    collator = handler.make_collator(proc)
    batch = collator([ds[i] for i in range(4)])
    assert isinstance(batch["pixel_values"], torch.Tensor)
    assert batch["pixel_values"].shape[0] == 4
    assert "labels" in batch


def test_compute_metrics_returns_accuracy_and_f1(handler) -> None:
    import numpy as np
    fn = handler.make_compute_metrics()
    assert fn is not None
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
    labels = np.array([1, 0, 0, 0])
    result = fn(type("EvalPred", (), {"predictions": logits, "label_ids": labels})())
    assert "accuracy" in result
    assert "f1" in result
