"""Image classification task handler."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from hf_train.tasks.base import TaskHandler


def _build_collator(processor):
    def collate(features):
        pixel_values = torch.stack(
            [
                f["pixel_values"]
                if isinstance(f["pixel_values"], torch.Tensor)
                else torch.as_tensor(f["pixel_values"], dtype=torch.float32)
                for f in features
            ]
        )
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}
    return collate


def _compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    # Macro-F1 without sklearn dependency
    classes = np.unique(np.concatenate([labels, preds]))
    f1_per_class = []
    for c in classes:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
    return {"accuracy": accuracy, "f1": float(np.mean(f1_per_class))}


class ImageClassificationHandler(TaskHandler):
    task_name = "image_classification"
    auto_model_class = AutoModelForImageClassification

    def prepare_preprocessor(self, model_name: str, trust_remote_code: bool = False) -> Any:
        return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    def prepare_model(self, model_cfg: Any, num_labels: Optional[int] = None) -> Any:
        if model_cfg.source == "hub":
            return AutoModelForImageClassification.from_pretrained(
                model_cfg.name,
                num_labels=num_labels,
                revision=model_cfg.revision,
                trust_remote_code=model_cfg.trust_remote_code,
                ignore_mismatched_sizes=True,
            )
        config = AutoConfig.from_pretrained(
            model_cfg.name,
            num_labels=num_labels,
            revision=model_cfg.revision,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        for k, v in (model_cfg.config_overrides or {}).items():
            setattr(config, k, v)
        return AutoModelForImageClassification.from_config(
            config, trust_remote_code=model_cfg.trust_remote_code
        )

    def prepare_dataset(self, data_cfg: Any, preprocessor: Any, split: str) -> Any:
        ds = load_dataset(data_cfg.name, data_cfg.config, split=split)
        image_col = data_cfg.image_column
        label_col = data_cfg.label_column

        def preprocess(batch):
            images = batch[image_col]
            inputs = preprocessor(images=images, return_tensors="pt")
            return {
                "pixel_values": [pv for pv in inputs["pixel_values"]],
                "labels": batch[label_col],
            }

        return ds.map(preprocess, batched=True, batch_size=16, remove_columns=ds.column_names)

    def make_collator(self, preprocessor: Any) -> Callable:
        return _build_collator(preprocessor)

    def make_compute_metrics(self) -> Callable:
        return _compute_metrics
