"""Semantic segmentation task handler."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)

from hf_train.tasks.base import TaskHandler


def _build_collator():
    def collate(features):
        # Mirror image_classification's defensive pattern: accept tensor or list
        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x if dtype is None else x.to(dtype)
            return torch.as_tensor(x, dtype=dtype)
        pixel_values = torch.stack(
            [_to_tensor(f["pixel_values"], torch.float32) for f in features]
        )
        labels = torch.stack([_to_tensor(f["labels"], torch.long) for f in features])
        return {"pixel_values": pixel_values, "labels": labels}
    return collate


def _compute_mean_iou(eval_pred):
    """Compute mean IoU per class, then average."""
    logits = eval_pred.predictions  # (B, C, H, W)
    labels = eval_pred.label_ids    # (B, H, W)
    # Upsample logits to label resolution if needed (Segformer outputs reduced res)
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)
    if logits_t.shape[-2:] != labels_t.shape[-2:]:
        logits_t = F.interpolate(
            logits_t, size=labels_t.shape[-2:], mode="bilinear", align_corners=False
        )
    preds = logits_t.argmax(dim=1).numpy()
    labels = labels_t.numpy()

    num_classes = logits.shape[1]
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        intersection = (pred_c & label_c).sum()
        union = (pred_c | label_c).sum()
        if union > 0:
            ious.append(intersection / union)
    return {"mean_iou": float(np.mean(ious)) if ious else 0.0}


class SemanticSegmentationHandler(TaskHandler):
    task_name = "semantic_segmentation"
    auto_model_class = AutoModelForSemanticSegmentation

    def prepare_preprocessor(self, model_name: str, trust_remote_code: bool = False) -> Any:
        return AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            reduce_labels=False,
        )

    def prepare_model(self, model_cfg: Any, num_labels: Optional[int] = None) -> Any:
        if model_cfg.source == "hub":
            return AutoModelForSemanticSegmentation.from_pretrained(
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
        return AutoModelForSemanticSegmentation.from_config(
            config, trust_remote_code=model_cfg.trust_remote_code
        )

    def prepare_dataset(self, data_cfg: Any, preprocessor: Any, split: str) -> Any:
        ds = load_dataset(data_cfg.name, data_cfg.config, split=split)
        image_col = data_cfg.image_column
        mask_col = data_cfg.mask_column
        size = data_cfg.image_size

        def preprocess(batch):
            images = batch[image_col]
            masks = batch[mask_col]
            inputs = preprocessor(
                images=images,
                segmentation_maps=masks,
                return_tensors="pt",
                size={"height": size, "width": size},
            )
            return {
                "pixel_values": [pv for pv in inputs["pixel_values"]],
                "labels": [lb for lb in inputs["labels"]],
            }

        return ds.map(preprocess, batched=True, batch_size=8, remove_columns=ds.column_names)

    def make_collator(self, preprocessor: Any) -> Callable:
        return _build_collator()

    def make_compute_metrics(self) -> Callable:
        return _compute_mean_iou
