"""Causal language modeling task handler."""

from __future__ import annotations

from typing import Any, Callable, Optional

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from hf_train.tasks.base import TaskHandler


class CausalLMHandler(TaskHandler):
    task_name = "causal_lm"
    auto_model_class = AutoModelForCausalLM

    def prepare_preprocessor(self, model_name: str, trust_remote_code: bool = False) -> Any:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def prepare_model(self, model_cfg: Any, num_labels: Optional[int] = None) -> Any:
        if model_cfg.source == "hub":
            return AutoModelForCausalLM.from_pretrained(
                model_cfg.name,
                revision=model_cfg.revision,
                trust_remote_code=model_cfg.trust_remote_code,
            )
        # from_config: load architecture, apply overrides, init random weights
        config = AutoConfig.from_pretrained(
            model_cfg.name,
            revision=model_cfg.revision,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        for k, v in (model_cfg.config_overrides or {}).items():
            setattr(config, k, v)
        return AutoModelForCausalLM.from_config(config, trust_remote_code=model_cfg.trust_remote_code)

    def prepare_dataset(self, data_cfg: Any, preprocessor: Any, split: str) -> Any:
        ds = load_dataset(data_cfg.name, data_cfg.config, split=split)
        text_col = data_cfg.text_column

        def tokenize(batch):
            return preprocessor(batch[text_col], return_special_tokens_mask=False)

        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

        # Group into fixed-length blocks
        block_size = data_cfg.max_seq_length

        def group(examples):
            concatenated = {k: sum(examples[k], []) for k in examples}
            total = (len(concatenated["input_ids"]) // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total, block_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = list(result["input_ids"])
            return result

        return ds.map(group, batched=True)

    def make_collator(self, preprocessor: Any) -> Callable:
        return DataCollatorForLanguageModeling(tokenizer=preprocessor, mlm=False)
