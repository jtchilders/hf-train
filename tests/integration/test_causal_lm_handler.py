"""Causal LM handler tests using sshleifer/tiny-gpt2 (tiny test fixture model)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.network


@pytest.fixture
def handler():
    from hf_train.tasks import get_handler
    return get_handler("causal_lm")


def test_prepare_preprocessor_returns_tokenizer(handler) -> None:
    tok = handler.prepare_preprocessor("sshleifer/tiny-gpt2")
    encoded = tok("hello world", return_tensors="pt")
    assert "input_ids" in encoded


def test_prepare_model_hub(handler) -> None:
    from transformers import GPT2LMHeadModel
    from hf_train.config.schema import ModelConfig
    cfg = ModelConfig(source="hub", name="sshleifer/tiny-gpt2")
    model = handler.prepare_model(cfg)
    assert isinstance(model, GPT2LMHeadModel)


def test_prepare_model_from_config_uses_overrides(handler) -> None:
    from hf_train.config.schema import ModelConfig
    cfg = ModelConfig(
        source="from_config",
        name="sshleifer/tiny-gpt2",
        config_overrides={"n_layer": 1},
    )
    model = handler.prepare_model(cfg)
    assert model.config.n_layer == 1


def test_prepare_dataset_tokenizes(handler) -> None:
    from hf_train.config.schema import DataConfig
    cfg = DataConfig(
        source="hub",
        name="wikitext",
        config="wikitext-2-raw-v1",
        text_column="text",
        max_seq_length=64,
    )
    tok = handler.prepare_preprocessor("sshleifer/tiny-gpt2")
    ds = handler.prepare_dataset(cfg, tok, split="train[:20]")
    assert len(ds) > 0
    item = ds[0]
    assert "input_ids" in item
    assert len(item["input_ids"]) == 64


def test_collator_produces_model_acceptable_batch(handler) -> None:
    from hf_train.config.schema import DataConfig
    cfg = DataConfig(
        source="hub", name="wikitext", config="wikitext-2-raw-v1",
        text_column="text", max_seq_length=64,
    )
    tok = handler.prepare_preprocessor("sshleifer/tiny-gpt2")
    ds = handler.prepare_dataset(cfg, tok, split="train[:8]")
    collator = handler.make_collator(tok)
    batch = collator([ds[i] for i in range(2)])
    assert "input_ids" in batch
    assert "labels" in batch
