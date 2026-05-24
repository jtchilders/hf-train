# hf-train — Design Spec

**Status:** Draft, approved 2026-05-24
**Repo:** https://github.com/jtchilders/hf-train

## Goal

A standalone Python package that runs one HuggingFace training experiment per invocation, configured by a YAML file, designed to be invoked from a future borealis-mcp ensemble application plugin via `cmd_template`. v1 targets a single Aurora GPU tile (one rank, one tile) and supports three task types: causal LM, image classification, semantic segmentation.

The package is independently useful — researchers can run experiments directly without the MCP layer — but its CLI surface and output layout are the load-bearing contract with the future MCP plugin.

## Non-goals (v1)

- Multi-tile / multi-node distributed training (extends later)
- Bring-your-own-dataset from local files (schema reserves space; implementation defers)
- Custom model architectures outside HF Hub (defers)
- PEFT/LoRA (defers; `training.trainer_kwargs` escape hatch exists)
- TensorBoard / W&B / MLflow integration (escape hatch via `training.trainer_kwargs`)
- Automatic Hub model card generation or `push_to_hub`
- Auto-retry on Hub transient failures or auto-resume on crash
- Instance segmentation, universal segmentation, QA, summarization, translation, audio, speech (all addable later via the task handler interface)

## Repository structure

```
hf-train/
├── src/hf_train/
│   ├── __init__.py
│   ├── cli.py                    # `hf-train run --config X` entry point (Click)
│   ├── backend/
│   │   ├── __init__.py           # auto-detect + factory
│   │   ├── base.py               # Backend abstract interface
│   │   ├── xpu.py                # Aurora (Intel GPU + IPEX)
│   │   ├── cuda.py               # commodity NVIDIA GPU
│   │   └── cpu.py                # fallback for testing
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py             # Pydantic v2 polymorphic schema
│   │   └── defaults.py           # per-task default fillers
│   ├── tasks/
│   │   ├── __init__.py           # HANDLERS registry + get_handler()
│   │   ├── base.py               # TaskHandler ABC
│   │   ├── causal_lm.py
│   │   ├── image_classification.py
│   │   └── semantic_segmentation.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── hub_loader.py         # datasets.load_dataset() wrapper
│   ├── model/
│   │   ├── __init__.py
│   │   └── loader.py             # hub + from_config loaders
│   ├── train/
│   │   ├── __init__.py
│   │   ├── runner.py             # orchestrates handlers + backend + Trainer
│   │   └── callbacks.py          # MetricsJsonlCallback, WalltimeWatchdog
│   └── output/
│       ├── __init__.py
│       └── writer.py             # config.resolved.yaml + results.json
├── tests/
│   ├── unit/                     # config, backend, output (~30 tests)
│   ├── integration/              # handler tests + smoke tests (~18 tests)
│   └── fixtures/configs/         # minimal valid YAML per task
├── examples/
│   ├── causal_lm_finetune_llama.yaml
│   ├── causal_lm_from_scratch.yaml
│   ├── image_classification_vit.yaml
│   └── semantic_segmentation_segformer.yaml
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

## Dependencies

**Hard requirements:** `transformers`, `datasets`, `accelerate`, `torch`, `pydantic>=2`, `click`, `pyyaml`, `evaluate`, `numpy`, `Pillow`.

**Conditional / environment-provided:**
- `intel-extension-for-pytorch` — provided by Aurora's `frameworks` module. Not pip-installed; the `backend/xpu.py` module imports it lazily so the package imports cleanly off-Aurora.
- Recommended Aurora install pattern: `module load frameworks` first, then `pip install hf-train --no-deps` (or install with `--ignore-installed` for the pure-Python additions only). README documents this clearly.

**Dev:** `pytest`, `pytest-cov`, `ruff`.

## CLI surface

```bash
hf-train run --config /path/to/run.yaml [--dry-run] [--log-level LEVEL]
```

Single subcommand for v1. Designed to grow (`inspect`, `eval`, `resume` later) but those are out of scope now.

**Behavior:**
1. Validate config against Pydantic schema; clear errors on failure
2. Resolve defaults (task-specific defaults merged in)
3. Write `<output_dir>/config.resolved.yaml` immediately (before model load)
4. Initialize backend (auto-detect XPU/CUDA/CPU; or honor explicit `training.backend`)
5. Hand off to `train.runner.run(config, backend)`
6. Exit with structured exit code (see table below)

**Flags:**
- `--config PATH` (required) — YAML config path
- `--dry-run` — validate + resolve + write `config.resolved.yaml` + print summary, then exit. No model load, no training. For MCP plugin's pre-submission validation.
- `--log-level [debug|info|warning|error]` — Python logging level for our code (default info)

**Startup banner (stderr):** `hf-train <version> | backend=<xpu|cuda|cpu> | task=<causal_lm|...>` so the first line of any captured log shows the environment.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Training completed cleanly (stopping criterion triggered, `results.json` written) |
| 1 | Config validation failed |
| 2 | Backend init failed (e.g., XPU requested but IPEX missing) |
| 3 | Dataset load failed (not found, wrong shape) |
| 4 | Model load failed (not found, AutoClass mismatch, num_labels missing for vision) |
| 5 | Training failed mid-run (caught exception) |
| 130 | SIGINT / SIGTERM (walltime kill, user interrupt) |

## Config schema (YAML)

```yaml
task: causal_lm | image_classification | semantic_segmentation

model:
  source: hub | from_config       # hub = pretrained; from_config = architecture-only, random init
  name: meta-llama/Llama-3.2-1B   # always a Hub identifier
  revision: main                  # default "main"
  trust_remote_code: false        # default false
  config_overrides:               # only meaningful for source=from_config
    n_layer: 6
    hidden_size: 384
  num_labels: null                # required for image_classification + semantic_segmentation
  id2label: null                  # optional label map for vision tasks

data:
  source: hub                     # v1: only "hub". Schema reserves "local"/"jsonl" for v2.
  name: wikitext                  # Hub dataset name
  config: wikitext-2-raw-v1       # optional Hub dataset config
  train_split: train              # default "train"
  eval_split: validation          # default "validation"; null = no eval
  text_column: text               # causal_lm only
  image_column: image             # image tasks only
  label_column: label             # image_classification only
  mask_column: annotation         # semantic_segmentation only
  max_seq_length: 1024            # causal_lm only
  image_size: 224                 # image tasks only

training:
  backend: auto                   # auto | xpu | cuda | cpu
  precision: bf16                 # fp32 | bf16 | fp16 (bf16 recommended on XPU)
  torch_compile: false            # warn if true on XPU backend in v1
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 1
  gradient_checkpointing: false
  max_epochs: null                # exactly one of max_epochs / max_steps required
  max_steps: null
  max_train_seconds: null         # wall-clock safety net (WalltimeWatchdog enforces)
  early_stopping:                 # optional
    patience: 3
    metric: eval_loss
    mode: min
    threshold: 0.0
  eval_strategy: epoch            # no | epoch | steps
  eval_steps: null                # required if eval_strategy == steps
  logging_steps: 50
  save_strategy: epoch            # no | epoch | steps
  save_steps: null                # required if save_strategy == steps
  save_total_limit: 3
  seed: 42
  trainer_kwargs: {}              # free-form escape hatch passed to HF TrainingArguments

output:
  output_dir: /lus/.../run42      # required, becomes run_dir
  overwrite: false                # if true, allow writing into a non-empty dir
  run_name: null                  # optional human label written into results.json
  notes: null                     # free-text user notes
```

### Validation rules

- Exactly one of `training.max_epochs` or `training.max_steps` non-null
- `model.num_labels` required when `task ∈ {image_classification, semantic_segmentation}`
- `data.text_column` required when `task == causal_lm`
- `data.mask_column` required when `task == semantic_segmentation`
- `training.eval_steps` required when `eval_strategy == "steps"`; same for save
- `model.source == "from_config"` with empty `config_overrides` → warning
- `training.precision == "fp16"` on XPU backend → warning (poor support)
- `training.torch_compile == true` on XPU backend → warning (unreliable with current PyTorch+IPEX)
- `output.output_dir` non-empty + `overwrite == false` → validation error

### Defaults

- Universal defaults declared in schema fields
- Task-specific defaults applied in `config/defaults.py` after the task discriminator is known (e.g., causal_lm gets `max_seq_length=1024`; vision tasks get `image_size=224`)
- All applied defaults are reflected in `config.resolved.yaml` (reproducibility intact)

## Task handler pattern

```python
class TaskHandler(ABC):
    task_name: str
    auto_model_class: type

    def prepare_model(self, model_cfg, num_labels=None) -> PreTrainedModel: ...
    def prepare_preprocessor(self, model_name) -> Any: ...                    # tokenizer or image_processor
    def prepare_dataset(self, data_cfg, preprocessor, split) -> Dataset: ...
    def make_collator(self, preprocessor) -> Callable: ...
    def make_compute_metrics(self) -> Callable | None: ...
    def validate_config(self, config) -> None: ...                            # optional, beyond schema-level
```

Registry in `tasks/__init__.py`:

```python
HANDLERS = {
    "causal_lm": CausalLMHandler,
    "image_classification": ImageClassificationHandler,
    "semantic_segmentation": SemanticSegmentationHandler,
}
```

Adding a task is: write one `tasks/<name>.py`, add one line to `HANDLERS`. No runner / schema / CLI changes.

### Per-task specifics

| Task | AutoClass | Preprocessor | Collator | Metric |
|---|---|---|---|---|
| `causal_lm` | `AutoModelForCausalLM` | `AutoTokenizer` | `DataCollatorForLanguageModeling(mlm=False)` | Perplexity (from eval_loss) |
| `image_classification` | `AutoModelForImageClassification` | `AutoImageProcessor` | Custom (stack pixel_values + labels) | Accuracy + macro F1 |
| `semantic_segmentation` | `AutoModelForSemanticSegmentation` | `AutoImageProcessor` | Custom (pad images + masks to common size) | Mean IoU (via `evaluate.load("mean_iou")`) |

**Vision tasks** split train/eval transforms: train applies augmentation (RandomResizedCrop + flip); eval applies deterministic resize + center crop. Masks (segmentation) are resized jointly with images, never normalized.

**Causal LM** tokenizes the `text_column`, truncates/packs into fixed-length blocks of `max_seq_length`, follows the standard HF causal-LM recipe.

## Backend abstraction

```python
class Backend(ABC):
    device: str                                # "xpu" | "cuda" | "cpu"

    def setup_env(self) -> None: ...           # accelerate env vars before Trainer init
    def optimize(self, model, optimizer) -> tuple[Model, Optimizer]: ...
    def report_capabilities(self) -> dict: ...  # supported dtypes, etc.
```

- **`xpu.py`**: imports IPEX lazily (only on instantiation, not module import). `setup_env` sets `ACCELERATE_USE_XPU=true`. `optimize` calls `ipex.optimize(model, optimizer=optimizer, dtype=<from precision>)`.
- **`cuda.py`**: no-op `setup_env`, no-op `optimize`.
- **`cpu.py`**: same as cuda, but with `device="cpu"`.

`__init__.py` provides `auto_detect()` returning the right backend; `from_config(training.backend)` honors explicit selection. Off-Aurora developers can import the package and run CPU-backed tests without IPEX installed.

## Output directory layout

```
<output_dir>/                          # = run_dir
├── config.resolved.yaml               # written at startup, before model load
├── metrics.jsonl                      # appended every logging_steps + every eval (minimal schema)
├── train.log                          # our Python logger output
├── checkpoints/
│   ├── checkpoint-500/                # HF Trainer native format
│   │   ├── (model weights, optimizer.pt, trainer_state.json, training_args.bin, preprocessor)
│   └── checkpoint-1000/
├── trainer_state.json                 # HF native, written at end of training
└── results.json                       # our final summary
```

### `metrics.jsonl` (minimal schema)

One JSON object per line. Different rows have different keys; only the keys that apply are included.

```jsonl
{"step": 50, "epoch": 0.1, "phase": "train", "loss": 2.41, "learning_rate": 4.9e-5, "wall_seconds": 12.3}
{"step": 100, "epoch": 0.2, "phase": "eval", "eval_loss": 2.05, "eval_runtime": 4.1, "eval_samples_per_second": 120.5}
```

Appended in real time so partial metrics survive crashes / walltime kills. Compatible with borealis-mcp's existing `parse_training_metrics` tool which already reads JSON-per-line.

Written by `MetricsJsonlCallback`, an HF `TrainerCallback` subclass with `on_log` + `on_evaluate` hooks.

### `results.json`

```json
{
  "status": "completed",
  "run_name": "llama_3_2_1b_finetune",
  "task": "causal_lm",
  "stopped_by": "early_stopping",
  "total_steps": 3500,
  "total_epochs": 3.5,
  "total_wall_seconds": 4287.2,
  "best_checkpoint": "/lus/.../run42/checkpoints/checkpoint-2500",
  "best_method": "early_stopping",
  "best_metric": {"name": "eval_loss", "value": 1.84, "mode": "min"},
  "final_metric": {"name": "eval_loss", "value": 1.91},
  "config_path": "/lus/.../run42/config.resolved.yaml"
}
```

- `stopped_by`: `max_epochs` | `max_steps` | `early_stopping` | `max_train_seconds` | `error`
- `best_method`: `"early_stopping"` when trainer_state has `best_model_checkpoint`, `"latest"` otherwise (defaults to latest checkpoint)
- On error: `status="error"`, `error_type`, `error_message`, truncated traceback also in `train.log`
- Written inside a top-level `try/finally` so it fires on success AND failure

### `WalltimeWatchdog`

Required for HPC: PBS sends SIGTERM at walltime mid-checkpoint. `TrainerCallback` polls `time.monotonic()` in `on_step_end`; when elapsed ≥ `max_train_seconds`, sets `control.should_save = True` and `control.should_training_stop = True`. Trainer saves final checkpoint and exits cleanly. We write `results.json` with `stopped_by="max_train_seconds"`.

Users are expected to set `max_train_seconds` 5–10% below PBS walltime (documented; not enforced).

### `train.log`

Standard Python `logging` module, file handler at `<run_dir>/train.log`. Startup banner mirrors CLI banner. INFO messages at phase boundaries (config loaded, dataset prepared, model loaded, training started, checkpoint saved, eval completed). Separate from stdout/stderr so ensemble_launcher's per-task log capture stays clean.

## Error handling

| Failure | Detection | Exit | results.json | Where agent sees it |
|---|---|---|---|---|
| Config validation | CLI startup (Pydantic) | 1 | Not written | stderr |
| Backend init | Backend factory | 2 | Not written | stderr |
| Dataset load | `handler.prepare_dataset` | 3 | `error_type="data_load"` | run_dir |
| Model load | `handler.prepare_model` | 4 | `error_type="model_load"` | run_dir |
| Training crash | `trainer.train()` | 5 | `error_type="training"` + traceback in train.log | run_dir |
| SIGTERM | Signal handler | 130 | `error_type="interrupted"` if past config-resolve phase | run_dir if applicable |

Top-level `try/finally` in `runner.run` ensures `results.json` always gets a structured summary. Custom `DataLoadError` and `ModelLoadError` exception classes wrap `huggingface_hub.errors.*` and `OSError` into a small known taxonomy.

**No retries in v1.** Failures surface immediately. Resume from checkpoint is a future `hf-train resume --run-dir X` subcommand.

## Testing strategy

**~50–70 tests, all runnable on a CPU dev machine (no Aurora, no GPU).**

### Layer 1: Unit tests (~30 tests, fast, no model loads)

- `tests/unit/test_config.py` (~12): valid configs per task, validation errors for each rule, defaults application, schema round-trip, `trainer_kwargs` passthrough, warning emission
- `tests/unit/test_backend.py` (~8): auto-detect with each backend mocked, explicit backend with missing deps, optimize is no-op on cpu/cuda, correct device strings
- `tests/unit/test_output.py` (~10): metrics.jsonl append + minimal schema, results.json for completed/error/interrupted, best_method derivation, config.resolved.yaml early write, overwrite behavior

### Layer 2: Task handler tests with tiny models (~15 tests, ~30 seconds)

Real HF models, trivially small. Test models:
- Causal LM: `sshleifer/tiny-gpt2`
- Image classification: `hf-internal-testing/tiny-random-ViTModel`
- Semantic segmentation: `hf-internal-testing/tiny-random-SegformerForSemanticSegmentation`

Per task (~5 tests each): `prepare_model`, `prepare_preprocessor`, `prepare_dataset` against tiny Hub dataset, `make_collator` produces accepted batches, `make_compute_metrics` returns expected keys.

Marked `pytest.mark.network` so CI can opt out when Hub is gated.

### Layer 3: End-to-end smoke tests (~3 tests, ~2 minutes)

One per task. Run full `hf-train run --config X` via `click.testing.CliRunner` for 2-3 steps. Assert: exit 0, `results.json` exists with `status="completed"`, `metrics.jsonl` has ≥2 train rows, `config.resolved.yaml` exists, `train.log` non-empty.

### Out of v1 test scope

- XPU backend execution tests (no hardware in CI; manual smoke on Aurora documented)
- Multi-GPU / distributed (v1 is single-tile only)
- Convergence ("does loss go down") — user's config concern
- Re-testing HF Trainer internals (trust upstream tests)

### CI

GitHub Actions, single Python 3.10 job, CPU-only. Runs Layer 1 + 2 + 3. Target wall time: <5 minutes. `pytest-cov` instrumented from day one; no coverage gate enforced in v1.

## Open implementation details (resolve during plan)

- Confirm exact `evaluate.load("mean_iou")` runtime behavior on Aurora compute nodes (Hub access at metric-load time)
- Confirm `accelerate` XPU plugin's preferred env-var name (`ACCELERATE_USE_XPU` may differ across accelerate versions)
- Confirm `sshleifer/tiny-gpt2` and `hf-internal-testing/*` tiny models are still on the Hub and have the schema we expect
- Confirm the exact `frameworks` module's PyTorch/IPEX/transformers versions to pin a known-good Python version (3.10 likely fine; verify)

## Decisions log

- **YAML config**, not pure CLI flags (single source of truth for reproducibility)
- **Flat run_dir layout** with conventional names (compatible with borealis-mcp observability tools)
- **Minimal metrics.jsonl schema** (only keys that apply per row)
- **`best_method = "latest"` fallback** when no early-stopping metric tracked
- **`overwrite: false` default** (prevents accidental wipes during sweeps)
- **No `peft` / `tensorboard` / `wandb` in v1** (escape hatch via `trainer_kwargs`)
- **Backend auto-detect, explicit override allowed** (`auto` in shipped examples; `xpu` in Aurora-specific examples)
- **Single `run` subcommand** (designed to grow; `inspect` / `eval` / `resume` defer)
