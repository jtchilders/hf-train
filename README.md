# hf-train

Run one HuggingFace training experiment per invocation. Configured by YAML, designed for Aurora.

## Install

On Aurora:
```bash
module load frameworks
pip install --no-deps .
# (frameworks provides torch + intel-extension-for-pytorch + transformers from a known-good Intel build)
```

Elsewhere (dev / CUDA / CPU):
```bash
pip install .
```

## Usage

```bash
hf-train run --config /path/to/run.yaml
```

See `examples/` for config samples and `docs/specs/2026-05-24-hf-train-design.md` for the full design.
