"""Click CLI entry point: `hf-train run --config X`."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import click

from hf_train import __version__
from hf_train.config import load_config
from hf_train.output import prepare_run_dir, write_config_resolved


def _setup_signal_handlers() -> None:
    def _raise(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _raise)
    signal.signal(signal.SIGINT, _raise)


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """hf-train: HuggingFace training tool."""


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--dry-run", is_flag=True, help="Validate config + write resolved YAML; do not train.")
@click.option("--log-level", type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False), default="info")
def run(config_path: Path, dry_run: bool, log_level: str) -> None:
    """Run one training experiment from a YAML config."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = load_config(config_path)
    except Exception as e:
        click.echo(f"Config validation failed:\n{e}", err=True)
        sys.exit(1)

    # Startup banner
    from hf_train.backend import get_backend
    backend = get_backend(config.training.backend)
    click.echo(
        f"hf-train {__version__} | backend={backend.device} | task={config.task}",
        err=True,
    )

    if dry_run:
        run_dir = prepare_run_dir(config.output.output_dir, overwrite=config.output.overwrite)
        write_config_resolved(run_dir, config.model_dump(mode="json"))
        click.echo(f"Dry run OK. Resolved config: {run_dir / 'config.resolved.yaml'}", err=True)
        sys.exit(0)

    _setup_signal_handlers()

    # Lazy import: defer heavy ML imports until we actually run
    from hf_train.train.runner import run as run_training
    sys.exit(run_training(config))


if __name__ == "__main__":
    main()
