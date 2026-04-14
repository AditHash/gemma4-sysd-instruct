"""Validate a JSONL dataset file before pushing to HuggingFace."""

import json
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from dataset_generator.schemas.sample import Sample

console = Console()
app = typer.Typer()

MIN_SAMPLES = 400


@app.command()
def validate(
    input: Path = typer.Option(..., "--input", "-i", help="Path to JSONL dataset file"),
    min_samples: int = typer.Option(MIN_SAMPLES, "--min-samples", help="Minimum required samples"),
) -> None:
    """Validate schema, count, and basic quality of a JSONL dataset."""
    if not input.exists():
        logger.error(f"File not found: {input}")
        raise typer.Exit(1)

    lines = input.read_text(encoding="utf-8").splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    valid: list[Sample] = []
    errors: list[tuple[int, str]] = []

    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
            sample = Sample(**data)
            valid.append(sample)
        except json.JSONDecodeError as exc:
            errors.append((i, f"JSON parse error: {exc}"))
        except Exception as exc:
            errors.append((i, str(exc)))

    # ── Summary table ──────────────────────────────────────────────────────────
    table = Table(title="Dataset Validation Report")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="green")
    table.add_column("Status")

    table.add_row("Total lines", str(len(lines)), "")
    table.add_row(
        "Valid samples",
        str(len(valid)),
        "✅" if len(valid) >= min_samples else "❌",
    )
    table.add_row(
        "Parse errors",
        str(len(errors)),
        "✅" if not errors else "❌",
    )
    table.add_row(
        "Min samples met",
        f"{len(valid)} >= {min_samples}",
        "✅" if len(valid) >= min_samples else "❌",
    )

    console.print(table)

    if errors:
        console.print("\n[red]Errors:[/red]")
        for lineno, msg in errors[:20]:
            console.print(f"  Line {lineno}: {msg}")

    if len(valid) < min_samples:
        logger.error(f"Too few valid samples: {len(valid)} < {min_samples}")
        raise typer.Exit(1)

    if errors:
        logger.error(f"{len(errors)} validation errors found")
        raise typer.Exit(1)

    logger.success(f"Dataset valid: {len(valid)} samples ready to push.")


if __name__ == "__main__":
    app()
