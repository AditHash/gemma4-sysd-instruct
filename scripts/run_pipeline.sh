#!/usr/bin/env bash
# One-shot dataset generation + validation + push to HuggingFace
# Usage: bash scripts/run_pipeline.sh [--dry-run]

set -euo pipefail

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[DRY RUN] Processing 1 topic, 5 samples"
fi

echo "=== Step 1: Run dataset generation pipeline ==="
uv run python dataset_generator/pipeline.py $DRY_RUN

echo ""
echo "=== Step 2: Validate dataset ==="
uv run python scripts/validate_dataset.py --input data/final/dataset.jsonl

echo ""
echo "=== Step 3: Push to HuggingFace ==="
if [[ -z "$DRY_RUN" ]]; then
    uv run python dataset_generator/push_to_hub.py
else
    echo "[DRY RUN] Skipping push to HuggingFace"
fi

echo ""
echo "=== Done ==="
