#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "  SmolLM-135M Base Model Evaluation Suite"
echo "=========================================="

echo ""
echo "=== Step 1: Perplexity on Medical Text ==="
uv run python evals/perplexity.py

echo ""
echo "=== Step 2: Medical MCQ Benchmarks ==="
uv run python evals/benchmarks.py

echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "  Results in: results/"
echo "=========================================="
