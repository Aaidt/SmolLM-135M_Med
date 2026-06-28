#!/bin/bash

# SmolLM-135M Medical Training Pipeline Runner
# This script orchestrates the complete pipeline:
# 1. Dataset loading
# 2. Pre-training evaluations
# 3. Model training
# 4. Post-training evaluations
# 5. Results comparison and logging

set -e  # Exit on error

echo "=========================================="
echo "SmolLM-135M Medical Training Pipeline"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check if required modules are available
echo "Checking dependencies..."
python -c "import torch; import transformers; import datasets; import unsloth" 2>/dev/null || {
    echo "⚠️  Some dependencies may be missing. Running: uv sync"
    uv sync
}

# Create results directory
mkdir -p results
echo "✓ Results directory ready: results/"
echo ""

# Check if dataset files exist
TRAIN_FILE="dataset/data/train.txt"
VAL_FILE="dataset/data/val.txt"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "⚠️  Dataset files not found."
    echo "   Run: python dataset/load.py"
    echo "   Then try again."
    exit 1
fi

echo "✓ Dataset files found"
echo ""

# Run the main pipeline
echo "Starting pipeline..."
echo "This may take several hours depending on your hardware."
echo ""

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="results/pipeline_run_${timestamp}.log"

python main.py 2>&1 | tee "$log_file"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Pipeline completed successfully!"
    echo "=========================================="
    echo "Log file: $log_file"
    echo "Results saved to: results/"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ Pipeline failed. Check log file:"
    echo "   $log_file"
    echo "=========================================="
    exit 1
fi
