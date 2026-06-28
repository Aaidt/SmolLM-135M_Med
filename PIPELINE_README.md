# SmolLM-135M Medical Training Pipeline

This pipeline orchestrates the complete training and evaluation workflow for the SmolLM-135M model on medical datasets.

## Pipeline Overview

The pipeline performs 5 main stages:

1. **Dataset Loading** - Loads and chunks the medical dataset
2. **Pre-Training Evaluation** - Runs benchmarks on the base model
3. **Training Setup** - Configures the model with LoRA adapters
4. **Model Training** - Trains the model using Continual Pre-Training (CPT)
5. **Post-Training Evaluation** - Runs benchmarks on the trained model
6. **Results Comparison** - Compares results and logs metrics

## Prerequisites

Before running the pipeline, ensure:

1. Dataset files are created:
   ```bash
   python dataset/load.py
   ```

2. Dependencies are installed:
   ```bash
   uv sync
   ```

3. GPU is available (recommended for faster training)

## Running the Pipeline

### Option 1: Using the Bash Script (Recommended)

```bash
./run_pipeline.sh
```

This will:
- Check dependencies
- Validate dataset files
- Run the complete pipeline
- Log all output to `results/pipeline_run_YYYYMMDD_HHMMSS.log`

### Option 2: Direct Python Execution

```bash
python main.py
```

## Output Files

All results are saved to the `results/` directory with the following structure:

### Evaluation Results
- `evals_before_training_YYYYMMDD_HHMMSS.json` - Base model evaluation scores
- `evals_after_training_YYYYMMDD_HHMMSS.json` - Trained model evaluation scores

### Training Metrics
- `training_stats_YYYYMMDD_HHMMSS.json` - Detailed training statistics (loss, learning rate, etc.)

### Comparison
- `comparison_YYYYMMDD_HHMMSS.json` - Side-by-side comparison with improvement metrics

### Logs
- `pipeline_run_YYYYMMDD_HHMMSS.log` - Complete console output

## Results Format

### Evaluation Results JSON
```json
{
  "model": "HuggingFaceTB/SmolLM-135M",
  "timestamp": "2024-06-28T10:30:45.123456",
  "stage": "before_training|after_training",
  "benchmarks": {
    "pubmedqa": {
      "accuracy": 0.65,
      "correct": 130,
      "total": 200
    },
    ...
  }
}
```

### Comparison Results JSON
```json
{
  "timestamp": "2024-06-28T10:30:45.123456",
  "benchmarks": {
    "pubmedqa": {
      "before": { "accuracy": 0.65, "correct": 130, "total": 200 },
      "after": { "accuracy": 0.72, "correct": 144, "total": 200 },
      "improvement": 0.07,
      "improvement_pct": 10.77
    },
    ...
  },
  "overall": {
    "avg_accuracy_before": 0.65,
    "avg_accuracy_after": 0.71,
    "avg_improvement": 0.06,
    "avg_improvement_pct": 9.23
  }
}
```

## Benchmarks Included

The pipeline evaluates on the following medical QA benchmarks:

1. **PubMedQA** - Biomedical literature QA (yes/no/maybe)
2. **MedMCQA** - Medical multiple-choice questions
3. **MedQA (USMLE)** - Medical board exam questions
4. **MedicationQA** - Consumer medication questions
5. **BioASQ** - Biomedical QA (yes/no subset)

Each benchmark is evaluated with configurable sample size (default: 100-200 samples per benchmark).

## Customization

### Modify Evaluation Sample Size

Edit `main.py` and change the `max_samples` parameter in the `run_evals()` calls:

```python
# Line ~160
results_before = run_evals(model, tokenizer, label="before_training", max_samples=50)

# Line ~190
results_after = run_evals(model, tokenizer, label="after_training", max_samples=50)
```

### Modify Training Configuration

Edit `model/train_cpt.py` to adjust training parameters:

```python
# Number of epochs
num_train_epochs=2

# Batch size
per_device_train_batch_size=16

# Learning rate
learning_rate=2e-4

# Max sequence length
MAX_SEQ_LENGTH = 512
```

## Monitoring Progress

During training, you can monitor:
- Training loss (decreases over time)
- Evaluation loss (at eval_steps intervals)
- Gradient norms (for stability checks)

The logs will show:
```
[Training Progress]
Step 10/500, Loss: 4.231, LR: 2.00e-4
Step 20/500, Loss: 4.102, LR: 1.99e-4
...
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size` in `model/train_cpt.py`
- Reduce `max_samples` in benchmark evaluations

### Dataset Not Found
```bash
python dataset/load.py
```

### Missing Dependencies
```bash
uv sync
```

### CUDA/GPU Issues
The pipeline automatically falls back to CPU if GPU is unavailable, but training will be slower.

## Expected Performance

On a single GPU (V100/A100):
- Pre-training evaluation: ~5-10 minutes
- Training: ~2-4 hours (depending on GPU and dataset size)
- Post-training evaluation: ~5-10 minutes
- Total: ~3-5 hours

## Results Interpretation

Compare `avg_accuracy_before` and `avg_accuracy_after`:
- **Positive improvement**: Model benefited from medical CPT
- **Negative improvement**: Model may have overfit or dataset size was too small
- **Improvement > 5%**: Strong positive transfer from medical pre-training

Check individual benchmarks to see which tasks improved most:
- Medical-specific tasks (MedMCQA, MedQA) often show larger improvements
- General QA tasks may show smaller improvements

## Support

For issues or questions:
1. Check logs in `results/pipeline_run_*.log`
2. Review individual benchmark errors in evaluation JSON files
3. Ensure dataset files are properly created with sufficient data
