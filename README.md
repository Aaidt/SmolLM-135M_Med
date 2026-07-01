 ```
  ▗▄▖           ▗▄▖  ▗▖   ▗▄ ▄▖      ▗▄   ▄▄▖ ▗▄▄▄ ▗▄ ▄▖     ▗▄ ▄▖        ▗▖
▗▛▀▜           ▝▜▌  ▐▌   ▐█ █▌      ▛█  ▐▀▀█▖▐▛▀▀ ▐█ █▌     ▐█ █▌        ▐▌
▐▙   ▐█▙█▖ ▟█▙  ▐▌  ▐▌   ▐███▌       █     ▟▌▐▙▄▖ ▐███▌     ▐███▌ ▟█▙  ▟█▟▌
  ▜█▙ ▐▌█▐▌▐▛ ▜▌ ▐▌  ▐▌   ▐▌█▐▌       █   ▐██ ▐▀▀█▖▐▌█▐▌     ▐▌█▐▌▐▙▄▟▌▐▛ ▜▌
    ▜▌▐▌█▐▌▐▌ ▐▌ ▐▌  ▐▌   ▐▌▀▐▌ ██▌   █     ▜▌   ▐▌▐▌▀▐▌     ▐▌▀▐▌▐▛▀▀▘▐▌ ▐▌
▐▄▄▟▘▐▌█▐▌▝█▄█▘ ▐▙▄ ▐▙▄▄▖▐▌ ▐▌     ▗▄█▄▖▐▄▄█▘▐▄▄█▘▐▌ ▐▌     ▐▌ ▐▌▝█▄▄▌▝█▄█▌
  ▀▀▘ ▝▘▀▝▘ ▝▀▘   ▀▀ ▝▀▀▀▘▝▘ ▝▘     ▝▀▀▀▘ ▀▀▘  ▀▀▘ ▝▘ ▝▘     ▝▘ ▝▘ ▝▀▀  ▝▀▝▘                  
                                                       ▀▀▀▀▀   

```

Medical-domain continued pre-training (CPT) pipeline for HuggingFaceTB's
SmolLM-135M model. Trains on PubMed, PMC, Medline, and FineWeb datasets
using LoRA adapters with the Unsloth framework.

---

## Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  1. DATA    │ ──► │  2. EVAL     │ ──► │  3. TRAIN  │
│  Phase      │     │  (baseline)  │     │  (LoRA)    │
└─────────────┘     └──────────────┘     └────────────┘
                          │                     │
                          ▼                     ▼
                   ┌──────────────┐     ┌──────────────┐
                   │  Perplexity  │     │  Perplexity  │
                   │  Benchmarks  │     │  Benchmarks  │
                   └──────────────┘     └──────────────┘
                          │                     │
                          └──────────┬──────────┘
                                     ▼
                            ┌────────────────┐
                            │  4. SAVE       │
                            │  Merged Model  │
                            └────────────────┘
```

### Step-by-Step

1. **Data** (`data.py`) — Downloads and tokenizes biomedical datasets, splits 90/10 train/val, writes to text files, then loads into HuggingFace Datasets.

2. **Baseline Eval** — Evaluates the untrained base model on perplexity (PubMed Abstracts, Medline) and benchmarks (PubMedQA, MedMCQA).

3. **Training** (`train.py`) — Loads base model, attaches LoRA adapters, runs 1 epoch of CPT with UnslothTrainer.

4. **Post-Training Eval** — Re-runs perplexity and benchmarks on the trained model.

5. **Export** — Saves a merged 16-bit model to `SmolLM-135M_Med_Merged/`.

---

## Configuration

All hyperparameters are set in `config.yaml`:

| Key | Value | Description |
|-----|-------|-------------|
| `MODEL_NAME` | `HuggingFaceTB/SmolLM-135M` | Base model |
| `SEED` | `42` | Random seed |
| `MAX_SEQ_LENGTH` | `512` | Max sequence length |
| `train_file` | `./data/train.txt` | Training data path |
| `val_file` | `./data/val.txt` | Validation data path |

---

## Training Details

### Model
- **Base**: HuggingFaceTB/SmolLM-135M (135M parameters)
- **Precision**: 4-bit loaded (NF4), merged to 16-bit on save
- **Sequence length**: 512 tokens

### LoRA Configuration
| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 32 |
| LoRA alpha | 32 |
| LoRA dropout | 0 |
| Bias | none |
| Gradient checkpointing | `"unsloth"` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `embed_tokens`, `lm_head` |

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Per-device batch size | 16 |
| Gradient accumulation | 2 |
| Effective batch size | 32 |
| Learning rate | 5e-5 |
| Embedding learning rate | 5e-6 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Optimizer | AdamW 8-bit |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Packing | Enabled |
| Eval strategy | Every 250 steps |
| Save strategy | Every 500 steps (keep 3) |
| Load best model at end | Yes |
| Metric for best model | Eval loss |

### Datasets (200,000 total samples)
| Source | Samples | Key |
|--------|---------|-----|
| PubMed Abstracts | 120,000 | `abstract` |
| PMC (PubMed Central) | 40,000 | `text` |
| Medline | 20,000 | `content` |
| FineWeb | 20,000 | `text` |

- **Split**: 90% train / 10% validation (per-source random split)
- **Pre-tokenization**: Tokenized with SmolLM-135M tokenizer to count tokens

### Evaluation
- **Perplexity**: Sliding-window (window=1024, stride=512) on PubMed Abstracts and Medline
- **Benchmarks**: PubMedQA (yes/no/maybe) and MedMCQA (4-option MCQ)
- **Results**: Saved to `./results/` as JSON with `_untrained` and `_trained` suffixes

---

## Usage

```bash
# Run the full pipeline
uv run main.py

### Dependencies

- Python >= 3.13
- unsloth
- datasets
- omegaconf
- evaluate

Install with uv:

```bash
uv sync
```

---

## Project Structure

```
├── main.py              # Pipeline entry point
├── train.py             # LoRA training with UnslothTrainer
├── data.py              # Dataset download & preprocessing
├── model_utils.py       # Shared model loading & config
├── config.yaml          # Configuration
├── pyproject.toml       # Project metadata & dependencies
├── evals/
│   ├── benchmarks.py    # PubMedQA & MedMCQA evaluation
│   └── perplexity.py    # Sliding-window perplexity
└── results/             # Evaluation outputs (JSON)
```
