import torch
from unsloth import FastLanguageModel
from omegaconf import OmegaConf
from pathlib import Path
from evals.perplexity import run_perplexity
from evals.benchmarks import run_all_benchmarks
from data import run_data

cfg = OmegaConf.load("config.yaml")

SEED = cfg.SEED
MAX_SEQ_LENGTH = cfg.MAX_SEQ_LENGTH
MODEL_NAME = cfg.MODEL_NAME
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_model():
    print("Loading base model ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=True,
    )
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Loaded in 4-bit: True")
    return model, tokenizer

def main():
    from train import run_training

    print("\n[1] Running data phase...")
    run_data()

    print("\n[2] Loading base model ...")
    model, tokenizer = load_model()

    print("\n[3] Running benchmarks on untrained model...")
    run_all_benchmarks(model, tokenizer, output_suffix="untrained")

    print("\n[4] Running perplexity on untrained model...")
    run_perplexity(model, tokenizer, output_suffix="untrained")

    print("\n[5] Starting training...")
    model, tokenizer = run_training()

    print("\n[6] Running benchmarks on trained model...")
    run_all_benchmarks(model, tokenizer, output_suffix="trained")

    print("\n[7] Running perplexity on trained model...")
    run_perplexity(model, tokenizer, output_suffix="trained")

    print("\nAll done! Results saved to ./results/")


if __name__ == "__main__":
    main()