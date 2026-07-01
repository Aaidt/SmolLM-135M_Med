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

def save_merged_model(model, tokenizer):
    FastLanguageModel.for_inference(model)

    model.save_pretrained_merged(
        "SmolLM-135M_Med_Merged",
        tokenizer,
        save_method="merged_16bit",
    )

def main():
    from train import run_training

    print("\nRunning data phase...")
    run_data()

    print("\nLoading base model ...")
    model, tokenizer = load_model()

    print("\nRunning benchmarks on untrained model...")
    run_all_benchmarks(model, tokenizer, output_suffix="untrained")

    print("\nRunning perplexity on untrained model...")
    run_perplexity(output_suffix="untrained")

    print("\nStarting training...")
    model, tokenizer = run_training()

    print(f"\nSaving the full merged model...")
    save_merged_model(model=model, tokenizer=tokenizer)

    print("\nRunning benchmarks on trained model...")
    run_all_benchmarks(model, tokenizer, output_suffix="trained")

    print("\nRunning perplexity on trained model...")
    run_perplexity(output_suffix="trained")

    print("\nAll done! Results saved to ./results/")


if __name__ == "__main__":
    main()