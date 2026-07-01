import torch
from unsloth import FastLanguageModel
from pathlib import Path
from evals.perplexity import run_perplexity
from evals.benchmarks import run_all_benchmarks
from data import run_data
from model_utils import load_model

def save_merged_model(model, tokenizer):
    FastLanguageModel.for_inference(model)

    model.save_pretrained_merged(
        "SmolLM-135M_Med_Merged",
        tokenizer,
        save_method="merged_16bit",
    )

def main():
    from train import run_training

    # print("\nRunning data phase...")
    # run_data()

    print("\nLoading base model ...")
    model, tokenizer = load_model()

    print("\nRunning benchmarks on untrained model...")
    run_all_benchmarks(model, tokenizer, output_suffix="untrained")

    print("\nRunning perplexity on untrained model...")
    run_perplexity(model, tokenizer, output_suffix="untrained")

    print("\nStarting training...")
    model, tokenizer = run_training()

    print(f"\nSaving the full merged model...")
    save_merged_model(model=model, tokenizer=tokenizer)

    print("\nRunning benchmarks on trained model...")
    run_all_benchmarks(model, tokenizer, output_suffix="trained")

    print("\nRunning perplexity on trained model...")
    run_perplexity(model, tokenizer, output_suffix="trained")

    print("\nAll done! Results saved to ./results/")


if __name__ == "__main__":
    main()