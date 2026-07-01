import torch
from pathlib import Path
from evals.perplexity import run_perplexity
from evals.benchmarks import run_all_benchmarks
from data import run_data
from model_utils import load_model

try:
    from unsloth import FastLanguageModel
except ImportError as e:
    raise ImportError(
        f"Unsloth is not installed: {e}\n"
        "Install with: pip install unsloth"
    ) from e


def save_merged_model(model, tokenizer):
    try:
        FastLanguageModel.for_inference(model)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to switch model to inference mode: {e}\n"
            "The model may be in an unexpected state."
        ) from e

    try:
        model.save_pretrained_merged(
            "SmolLM-135M_Med_Merged",
            tokenizer,
            save_method="merged_16bit",
        )
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "GPU out of memory while merging LoRA weights.\n"
            "The merge requires additional VRAM on top of the 4-bit loaded model.\n"
            "Solutions:\n"
            "  - Use save_method='lora' instead of 'merged_16bit'\n"
            "  - Free up GPU memory before merging\n"
            "  - Use a GPU with more VRAM"
        ) from e
    except OSError as e:
        if "No space left" in str(e):
            raise OSError(
                "Insufficient disk space to save the merged model.\n"
                f"Free space and try again. Error: {e}"
            ) from e
        raise

def main():
    from train import run_training

    # print("\nLoading base model ...")
    # model, tokenizer = load_model()

    # print("\nRunning benchmarks on untrained model...")
    # run_all_benchmarks(model, tokenizer, output_suffix="untrained")

    # print("\nRunning perplexity on untrained model...")
    # run_perplexity(model, tokenizer, output_suffix="untrained")

    print("\nStarting training...")
    try:
        model, tokenizer = run_training()
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        print("Skipping merge and evaluation.")
        raise

    print(f"\nSaving the full merged model...")
    save_merged_model(model=model, tokenizer=tokenizer)

    print("\nRunning benchmarks on trained model...")
    run_all_benchmarks(model, tokenizer, output_suffix="trained")

    print("\nRunning perplexity on trained model...")
    run_perplexity(model, tokenizer, output_suffix="trained")

    print("\nAll done! Results saved to ./results/")


if __name__ == "__main__":
    main()