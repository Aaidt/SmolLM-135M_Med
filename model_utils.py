import torch
from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")

SEED = cfg.SEED
MAX_SEQ_LENGTH = cfg.MAX_SEQ_LENGTH
MODEL_NAME = cfg.MODEL_NAME
# DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. Unsloth needs a GPU for 4-bit loading and training."
        )

    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            f"Unsloth is not installed or missing dependencies: {e}\n"
            "Install with: pip install unsloth"
        ) from e

    print("Loading base model ...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"GPU out of memory loading {MODEL_NAME} in 4-bit.\n"
            f"Try reducing max_seq_length (currently {MAX_SEQ_LENGTH}) "
            "or use a GPU with more VRAM."
        ) from e
    except ValueError as e:
        if "unsloth does not support" in str(e).lower():
            raise ValueError(
                f"{MODEL_NAME} is not supported by Unsloth. "
                "Check https://huggingface.co/unsloth for supported models."
            ) from e
        raise
    except OSError as e:
        raise OSError(
            f"Could not load {MODEL_NAME}. Check:\n"
            "  1. Internet connection (if downloading first time)\n"
            "  2. Model name is correct\n"
            f"  3. You have access to the model\n"
            f"Error: {e}"
        ) from e
    except RuntimeError as e:
        if "bfloat16" in str(e) or "bf16" in str(e):
            raise RuntimeError(
                "bfloat16 is not supported on this GPU. "
                "Try setting dtype=torch.float16 in config or "
                "use a newer GPU (Ampere+)."
            ) from e
        if "bitsandbytes" in str(e).lower():
            raise RuntimeError(
                f"bitsandbytes error during 4-bit loading.\n"
                f"Try: pip install bitsandbytes --upgrade\n"
                f"Error: {e}"
            ) from e
        raise

    print(f"  Model: {MODEL_NAME}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Loaded in 4-bit: True")
    return model, tokenizer