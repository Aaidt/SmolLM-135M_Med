import torch
from unsloth import FastLanguageModel
from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")

SEED = cfg.SEED
MAX_SEQ_LENGTH = cfg.MAX_SEQ_LENGTH
MODEL_NAME = cfg.MODEL_NAME
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


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