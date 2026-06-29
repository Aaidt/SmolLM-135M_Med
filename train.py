import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
from omegaconf import OmegaConf

train_cfg = OmegaConf.load("config/train.yaml")
data_cfg = OmegaConf.load("config/data.yaml")

SEED = train_cfg.seed
MAX_SEQ_LENGTH = train_cfg.MAX_SEQ_LENGTH
MODEL_NAME = train_cfg.MODEL_NAME
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

TRAIN_FILE = data_cfg.train_file
VAL_FILE = data_cfg.val_file
CHUNK_SIZE = data_cfg.chunk_size
OVERLAP = data_cfg.overlap


def chunk_texts(texts, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    step = int(chunk_size * (1 - overlap))
    all_chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), step):
            chunk = words[i:i + chunk_size]
            if len(chunk) > 50:
                all_chunks.append(" ".join(chunk))
    return all_chunks


def load_chunked_dataset():
    print("[1/5] Loading and chunking dataset ...")
    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError(
            "Data files not found. Run `uv run python dataset/load.py` first."
        )

    with open(TRAIN_FILE) as f:
        train_lines = f.readlines()
    with open(VAL_FILE) as f:
        val_lines = f.readlines()

    print(f"  Raw lines: {len(train_lines):>8,} train, {len(val_lines):>8,} val")

    train_chunks = chunk_texts(train_lines)
    val_chunks = chunk_texts(val_lines)

    train_dataset = Dataset.from_dict({"text": train_chunks})
    val_dataset = Dataset.from_dict({"text": val_chunks})

    print(f"  Chunked:   {len(train_dataset):>8,} train, {len(val_dataset):>8,} val")
    return train_dataset, val_dataset


def load_model():
    print("[2/5] Loading base model ...")
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


def add_lora_adapters(model):
    print("[3/5] Adding LoRA adapters ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
        use_rslora=False,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total params:    {total_params:>12,}")
    print(f"  Trainable (LoRA): {trainable_params:>12,}  ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Frozen (base):   {frozen_params:>12,}  ({100 * frozen_params / total_params:.2f}%)")
    print(f"  Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, embed_tokens, lm_head")
    return model


def configure_trainer(model, tokenizer, train_dataset, val_dataset):
    print("[4/5] Configuring trainer ...")

    training_args = UnslothTrainingArguments(
        output_dir="./cpt_sec_filings",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        embedding_learning_rate=2e-5,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        dataset_text_field="text",
        dataset_num_proc=2,
        eval_strategy="steps",
        eval_steps=25,
        per_device_eval_batch_size=16,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        seed=SEED,
        report_to="none",
    )

    trainer = UnslothTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    print(f"  Training samples: {len(train_dataset):>8,}")
    print(f"  Eval samples:     {len(val_dataset):>8,}")
    print(f"  Batch size:       {training_args.per_device_train_batch_size}")
    print(f"  Grad accum steps: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch:  {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate:    {training_args.learning_rate}")
    print(f"  Embedding LR:     {training_args.embedding_learning_rate}")
    print(f"  Epochs:           {training_args.num_train_epochs}")
    print(f"  Warmup steps:     {training_args.warmup_steps}")
    print(f"  Scheduler:        {training_args.lr_scheduler_type}")
    print(f"  Packing:          {training_args.packing}")

    return trainer


def main():
    print("=" * 58)
    print("  SmolLM-135M  CPT  Training")
    print("=" * 58)

    train_dataset, val_dataset = load_chunked_dataset()
    model, tokenizer = load_model()
    model = add_lora_adapters(model)
    trainer = configure_trainer(model, tokenizer, train_dataset, val_dataset)

    print("[5/5] Starting training ...")
    print("-" * 58)
    trainer_stats = trainer.train()

    print("-" * 58)
    print("  Training complete!")
    print(f"  Best eval loss: {trainer.state.best_metric:.4f}")
    print(f"  Model saved to: ./cpt_sec_filings")
    print("=" * 58)


if __name__ == "__main__":
    main()
