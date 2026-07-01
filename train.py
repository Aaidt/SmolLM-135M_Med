import torch
from data import run_data
from omegaconf import OmegaConf
from model_utils import load_model

try:
    from unsloth import FastLanguageModel
    from unsloth import UnslothTrainer, UnslothTrainingArguments
except ImportError as e:
    raise ImportError(
        f"Unsloth is not installed: {e}\n"
        "Install with: pip install unsloth"
    ) from e

cfg = OmegaConf.load("config.yaml")

SEED = cfg.SEED
MAX_SEQ_LENGTH = cfg.MAX_SEQ_LENGTH


def tokenize_dataset_for_packing(dataset, tokenizer, split_name):
    print(f"  Tokenizing {split_name} dataset for packing ...")
    chunk_size = MAX_SEQ_LENGTH
    original_size = len(dataset)

    def tokenize_examples(batch):
        input_ids = []
        attention_mask = []
        labels = []

        for text in batch["text"]:
            text = text.strip()
            if not text:
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, len(token_ids), chunk_size):
                chunk_ids = token_ids[start : start + chunk_size]
                if not chunk_ids:
                    continue

                input_ids.append(chunk_ids)
                attention_mask.append([1] * len(chunk_ids))
                labels.append(list(chunk_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(
        tokenize_examples,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split_name} dataset",
    )

    print(f"  {split_name.capitalize()} sequences: {original_size:>8,} -> {len(dataset):>8,}")
    if len(dataset) == 0:
        raise ValueError(f"{split_name} dataset is empty after tokenization.")
    return dataset


def add_lora_adapters(model):
    print("Adding LoRA adapters ...")

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ]

    # missing_modules = [
    #     m for m in target_modules
    #     if m not in dict(model.named_modules())
    # ]
    # if missing_modules:
    #     raise ValueError(
    #         f"Target modules not found in model: {missing_modules}\n"
    #         f"Available modules: {list(dict(model.named_modules()).keys())[:50]}"
    #     )

    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=target_modules,
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
            use_rslora=False,
        )
    except RuntimeError as e:
        if "gradient_checkpointing" in str(e).lower():
            raise RuntimeError(
                "Unsloth gradient checkpointing failed. "
                "Try setting use_gradient_checkpointing=True instead of 'unsloth'."
            ) from e
        raise
    except ValueError as e:
        if "target_modules" in str(e).lower():
            raise ValueError(
                f"Invalid target modules. Error: {e}"
            ) from e
        raise

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total params:    {total_params:>12,}")
    print(f"  Trainable (LoRA): {trainable_params:>12,}  ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Frozen (base):   {frozen_params:>12,}  ({100 * frozen_params / total_params:.2f}%)")
    print(f"  Target modules: {', '.join(target_modules)}")
    return model


def configure_trainer(model, tokenizer, train_dataset, val_dataset):
    print("  Configuring trainer ...")

    try:
        training_args = UnslothTrainingArguments(
            output_dir="./SmolLM-135M_Med",

            num_train_epochs=1,
            per_device_train_batch_size=128,
            gradient_accumulation_steps=1,

            learning_rate=5e-5,
            embedding_learning_rate=5e-6,

            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,

            # max_length=MAX_SEQ_LENGTH,
            # packing=True,
            dataset_num_proc=2,

            eval_strategy="steps",
            eval_steps=1000,
            per_device_eval_batch_size=16,

            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",

            logging_steps=100,
            seed=SEED,

            report_to="none",
        )
    except TypeError as e:
        raise TypeError(
            f"Invalid argument in UnslothTrainingArguments: {e}\n"
            "Check that all parameter names match the Unsloth API "
            "(especially embedding_learning_rate if using an older version)."
        ) from e

    try:
        trainer = UnslothTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer
        )
    except ValueError as e:
        if "columns" in str(e).lower():
            available = list(train_dataset.features.keys()) if train_dataset else "N/A"
            raise ValueError(
                f"Dataset format error. Expected 'input_ids', 'attention_mask', 'labels' columns.\n"
                f"Train columns: {available}\n"
                f"Error: {e}"
            ) from e
        raise

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


def run_training():
    print("=" * 58)
    print("  SmolLM-135M  CPT  Training")
    print("=" * 58)

    train_dataset, val_dataset = run_data()
    model, tokenizer = load_model()
    train_dataset = tokenize_dataset_for_packing(train_dataset, tokenizer, "train")
    val_dataset = tokenize_dataset_for_packing(val_dataset, tokenizer, "eval")
    model = add_lora_adapters(model)
    trainer = configure_trainer(model, tokenizer, train_dataset, val_dataset)

    print("  Starting training ...")
    print("-" * 58)
    try:
        trainer_stats = trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "GPU out of memory during training.\n"
            f"  Effective batch size: {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}\n"
            # f"  Max seq length: {trainer.args.max_length}\n"
            "  Solutions:\n"
            "    - Reduce per_device_train_batch_size (try 8 or 4)\n"
            "    - Reduce gradient_accumulation_steps\n"
            "    - Reduce max_seq_length in config.yaml"
        ) from e
    except RuntimeError as e:
        if "NaN" in str(e) or "nan" in str(e):
            raise RuntimeError(
                "NaN loss detected during training.\n"
                "  Solutions:\n"
                "    - Reduce learning_rate (try 2e-5 or 1e-5)\n"
                "    - Reduce embedding_learning_rate (try 2e-6)\n"
                "    - Increase gradient_accumulation_steps\n"
                "    - Check dataset for corrupted/empty samples"
            ) from e
        if "device-side assert" in str(e):
            raise RuntimeError(
                "CUDA device-side assert triggered. This usually means token IDs\n"
                "exceed the model's vocabulary size. Check tokenization output.\n"
                f"Tokenizer vocab size: {tokenizer.vocab_size}"
            ) from e
        raise

    print("-" * 58)
    print("[DONE]  Training complete!")
    best = trainer.state.best_metric
    if best is not None:
        print(f"  Best eval loss: {best:.4f}")
    print("=" * 58)

    return model, tokenizer


if __name__ == "__main__":
    run_training()