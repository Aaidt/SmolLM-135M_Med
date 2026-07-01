from data import run_data
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
from omegaconf import OmegaConf
from main import load_model

cfg = OmegaConf.load("config.yaml")

SEED = cfg.SEED
MAX_SEQ_LENGTH = cfg.MAX_SEQ_LENGTH


def add_lora_adapters(model):
    print("  Adding LoRA adapters ...")
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
        use_gradient_checkpointing="unsloth",
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


def configure_trainer(model, train_dataset, val_dataset):
    print("  Configuring trainer ...")

    training_args = UnslothTrainingArguments(
        output_dir="./SmolLM-135M_Med",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        embedding_learning_rate=5e-6,
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
        eval_steps=250,
        per_device_eval_batch_size=16,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
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


def run_training():
    print("=" * 58)
    print("  SmolLM-135M  CPT  Training")
    print("=" * 58)

    train_dataset, val_dataset = run_data()
    model, tokenizer = load_model()
    model = add_lora_adapters(model)
    trainer = configure_trainer(model, train_dataset, val_dataset)

    print("  Starting training ...")
    print("-" * 58)
    trainer_stats = trainer.train()

    print("-" * 58)
    print("[DONE]  Training complete!")
    print(f"  Best eval loss: {trainer.state.best_metric:.4f}")
    print("=" * 58)

    return model, tokenizer


def main():
    run_training()


if __name__ == "__main__":
    main()