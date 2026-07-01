from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import random
import time

def run_data():
    cfg = OmegaConf.load("config.yaml")

    train_file = Path(cfg.train_file)
    val_file = Path(cfg.val_file)
    train_file.parent.mkdir(parents=True, exist_ok=True)

    CHUNK_SIZE = cfg.MAX_SEQ_LENGTH
    SEED = cfg.SEED

    SPLIT = 0.1
    random.seed(SEED)

    PUBMED_ABSTRACT_DOCS = 120_000
    PMC_DOCS = 40_000
    MEDLINE_DOCS = 20_000
    FINEWEB_DOCS = 20_000

    total_tokens = 0

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    if train_file.exists() and val_file.exists() and train_file.stat().st_size > 0 and val_file.stat().st_size > 0:
        print(f"Dataset files already exist. Dataset phase done!!!.\n")
    else:
        print(f"Creating dataset files...\n")
        datasets = [
            ("PubMed Abstracts", lambda: load_dataset("uiyunkim-hub/pubmed-abstract", split="train", streaming=True), "abstract", PUBMED_ABSTRACT_DOCS),
            ("PMC", lambda: load_dataset("hatakeyama-llm-team/PMC", split="PMC002xxxxxx_0", streaming=True), "text", PMC_DOCS),
            ("Medline", lambda: load_dataset("cyrilzakka/pubmed-medline", split="train", streaming=True), "content", MEDLINE_DOCS),
            ("FineWeb", lambda: load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True), "text", FINEWEB_DOCS),
        ]

        total_samples = sum(n for _, _, _, n in datasets)

        start_time = time.time()

        with open(train_file, "w") as train, open(val_file, "w") as val:
            with tqdm(total=total_samples, desc="Total progress", unit=" samples") as total_pbar:

                for name, loader, key, max_samples in datasets:
                    ds = loader()

                    inner_pbar = tqdm(total=max_samples, desc=f"  Downloading {name}", unit=" samples", leave=False)

                    for i, item in enumerate(ds):
                        if i >= max_samples:
                            break

                        text = item[key].strip()
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        ids.append(tokenizer.eos_token_id)
                        total_tokens += len(ids)

                        out = val if random.random() < SPLIT else train
                        out.write(text + "\n")

                        inner_pbar.update(1)
                        total_pbar.update(1)
                    inner_pbar.close()
        
        total_time = time.time() - start_time
        print(f"========== TOTAL NUMBER OF TOKENS: {total_tokens} | TIME TAKEN TO LOAD DATA: {total_time} ============")

    print(f"Converting dataset into Dataset format for Unsloth...\n")
    datasets = load_dataset(
        "text",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )

    train_ds = datasets["train"].shuffle(seed=SEED)
    val_ds = datasets["validation"].shuffle(seed=SEED)

    print(f"Data phase completed.")
    return train_ds, val_ds


if __name__ == "__main__":
    run_data()