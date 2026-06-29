from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import random

cfg = OmegaConf.load("config.yaml")

data_file = Path(cfg.data_file)
train_file = Path(cfg.train_file)
val_file = Path(cfg.val_file)

data_file.parent.mkdir(parents=True, exist_ok=True)

OVERLAP = cfg.OVERLAP
CHUNK_SIZE = cfg.CHUNK_SIZE
SEED = cfg.SEED

SPLIT = 0.1
random.seed(SEED)

PUBMED_ABSTRACT_DOCS = 6000
PMC_DOCS = 2000
MEDLINE_DOCS = 1000
FINEWEB_DOCS = 1000

TOTAL = 10_000

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

if data_file.exists() and data_file.stat().st_size > 0:
    print(f"[1] dataset file already exists. Skipping dataset streaming")
else:
    print(f"[1] Creating dataset files at {data_file}...")

    datasets = [
        ("PubMed Abstracts", lambda: load_dataset("uiyunkim-hub/pubmed-abstract", split="train", streaming=True), "abstract", PUBMED_ABSTRACT_DOCS),
        ("PMC", lambda: load_dataset("hatakeyama-llm-team/PMC", split="PMC002xxxxxx_0", streaming=True), "text", PMC_DOCS),
        ("Medline", lambda: load_dataset("cyrilzakka/pubmed-medline", split="train", streaming=True), "content", MEDLINE_DOCS),
        ("FineWeb", lambda: load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True), "text", FINEWEB_DOCS),
    ]

    total_samples = sum(n for _, _, _, n in datasets)

    with open(data_file, "w") as f:
        with tqdm(total=total_samples, desc="Total progress", unit=" samples") as total_pbar:
            for name, loader, key, max_samples in datasets:
                ds = loader()
                inner_pbar = tqdm(total=max_samples, desc=f"  Downloading {name}", unit=" samples", leave=False)
                for i, item in enumerate(ds):
                    if i >= max_samples:
                        break
                    f.write(item[key] + "\n")
                    inner_pbar.update(1)
                    total_pbar.update(1)
                inner_pbar.close()

print(f"[2] Lets start chunking this data to pass it to Unsloth trainer...")

if train_file.exists() and val_file.exists() and train_file.stat().st_size > 0 and val_file.stat().st_size > 0:
    print(f"training and validation datasets already created...Done!!")
else:
    with open(train_file, "w") as train, open(val_file, "w") as val:
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                if len(ids) < CHUNK_SIZE:
                    continue

                out = val if random.random() < SPLIT else train
                step = CHUNK_SIZE - OVERLAP
                for start in range(0, len(ids) - CHUNK_SIZE + 1, step):
                    chunk = ids[start : start + CHUNK_SIZE]

                    if len(chunk) < CHUNK_SIZE:
                        continue

                    out.write(tokenizer.decode(chunk) + "\n")

print(f"[3] Converting the dataset into Dataset format for Unsloth...")

datasets = load_dataset(
    "text",
    data_files={
        "train": str(train_file),
        "validation": str(val_file),
    },
)

train_ds = datasets["train"]
val_ds = datasets["validation"]

print(f"Dataset phase completed")