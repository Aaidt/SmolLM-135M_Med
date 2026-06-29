from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
import random

cfg = OmegaConf.load("config.yaml")

data_file = Path(cfg.data_file)
train_file = Path(cfg.train_file)
val_file = Path(cfg.val_file)

data_file.parent.mkdir(parents=True, exist_ok=True)

OVERLAP = cfg.OVERLAP
CHUNK_SIZE = cfg.CHUNK_SIZE

PUBMED_ABSTRACT_DOCS = 6000
PMC_DOCS = 2000
MEDLINE_DOCS = 1000
FINEWEB_DOCS = 1000

TOTAL = 10_000

if data_file.exists() and data_file.stat().st_size > 0:
    print(f"dataset file already exists. Skipping dataset streaming")
else:
    print(f"Creating dataset files at {data_file}...")

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