from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

data_cfg = OmegaConf.load("configs/data/CPT_data.yaml")
train_file = data_cfg.train_file
validation_file = data_cfg.val_file

PUBMED_ABSTRACT_DOCS = 6_000_000
PMC_DOCS = 2_000_000
MEDLINE_DOCS = 1_000_000
FINEWEB_DOCS = 1_000_000

streaming = True

if train_file.exists() and validation_file.exists() and train_file.stat().st_size > 0 and validation_file.stat().st_size > 0:
    print(f"Train and validation files already exist at {train_file} and {validation_file}. Skipping dataset creation.")
    streaming = False

if streaming:
    print(f"Creating train and validation files at {train_file} and {validation_file}...")

    datasets = [
        ("PubMed Abstracts", lambda: load_dataset("uiyunkim-hub/pubmed-abstract", split="train", streaming=True), "abstract", PUBMED_ABSTRACT_DOCS),
        ("PMC", lambda: load_dataset("hatakeyama-llm-team/PMC", split="PMC002xxxxxx_0", streaming=True), "text", PMC_DOCS),
        ("Medline", lambda: load_dataset("cyrilzakka/pubmed-medline", split="train", streaming=True), "content", MEDLINE_DOCS),
        ("FineWeb", lambda: load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True), "text", FINEWEB_DOCS),
    ]

    total_samples = sum(n for _, _, _, n in datasets)

    with open(train_file, "w") as f:
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
