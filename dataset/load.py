from datasets import load_dataset
from omegaconf import OmegaConf

data_cfg = OmegaConf.load("configs/data/CPT_data.yaml")
train_file = data_cfg.train_file
validation_file = data_cfg.val_file

PUBMED_ABSTRACT_TOKENS = 6_000_000
PMC_TOKENS = 2_000_000
MEDLINE_TOKENS = 1_000_000
FINEWEB_TOKENS = 1_000_000

if train_file.exists() and validation_file.exists() and train_file.stat().st_size > 0 and validation_file.stat().st_size > 0:
    print(f"Train and validation files already exist at {train_file} and {validation_file}. Skipping dataset creation.")

pubmed = load_dataset("uiyunkim-hub/pubmed-abstract", split="train", streaming=True)
with open(train_file, "w") as f:
    for i, item in enumerate(pubmed):
        if i >= PUBMED_ABSTRACT_TOKENS:
            break
        f.write(item["abstract"] + "\n")

pmc = load_dataset("hatakeyama-llm-team/PMC", split="PMC002xxxxxx_0", streaming=True)
with open(train_file, "w") as f:
    for i, item in enumerate(pmc):
        if i >= PMC_TOKENS:
            break
        f.write(item["text"] + "\n")
    
medline = load_dataset("cyrilzakka/pubmed-medline", split="train", streaming=True)
with open(train_file, "w") as f:
    for i, item in enumerate(medline):
        if i >= MEDLINE_TOKENS:
            break
        f.write(item["content"] + "\n")

fineweb = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
with open(train_file, "w") as f:
    for i, item in enumerate(fineweb):
        if i >= FINEWEB_TOKENS:
            break
        f.write(item["text"] + "\n")

