from omegaconf import OmegaConf
from datasets import Dataset

data_cfg = OmegaConf.load("configs/data/CPT_data.yaml")
train_file = data_cfg.train_file
val_file = data_cfg.val_file
CHUNK_SIZE = data_cfg.CHUNK_SIZE
OVERLAP = data_cfg.OVERLAP


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


if train_file.exists() and val_file.exists():
    with open(train_file, "r") as f:
        train_lines = f.readlines()
    with open(val_file, "r") as f:
        val_lines = f.readlines()

    train_chunks = chunk_texts(train_lines, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    val_chunks = chunk_texts(val_lines, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    train_dataset = Dataset.from_dict({"text": train_chunks})
    val_dataset = Dataset.from_dict({"text": val_chunks})

    print(f"\nTrain chunks: {len(train_chunks)}")
    print(f"Val chunks:   {len(val_chunks)}")
    print(f"\nSample chunk (first 100 words):")
    print(" ".join(train_chunks[0].split()[:100]))
else:
    print(f"Train or val file not found. Run load.py first.")
