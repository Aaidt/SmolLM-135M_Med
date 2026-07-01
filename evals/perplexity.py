import torch
import torch.nn.functional as F
import math
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path
from omegaconf import OmegaConf

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

cfg = OmegaConf.load("config.yaml")

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_LENGTH = cfg.MAX_SEQ_LENGTH
STRIDE = max(1, MAX_LENGTH // 2)

def sliding_window_ppl(model, tokenizer, texts, max_length=MAX_LENGTH, stride=STRIDE):
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to(model.device)
            seq_len = input_ids.size(1)

            if seq_len < 10:
                continue

            nll = 0.0
            n_tokens = 0

            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                chunk_ids = input_ids[:, begin:end]
                labels = chunk_ids.clone()
                if begin > 0:
                    prev_tokens = max(0, max_length - stride)
                    labels[:, :prev_tokens] = -100
                elif end < seq_len:
                    pass

                outputs = model(input_ids=chunk_ids, use_cache=False)
                shift_logits = outputs.logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()
                valid = (labels[:, 1:] != -100).sum().item()
                if valid > 0:
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    nll += loss.item()
                    n_tokens += valid

                if end == seq_len:
                    break

            if n_tokens > 0:
                total_nll += nll
                total_tokens += n_tokens

    if total_tokens == 0:
        raise ValueError("No valid tokens found while computing perplexity.")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, total_tokens


def run_perplexity(model, tokenizer, output_suffix="untrained"):
    print(f"Evaluating perplexity ...")
    model.eval()

    datasets_to_eval = [
        ("PubMed Abstracts", "uiyunkim-hub/pubmed-abstract", "abstract", 1000),
        ("Medline", "cyrilzakka/pubmed-medline", "content", 1000),
    ]

    results = {"model": MODEL_NAME, "perplexity_results": {}}

    for name, ds_name, key, max_samples in datasets_to_eval:
        print(f"\n--- Loading {name} ---")
        ds = load_dataset(ds_name, split="train", streaming=True)
        texts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            texts.append(item[key])

        print(f"Evaluating perplexity on {name} ({len(texts)} samples)...")
        ppl, avg_nll, total_tokens = sliding_window_ppl(model, tokenizer, texts)
        print(f"  Perplexity: {ppl:.4f}")
        print(f"  Avg NLL:    {avg_nll:.4f}")
        print(f"  Tokens:     {total_tokens}")

        results["perplexity_results"][name] = {
            "perplexity": round(ppl, 4),
            "avg_nll": round(avg_nll, 4),
            "num_tokens": total_tokens,
            "num_samples": len(texts),
        }

    out_path = RESULTS_DIR / f"perplexity_{output_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results
