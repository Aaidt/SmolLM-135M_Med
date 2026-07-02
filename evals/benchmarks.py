import torch
import json
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

cfg = OmegaConf.load("config.yaml")

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16
MAX_SEQ_LENGTH = cfg.MAX_SEQ_LENGTH


def _choice_token_start(full_ids, prompt, tokenizer):
    """Index in full_ids where choice tokens begin (handles BPE merges at boundary)."""
    for start in range(len(full_ids) + 1):
        if tokenizer.decode(full_ids[:start]) == prompt:
            return start
    return len(full_ids)


def _sequence_log_prob(log_probs, token_ids, start_idx):
    """Sum log P(token_i | tokens_<i) for tokens at positions [start_idx, end)."""
    return sum(
        log_probs[pos - 1, token_ids[pos]].item()
        for pos in range(start_idx, len(token_ids))
    )


def choice_log_probs(model, tokenizer, prompt, choices):
    """Return log probability of each choice given the prompt.

    Uses one forward pass per choice: encode prompt+choice as a single sequence,
    then sum the log-prob of each choice token from the logits at the prior position.
    All choices are batched into a single forward pass when possible.
    """
    sequences = []
    start_indices = []

    for choice in choices:
        if not choice:
            sequences.append(None)
            start_indices.append(None)
            continue

        full_ids = tokenizer(
            prompt + choice,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).input_ids

        start = _choice_token_start(full_ids, prompt, tokenizer)
        if start >= len(full_ids):
            sequences.append(None)
            start_indices.append(None)
            continue

        sequences.append(full_ids)
        start_indices.append(start)

    choice_logprobs = [-float("inf")] * len(choices)
    valid = [
        (i, seq, start)
        for i, (seq, start) in enumerate(zip(sequences, start_indices))
        if seq is not None
    ]
    if not valid:
        return choice_logprobs

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    max_len = max(len(seq) for _, seq, _ in valid)
    batch_ids, batch_mask = [], []
    for _, seq, _ in valid:
        pad_len = max_len - len(seq)
        batch_ids.append(seq + [pad_id] * pad_len)
        batch_mask.append([1] * len(seq) + [0] * pad_len)

    input_ids = torch.tensor(batch_ids, device=model.device)
    attention_mask = torch.tensor(batch_mask, device=model.device)

    with torch.no_grad():
        log_probs = torch.log_softmax(
            model(input_ids, attention_mask=attention_mask).logits.float(),
            dim=-1,
        )

        for batch_idx, (choice_idx, seq, start) in enumerate(valid):
            choice_logprobs[choice_idx] = _sequence_log_prob(
                log_probs[batch_idx], seq, start
            )

    log_probs_tensor = torch.tensor(choice_logprobs)
    log_probs_tensor = log_probs_tensor - log_probs_tensor.logsumexp(0)
    return log_probs_tensor.tolist()


def eval_pubmedqa(model, tokenizer, max_samples=None):
    """PubMedQA: yes/no/maybe biomedical questions."""
    print("\n=== PubMedQA ===")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    results = []

    for item in tqdm(ds, desc="PubMedQA"):
        item = dict(item)
        question = item["question"]
        context = " ".join(item["context"]["contexts"])
        answer = item["final_decision"]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        choices = ["yes", "no", "maybe"]
        log_probs = choice_log_probs(model, tokenizer, prompt, choices)

        predicted = choices[int(torch.tensor(log_probs).argmax().item())]
        is_correct = predicted == answer.lower()
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "true_answer": answer,
            "predicted": predicted,
            "correct": is_correct,
            "choice_log_probs": {c: round(lp, 4) for c, lp in zip(choices, log_probs)},
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    return {"accuracy": round(accuracy, 4), "correct": correct, "total": total, "details": results}


def eval_medmcqa(model, tokenizer, max_samples=None):
    """MedMCQA: medical MCQ with 4 options."""
    print("\n=== MedMCQA ===")
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    results = []

    choice_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for item in tqdm(ds, desc="MedMCQA"):
        item = dict(item)
        question = item["question"]
        opts = [item["opa"], item["opb"], item["opc"], item["opd"]]
        correct_idx = item["cop"] if item["cop"] is not None else item["correct"]
        true_answer = choice_map[correct_idx]

        prompt = f"Question: {question}\nA: {opts[0]}\nB: {opts[1]}\nC: {opts[2]}\nD: {opts[3]}\nAnswer:"
        choices = ["A", "B", "C", "D"]
        log_probs = choice_log_probs(model, tokenizer, prompt, choices)

        predicted = choices[int(torch.tensor(log_probs).argmax().item())]
        is_correct = predicted == true_answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "true_answer": true_answer,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    return {"accuracy": round(accuracy, 4), "correct": correct, "total": total, "details": results}


def run_all_benchmarks(model, tokenizer, output_suffix="untrained"):
    print(f"Running benchmarks ...")
    print(f"Device: {model.device}")

    all_results = {"model": MODEL_NAME, "benchmarks": {}}

    benchmarks = [
        ("pubmedqa", eval_pubmedqa, 1000),
        ("medmcqa", eval_medmcqa, 1000),
        # ("medqa", eval_medqa, 200),
        # ("medication_qa", eval_medication_qa, 200),
        # ("bioasq_yesno", eval_bioasq, 200),
    ]

    for name, fn, limit in benchmarks:
        try:
            result = fn(model, tokenizer, max_samples=limit)
            all_results["benchmarks"][name] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }
        except Exception as e:
            print(f"  ERROR running {name}: {e}")
            all_results["benchmarks"][name] = {"error": str(e)}

    out_path = RESULTS_DIR / f"benchmarks_{output_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {out_path}")
    return all_results
