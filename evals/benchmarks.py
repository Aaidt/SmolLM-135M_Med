import torch
import json
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def choice_log_probs(model, tokenizer, prompt, choices):
    """Return log probability of each choice given the prompt."""
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0, -1, :]

    choice_logprobs = []
    for choice in choices:
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)
        if not choice_ids:
            choice_logprobs.append(-float("inf"))
            continue
        lp = 0.0
        for i, cid in enumerate(choice_ids):
            if i == 0:
                lp += logits[cid].item()
            else:
                sub_prompt = prompt + tokenizer.decode(choice_ids[:i])
                sub_encoded = tokenizer(sub_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    sub_logits = model(**sub_encoded).logits[0, -1, :]
                lp += sub_logits[cid].item()
        choice_logprobs.append(lp)

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
        ("pubmedqa", eval_pubmedqa, 200),
        ("medmcqa", eval_medmcqa, 200),
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