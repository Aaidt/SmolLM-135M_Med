import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

RESULTS_DIR = Path("/results") if Path("/results").exists() else Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map="auto"
    )
    model.eval()
    return model, tokenizer


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
        question = item["question"]
        context = " ".join(item["context"]["contexts"])
        answer = item["final_decision"]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        choices = ["yes", "no", "maybe"]
        log_probs = choice_log_probs(model, tokenizer, prompt, choices)

        predicted = choices[torch.tensor(log_probs).argmax().item()]
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
    ds = load_dataset("medmcqa", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    results = []

    choice_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for item in tqdm(ds, desc="MedMCQA"):
        question = item["question"]
        opts = [item["opa"], item["opb"], item["opc"], item["opd"]]
        correct_idx = item["cop"] if item["cop"] is not None else item["correct"]
        true_answer = choice_map[correct_idx]

        prompt = f"Question: {question}\nA: {opts[0]}\nB: {opts[1]}\nC: {opts[2]}\nD: {opts[3]}\nAnswer:"
        choices = ["A", "B", "C", "D"]
        log_probs = choice_log_probs(model, tokenizer, prompt, choices)

        predicted = choices[torch.tensor(log_probs).argmax().item()]
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


def eval_medqa(model, tokenizer, max_samples=None):
    """MedQA (USMLE): 4-option medical board questions."""
    print("\n=== MedQA (USMLE) ===")
    ds = load_dataset("bigbio/med_qa", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    results = []
    choice_labels = ["A", "B", "C", "D", "E"]

    for item in tqdm(ds, desc="MedQA"):
        question = item["question"]
        options = item["options"]
        answer_key = item["answer"].strip().upper()

        prompt_parts = [f"Question: {question}"]
        valid_labels = []
        for opt in options:
            label = opt["key"].strip().upper()
            value = opt["value"]
            prompt_parts.append(f"{label}: {value}")
            valid_labels.append(label)
        prompt = "\n".join(prompt_parts) + "\nAnswer:"

        log_probs = choice_log_probs(model, tokenizer, prompt, valid_labels)
        predicted = valid_labels[torch.tensor(log_probs).argmax().item()]
        is_correct = predicted == answer_key
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100],
            "true_answer": answer_key,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    return {"accuracy": round(accuracy, 4), "correct": correct, "total": total, "details": results}


def eval_medication_qa(model, tokenizer, max_samples=None):
    """MedicationQA: consumer medication questions."""
    print("\n=== MedicationQA ===")
    ds = load_dataset("bigbio/medication_qa", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    results = []

    for item in tqdm(ds, desc="MedicationQA"):
        question = item["question"]
        answer = item["answer"]
        choices_list = item.get("choices", [])
        if not choices_list or len(choices_list) < 2:
            continue

        choices = [c["value"] for c in choices_list]
        prompt = f"Question: {question}\nAnswer:"

        log_probs = choice_log_probs(model, tokenizer, prompt, choices)
        predicted = choices[torch.tensor(log_probs).argmax().item()]
        is_correct = predicted.lower().strip() == answer.lower().strip()
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100],
            "true_answer": answer,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    return {"accuracy": round(accuracy, 4), "correct": correct, "total": total, "details": results}


def eval_bioasq(model, tokenizer, max_samples=None):
    """BioASQ: biomedical QA (factoid yes/no)."""
    print("\n=== BioASQ ===")
    ds = load_dataset("bigbio/bioasq", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    results = []

    for item in tqdm(ds, desc="BioASQ"):
        question = item["question"]
        type_ = item.get("type", "")
        if type_ != "yesno":
            continue
        answer = item.get("ideal_answer", "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        true_label = answer.strip().lower()
        if true_label not in ("yes", "no"):
            continue

        prompt = f"Question: {question}\nAnswer:"
        choices = ["yes", "no"]
        log_probs = choice_log_probs(model, tokenizer, prompt, choices)

        predicted = choices[torch.tensor(log_probs).argmax().item()]
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100],
            "true_answer": true_label,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    return {"accuracy": round(accuracy, 4), "correct": correct, "total": total, "details": results}


def main():
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = load_model()
    print(f"Device: {model.device}")

    all_results = {"model": MODEL_NAME, "benchmarks": {}}

    benchmarks = [
        ("pubmedqa", eval_pubmedqa, 200),
        ("medmcqa", eval_medmcqa, 200),
        ("medqa", eval_medqa, 200),
        ("medication_qa", eval_medication_qa, 200),
        ("bioasq_yesno", eval_bioasq, 200),
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

    out_path = RESULTS_DIR / "benchmarks_base_model.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {out_path}")


if __name__ == "__main__":
    main()
