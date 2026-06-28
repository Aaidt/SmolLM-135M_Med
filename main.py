import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import training and evaluation modules
from model.train_cpt import load_chunked_dataset, load_model as load_base_model, add_lora_adapters, configure_trainer
from evals.benchmarks import (
    eval_pubmedqa, eval_medmcqa, eval_medqa, eval_medication_qa, eval_bioasq
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def setup_device():
    """Setup device and print info."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def run_evals(model, tokenizer, label: str = "base", max_samples: int = 200) -> Dict[str, Any]:
    """Run all benchmarks and return results."""
    print(f"\n{'='*60}")
    print(f"Running Benchmarks: {label.upper()}")
    print(f"{'='*60}")
    
    all_results = {
        "model": "HuggingFaceTB/SmolLM-135M",
        "timestamp": datetime.now().isoformat(),
        "stage": label,
        "benchmarks": {}
    }

    benchmarks = [
        ("pubmedqa", eval_pubmedqa),
        ("medmcqa", eval_medmcqa),
        ("medqa", eval_medqa),
        ("medication_qa", eval_medication_qa),
        ("bioasq_yesno", eval_bioasq),
    ]

    for name, eval_fn in benchmarks:
        try:
            print(f"\nRunning {name}...")
            result = eval_fn(model, tokenizer, max_samples=max_samples)
            all_results["benchmarks"][name] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }
            print(f"✓ {name} complete")
        except Exception as e:
            print(f"✗ ERROR running {name}: {e}")
            all_results["benchmarks"][name] = {"error": str(e)}

    return all_results


def compare_results(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compare evaluation results before and after training."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {}
    }

    for benchmark_name in before["benchmarks"]:
        before_result = before["benchmarks"][benchmark_name]
        after_result = after["benchmarks"].get(benchmark_name, {})

        if "error" in before_result or "error" in after_result:
            comparison["benchmarks"][benchmark_name] = {
                "before": before_result,
                "after": after_result,
                "status": "error"
            }
            continue

        before_acc = before_result.get("accuracy", 0)
        after_acc = after_result.get("accuracy", 0)
        improvement = after_acc - before_acc
        improvement_pct = (improvement / before_acc * 100) if before_acc > 0 else 0

        comparison["benchmarks"][benchmark_name] = {
            "before": before_result,
            "after": after_result,
            "improvement": round(improvement, 4),
            "improvement_pct": round(improvement_pct, 2),
        }

    # Calculate overall stats
    before_accs = [
        v["accuracy"] for v in before["benchmarks"].values() 
        if "accuracy" in v
    ]
    after_accs = [
        v["accuracy"] for v in after["benchmarks"].values() 
        if "accuracy" in v
    ]

    if before_accs and after_accs:
        avg_before = sum(before_accs) / len(before_accs)
        avg_after = sum(after_accs) / len(after_accs)
        comparison["overall"] = {
            "avg_accuracy_before": round(avg_before, 4),
            "avg_accuracy_after": round(avg_after, 4),
            "avg_improvement": round(avg_after - avg_before, 4),
            "avg_improvement_pct": round((avg_after - avg_before) / avg_before * 100, 2) if avg_before > 0 else 0,
        }

    return comparison


def main():
    print("=" * 60)
    print("SmolLM-135M Medical Training Pipeline")
    print("=" * 60)
    
    # Setup
    device = setup_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Stage 1: Load dataset
    print("\n[1/5] Loading and preparing dataset...")
    try:
        train_dataset, val_dataset = load_chunked_dataset()
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Stage 2: Load base model and run initial evals
    print("\n[2/5] Loading base model and running initial evaluations...")
    try:
        model, tokenizer = load_base_model()
        print(f"✓ Base model loaded on {model.device}")
        
        results_before = run_evals(model, tokenizer, label="before_training", max_samples=100)
        
        # Save pre-training results
        before_path = RESULTS_DIR / f"evals_before_training_{timestamp}.json"
        with open(before_path, "w") as f:
            json.dump(results_before, f, indent=2)
        print(f"✓ Pre-training evaluation results saved to {before_path}")
    except Exception as e:
        print(f"✗ Failed during pre-training evaluation: {e}")
        return
    finally:
        # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 3: Setup training
    print("\n[3/5] Setting up training...")
    try:
        model, tokenizer = load_base_model()
        model = add_lora_adapters(model)
        trainer = configure_trainer(model, tokenizer, train_dataset, val_dataset)
        print("✓ Training setup complete")
    except Exception as e:
        print(f"✗ Failed to setup training: {e}")
        return

    # Stage 4: Train model
    print("\n[4/5] Training model...")
    try:
        print("-" * 60)
        trainer_stats = trainer.train()
        print("-" * 60)
        print(f"✓ Training complete")
        print(f"  Best eval loss: {trainer.state.best_metric:.4f}")
        
        # Save training stats
        train_stats_path = RESULTS_DIR / f"training_stats_{timestamp}.json"
        with open(train_stats_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"  Training stats saved to {train_stats_path}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return

    # Stage 5: Run post-training evals
    print("\n[5/5] Running evaluations on trained model...")
    try:
        results_after = run_evals(model, tokenizer, label="after_training", max_samples=100)
        
        # Save post-training results
        after_path = RESULTS_DIR / f"evals_after_training_{timestamp}.json"
        with open(after_path, "w") as f:
            json.dump(results_after, f, indent=2)
        print(f"✓ Post-training evaluation results saved to {after_path}")
    except Exception as e:
        print(f"✗ Failed during post-training evaluation: {e}")
        return
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    comparison = compare_results(results_before, results_after)
    comparison_path = RESULTS_DIR / f"comparison_{timestamp}.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"✓ Comparison results saved to {comparison_path}")
    
    # Print summary
    if "overall" in comparison:
        print(f"\nOverall Results:")
        print(f"  Average accuracy before: {comparison['overall']['avg_accuracy_before']:.4f}")
        print(f"  Average accuracy after:  {comparison['overall']['avg_accuracy_after']:.4f}")
        print(f"  Improvement:             {comparison['overall']['avg_improvement']:+.4f} ({comparison['overall']['avg_improvement_pct']:+.2f}%)")
    
    print(f"\nBenchmark-by-benchmark:")
    for benchmark_name, result in comparison["benchmarks"].items():
        if "error" not in result:
            print(f"  {benchmark_name:20} {result['before']['accuracy']:.4f} → {result['after']['accuracy']:.4f} ({result['improvement']:+.4f})")
        else:
            print(f"  {benchmark_name:20} ERROR")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Results directory: {RESULTS_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
