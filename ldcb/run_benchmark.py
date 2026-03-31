import json
import os
import torch
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

from ldcb.methods.fullkv import FullKVMethod
from ldcb.methods.kivi import KIVIMethod
from ldcb.methods.apkvc import APKVCMethod
from ldcb.tasks.continuation import run_continuation
from ldcb.tasks.reasoning import run_reasoning
from ldcb.tasks.multiturn import run_multiturn
from ldcb.plots import plot1_compression_vs_length, plot2_pareto_frontier, plot3_vram_over_turns

def load_model(model_id, device_map="cuda"):
    print(f"Loading model {model_id} (device_map={device_map})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run Long-Decode Compression Benchmark (LDCB)")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model ID to benchmark")
    parser.add_argument("--output_dir", type=str, default="ldcb/results", help="Directory to save results")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map to use (e.g. 'auto', 'cuda:0')")
    parser.add_argument("--tasks", type=str, default="continuation,reasoning,multiturn", help="Comma-separated list of tasks to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, tokenizer = load_model(args.model_id, args.device_map)
    
    # Detect architectural context limit and calculate safe benchmark targets
    max_pos = getattr(model.config, "max_position_embeddings", 2048)
    print(f"Model capacity: {max_pos} tokens")

    # Continuation: Reserve ~128 tokens for prompt
    safe_continuation_max = min(4000, max_pos - 128)
    # Reasoning: Reserve ~128 tokens for prompt 
    safe_reasoning_max = min(2000, max_pos - 128)
    # Multiturn: Ensure (turns * 150) + prompt < max_pos
    # Heuristic: 100 tokens response + 50 tokens avg user msg
    safe_multiturn_turns = min(15, (max_pos - 100) // 150)

    # Define methods to benchmark
    methods = {
        "FullKV":          FullKVMethod(),
        "KIVI-int4":       KIVIMethod(bits=4),
        "KIVI-int2":       KIVIMethod(bits=2),
        "APKVC-identity":  APKVCMethod(predictor_type="identity"),
        "APKVC-linear":    APKVCMethod(predictor_type="linear"),
    }

    all_results = {}
    selected_tasks = args.tasks.split(",")

    # ----- Task 1: Continuation -----
    if "continuation" in selected_tasks:
        print("\n" + "=" * 60)
        print("TASK 1: Long continuation")
        print("=" * 60)
        task1_results = {}
        for name, method in methods.items():
            print(f"\nRunning {name}...")
            task1_results[name] = run_continuation(method, model, tokenizer, max_new_tokens=safe_continuation_max)
            # Memory Cleanup
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        all_results["task1_continuation"] = task1_results

    # ----- Task 2: Reasoning -----
    if "reasoning" in selected_tasks:
        print("\n" + "=" * 60)
        print("TASK 2: Structured reasoning")
        print("=" * 60)
        task2_results = {}
        for name, method in methods.items():
            print(f"\nRunning {name}...")
            task2_results[name] = run_reasoning(method, model, tokenizer, max_new_tokens=safe_reasoning_max)
            # Memory Cleanup
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        all_results["task2_reasoning"] = task2_results

    # ----- Task 3: Multi-turn -----
    if "multiturn" in selected_tasks:
        print("\n" + "=" * 60)
        print("TASK 3: Multi-turn simulation")
        print("=" * 60)
        task3_results = {}
        for name, method in methods.items():
            print(f"\nRunning {name}...")
            task3_results[name] = run_multiturn(method, model, tokenizer, n_turns=safe_multiturn_turns)
            # Memory Cleanup
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        all_results["task3_multiturn"] = task3_results

    # ----- Save raw results -----
    results_path = os.path.join(args.output_dir, f"ldcb_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ----- Generate plots -----
    if "continuation" in all_results:
        plot1_compression_vs_length(
            all_results["task1_continuation"],
            task_name="Continuation",
            save_path=os.path.join(args.output_dir, f"plot1_continuation_{timestamp}.png"),
        )
        # Pareto data
        pareto_data = []
        for name, r in all_results["task1_continuation"].items():
            pareto_data.append({
                "method": name,
                "compression_ratio": r.get("final_compression_ratio", {}).get("mean", 1.0),
                "perplexity": r.get("perplexity", {}).get("mean", 0.0),
                "rouge_l": 1.0, # Placeholder or compute against FullKV
                "config_label": name,
            })
        plot2_pareto_frontier(pareto_data, save_path=os.path.join(args.output_dir, f"plot2_pareto_{timestamp}.png"))

    if "task3_multiturn" in all_results:
        plot3_vram_over_turns(
            {name: r["turns"] for name, r in all_results["task3_multiturn"].items()},
            save_path=os.path.join(args.output_dir, f"plot3_vram_{timestamp}.png"),
        )

    print("Plots saved to", args.output_dir)

    # ----- Summary Table -----
    if "task1_continuation" in all_results:
        print("\n" + "=" * 60)
        print("SUMMARY — Task 1 (Continuation)")
        print("=" * 60)
        print(f"{'Method':<20} {'Compression':>12} {'Perplexity':>12}")
        print("-" * 46)
        for name, r in all_results["task1_continuation"].items():
            cr = r.get("final_compression_ratio", {}).get("mean", 0)
            pp = r.get("perplexity", {}).get("mean", 0)
            print(f"{name:<20} {cr:>12.3f} {pp:>12.2f}")

if __name__ == "__main__":
    main()
