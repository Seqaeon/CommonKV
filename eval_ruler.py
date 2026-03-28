import os
import json
import argparse
import glob
import numpy as np
import csv
from tqdm import tqdm

from metrics import (
    string_match_all
)

def find_case_insensitive_path(path):
    if not path or os.path.exists(path):
        return path
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        return path
    target = os.path.basename(path).lower()
    try:
        for entry in os.listdir(parent):
            if entry.lower() == target:
                new_path = os.path.join(parent, entry)
                print(f"[INFO] Path mismatch detected. Using case-insensitive match: {new_path}")
                return new_path
    except Exception:
        pass
    return path

def discover_methods_and_datasets(results_dir: str):
    methods = set()
    datasets = set()
    # RULER structure: results_dir / dataset / method.json
    dataset_dirs = [d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d)]
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        has_jsons = False
        for fp in glob.glob(os.path.join(dataset_dir, "*.json")):
            name = os.path.basename(fp)
            if name == "metrics.json" or name.endswith(".pretty.json"):
                continue
            if os.path.getsize(fp) > 0:
                methods.add(name[:-5])
                has_jsons = True
        if has_jsons:
            datasets.add(dataset_name)
    return sorted(methods), sorted(datasets)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--methods', type=str, default=None, help="Comma-separated list of methods to evaluate")
    parser.add_argument('--datasets', type=str, default=None, help="Comma-separated list of datasets to evaluate")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    
    # Discovery / Argument handling
    args.results_dir = find_case_insensitive_path(args.results_dir)
    discovered_methods, discovered_datasets = discover_methods_and_datasets(args.results_dir)
    
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = discovered_methods
            
    dataset_list = discovered_datasets if not args.datasets else [d.strip() for d in args.datasets.split(",")]
    
    if not dataset_list:
        print(f"\n[ERROR] No datasets found in: {args.results_dir}")
        print("Expected structure: results_dir/dataset_name/method_name.json")
        print("Please check your results folder or --results_dir argument.\n")
        import sys
        sys.exit(1)

    print(f"Datasets to evaluate: {dataset_list}")
    print(f"Methods to evaluate: {methods}")

    results_list = []
    cr_list = []
    long_results = []
    
    # Wide format headers
    results_list.append(["dataset"] + methods)
    cr_list.append(["dataset"] + methods)

    for dataset in tqdm(dataset_list):
        row = [dataset]
        cr_row = [dataset]
        
        for method in methods:
            try:
                args.method = method
                args.dataset = dataset
                eval_file = os.path.join(args.results_dir, dataset, f"{method}.json")
                
                if not os.path.exists(eval_file) or os.path.getsize(eval_file) == 0:
                    row.append(-1)
                    cr_row.append(-1)
                    continue

                predictions, answers, lengths, ratios, latencies, tpss = [], [], [], [], [], []
                with open(eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            # RULER JSON files can be line-delimited JSON or a single JSON list
                            # Standard run_ruler.py saves as line-delimited (JSONL named .json)
                            data = json.loads(line)
                            predictions.append(data.get("pred", ""))
                            answers.append(data.get("answers", []))
                            if "length" in data:
                                lengths.append(data["length"])
                            if "compression_ratio" in data:
                                ratios.append(data["compression_ratio"])
                            if "latency" in data:
                                latencies.append(data["latency"])
                            if "tps" in data:
                                tpss.append(data["tps"])
                        except Exception:
                            continue

                if len(predictions) == 0:
                    row.append(-1)
                    cr_row.append(-1)
                    continue
                
                # RULER uses string_match_all
                score = string_match_all(predictions, answers)
                cr = np.mean(ratios) if ratios else 1.0
                
                row.append(score)
                cr_row.append(cr)
                
                avg_latency = np.mean(latencies) if latencies else 0.0
                avg_tps = np.mean(tpss) if tpss else 0.0
                
                long_results.append({
                    "dataset": dataset, 
                    "method": method, 
                    "score": score, 
                    "compression_ratio": round(float(cr), 4),
                    "latency": round(float(avg_latency), 4),
                    "tps": round(float(avg_tps), 2)
                })
                
                # per-method-dataset metrics file
                scores = {
                    dataset: score, 
                    "compression_ratio": cr,
                    "latency": avg_latency,
                    "tps": avg_tps
                }
                output_dir = os.path.dirname(eval_file)
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
            
                # print(f"dataset {dataset} method {method} scores {score}")
            except Exception as e:
                row.append(-1)
                cr_row.append(-1)
                print(f"dataset {dataset} method {method} scores {None} (Error: {e})")
        
        results_list.append(row)
        cr_list.append(cr_row)
                
    # 1. Wide Results (Score only)
    with open(os.path.join(args.results_dir, "results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)
    
    # 2. Wide Compression Ratios
    with open(os.path.join(args.results_dir, "compression_ratios.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(cr_list)
        
    # 3. Long Format (Best for Pandas - The user's notebook specifically asks for results_ruler.jsonl)
    with open(os.path.join(args.results_dir, "results_ruler.jsonl"), 'w') as fp:
        for entry in long_results:
            fp.write(json.dumps(entry) + "\n")

    print(f"\n[DONE] Evaluation complete. Summary saved to: {args.results_dir}")
    print(f"- Wide scores: results.csv")
    print(f"- Wide ratios: compression_ratios.csv")
    print(f"- Long format: results_ruler.jsonl")
