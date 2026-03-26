import os
import json
import argparse
import glob
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)
from tqdm import tqdm

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# These lists will be dynamically patched by run_eval_dynamic.py
methods = ["FullKV", "random", "SnapKV", "StreamingLLM", "H2O", "PyramidKV", "L2Norm","CAM","ThinK"]
dataset_list = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p"
]

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
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes, ratios=None):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
    avg_ratio = np.mean(ratios) if ratios is not None and len(ratios) > 0 else 1.0
    final_scores = {}
    for key in scores.keys():
        final_scores[key] = round(100 * np.mean(scores[key]), 2) if len(scores[key]) > 0 else 0.0
    return final_scores, avg_ratio

def scorer(dataset, predictions, answers, all_classes, ratios=None):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    
    avg_score = round(100 * total_score / len(predictions), 2) if len(predictions) > 0 else 0.0
    avg_ratio = np.mean(ratios) if ratios is not None and len(ratios) > 0 else 1.0
    return avg_score, avg_ratio

if __name__ == '__main__':
    args = parse_args()
    
    # Discovery / Argument handling
    args.results_dir = find_case_insensitive_path(args.results_dir)
    discovered_methods, discovered_datasets = discover_methods_and_datasets(args.results_dir)
    
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        # If the global 'methods' list looks unpatched/legacy, use discovered ones
        legacy_methods = ["FullKV", "random", "SnapKV", "StreamingLLM", "H2O", "PyramidKV", "L2Norm","CAM","ThinK"]
        if methods == legacy_methods:
            methods = discovered_methods
            
    dataset_list = discovered_datasets if not args.datasets else [d.strip() for d in args.datasets.split(",")]
    
    if not dataset_list:
        print(f"\n[ERROR] No datasets found in: {args.results_dir}")
        print("Expected structure: results_dir/dataset_name/method_name.json")
        print("Please check your results folder or --results_dir argument.\n")
        import sys
        sys.exit(1)

    # method_scores[method][dataset] = float_score
    method_scores = {m: {} for m in methods}

    print(f"Datasets to evaluate: {dataset_list}")
    print(f"Methods to evaluate: {methods}")

    results_list = []
    cr_list = []
    long_results = []
    
    for dataset in tqdm(dataset_list):
        row = [dataset]
        cr_row = [dataset]
        
        for method in methods:
            try:
                args.method = method
                args.dataset = dataset
                args.eval_file = os.path.join(args.results_dir, dataset, f"{method}.json")
                
                if not os.path.exists(args.eval_file) or os.path.getsize(args.eval_file) == 0:
                    row.append(-1)
                    cr_row.append(-1)
                    continue

                predictions, answers, lengths, ratios, latencies, tpss = [], [], [], [], [], []
                with open(args.eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            all_classes = data["all_classes"]
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

                if args.longbench_e:
                    score, cr = scorer_e(args.dataset, predictions, answers, lengths, all_classes, ratios=ratios)
                else:
                    score, cr = scorer(args.dataset, predictions, answers, all_classes, ratios=ratios)
                    
                row.append(score)
                cr_row.append(cr)
                
                avg_latency = np.mean(latencies) if latencies else 0.0
                avg_tps = np.mean(tpss) if tpss else 0.0
                
                long_results.append({
                    "dataset": dataset, 
                    "method": method, 
                    "score": score, 
                    "compression_ratio": cr,
                    "latency": round(float(avg_latency), 4),
                    "tps": round(float(avg_tps), 2)
                })
                
                scores = {
                    args.dataset: score, 
                    "compression_ratio": cr,
                    "latency": avg_latency,
                    "tps": avg_tps
                }
                output_dir = os.path.dirname(args.eval_file)
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
            
                print(f"dataset {args.dataset} method {args.method} scores {scores}")
            except Exception as e:
                row.append(-1)
                cr_row.append(-1)
                print(f"dataset {args.dataset} method {args.method} scores {None} (Error: {e})")
        
        results_list.append(row)
        cr_list.append(cr_row)
                
    import csv
    # 1. Wide Results (Score only)
    with open(os.path.join(args.results_dir, "results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(["dataset"] + methods)
        writer.writerows(results_list)
    
    # 2. Wide Compression Ratios
    with open(os.path.join(args.results_dir, "compression_ratios.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(["dataset"] + methods)
        writer.writerows(cr_list)
        
    # 3. Long Format (Best for Pandas)
    with open(os.path.join(args.results_dir, "results_long.jsonl"), 'w') as fp:
        for entry in long_results:
            fp.write(json.dumps(entry) + "\n")

    # Generate Tie Table (Win/Loss/Tie Matrix)
    print("\n" + "="*50)
    print("      TIE TABLE (Win / Loss / Tie)")
    print("="*50)
    
    # Header
    header = " " * 15
    for m in methods:
        header += f"{m[:10]:>12}"
    print(header)

    tie_results = []
    for m1 in methods:
        row = [m1]
        print_row = f"{m1[:12]:<15}"
        for m2 in methods:
            if m1 == m2:
                print_row += f"{'-':>12}"
                row.append("-")
            else:
                wins, losses, ties = 0, 0, 0
                for d in dataset_list:
                    s1 = method_scores[m1].get(d, -1)
                    s2 = method_scores[m2].get(d, -1)
                    if s1 == -1 or s2 == -1: continue
                    
                    if s1 > s2 + 0.1: wins += 1
                    elif s2 > s1 + 0.1: losses += 1
                    else: ties += 1
                
                cell = f"{wins}/{losses}/{ties}"
                print_row += f"{cell:>12}"
                row.append(cell)
        print(print_row)
        tie_results.append(row)

    with open(os.path.join(args.results_dir, "tie_table.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Method"] + methods)
        writer.writerows(tie_results)
    print("="*50 + "\n")
