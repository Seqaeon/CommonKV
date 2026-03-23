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

def scorer_e(dataset, predictions, answers, lengths, all_classes):
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
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    if len(predictions) == 0:
        return 0.0
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    
    # Discovery / Argument handling
    discovered_methods, discovered_datasets = discover_methods_and_datasets(args.results_dir)
    
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        # If the global 'methods' list looks unpatched/legacy, use discovered ones
        legacy_methods = ["FullKV", "random", "SnapKV", "StreamingLLM", "H2O", "PyramidKV", "L2Norm","CAM","ThinK"]
        if methods == legacy_methods:
            methods = discovered_methods
            
    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(",")]
    else:
        # If the global 'dataset_list' looks unpatched/legacy and we found others, prefer found
        if len(discovered_datasets) > 0:
            dataset_list = discovered_datasets

    print(f"Datasets to evaluate: {dataset_list}")
    print(f"Methods to evaluate: {methods}")

    results_list = [["dataset"]] + [[m] for m in methods]
    
    for dataset in dataset_list:
        results_list[0].append(dataset)
        
        for idx, method in enumerate(methods):
            try:
                args.method = method
                args.dataset = dataset
                args.eval_file = os.path.join(args.results_dir, dataset, f"{method}.json")
                
                if not os.path.exists(args.eval_file) or os.path.getsize(args.eval_file) == 0:
                    results_list[idx+1].append(-1)
                    continue

                predictions, answers, lengths = [], [], []
                with open(args.eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            all_classes = data["all_classes"]
                            if "length" in data:
                                lengths.append(data["length"])
                        except Exception:
                            continue

                if len(predictions) == 0:
                    results_list[idx+1].append(-1)
                    continue

                if args.longbench_e:
                    score = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                else:
                    score = scorer(args.dataset, predictions, answers, all_classes)
                    
                results_list[idx+1].append(score)
                
                scores = {args.dataset: score}
                output_dir = os.path.dirname(args.eval_file)
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
            
                print(f"dataset {args.dataset} method {args.method} scores {scores}")
            except Exception as e:
                results_list[idx+1].append(-1)
                print(f"dataset {args.dataset} method {args.method} scores {None} (Error: {e})")
                
    import csv
    with open(os.path.join(args.results_dir, f"results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)
