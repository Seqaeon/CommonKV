import argparse
import glob
import os
import re
import subprocess
import tempfile


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
            # Skip empty or corrupted result files
            if os.path.getsize(fp) == 0:
                continue
            methods.add(name[:-5])
            has_jsons = True
        if has_jsons:
            datasets.add(dataset_name)
    return sorted(methods), sorted(datasets)


def patch_list(eval_source: str, list_name: str, items: list):
    items_repr = "[" + ", ".join(repr(m) for m in items) + "]"
    patterns = [
        rf"(?ms)^{list_name}\s*=\s*\[.*?\]",
        rf"(?ms)^{list_name}\s*:\s*list\s*=\s*\[.*?\]",
        rf"(?ms)^{list_name.upper()}\s*=\s*\[.*?\]",
    ]
    for pattern in patterns:
        patched, count = re.subn(pattern, f"{list_name} = {items_repr}", eval_source, count=1)
        if count:
            return patched, True
    return eval_source, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str)
    parser.add_argument("--eval_py", required=True, type=str)
    args, extra = parser.parse_known_args()

    methods, datasets = discover_methods_and_datasets(args.results_dir)
    if not methods:
        raise ValueError(f"No method json files found under {args.results_dir}")
    print(f"[dynamic-eval] methods discovered from results: {methods}")
    print(f"[dynamic-eval] datasets discovered from results: {datasets}")

    with open(args.eval_py, "r", encoding="utf-8") as f:
        source = f.read()

    patched_source, patched_m = patch_list(source, "methods", methods)
    patched_source, patched_d = patch_list(patched_source, "dataset_list", datasets)

    if not patched_m:
        print("[dynamic-eval][WARN] Could not find a top-level 'methods = [...]' list to patch; running evaluator as-is.")
        cmd = ["python3", args.eval_py, "--results_dir", args.results_dir, *extra]
        raise SystemExit(subprocess.call(cmd))

    with tempfile.TemporaryDirectory() as tmpdir:
        patched_eval = os.path.join(tmpdir, "eval_dynamic_methods.py")
        with open(patched_eval, "w", encoding="utf-8") as f:
            f.write(patched_source)
        cmd = ["python3", patched_eval, "--results_dir", args.results_dir, *extra]
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
