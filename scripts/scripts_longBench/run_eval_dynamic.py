import argparse
import glob
import os
import re
import subprocess
import tempfile


def discover_methods(results_dir: str):
    methods = set()
    dataset_dirs = [d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d)]
    for dataset_dir in dataset_dirs:
        for fp in glob.glob(os.path.join(dataset_dir, "*.json")):
            name = os.path.basename(fp)
            if name == "metrics.json" or name.endswith(".pretty.json"):
                continue
            methods.add(name[:-5])
    return sorted(methods)


def patch_methods_list(eval_source: str, methods):
    methods_repr = "[" + ", ".join(repr(m) for m in methods) + "]"
    patterns = [
        r"(?m)^methods\s*=\s*\[[^\]]*\]",
        r"(?m)^methods\s*=\s*\[[^\]]*\]",
    ]
    for pattern in patterns:
        patched, count = re.subn(pattern, f"methods = {methods_repr}", eval_source, count=1)
        if count:
            return patched, True
    return eval_source, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str)
    parser.add_argument("--eval_py", required=True, type=str)
    args, extra = parser.parse_known_args()

    methods = discover_methods(args.results_dir)
    if not methods:
        raise ValueError(f"No method json files found under {args.results_dir}")
    print(f"[dynamic-eval] methods discovered from results: {methods}")

    with open(args.eval_py, "r", encoding="utf-8") as f:
        source = f.read()
    patched_source, patched = patch_methods_list(source, methods)
    if not patched:
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
