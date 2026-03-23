results_dir=$1  # results directory
eval_py=${2:-eval.py}  # optional absolute/relative path to LongBench eval.py
dynamic_methods=${3:-1}  # set to 0 to disable dynamic method patching
script_dir="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${results_dir}" ]; then
    echo "Usage: sh scripts/scripts_longBench/metrics.sh <results_dir> [eval_py_path]"
    exit 1
fi

if [ ! -f "${eval_py}" ]; then
    echo "Error: eval.py not found at '${eval_py}'."
    echo "Pass an explicit evaluator path, e.g.:"
    echo "  sh scripts/scripts_longBench/metrics.sh <results_dir> /kaggle/working/LongBench/eval.py"
    exit 1
fi

if [ "${dynamic_methods}" = "1" ]; then
    python3 "${script_dir}/run_eval_dynamic.py" \
        --results_dir "${results_dir}" \
        --eval_py "${eval_py}"
else
    python3 "${eval_py}" \
        --results_dir "${results_dir}"
fi
