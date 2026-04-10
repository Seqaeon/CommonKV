import gc
import json
import os
import torch
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ldcb.methods.fullkv import FullKVMethod
from ldcb.methods.kivi import KIVIMethod
# from ldcb.methods.apkvc import APKVCMethod  # commented out — replaced by IAVQ-KC
from ldcb.methods.iavq_kc import IAVQKCMethod
from ldcb.methods.commvq import CommVQMethod
from ldcb.tasks.continuation import run_continuation
from ldcb.tasks.reasoning import run_reasoning
from ldcb.tasks.multiturn import run_multiturn
from ldcb.plots import plot1_compression_vs_length, plot2_pareto_frontier, plot3_vram_over_turns


def load_model(model_id, device_map="auto", load_in_8bit=False, load_in_4bit=False,
               attn_implementation="eager"):
    print(f"Loading model {model_id} (device_map={device_map}, attn={attn_implementation})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = None
    if load_in_4bit:
        print("  Using 4-bit NF4 weight quantization (bitsandbytes)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        print("  Using 8-bit weight quantization (bitsandbytes)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if bnb_config is None else None,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Codebook calibration
# ---------------------------------------------------------------------------

CALIBRATION_PROMPTS = [
    "The history of artificial intelligence spans decades of research, beginning with early work on symbolic reasoning and logic-based systems. Researchers initially believed that human intelligence could be reduced to symbol manipulation. Over time, it became clear that learning from data was essential, giving rise to the field of machine learning.",
    "Climate change poses one of the most significant challenges of the twenty-first century. Rising global temperatures are causing glaciers to melt, sea levels to rise, and extreme weather events to become more frequent. Scientists stress the urgency of transitioning to renewable energy sources and reducing greenhouse gas emissions.",
    "The development of large language models has transformed natural language processing. Models trained on vast corpora of text have demonstrated remarkable abilities in translation, summarisation, question answering, and creative writing. These capabilities emerge from learning statistical patterns across billions of tokens.",
    "Modern operating systems manage hardware resources through layers of abstraction. The kernel handles memory allocation, process scheduling, and I/O operations. User applications interact with the kernel through system calls, which provide a controlled interface to hardware without exposing low-level details.",
    "Economic inequality has grown substantially in many countries over the past four decades. While technological progress has created enormous wealth, the gains have been concentrated among a small fraction of the population. Policymakers debate the appropriate role of taxation, education, and social programmes in addressing this disparity.",
    "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against pathogens. When a foreign antigen is detected, B cells produce antibodies while T cells coordinate the adaptive immune response. Vaccines exploit this mechanism by training the immune system without causing disease.",
    "Quantum computing leverages the principles of superposition and entanglement to perform computations that are intractable for classical computers. A qubit can represent both zero and one simultaneously, allowing quantum algorithms to explore many solutions in parallel. Error correction remains one of the central engineering challenges in building fault-tolerant quantum computers.",
    "The Renaissance was a period of profound cultural and intellectual transformation in Europe, spanning roughly the fourteenth to seventeenth centuries. It witnessed a revival of interest in classical antiquity, advancements in art and architecture, and the emergence of humanism as a philosophical framework. Figures such as Leonardo da Vinci, Michelangelo, and Galileo exemplified the era's spirit of inquiry.",
]


def calibrate_apkvc(model, tokenizer, output_dir, prompts,
                    gen_tokens=300, K_num_codebooks=4, V_num_codebooks=2,
                    codebook_size=256, iters=25, per_layer=True,
                    trainer="kmeans", codebook_structure="unconstrained"):
    """
    Run a short tracing pass to collect KV residual statistics, then
    fit RVQ codebooks via K-means.  Returns the path to the saved .pt file.

    Strategy
    --------
    1. Run APKVCMethod with trace_output_path set, generating a few hundred
       tokens per calibration prompt.  The cluster's _append_trace_samples()
       collects normalised residual vectors for every non-anchor decode token.
    2. Force-flush the traces via atexit handler by importing the cluster class
       and calling its dump method directly.
    3. Pass the trace file to the calibration K-means routine.
    4. Return the calibration path so APKVCMethod can load it.
    """
    from scripts.calibrate_apkvc_codebooks import (
        _load_trace_tensors, _residual_kmeans
    )
    from attention_aware_predictive_kv import AttentionAwarePredictiveKVCluster

    trace_path = os.path.join(output_dir, "apkvc_calibration_trace.pt")
    calib_path = os.path.join(output_dir, "apkvc_codebooks.pt")

    # Skip if already calibrated for this run
    if os.path.isfile(calib_path):
        print(f"[APKVC] Found existing codebooks: {calib_path}  (skipping calibration)")
        return calib_path

    print(f"\n{'='*60}")
    print("APKVC CODEBOOK CALIBRATION")
    print(f"{'='*60}")
    n_prompts = len(prompts)
    print(f"Collecting residual traces from {n_prompts} prompts × {gen_tokens} tokens...")

    # Reset class-level trace state so we get a clean trace
    AttentionAwarePredictiveKVCluster._trace_delta_k = {}
    AttentionAwarePredictiveKVCluster._trace_delta_v = {}
    AttentionAwarePredictiveKVCluster._trace_output_path = trace_path
    AttentionAwarePredictiveKVCluster._trace_registered = True  # prevent re-register

    trace_method = APKVCMethod(
        predictor_type="identity",
        trace_output_path=trace_path,
        trace_max_samples=200_000,
        per_layer_codebooks=per_layer,
        K_num_codebooks=K_num_codebooks,
        V_num_codebooks=V_num_codebooks,
    )

    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}/{n_prompts}...")
        with torch.no_grad():
            trace_method.generate(
                model, tokenizer, prompt,
                max_new_tokens=gen_tokens,
                checkpoint_steps=[gen_tokens],
            )
        torch.cuda.empty_cache()
        gc.collect()

    # Flush traces from class-level buffers to disk
    print("  Flushing trace buffers...")
    AttentionAwarePredictiveKVCluster._dump_traces_at_exit()

    if not os.path.isfile(trace_path):
        print("[APKVC][WARN] Trace file not written — codebooks will remain random.")
        return None

    # Fit codebooks
    print(f"  Fitting codebooks (per_layer={per_layer}, K×{K_num_codebooks}, V×{V_num_codebooks}, size={codebook_size})...")
    K_dicts = _load_trace_tensors([trace_path], "delta_K_base", max_samples=200_000)
    V_dicts = _load_trace_tensors([trace_path], "delta_V",       max_samples=200_000)

    head_dim = next(iter(K_dicts.values())).shape[-1]

    if per_layer:
        n_layers = max(list(K_dicts.keys()) + list(V_dicts.keys())) + 1
        K_cbs_list, V_cbs_list = [], []
        for l in range(n_layers):
            kl = K_dicts.get(l, K_dicts.get(0))
            vl = V_dicts.get(l, V_dicts.get(0))
            K_cbs_list.append(_residual_kmeans(kl, K_num_codebooks, codebook_size, iters, trainer=trainer, commutative_2x2=(codebook_structure == "rope_commutative_2x2")))
            V_cbs_list.append(_residual_kmeans(vl, V_num_codebooks, codebook_size, iters, trainer=trainer, commutative_2x2=(codebook_structure == "rope_commutative_2x2")))
            if (l + 1) % 4 == 0:
                print(f"    ...layer {l+1}/{n_layers}")
        K_codebooks = torch.stack(K_cbs_list, dim=0)
        V_codebooks = torch.stack(V_cbs_list, dim=0)
    else:
        K_all = torch.cat(list(K_dicts.values()), dim=0)
        V_all = torch.cat(list(V_dicts.values()), dim=0)
        K_codebooks = _residual_kmeans(K_all, K_num_codebooks, codebook_size, iters, trainer=trainer, commutative_2x2=(codebook_structure == "rope_commutative_2x2"))
        V_codebooks = _residual_kmeans(V_all, V_num_codebooks, codebook_size, iters, trainer=trainer, commutative_2x2=(codebook_structure == "rope_commutative_2x2"))

    torch.save({
        "K_codebooks": K_codebooks.half(),
        "V_codebooks": V_codebooks.half(),
        "metadata": {
            "K_num_codebooks": K_num_codebooks,
            "V_num_codebooks": V_num_codebooks,
            "codebook_size":   codebook_size,
            "per_layer":       per_layer,
            "head_dim":        head_dim,
            "trainer":         trainer,
            "codebook_structure": codebook_structure,
        },
    }, calib_path)
    print(f"  Saved calibrated codebooks → {calib_path}")
    return calib_path


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Long-Decode Compression Benchmark (LDCB)")
    parser.add_argument("--model_id",   type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="ldcb/results")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--tasks",      type=str, default="continuation,reasoning,multiturn")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
                        help="Attention backend. Defaults to 'eager' (required for IAVQ-KC "
                             "which needs output_attentions=True at prefill). Use 'sdpa', "
                             "'flash_attention_2', or 'flash_attention_3' only if you are "
                             "not running IAVQ-KC.")
    # Memory-saving options
    parser.add_argument("--low_memory", action="store_true",
                        help="Enable all memory-saving options at once: 8-bit weights, "
                             "KIVI CPU offload, reduced calibration (5x500), "
                             "max 2000 continuation tokens")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the most recent results JSON in output_dir. "
                             "Methods already present in that file are skipped.")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Quantize model weights to 8-bit with bitsandbytes")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Quantize model weights to 4-bit NF4 with bitsandbytes")
    parser.add_argument("--max_continuation_tokens", type=int, default=None,
                        help="Override max tokens for continuation task (default: model max - 128)")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip APKVC codebook calibration (uses random codebooks)")
    parser.add_argument("--calibrate_only", action="store_true",
                        help="Exit after calibration; do not run benchmark tasks.")
    parser.add_argument("--reuse_calibration_path", type=str, default=None,
                        help="Path to a previously saved apkvc_codebooks.pt — skips calibration "
                             "and reuses these codebooks directly. Example: "
                             "ldcb/results/llama31_8b/apkvc_codebooks.pt")
    parser.add_argument("--calibration_prompts", type=int, default=None,
                        help="Number of prompts to use for APKVC calibration")
    parser.add_argument("--calibration_tokens", type=int, default=None,
                        help="Tokens to generate per calibration prompt")
    # New calibration dataset flags
    parser.add_argument("--calibration_dataset", type=str, default=None,
                        help="HuggingFace dataset for calibration (e.g., 'wikitext')")
    parser.add_argument("--calibration_subset", type=str, default=None,
                        help="Dataset subset/config (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--calibration_split", type=str, default="train",
                        help="Dataset split for calibration (e.g., 'train', 'validation')")
    parser.add_argument("--calibration_text_column", type=str, default="text",
                        help="Column name in dataset containing text")
    # APKVC specialized options
    parser.add_argument("--apkvc_prefill_compression", type=str, default="int8", choices=["int8", "vq", "fp16"])
    parser.add_argument("--apkvc_codebook_structure", type=str, default="unconstrained", choices=["unconstrained", "rope_commutative_2x2"])
    parser.add_argument("--apkvc_enable_code_attention_lookup", action="store_true")
    parser.add_argument("--apkvc_calib_trainer", type=str, default="kmeans", choices=["kmeans", "em_closed_form"])
    # CommVQ options (requires meta-llama/Llama-3.1-8B-Instruct as --model_id)
    parser.add_argument("--add_commvq", action="store_true",
                        help="Add CommVQ-2bit to the benchmark. Requires --model_id meta-llama/Llama-3.1-8B-Instruct.")
    parser.add_argument("--commvq_bits", type=int, default=2, choices=[1, 2],
                        help="CommVQ key quantization bits (default: 2).")
    # LongBench (task4) options
    parser.add_argument("--lb_data_dir", type=str, default="data/LongBench",
                        help="Directory containing LongBench .jsonl files (one per dataset).")
    parser.add_argument("--lb_save_dir", type=str, default=None,
                        help="Directory to write run_longbench.py prediction files. "
                             "Defaults to <output_dir>/longbench_preds.")
    parser.add_argument("--lb_steps", type=int, default=-1,
                        help="Max examples per dataset in LongBench (passed as --steps to "
                             "run_longbench.py). -1 = unlimited.")
    parser.add_argument("--lb_max_datasets", type=int, default=-1,
                        help="Max number of LongBench datasets to evaluate. -1 = all 16.")
    parser.add_argument("--lb_max_capacity", type=int, default=512,
                        help="max_capacity_prompts passed to run_longbench.py for KIVI (default 512).")
    args = parser.parse_args()

    # Apply --low_memory defaults
    if args.low_memory:
        if not args.load_in_4bit:
            args.load_in_8bit = True
        if args.calibration_prompts is None:
            args.calibration_prompts = 5
        if args.calibration_tokens is None:
            args.calibration_tokens = 500
        if args.max_continuation_tokens is None:
            args.max_continuation_tokens = 2000
    # Apply regular defaults
    if args.calibration_prompts is None:
        args.calibration_prompts = 8
    if args.calibration_tokens is None:
        args.calibration_tokens = 1000

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Helper: decide whether a previously-saved result for a method
    # counts as "done" and can be skipped by --resume.
    # A result is NOT done if it has status="failed" or status="skipped"
    # (so those methods are always retried on the next run).
    # ------------------------------------------------------------------
    def _is_done(result) -> bool:
        if not result:          # missing or empty
            return False
        if isinstance(result, dict):
            # Task-4-style status dicts
            status = result.get("status", "")
            if status in ("failed", "skipped"):
                return False
            # Task-4 predictions_done also counts as done
            if status in ("predictions_done",):
                return True
        # Anything else (aggregated metric dict from tasks 1-3) counts as done
        return True

    attn_impl = args.attn_implementation or "eager"
    model, tokenizer = load_model(args.model_id, args.device_map,
                                   load_in_8bit=getattr(args, "load_in_8bit", False),
                                   load_in_4bit=getattr(args, "load_in_4bit", False),
                                   attn_implementation=attn_impl)

    max_pos = getattr(model.config, "max_position_embeddings", 2048)
    print(f"Model capacity: {max_pos} tokens")

    safe_continuation_max = args.max_continuation_tokens or min(4000, max_pos - 128)
    safe_reasoning_max    = min(2000, max_pos - 128)
    safe_multiturn_turns  = min(15, (max_pos - 100) // 150)

    # ---- Calibrate APKVC codebooks before any benchmark tasks ----
    calibration_path = None
    if getattr(args, "reuse_calibration_path", None):
        # Reuse a previously saved codebook — skip the calibration pass entirely.
        if not os.path.isfile(args.reuse_calibration_path):
            raise FileNotFoundError(
                f"--reuse_calibration_path: file not found: {args.reuse_calibration_path}")
        calibration_path = args.reuse_calibration_path
        print(f"[APKVC] Reusing codebooks from: {calibration_path}")
    elif not args.skip_calibration:
        # Load calibration prompts
        calib_prompts = []
        if args.calibration_dataset:
            try:
                from datasets import load_dataset
                print(f"[APKVC] Loading calibration dataset: {args.calibration_dataset} ({args.calibration_subset or 'default'})...")
                ds = load_dataset(args.calibration_dataset, args.calibration_subset, split=args.calibration_split, streaming=True)
                count = 0
                for item in ds:
                    text = item.get(args.calibration_text_column, "")
                    if len(text.strip()) > 100: # skip very short snippets
                        calib_prompts.append(text)
                        count += 1
                        if count >= args.calibration_prompts: break
                if len(calib_prompts) == 0:
                    print("[WARNING] Dataset loading returned no prompts! Falling back to defaults.")
                    calib_prompts = (CALIBRATION_PROMPTS * ((args.calibration_prompts // len(CALIBRATION_PROMPTS)) + 1))[:args.calibration_prompts]
                else:
                    print(f"  Loaded {len(calib_prompts)} prompts from dataset.")
            except ImportError:
                print("[ERROR] 'datasets' library not found. Run 'pip install datasets'. Falling back to default prompts.")
                calib_prompts = (CALIBRATION_PROMPTS * ((args.calibration_prompts // len(CALIBRATION_PROMPTS)) + 1))[:args.calibration_prompts]
            except Exception as e:
                print(f"[ERROR] Failed to load dataset: {e}. Falling back to default prompts.")
                calib_prompts = (CALIBRATION_PROMPTS * ((args.calibration_prompts // len(CALIBRATION_PROMPTS)) + 1))[:args.calibration_prompts]
        else:
            print("[APKVC] Using built-in calibration prompts.")
            calib_prompts = (CALIBRATION_PROMPTS * ((args.calibration_prompts // len(CALIBRATION_PROMPTS)) + 1))[:args.calibration_prompts]

        calibration_path = calibrate_apkvc(
            model, tokenizer,
            output_dir=args.output_dir,
            prompts=calib_prompts,
            gen_tokens=args.calibration_tokens,
            per_layer=True,
            trainer=args.apkvc_calib_trainer,
            codebook_structure=args.apkvc_codebook_structure,
        )
        if args.calibrate_only:
            print(f"\n[APKVC] Calibration complete. Exiting (--calibrate_only).")
            return
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("[APKVC] Skipping calibration (--skip_calibration). Using random codebooks.")

    # ---- Define methods ----
    kivi_offload = getattr(args, "low_memory", False)

    methods = {
        "FullKV":    FullKVMethod(),
        "KIVI-int4": KIVIMethod(bits=4, group_size=32, residual_length=128,
                                cpu_offload_quant=kivi_offload),
        # IAVQ-KC: Importance-Aware Vector Quantization Key Cache.
        # No offline calibration needed — codebook is built online from each
        # prompt's own prefill attention statistics.
        # See IAVQ_KC_design.md for the full design specification.
        "IAVQ-KC": IAVQKCMethod(
            codebook_size=256,        # A — total centroids, split evenly across layers
            recency_window=64,        # R — last R decode tokens stored as full fp16
            importance_anchors=128,   # M — top-M old tokens kept as full fp16
            kmeans_iters=3,
            update_strategy="recency_gated",
            update_importance_during_decode=False,
        ),
        # ----- APKVC (commented out — replaced by IAVQ-KC) -----
        # "APKVC": APKVCMethod(**{
        #     **apkvc_extra,
        #     "predictor_type":               "identity",
        #     "codebook_structure":           "unconstrained",
        #     "prefill_compression":          args.apkvc_prefill_compression,
        #     "enable_code_attention_lookup": args.apkvc_enable_code_attention_lookup,
        # }),
    }

    # CommVQ: only included when --add_commvq is set.
    # Requires meta-llama/Llama-3.1-8B-Instruct as --model_id.
    # Codebooks are auto-downloaded from huggingface.co/senfu on first run.
    if getattr(args, "add_commvq", False):
        if "llama-3" not in args.model_id.lower() and "llama3" not in args.model_id.lower():
            print("[WARNING] --add_commvq is designed for meta-llama/Llama-3.1-8B-Instruct. "
                  "Other models will likely fail codebook loading.")
        methods["CommVQ-2bit"] = CommVQMethod(
            bits=args.commvq_bits,
            residual_length=128,
            cpu_offload=kivi_offload,
        )

    all_results = {}
    selected_tasks = args.tasks.split(",")

    # ---- Resume: load most recent results snapshot if --resume is set ----
    if getattr(args, "resume", False):
        existing_jsons = sorted(
            [f for f in os.listdir(args.output_dir) if f.startswith("ldcb_") and f.endswith(".json")]
        )
        if existing_jsons:
            resume_path = os.path.join(args.output_dir, existing_jsons[-1])
            print(f"[RESUME] Loading previous results from: {resume_path}")
            with open(resume_path) as f:
                all_results = json.load(f)
            # Print what's already done
            for task_key, task_res in all_results.items():
                done = list(task_res.keys())
                print(f"  {task_key}: already done → {done}")
        else:
            print("[RESUME] No previous results found in output_dir — starting fresh.")

    def save_results_snapshot():
        results_path = os.path.join(args.output_dir, f"ldcb_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        return results_path

    def render_available_plots():
        if "task1_continuation" in all_results:
            plot1_compression_vs_length(
                all_results["task1_continuation"],
                task_name="Continuation",
                save_path=os.path.join(args.output_dir, f"plot1_continuation_{timestamp}.png"),
            )
            pareto_data = []
            for name, r in all_results["task1_continuation"].items():
                pareto_data.append({
                    "method":           name,
                    "compression_ratio": r.get("final_compression_ratio", {}).get("mean", 1.0),
                    "perplexity":        r.get("base_ppl",  {}).get("mean", float("nan")),
                    "output_kl":         r.get("output_kl", {}).get("mean", float("nan")),
                    "config_label":      name,
                })
            plot2_pareto_frontier(
                pareto_data,
                save_path=os.path.join(args.output_dir, f"plot2_pareto_{timestamp}.png"),
            )

        if "task3_multiturn" in all_results:
            plot3_vram_over_turns(
                {name: r["turns"] for name, r in all_results["task3_multiturn"].items()},
                save_path=os.path.join(args.output_dir, f"plot3_vram_{timestamp}.png"),
            )

    # ----- Task 1: Continuation -----
    if "continuation" in selected_tasks:
        print("\n" + "=" * 60)
        print("TASK 1: Long continuation")
        print("=" * 60)
        task1_results = all_results.get("task1_continuation", {})
        # Recover FullKV texts for OutputKL/DeltaPPL baseline if we're resuming
        fullkv_texts = None
        for name, method in methods.items():
            if _is_done(task1_results.get(name)):
                print(f"  Skipping {name} (already in results).")
                if name == "FullKV":
                    # Can't recover raw text from JSON — run FullKV without
                    # text-aligned baseline metrics
                    fullkv_texts = None
                continue
            print(f"\nRunning {name}...")
            aggregated, gen_texts = run_continuation(
                method, model, tokenizer,
                max_new_tokens=safe_continuation_max,
                reference_texts=fullkv_texts,
            )
            task1_results[name] = aggregated
            if name == "FullKV":
                fullkv_texts = gen_texts
            
            # Save incrementally after each method
            all_results["task1_continuation"] = task1_results
            save_results_snapshot()
            render_available_plots()

            torch.cuda.empty_cache()
            gc.collect()

    # ----- Task 2: Reasoning -----
    if "reasoning" in selected_tasks:
        print("\n" + "=" * 60)
        print("TASK 2: Structured reasoning")
        print("=" * 60)
        task2_results = all_results.get("task2_reasoning", {})
        for name, method in methods.items():
            if _is_done(task2_results.get(name)):
                print(f"  Skipping {name} (already in results).")
                continue
            print(f"\nRunning {name}...")
            task2_results[name] = run_reasoning(method, model, tokenizer,
                                                max_new_tokens=safe_reasoning_max)
            
            # Save incrementally after each method
            all_results["task2_reasoning"] = task2_results
            save_results_snapshot()
            render_available_plots()

            torch.cuda.empty_cache()
            gc.collect()

    # ----- Task 3: Multi-turn -----
    if "multiturn" in selected_tasks:
        print("\n" + "=" * 60)
        print("TASK 3: Multi-turn simulation")
        print("=" * 60)
        task3_results = all_results.get("task3_multiturn", {})
        for name, method in methods.items():
            if _is_done(task3_results.get(name)):
                print(f"  Skipping {name} (already in results).")
                continue
            print(f"\nRunning {name}...")
            task3_results[name] = run_multiturn(method, model, tokenizer,
                                                n_turns=safe_multiturn_turns)
            
            # Save incrementally after each method
            all_results["task3_multiturn"] = task3_results
            save_results_snapshot()
            render_available_plots()

            torch.cuda.empty_cache()
            gc.collect()

    # ----- Task 4: LongBench -----
    if "longbench" in selected_tasks:
        import subprocess, sys

        print("\n" + "=" * 60)
        print("TASK 4: LongBench evaluation")
        print("=" * 60)

        lb_save_dir = args.lb_save_dir or os.path.join(args.output_dir, "longbench_preds")
        os.makedirs(lb_save_dir, exist_ok=True)

        # Build dataset list (mirrors run_longbench.py's hardcoded list, capped by --lb_max_datasets)
        _all_lb_datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
            "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
            "trec", "triviaqa", "samsum", "passage_count",
            "passage_retrieval_en", "lcc", "repobench-p",
        ]
        lb_datasets = (
            _all_lb_datasets[:args.lb_max_datasets]
            if args.lb_max_datasets != -1
            else _all_lb_datasets
        )
        # Filter to datasets for which data files actually exist
        lb_datasets = [
            d for d in lb_datasets
            if os.path.exists(os.path.join(args.lb_data_dir, f"{d}.jsonl"))
        ]
        if not lb_datasets:
            print(f"  [WARN] No LongBench .jsonl files found in {args.lb_data_dir}. "
                  "Set --lb_data_dir to the folder containing narrativeqa.jsonl etc. "
                  "Skipping task4.")
        else:
            task4_results = all_results.get("task4_longbench", {})
            # Resolve model save-dir subdir name (matches run_longbench.py convention)
            model_name = args.model_id.split("/")[-1]
            lb_pred_subdir = os.path.join(
                lb_save_dir,
                f"{model_name}_1024_2_v4",   # rank=1024, layer_step=2 defaults
            )
            os.makedirs(lb_pred_subdir, exist_ok=True)

            # Helper: build the base run_longbench.py subprocess command
            # KIVI is invoked as --method kivi (monkeypatch); FullKV as --method FullKV
            _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            def _lb_cmd(method_flag, extra_args=None):
                cmd = [
                    sys.executable, os.path.join(_REPO_ROOT, "run_longbench.py"),
                    "--model_path",      args.model_id,
                    "--method",          method_flag,
                    "--dataset",         "all",
                    "--save_dir",        lb_save_dir,
                    "--attn_implementation", "eager",
                    "--max_capacity_prompts", str(args.lb_max_capacity),
                    "--steps",           str(args.lb_steps),
                    "--max_datasets",    str(args.lb_max_datasets),
                    "--eval_batch_size", "1",
                ]
                if extra_args:
                    cmd += extra_args
                return cmd

            # --- FullKV via run_longbench.py ---
            if not _is_done(task4_results.get("FullKV")):
                print("\n[LongBench] Running FullKV via run_longbench.py ...")
                ret = subprocess.run(_lb_cmd("FullKV"), cwd=_REPO_ROOT)
                if ret.returncode != 0:
                    task4_results["FullKV"] = {"status": "failed", "reason": "prediction_run_failed"}
                    print("  [WARN] run_longbench.py exited non-zero for FullKV.")
                else:
                    task4_results["FullKV"] = {"status": "predictions_done"}
                all_results["task4_longbench"] = task4_results
                save_results_snapshot()
            else:
                print("  Skipping FullKV (already in results).")

            # --- KIVI-int4 via run_longbench.py (method=kivi, monkeypatch) ---
            if not _is_done(task4_results.get("KIVI-int4")):
                print("\n[LongBench] Running KIVI-int4 via run_longbench.py ...")
                ret = subprocess.run(_lb_cmd("kivi"), cwd=_REPO_ROOT)
                if ret.returncode != 0:
                    task4_results["KIVI-int4"] = {"status": "failed", "reason": "prediction_run_failed"}
                    print("  [WARN] run_longbench.py exited non-zero for KIVI.")
                else:
                    task4_results["KIVI-int4"] = {"status": "predictions_done"}
                all_results["task4_longbench"] = task4_results
                save_results_snapshot()
            else:
                print("  Skipping KIVI-int4 (already in results).")

            # --- IAVQ-KC via its own generate() loop ---
            if not _is_done(task4_results.get("IAVQ-KC")) and "IAVQ-KC" in methods:
                print("\n[LongBench] Running IAVQ-KC (custom generate loop) ...")
                from ldcb.tasks.longbench import run_longbench_iavqkc
                try:
                    run_longbench_iavqkc(
                        method=methods["IAVQ-KC"],
                        model=model,
                        tokenizer=tokenizer,
                        datasets=lb_datasets,
                        data_dir=args.lb_data_dir,
                        save_dir=lb_pred_subdir,
                        steps=args.lb_steps,
                    )
                    task4_results["IAVQ-KC"] = {"status": "predictions_done"}
                except Exception as e:
                    task4_results["IAVQ-KC"] = {"status": "failed", "reason": str(e)}
                    print(f"  [WARN] IAVQ-KC LongBench run failed: {e}")
                all_results["task4_longbench"] = task4_results
                save_results_snapshot()
            elif not _is_done(task4_results.get("IAVQ-KC")):
                task4_results["IAVQ-KC"] = {"status": "skipped",
                                             "reason": "not in methods list"}

            # --- Score all methods with eval.py ---
            print("\n[LongBench] Scoring with eval.py ...")
            methods_to_score = ",".join([
                n.replace("KIVI-int4", "kivi")  # match run_longbench output filename
                for n, v in task4_results.items()
                if isinstance(v, dict) and v.get("status") == "predictions_done"
            ])
            if methods_to_score:
                eval_cmd = [
                    sys.executable, os.path.join(_REPO_ROOT, "eval.py"),
                    "--results_dir", lb_pred_subdir,
                    "--methods",     methods_to_score,
                ]
                eval_proc = subprocess.run(
                    eval_cmd, cwd=_REPO_ROOT,
                    capture_output=True, text=True,
                )
                if eval_proc.returncode != 0:
                    print(f"  [WARN] eval.py failed:\n{eval_proc.stderr[-800:]}")
                else:
                    # Parse results.csv for a summary dict keyed by method
                    results_csv = os.path.join(lb_pred_subdir, "results.csv")
                    if os.path.exists(results_csv):
                        import csv
                        with open(results_csv) as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        # Pivot: task4_results[method]["lb_scores"][dataset] = score
                        for row in rows:
                            dataset = row.get("dataset", "")
                            for method_col, score_str in row.items():
                                if method_col == "dataset":
                                    continue
                                # Remap kivi -> KIVI-int4
                                m = "KIVI-int4" if method_col.lower() == "kivi" else method_col
                                if m not in task4_results:
                                    task4_results[m] = {}
                                task4_results[m].setdefault("lb_scores", {})[dataset] = (
                                    float(score_str) if score_str not in ("", "-1") else None
                                )
                        # Compute mean score per method
                        for m in list(task4_results.keys()):
                            scores = task4_results[m].get("lb_scores", {})
                            valid = [v for v in scores.values() if v is not None and v >= 0]
                            if valid:
                                task4_results[m]["lb_mean_score"] = round(sum(valid) / len(valid), 2)
                        print("\n[LongBench] Mean scores:")
                        for m, v in task4_results.items():
                            if "lb_mean_score" in v:
                                print(f"  {m}: {v['lb_mean_score']:.2f}")

            all_results["task4_longbench"] = task4_results
            save_results_snapshot()

    results_path = save_results_snapshot()
    print(f"\nResults saved to {results_path}")
    render_available_plots()
    print("Plots saved to", args.output_dir)

    # ----- Summary Table -----
    if "task1_continuation" in all_results:
        print("\n" + "=" * 60)
        print("SUMMARY — Task 1 (Continuation)")
        print("=" * 60)
        first = next(iter(all_results["task1_continuation"].values()))
        sanity_ppl = first.get("base_ppl", {}).get("mean", float("nan"))
        print(f"Base model PPL on WikiText-2 (sanity check, same for all): {sanity_ppl:.2f}")
        calib_note = f" [calibrated: {calibration_path}]" if calibration_path else " [random codebooks]"
        print(f"APKVC codebooks:{calib_note}")
        print()
        print(f"{'Method':<20} {'Compression':>12} {'OutputKL vs FullKV':>20} {'DeltaPPL':>12}")
        print("-" * 70)
        for name, r in all_results["task1_continuation"].items():
            cr = r.get("final_compression_ratio", {}).get("mean", 0)
            ok = r.get("output_kl", {}).get("mean", float("nan"))
            dp = r.get("delta_ppl", {}).get("mean", float("nan"))
            ok_str = f"{ok:.6f}" if ok == ok else "N/A (FullKV baseline)"
            dp_str = f"{dp:.3f}" if dp == dp else "N/A"
            print(f"{name:<20} {cr:>12.3f} {ok_str:>20} {dp_str:>12}")


if __name__ == "__main__":
    main()
