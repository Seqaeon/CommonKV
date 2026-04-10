import torch
from ..utils import get_total_vram_gb
from ..metrics import (
    compute_perplexity_on_reference,
    compute_output_kl_on_text_pair,
    aggregate_task_results,
)

CHECKPOINT_STEPS = [250, 500, 1000, 2000, 4000]
MAX_NEW_TOKENS = 4000

CONTINUATION_PROMPTS = [
    "Write a detailed technical report on the history of transformer "
    "architectures, covering all major developments from 2017 to present.",

    "Write a comprehensive essay on the political economy of colonial West "
    "Africa, examining land tenure, taxation, and labour extraction in depth.",

    "Write a detailed explanation of how operating system kernels manage "
    "memory, covering virtual address spaces, paging, and page replacement.",

    "Write a thorough account of how deep learning changed the field of "
    "natural language processing between 2013 and 2023.",

    "Write a detailed analysis of how monetary policy transmission works "
    "in an economy with a large informal sector.",
]


def run_continuation(method, model, tokenizer, max_new_tokens=None,
                     reference_texts=None) -> tuple:
    """
    Run the continuation task for a single method.

    Parameters
    ----------
    reference_texts : list[str] or None
        One generated string per prompt from FullKV, used as the baseline for
        OutputKL and DeltaPPL metrics. Pass None for the FullKV run itself.

    Returns
    -------
    (aggregated_results : dict, generated_texts : list[str])
        generated_texts contains the raw decoded output for each prompt,
        in the same order as CONTINUATION_PROMPTS.  The caller should
        capture this from the FullKV run and pass it as reference_texts
        for all subsequent methods.
    """
    all_results = []
    generated_texts = []

    # Calculate limits based on model capacity if provided
    limit = max_new_tokens or MAX_NEW_TOKENS
    # Filter and cap checkpoint steps
    active_steps = [s for s in CHECKPOINT_STEPS if s < limit]
    active_steps.append(limit)

    for prompt_idx, prompt in enumerate(CONTINUATION_PROMPTS):
        # Tokenize and verify prompt is short
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        assert input_ids.shape[1] <= 128, \
            f"Prompt too long: {input_ids.shape[1]} tokens. Trim it."

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        generated_text, snapshots, final_state = method.generate(
            model, tokenizer, prompt,
            max_new_tokens=limit,
            checkpoint_steps=active_steps,
        )
        generated_texts.append(generated_text)

        result = {
            "prompt": prompt[:60] + "...",
            "snapshots": [
                {
                    "tokens_generated": step,
                    "compression_ratio": snap.compressed_bytes / snap.fullkv_bytes,
                    "anchor_rate": snap.anchor_count / max(snap.anchor_count + snap.residual_count, 1),
                }
                for step, snap in zip(active_steps, snapshots)
            ],
            "final_compression_ratio": final_state.compressed_bytes / final_state.fullkv_bytes,
            "peak_vram_gb": get_total_vram_gb(),
            # WikiText-2 reference perplexity — identical for all methods (same
            # base model, no cache state involved).  Use as a sanity check.
            "base_ppl": compute_perplexity_on_reference(model, tokenizer, n_tokens=2048),
            "distortion_mean": float(torch.tensor(final_state.distortions).mean())
                               if final_state.distortions else 0.0,
            "distortion_p95": float(torch.tensor(final_state.distortions).quantile(0.95))
                              if final_state.distortions else 0.0,
        }

        # Replace ROUGE with IAVQ-KC stack end metrics:
        #  - Level 4: Output distribution KL vs FullKV continuation
        #  - Level 5 proxy: Delta perplexity (first-order) from KL:
        #      delta_ppl ≈ ppl_ref * KL(p||q)
        # This avoids exponential blow-ups when KL is large/off-distribution.
        if reference_texts is not None and prompt_idx < len(reference_texts):
            reference_text = reference_texts[prompt_idx]
            result["output_kl"] = compute_output_kl_on_text_pair(
                model, tokenizer, reference_text, generated_text
            )
            ref_ppl = result["base_ppl"]
            output_kl = result["output_kl"]
            if output_kl == output_kl:  # not NaN
                result["delta_ppl"] = ref_ppl * float(output_kl)
            else:
                result["delta_ppl"] = float("nan")

        all_results.append(result)

    return aggregate_task_results(all_results), generated_texts
