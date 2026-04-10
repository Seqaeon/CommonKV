import torch
from ..utils import get_total_vram_gb
from ..metrics import (
    compute_perplexity_on_reference,
    compute_text_ppl_delta,
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
        delta-PPL. Pass None for the FullKV run itself.

    Returns
    -------
    (aggregated_results : dict, generated_texts : list[str])
        generated_texts contains the raw decoded output for each prompt,
        in the same order as CONTINUATION_PROMPTS.  The caller should
        capture this from the FullKV run and pass it as reference_texts
        for all subsequent methods.

    Metrics (per-prompt, then aggregated across prompts)
    -------
    base_ppl     : WikiText-2 test-set PPL — sanity check, should be ~identical
                   for all methods (no generation cache involved).
    gen_ppl      : Model's teacher-forcing PPL on *this method's* generated text.
                   Measures how natural/predictable the output is.
    gen_ppl_ref  : Same for the FullKV reference text (only when reference_texts
                   is provided, i.e. not the FullKV run).
    delta_ppl    : gen_ppl - gen_ppl_ref.
                   ~0 = compression is transparent.  >2 = noticeable degradation.
                   >5 = serious quality loss.  This is the correct Level-5 metric
                   (see IAVQ_KC_metrics.md).  The previously-used formula
                   base_ppl * (exp(output_kl) - 1) was wrong: it used KL computed
                   between distributions on two *different* texts, giving ~11 nats
                   of noise (≈ ln(vocab_size)), inflating delta_ppl into the millions.
    """
    all_results = []
    generated_texts = []

    limit = max_new_tokens or MAX_NEW_TOKENS
    active_steps = [s for s in CHECKPOINT_STEPS if s < limit]
    active_steps.append(limit)

    for prompt_idx, prompt in enumerate(CONTINUATION_PROMPTS):
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
            "base_ppl": compute_perplexity_on_reference(model, tokenizer, n_tokens=2048),
            "distortion_mean": float(torch.tensor(final_state.distortions).mean())
                               if final_state.distortions else 0.0,
            "distortion_p95": float(torch.tensor(final_state.distortions).quantile(0.95))
                              if final_state.distortions else 0.0,
        }

        if reference_texts is not None and prompt_idx < len(reference_texts):
            # Compressed method run: compare to FullKV reference text
            reference_text = reference_texts[prompt_idx]
            gen_ppl_comp, gen_ppl_ref, delta_ppl = compute_text_ppl_delta(
                model, tokenizer, generated_text, reference_text
            )
            result["gen_ppl"]     = gen_ppl_comp
            result["gen_ppl_ref"] = gen_ppl_ref
            result["delta_ppl"]   = delta_ppl
        else:
            # FullKV run: just record its own gen_ppl as baseline reference
            gen_ppl_self, _, _ = compute_text_ppl_delta(
                model, tokenizer, generated_text, generated_text
            )
            result["gen_ppl"] = gen_ppl_self

        all_results.append(result)

    return aggregate_task_results(all_results), generated_texts
