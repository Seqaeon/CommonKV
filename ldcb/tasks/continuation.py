import torch
from ..utils import get_total_vram_gb
from ..metrics import compute_perplexity, aggregate_task_results

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

def run_continuation(method, model, tokenizer, max_new_tokens=None) -> dict:
    all_results = []
    
    # Calculate limits based on model capacity if provided
    limit = max_new_tokens or MAX_NEW_TOKENS
    # Filter and cap checkpoint steps
    active_steps = [s for s in CHECKPOINT_STEPS if s < limit]
    active_steps.append(limit)

    for prompt in CONTINUATION_PROMPTS:
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
            "perplexity": compute_perplexity(model, tokenizer, generated_text),
            "distortion_mean": float(torch.tensor(final_state.distortions).mean())
                               if final_state.distortions else 0.0,
            "distortion_p95": float(torch.tensor(final_state.distortions).quantile(0.95))
                              if final_state.distortions else 0.0,
        }
        all_results.append(result)

    return aggregate_task_results(all_results)
