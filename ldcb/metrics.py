import torch
import numpy as np
from rouge_score import rouge_scorer

def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    Compute perplexity of the generated text under the model.
    Lower = model considers the text more probable = better quality.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    max_length = 2048
    stride = 512
    nlls = []

    for i in range(0, input_ids.shape[1], stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.shape[1])
        trg_len = end_loc - i
        chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L between compressed-method output and FullKV output.
    Reference = FullKV generation for the same prompt.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_compression_ratio(compressed_bytes: int, fullkv_bytes: int) -> float:
    return compressed_bytes / fullkv_bytes


def estimate_cache_bytes(n_tokens: int, n_layers: int, n_heads: int,
                          head_dim: int, dtype_bytes: int = 2) -> int:
    """
    Estimate FullKV cache size in bytes.
    dtype_bytes = 2 for fp16.
    """
    # K and V, per layer
    return 2 * n_tokens * n_layers * n_heads * head_dim * dtype_bytes


def aggregate_task_results(results: list) -> dict:
    """Average metrics across prompts within a task."""
    keys = [k for k in results[0] if k != "prompt" and k != "snapshots"]
    aggregated = {}
    for k in keys:
        vals = [r[k] for r in results if isinstance(r[k], (int, float))]
        if vals:
            aggregated[k] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }
    # Aggregate snapshots separately
    if "snapshots" in results[0]:
        n_steps = len(results[0]["snapshots"])
        aggregated["snapshots"] = []
        for step_idx in range(n_steps):
            step_data = [r["snapshots"][step_idx] for r in results]
            aggregated["snapshots"].append({
                "tokens_generated": step_data[0]["tokens_generated"],
                "compression_ratio_mean": float(np.mean([s["compression_ratio"] for s in step_data])),
                "compression_ratio_std":  float(np.std( [s["compression_ratio"] for s in step_data])),
                "anchor_rate_mean":       float(np.mean([s["anchor_rate"] for s in step_data])),
            })
    return aggregated
