import torch
import numpy as np
from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Reference text: load once, cache in module-level variable
# ---------------------------------------------------------------------------
_REFERENCE_IDS_CACHE: dict = {}  # tokenizer_id -> tensor


def load_reference_ids(tokenizer, n_tokens: int = 2048):
    """
    Load a fixed slice of WikiText-2 test set and tokenize it.
    Result is cached so only one network/disk fetch per process.

    Returns: LongTensor of shape (1, n_tokens)
    """
    key = (id(tokenizer), n_tokens)
    if key in _REFERENCE_IDS_CACHE:
        return _REFERENCE_IDS_CACHE[key]

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install 'datasets' to use reference perplexity: pip install datasets"
        ) from e

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(t for t in dataset["text"] if t.strip())
    ids = tokenizer(full_text, return_tensors="pt").input_ids
    # Take up to n_tokens; if the corpus is shorter just use what we have
    ids = ids[:, :n_tokens]
    _REFERENCE_IDS_CACHE[key] = ids
    return ids


def compute_perplexity_on_reference(model, tokenizer, n_tokens: int = 2048) -> float:
    """
    Compute perplexity of `model` on a fixed WikiText-2 test slice.

    This is the **correct** perplexity metric for comparing KV-cache methods:
    the same reference text is used for every method so results are comparable.
    Lower is better; FullKV on LLaMA-2-7B should be ~5.5 on WikiText-2.

    NOTE: This performs a *fresh* forward pass on the reference text — it does
    NOT use any cached KV state from generation. It measures the model's
    language-modeling quality in isolation.
    """
    reference_ids = load_reference_ids(tokenizer, n_tokens=n_tokens)
    reference_ids = reference_ids.to(model.device)

    max_pos = getattr(model.config, "max_position_embeddings", 2048)
    max_length = min(1024, max_pos)
    stride = 512
    nlls = []
    seq_len = reference_ids.shape[1]

    prev_end = 0
    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - max(begin, prev_end)

            chunk = reference_ids[:, begin:end]
            labels = chunk.clone()
            # Mask context-only tokens so loss is computed on the new tokens
            labels[:, :-target_len] = -100

            outputs = model(chunk, labels=labels)
            nlls.append(outputs.loss.detach() * target_len)
            prev_end = end

            if end == seq_len:
                break

    return torch.exp(torch.stack(nlls).sum() / seq_len).item()


# ---------------------------------------------------------------------------
# Legacy self-evaluation perplexity — kept for backward compatibility but
# should NOT be used to compare methods (each method gets different text).
# ---------------------------------------------------------------------------
def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    DEPRECATED for cross-method comparison.

    Computes perplexity of the model on `text` — which is the *generated* text
    itself.  Because each method generates different text, results are not
    comparable across methods (KIVI can beat FullKV simply by generating more
    repetitive/predictable output).

    Use `compute_perplexity_on_reference` instead.
    """
    import gc
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    max_pos = getattr(model.config, "max_position_embeddings", 2048)
    max_length = min(1024, max_pos)
    stride = 512
    nlls = []

    if input_ids.shape[1] > max_pos:
        input_ids = input_ids[:, :max_pos]

    end_loc = 0
    for i in range(0, input_ids.shape[1], stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.shape[1])
        trg_len = end_loc - i
        chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        del outputs
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()

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
