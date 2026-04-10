# IAVQ-KC: Compression Evaluation Metric Stack

## Overview

IAVQ-KC compression perturbs the Key cache, introducing error at the attention score level that propagates through value aggregation into the residual stream and eventually into the output token distribution. Evaluating compression quality therefore requires metrics at **each level of this causal chain** — not just at the output surface.

The causal chain is:

```
K → QK^T (attention logits) → softmax(QK^T/√d) (attention weights)
  → weighted sum of V (context vector) → residual stream → output logits → token distribution
```

IAVQ-KC injects error at the first arrow. Every metric below measures how far that error propagates before being absorbed.

---

## Metric Stack

### Level 1 — Attention Logit Reconstruction Error

**What it measures:** How wrong the pre-softmax attention scores are after compression. This is the most direct measure of what IAVQ-KC actually perturbs — error at the source before any nonlinearity.

**Why it matters:** Two different $\hat{K}$ matrices can produce the same $Q\hat{K}^T$ if their projections onto the query subspace are similar. This metric rewards functional equivalence in attention-logit space, not direct Key vector replication. A compressed cache that changes $K$ values but preserves $QK^T$ is a perfect compression for this purpose.

**Formula:**
$$\text{RelErr}_K(l, h) = \frac{\|QK^T - Q\hat{K}^T\|_F}{\|QK^T\|_F}$$

Computed per layer $l$, per head $h$, averaged across tokens and batch.

**Implementation:**
```python
def attention_logit_rel_error(Q, K_full, K_compressed):
    """
    Q:             [batch, heads, T_q, head_dim]
    K_full:        [batch, heads, T_k, head_dim]
    K_compressed:  [batch, heads, T_k, head_dim]  — reconstructed from IAVQ-KC
    Returns:       scalar relative error
    """
    scores_full = torch.matmul(Q, K_full.transpose(-2, -1))        # [B, H, T_q, T_k]
    scores_comp = torch.matmul(Q, K_compressed.transpose(-2, -1))  # [B, H, T_q, T_k]
    
    num = torch.norm(scores_full - scores_comp, p='fro')
    den = torch.norm(scores_full, p='fro')
    return (num / den).item()
```

**Target threshold:** < 5% for acceptable quality. > 10% indicates the codebook or residual projections are inadequate for this layer/head.

**Diagnostic use:** Plot per-layer to identify problem layers. Late layers typically show higher error due to more structured, outlier-heavy key distributions. If early layers show high error, the codebook size $A$ is likely too small.

---

### Level 2 — Attention Distribution KL Divergence

**What it measures:** How much the attention weight distribution shifts after softmax. Captures the effect of the nonlinearity — equal pre-softmax error can mean very different post-softmax error depending on the sharpness of the distribution.

**Why it matters:** Level 1 measures error in logit space, which is linear. But what the model actually uses is the softmax output — a probability distribution over past tokens. A small logit perturbation on a peaked distribution (confident attention) causes little distributional shift; the same perturbation on a flat distribution causes a large shift. KL divergence captures this sensitivity correctly.

**Formula:**
$$\text{AttKL}(l, h) = \frac{1}{T_q} \sum_{t=1}^{T_q} \text{KL}\left(A_t \,\|\, \hat{A}_t\right)$$

where $A_t = \text{softmax}(QK^T / \sqrt{d})_{t,:}$ is the attention distribution for query position $t$ under full precision, and $\hat{A}_t$ is the same under compressed cache.

**Implementation:**
```python
def attention_kl_divergence(Q, K_full, K_compressed, scale):
    """
    scale: 1 / sqrt(head_dim)
    Returns: mean per-query KL divergence, scalar
    """
    scores_full = torch.matmul(Q, K_full.transpose(-2, -1)) * scale
    scores_comp = torch.matmul(Q, K_compressed.transpose(-2, -1)) * scale
    
    A_full = torch.softmax(scores_full, dim=-1)   # [B, H, T_q, T_k]
    A_comp = torch.softmax(scores_comp, dim=-1)   # [B, H, T_q, T_k]
    
    # KL(full || compressed) per query position, averaged
    # Add epsilon for numerical stability
    eps = 1e-9
    kl = (A_full * (torch.log(A_full + eps) - torch.log(A_comp + eps))).sum(dim=-1)
    return kl.mean().item()
```

**Target threshold:** Mean KL < 0.01 nats for strong compression quality. KL > 0.1 nats indicates the attention pattern has shifted substantially and downstream hidden states will likely be affected.

**Diagnostic use:** High KL on certain layers/heads despite low Level 1 RelErr indicates that those attention distributions are very peaked — small logit errors are landing on confident decisions. This is actually good news: the softmax is acting as a noise gate. Conversely, high KL with low Level 1 RelErr on flat distributions suggests the model is uncertain and the compression may push it toward wrong answers.

---

### Level 3 — Hidden State Cosine Similarity

**What it measures:** Whether the perturbation from Levels 1–2 propagates into the residual stream in a meaningful way. Even if attention distributions shift (Level 2), the value aggregation and residual connection may absorb the error. This metric checks whether the hidden state the next layer receives is functionally equivalent.

**Why it matters:** Transformers use residual connections: $h_t^{(l+1)} = h_t^{(l)} + \text{Attn}^{(l)}(h^{(l)}) + \text{FFN}^{(l)}(\cdot)$. Even if the attention output changes, adding it to a large residual stream may leave the sum nearly unchanged. High cosine similarity here means the compression is functionally transparent to subsequent layers even if intermediate values differ.

**Formula:**
$$\text{HiddenSim}(l) = \frac{1}{B \cdot T} \sum_{b,t} \frac{h_{b,t}^{(l)} \cdot \hat{h}_{b,t}^{(l)}}{\|h_{b,t}^{(l)}\| \cdot \|\hat{h}_{b,t}^{(l)}\|}$$

Computed at the output of each transformer block (after attention + FFN + residual add).

**Implementation:**
```python
def hidden_state_cosine_similarity(h_full, h_compressed):
    """
    h_full:       [batch, seq_len, hidden_dim] — residual stream, full precision
    h_compressed: [batch, seq_len, hidden_dim] — residual stream, compressed cache
    Returns:      mean cosine similarity across batch and sequence, scalar
    """
    cos_sim = torch.nn.functional.cosine_similarity(
        h_full, h_compressed, dim=-1
    )  # [batch, seq_len]
    return cos_sim.mean().item()
```

**Collection:** Requires hooking into the model's residual stream at each layer boundary. In HuggingFace models this is done via `output_hidden_states=True` in both full-precision and compressed forward passes.

**Target threshold:** Mean cosine similarity > 0.99 across all layers for strong quality. Degradation below 0.95 at any layer indicates that layer's attention error is not being absorbed by the residual connection and will compound in subsequent layers.

**Diagnostic use:** Plot cosine similarity across layers for the full vs compressed run. If similarity is high in early layers but drops sharply at a specific layer, that layer's codebook needs more centroids or the residual projections are insufficient. This is a precise localization tool for debugging reconstruction quality.

---

### Level 4 — Output Distribution KL Divergence

**What it measures:** How much the compressed cache shifts the model's full output token distribution — not just the argmax, but the entire probability vector over the vocabulary.

**Why it matters:** This is the right end-to-end measure of whether compression changes what the model believes. A compressed cache that consistently shifts probability mass from the correct token to plausible alternatives is degrading quality even if the argmax doesn't change. Argmax-based metrics (ROUGE, accuracy) are blind to this. KL divergence at the output level is sensitive to the full distribution.

**Formula:**
$$\text{OutputKL} = \frac{1}{T} \sum_{t=1}^{T} \text{KL}\left(p_\theta(\cdot \mid x, K)_t \;\|\; p_\theta(\cdot \mid x, \hat{K})_t\right)$$

where $p_\theta(\cdot \mid x, K)_t$ is the output softmax distribution at position $t$ under full precision cache, and $p_\theta(\cdot \mid x, \hat{K})_t$ is the same under compressed cache.

**Implementation:**
```python
def output_distribution_kl(logits_full, logits_compressed, temperature=1.0):
    """
    logits_full:       [batch, seq_len, vocab_size]
    logits_compressed: [batch, seq_len, vocab_size]
    Returns:           mean per-position KL divergence, scalar
    """
    p_full = torch.softmax(logits_full / temperature, dim=-1)
    p_comp = torch.softmax(logits_compressed / temperature, dim=-1)
    
    eps = 1e-9
    kl = (p_full * (torch.log(p_full + eps) - torch.log(p_comp + eps))).sum(dim=-1)
    return kl.mean().item()  # [batch, seq_len] -> scalar
```

**Target threshold:** Mean OutputKL < 0.001 nats for strong quality (output distributions nearly identical). > 0.01 nats indicates meaningful distribution shift that will affect generation quality on longer sequences.

**Diagnostic use:** If OutputKL is low despite non-trivial hidden state divergence (Level 3), the final LayerNorm and LM head are absorbing the perturbation — compression is safe. If OutputKL is high despite low hidden state divergence, the LM head is amplifying small representation differences — consider keeping the last 1–2 layers' KV cache at higher precision.

---

### Level 5 — Perplexity Delta

**What it measures:** The increase in perplexity on a held-out corpus when using the compressed cache versus full precision. This integrates the output distribution KL (Level 4) across a full dataset and is the standard benchmark used by all KV cache compression papers (KIVI, H2O, SnapKV, KIVI-2).

**Why it matters:** Perplexity is the geometric mean of the inverse token probabilities — it penalizes the model for assigning low probability to correct tokens. A compressed cache that shifts the distribution away from correct tokens will show up as higher perplexity even if ROUGE is unaffected. It is also directly comparable to numbers reported in the existing literature.

**Formula:**
$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}, \hat{K})\right)$$

$$\Delta\text{PPL} = \text{PPL}_{\text{compressed}} - \text{PPL}_{\text{full fp16}}$$

**Implementation:**
```python
def perplexity(model, tokenized_corpus, compressed_cache, max_length=2048):
    """
    Compute perplexity of model on tokenized_corpus under compressed KV cache.
    Returns: scalar perplexity value
    """
    total_log_prob = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tokenized_corpus:
            input_ids = batch['input_ids']  # [1, T]
            
            # Run with compressed cache
            logits = model_forward_compressed(
                input_ids, compressed_cache
            )  # [1, T, vocab_size]
            
            # Shift: predict token t from tokens 0..t-1
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            total_log_prob += token_log_probs.sum().item()
            total_tokens += shift_labels.numel()
    
    return torch.exp(torch.tensor(-total_log_prob / total_tokens)).item()
```

**Evaluation corpora:**
- **WikiText-103**: standard long-document benchmark, good coverage of general language patterns
- **LongBench**: specifically designed for long-context evaluation — critical here since KV cache compression matters most at long sequence lengths
- **PG-19**: long-form book text, tests performance at T > 8192

**Target threshold:** $\Delta\text{PPL}$ < 0.5 for strong compression quality. > 2.0 indicates unacceptable degradation. All results should be reported at multiple sequence lengths (T = 512, 1024, 2048, 4096, 8192) since compression quality typically degrades with sequence length as the anchor fraction falls.

---

## Evaluation Protocol

### Standard Evaluation Run

```python
def evaluate_iavq_compression(model, corpus, R, M, A, sequence_lengths):
    """Full evaluation across all metric levels."""
    
    results = defaultdict(dict)
    
    for T in sequence_lengths:
        # Run full precision forward pass, collect all intermediate states
        full_outputs = run_full_precision(model, corpus, T,
                                          output_attentions=True,
                                          output_hidden_states=True)
        
        # Build compressed cache and run compressed forward pass
        compressed_outputs = run_compressed(model, corpus, T, R, M, A,
                                             output_attentions=True,
                                             output_hidden_states=True)
        
        # Level 1: Attention logit relative error, per layer
        results[T]['rel_err_per_layer'] = [
            attention_logit_rel_error(
                full_outputs.queries[l],
                full_outputs.keys[l],
                compressed_outputs.keys[l]
            )
            for l in range(model.config.num_hidden_layers)
        ]
        
        # Level 2: Attention KL divergence, per layer
        results[T]['attn_kl_per_layer'] = [
            attention_kl_divergence(
                full_outputs.queries[l],
                full_outputs.keys[l],
                compressed_outputs.keys[l],
                scale=1.0 / (model.config.head_dim ** 0.5)
            )
            for l in range(model.config.num_hidden_layers)
        ]
        
        # Level 3: Hidden state cosine similarity, per layer
        results[T]['hidden_sim_per_layer'] = [
            hidden_state_cosine_similarity(
                full_outputs.hidden_states[l],
                compressed_outputs.hidden_states[l]
            )
            for l in range(model.config.num_hidden_layers)
        ]
        
        # Level 4: Output distribution KL
        results[T]['output_kl'] = output_distribution_kl(
            full_outputs.logits,
            compressed_outputs.logits
        )
        
        # Level 5: Perplexity delta
        ppl_full = perplexity(model, corpus, full_cache=True, T=T)
        ppl_comp = perplexity(model, corpus, compressed_cache=True, T=T)
        results[T]['ppl_full'] = ppl_full
        results[T]['ppl_compressed'] = ppl_comp
        results[T]['delta_ppl'] = ppl_comp - ppl_full
    
    return results
```

### Reporting Format

For each experimental configuration (A, R, M), report:

| Sequence Length | RelErr (mean across layers) | AttKL (mean) | HiddenSim (min across layers) | OutputKL | PPL Full | PPL Compressed | ΔPPL |
|---|---|---|---|---|---|---|---|
| 512 | | | | | | | |
| 1024 | | | | | | | |
| 2048 | | | | | | | |
| 4096 | | | | | | | |
| 8192 | | | | | | | |

Additionally report per-layer profiles of RelErr and HiddenSim as plots for diagnostic purposes.

---

## Metric Relationships and Failure Mode Taxonomy

Understanding how these metrics relate to each other identifies the root cause of any quality degradation:

| Pattern | Diagnosis | Fix |
|---|---|---|
| High Level 1, High Level 2, Low Level 3 | Attention error absorbed by residual connection — compression is safe despite logit error | No fix needed |
| High Level 1, High Level 2, High Level 3, Low Level 4 | Residual stream perturbed but LM head absorbs it — borderline, monitor | Consider higher-precision anchors in last 2 layers |
| High Level 1, Low Level 2 | Logit error lands in already-confident attention distributions — softmax acting as noise gate | Benign, compression is functionally safe |
| Low Level 1, High Level 2 | Attention distributions are very flat — small logit errors cause large distributional shifts | Increase $M$ (more importance anchors) for tokens with flat attention |
| Low Levels 1–4, High Level 5 | Perplexity sensitive to rare tokens receiving lower probability — hard to detect locally | Increase $R$ or evaluate at shorter sequence lengths |
| Levels 1–2 degrade sharply at specific layers | Those layers have outlier key distributions poorly covered by shared codebook | Increase $m$ (centroids per layer) for problem layers, or use per-head codebook for those layers |

---

## Why Not ROUGE

ROUGE measures n-gram overlap between generated text strings. It is sensitive only to argmax token changes — if the argmax is unchanged, ROUGE reports zero degradation regardless of how much the underlying distribution has shifted. Specifically:

- A compressed cache could shift the correct token's probability from 0.95 to 0.60 (significant degradation) while leaving the argmax unchanged — ROUGE sees nothing
- ROUGE cannot distinguish "K values changed but $QK^T$ was preserved" from "K values changed and $QK^T$ was substantially perturbed" — it is blind to the entire causal chain above Level 5
- ROUGE is not used as a primary metric in any KV cache compression paper in the literature (KIVI, H2O, SnapKV, MiniCache all use perplexity + downstream task accuracy)

ROUGE may be included as a supplementary metric on long-form generation tasks (summarization, open-ended QA) as a downstream sanity check, but it should never be the primary evaluation signal. A result where Levels 1–4 are healthy will always produce acceptable ROUGE; a result where ROUGE is bad indicates a catastrophic failure that the upstream metrics will have already flagged more precisely.

---

*IAVQ-KC Metrics Specification v1.0*
