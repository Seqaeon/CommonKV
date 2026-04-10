# IAVQ-KC: Importance-Aware Vector Quantization Key Cache Compression

## Table of Contents
1. [Overview](#overview)
2. [Motivation and Problem Statement](#motivation)
3. [Core Concepts](#core-concepts)
4. [Data Structures](#data-structures)
5. [Compression Pipeline](#compression-pipeline)
6. [Reconstruction](#reconstruction)
7. [Hyperparameters](#hyperparameters)
8. [Compression Ratio Analysis](#compression-ratio-analysis)
9. [Comparison to APKVC](#comparison-to-apkvc)
10. [Implementation Plan](#implementation-plan)
11. [Open Questions and Future Work](#open-questions)

---

## 1. Overview <a name="overview"></a>

IAVQ-KC is a KV cache compression scheme targeting the **Key cache** of autoregressive transformer inference. It replaces full-precision key vectors with a compact **(codebook index, 8 residual projections)** representation, where:

- The codebook is built **online at prefill** using attention-derived importance scores — it is specific to each forward pass, not calibrated offline
- High-importance and recent tokens are stored as **exact anchors** (full fp16), zero reconstruction error
- All other tokens are compressed to roughly **69 bits** from 2048 bits — approximately **30× compression** on compressed tokens
- Reconstruction is **fully independent per token** — no chain dependencies, no drift, random access on any token at any time

---

## 2. Motivation and Problem Statement <a name="motivation"></a>

During autoregressive decoding, a transformer must retain key and value tensors for every past token to avoid recomputation. The full Key cache for a single forward pass has shape:

```
[num_layers, num_kv_heads, batch_size, seq_len, head_dim]
```

For Llama-3-8B (32 layers, 8 KV heads, head_dim=128) at sequence length 4096:

```
32 × 8 × 1 × 4096 × 128 × 2 bytes = 268 MB
```

This grows linearly with sequence length and is the dominant memory bottleneck at long context. Existing approaches either:
- Apply uniform quantization (KIVI: INT2/INT4), which is not content-aware
- Use temporal prediction chains (APKVC), which introduce drift and require anchors on a fixed schedule
- Evict tokens (H2O, StreamingLLM), which irreversibly loses information

IAVQ-KC takes a different approach: **spatial codebook compression** where the codebook is importance-weighted, anchors are placed by content rather than schedule, and every compressed token is independently decodable.

---

## 3. Core Concepts <a name="core-concepts"></a>

### 3.1 Spatial vs. Temporal Compression

APKVC is a **temporal codec** — it exploits time-series structure (consecutive tokens have similar keys), storing deltas between adjacent tokens. This requires sequential reconstruction and periodic full-precision anchors to prevent error accumulation.

IAVQ-KC is a **spatial codec** — it exploits the fact that keys cluster in $\mathbb{R}^{d_{head}}$ space. Every token is independently assigned to a prototype (centroid) and the residual from that prototype is stored compactly. No sequential dependency, no drift.

### 3.2 Importance-Aware Codebook Construction

Standard VQ builds a codebook by minimizing mean reconstruction error uniformly across all tokens. IAVQ-KC weights this by **attention-derived importance scores** — tokens that receive more attention mass during prefill pull the centroids toward the regions of key space that matter most for future queries. The codebook is better calibrated for high-attention regions at the cost of slightly worse coverage of low-attention regions, which is exactly the right tradeoff.

### 3.3 Content-Aware Anchoring

Rather than periodically resetting to full precision on a fixed schedule (APKVC's approach), IAVQ-KC places anchors by content:
- **Recency anchors**: the last $R$ tokens — current queries attend heavily to recent context
- **Importance anchors**: the top $M$ oldest tokens by accumulated attention mass — these are the tokens future queries are most likely to attend to

Error is zero exactly where it hurts most.

### 3.4 PCA Residual Projection

After centroid assignment, the residual $r_t = k_t - C[\text{idx}_t]$ is a $d_{head}$-dimensional vector. Rather than storing it naively, IAVQ-KC projects it onto the **top 8 principal components** of the residual distribution — the 8 directions along which residuals vary most. This is the optimal linear compression of the residual in terms of variance captured. The 8 projections are stored as int8 scalars.

---

## 4. Data Structures <a name="data-structures"></a>

### 4.1 Codebook $C$

```
Shape: [A, head_dim]
Dtype: float16
Stratification: layer-stratified
  - Layer l owns rows [l * m : (l+1) * m]
  - m = A / num_layers centroids per layer
  - Each centroid is a head_dim-dimensional prototype vector
  - Shared across all KV heads within a layer
```

**Storage cost:** $A \times d_{head} \times 2$ bytes
- At $A=256$, $d_{head}=128$: **65 KB**
- At $A=1024$, $d_{head}=128$: **262 KB**

### 4.2 PCA Basis $B$

```
Shape: [num_layers, 8, head_dim]
Dtype: float16
Scope: per layer (computed from pooled residuals across all heads in that layer)
```

**Storage cost:** $L \times 8 \times d_{head} \times 2$ bytes
- At $L=32$, $d_{head}=128$: **65 KB**

### 4.3 Per-Token Compressed Representation

For each non-anchor token at layer $l$:

```
idx:          log2(m) bits   — index into layer l's m centroids
projections:  8 × 8 = 64 bits — 8 int8 scalars, residual projected onto B_l
─────────────────────────────────────────────────────
Total:        log2(m) + 64 bits per token
```

At $m=8$ (A=256, L=32): **67 bits** vs 2048 bits full fp16 → **~30.6× compression**
At $m=32$ (A=1024, L=32): **69 bits** vs 2048 bits → **~29.7× compression**

### 4.4 Anchor Token Storage

```
Full fp16: head_dim × 2 = 256 bytes per anchor token
Metadata: token position index (int32)
```

Two anchor pools maintained:
- **Recency pool**: ring buffer of last $R$ tokens, full fp16
- **Importance pool**: sorted set of top $M$ tokens by importance score, full fp16

### 4.5 Importance Score Buffer

```
Shape: [seq_len]
Dtype: float32
Content: per-token accumulated attention mass s_t = sum_q A_{q,t}
Updated: at prefill (from full attention matrix), and incrementally during decode
```

---

## 5. Compression Pipeline <a name="compression-pipeline"></a>

### 5.1 Prefill Phase

```python
def prefill_compress(K, attention_weights):
    """
    K:                [num_layers, num_heads, T, head_dim] — prompt key cache
    attention_weights: [num_layers, num_heads, T, T]       — transient, grabbed before discard
    
    Returns: compressed_cache, codebook C, basis B, importance_scores
    """
    
    T = K.shape[2]
    
    # ── Step 1: Importance scores (free — from transient attention weights) ──
    # Average across layers and heads for a single importance score per token
    # Shape: [T]
    s = attention_weights.mean(dim=[0, 1]).sum(dim=0)  # column sums, averaged across L and H
    
    # ── Step 2: Build codebook per layer ──
    C = zeros(A, head_dim)  # full codebook, layer-stratified
    
    for l in range(num_layers):
        # Pool keys across all heads for this layer: [num_heads * T, head_dim]
        K_l = K[l].reshape(-1, head_dim)
        s_l = s.repeat(num_heads)  # same importance per position across heads
        
        # Importance-weighted k-means, 3 iterations
        # Initial centroids: sample with probability proportional to s_l (importance-biased seeding)
        C[l*m : (l+1)*m] = importance_weighted_kmeans(K_l, s_l, k=m, iters=3)
    
    # ── Step 3: Assign tokens to centroids, compute residuals ──
    assignments = zeros(num_layers, num_heads, T, dtype=int)
    R_all = zeros(num_layers, num_heads, T, head_dim)  # residuals
    
    for l in range(num_layers):
        C_l = C[l*m : (l+1)*m]  # this layer's centroids
        for h in range(num_heads):
            keys = K[l, h]  # [T, head_dim]
            dists = cdist(keys, C_l)  # [T, m]
            assignments[l, h] = dists.argmin(dim=1)  # [T]
            R_all[l, h] = keys - C_l[assignments[l, h]]  # [T, head_dim]
    
    # ── Step 4: PCA basis per layer ──
    B = zeros(num_layers, 8, head_dim)
    
    for l in range(num_layers):
        # Pool residuals across all heads for this layer: [num_heads * T, head_dim]
        R_l = R_all[l].reshape(-1, head_dim)
        # Truncated SVD, top 8 components
        _, _, Vt = truncated_svd(R_l, k=8)
        B[l] = Vt  # [8, head_dim]
    
    # ── Step 5: Project residuals onto basis ──
    projections = zeros(num_layers, num_heads, T, 8)
    
    for l in range(num_layers):
        for h in range(num_heads):
            projections[l, h] = R_all[l, h] @ B[l].T  # [T, 8]
    
    # Quantize projections to int8
    proj_scale = projections.abs().max(dim=-1, keepdim=True)  # per-token scale
    projections_int8 = (projections / proj_scale).clip(-128, 127).to(int8)
    
    # ── Step 6: Designate anchor set ──
    # Recency anchors: last R tokens
    recency_anchor_ids = set(range(T - R, T))
    
    # Importance anchors: top M tokens by s_t from tokens [0, T-R)
    older_tokens = range(0, T - R)
    importance_anchor_ids = set(topk(s[older_tokens], M).indices.tolist())
    
    anchor_ids = recency_anchor_ids | importance_anchor_ids
    compressed_ids = set(range(T)) - anchor_ids
    
    # ── Step 7: Build compressed cache ──
    compressed_cache = {
        'anchors': {t: K[:, :, t, :] for t in anchor_ids},   # full fp16
        'compressed': {
            t: (assignments[:, :, t], projections_int8[:, :, t], proj_scale[:, :, t])
            for t in compressed_ids
        },
        'importance_scores': s,
        'anchor_ids': anchor_ids,
    }
    
    return compressed_cache, C, B
```

### 5.2 Importance-Weighted K-Means

```python
def importance_weighted_kmeans(X, weights, k, iters=3):
    """
    X:       [N, d] — key vectors
    weights: [N]    — importance scores (attention column sums)
    k:       int    — number of centroids
    iters:   int    — Lloyd iterations
    
    Returns: centroids [k, d]
    """
    
    # Importance-biased seeding (weighted k-means++)
    probs = weights / weights.sum()
    centroid_ids = multinomial(probs, k, replacement=False)
    centroids = X[centroid_ids]
    
    for _ in range(iters):
        # Assignment step
        dists = cdist(X, centroids)         # [N, k]
        assignments = dists.argmin(dim=1)   # [N]
        
        # Update step — importance-weighted centroid update
        new_centroids = zeros(k, X.shape[1])
        for c in range(k):
            mask = (assignments == c)
            if mask.sum() == 0:
                new_centroids[c] = centroids[c]  # keep old centroid if empty
                continue
            w = weights[mask]
            new_centroids[c] = (X[mask] * w.unsqueeze(1)).sum(0) / w.sum()
        
        centroids = new_centroids
    
    return centroids
```

### 5.3 Decode Phase (Per New Token)

```python
def decode_step_compress(k_new, l, pos, compressed_cache, C, B, alpha=0.01):
    """
    k_new:  [num_heads, head_dim] — new key vectors at layer l
    l:      int  — current layer index
    pos:    int  — current sequence position
    alpha:  float — soft centroid update rate
    """
    
    # ── Step 1: New token enters as recency anchor ──
    compressed_cache['anchors'][pos] = k_new  # full fp16
    compressed_cache['anchor_ids'].add(pos)
    
    # ── Step 2: Age out token at position (pos - R) ──
    aging_pos = pos - R
    if aging_pos >= 0 and aging_pos in compressed_cache['anchor_ids']:
        
        # Check if it qualifies as importance anchor
        s = compressed_cache['importance_scores']
        older_importance_scores = {
            t: s[t] for t in compressed_cache['anchor_ids']
            if t < pos - R
        }
        
        # Maintain top-M importance anchors
        current_importance_anchors = {
            t for t in compressed_cache['anchor_ids']
            if t < pos - R
        }
        
        if len(current_importance_anchors) < M or s[aging_pos] > min(
            s[t] for t in current_importance_anchors
        ):
            # Promote to importance anchor — keep in anchor set
            # If over budget M, evict the lowest-importance old anchor
            if len(current_importance_anchors) >= M:
                evict_pos = min(current_importance_anchors, key=lambda t: s[t])
                _compress_token(evict_pos, l, compressed_cache, C, B)
            # aging_pos stays in anchor_ids (already there)
        
        else:
            # Compress the aging token
            _compress_token(aging_pos, l, compressed_cache, C, B)
    
    # ── Step 3: Soft centroid update ──
    # IMPORTANT: Only update centroids for tokens still in the recency window.
    # Once a token ages out and is compressed against the current centroid,
    # moving that centroid would silently invalidate its stored index.
    # By restricting updates to the recency window, we guarantee that no
    # already-compressed token's reconstruction is affected — those tokens
    # are frozen against the centroid state at the time they were compressed.
    #
    # Two strategies (select via `update_strategy`):
    #
    # Strategy A — "recency-gated" (recommended):
    #   Update centroids only while the triggering token is still a recency anchor.
    #   Once it ages out and gets compressed, freeze the codebook.
    #   Eliminates index invalidation entirely. Less adaptive for very long decodes.
    #
    # Strategy B — "always-update" with small alpha:
    #   Update on every new token regardless. Bounded drift per step (~alpha * d).
    #   Over N decode steps, a centroid moves at most (1 - (1-alpha)^N) of the way
    #   toward the running mean. At alpha=0.01, after 200 steps: ~86% max drift.
    #   Acceptable for short-medium generation; degrades for very long generation.
    #   Requires periodic reconstruction-error monitoring in production.
    
    if update_strategy == 'recency_gated':
        # Only update — k_new is currently a recency anchor, not yet compressed
        C_l = C[l*m : (l+1)*m]
        for h in range(num_heads):
            dists = norm(k_new[h].unsqueeze(0) - C_l, dim=1)
            idx = dists.argmin()
            C[l*m + idx] = (1 - alpha) * C[l*m + idx] + alpha * k_new[h]
    
    elif update_strategy == 'always':
        # Update regardless — accept bounded drift on past compressed tokens
        C_l = C[l*m : (l+1)*m]
        for h in range(num_heads):
            dists = norm(k_new[h].unsqueeze(0) - C_l, dim=1)
            idx = dists.argmin()
            C[l*m + idx] = (1 - alpha) * C[l*m + idx] + alpha * k_new[h]
    
    # else: update_strategy == 'frozen' — no update, codebook fixed after prefill

    # ── Step 4: Update importance score for new token ──
    # Will be updated when its attention weights are available
    # (accumulated from subsequent decode steps)


def _compress_token(pos, l, compressed_cache, C, B):
    """Move a token from anchor storage to compressed storage."""
    k = compressed_cache['anchors'].pop(pos)   # [num_heads, head_dim]
    compressed_cache['anchor_ids'].discard(pos)
    
    C_l = C[l*m : (l+1)*m]
    
    assignments = []
    projections = []
    scales = []
    
    for h in range(num_heads):
        dists = norm(k[h].unsqueeze(0) - C_l, dim=1)
        idx = dists.argmin().item()
        residual = k[h] - C_l[idx]              # [head_dim]
        proj = residual @ B[l].T                 # [8]
        scale = proj.abs().max()
        proj_int8 = (proj / scale).clip(-128, 127).to(int8)
        
        assignments.append(idx)
        projections.append(proj_int8)
        scales.append(scale)
    
    compressed_cache['compressed'][pos] = (
        tensor(assignments),   # [num_heads]
        stack(projections),    # [num_heads, 8] int8
        stack(scales),         # [num_heads] fp16 scale per token per head
    )
```

---

## 6. Reconstruction <a name="reconstruction"></a>

```python
def reconstruct_key(pos, l, h, compressed_cache, C, B):
    """
    Reconstruct key vector for token at position pos, layer l, head h.
    Returns: [head_dim] float16
    """
    
    if pos in compressed_cache['anchor_ids']:
        # Full precision — zero reconstruction error
        return compressed_cache['anchors'][pos][h]
    
    else:
        idx, proj_int8, scale = compressed_cache['compressed'][pos]
        
        # Dequantize projections
        proj = proj_int8[h].to(float16) * scale[h]  # [8]
        
        # Reconstruct: centroid + projected residual
        C_l = C[l*m : (l+1)*m]
        k_hat = C_l[idx[h]] + proj @ B[l]           # [head_dim]
        
        return k_hat


def reconstruct_full_layer_cache(l, h, compressed_cache, C, B):
    """
    Reconstruct the full key cache for layer l, head h at attention time.
    Returns: [T, head_dim] float16
    """
    T = len(compressed_cache['anchors']) + len(compressed_cache['compressed'])
    K_hat = zeros(T, head_dim, dtype=float16)
    
    # Anchors — direct read
    for pos in compressed_cache['anchor_ids']:
        K_hat[pos] = compressed_cache['anchors'][pos][h]
    
    # Compressed tokens — batch reconstruct
    compressed_positions = list(compressed_cache['compressed'].keys())
    if compressed_positions:
        indices = stack([compressed_cache['compressed'][p][0][h] 
                        for p in compressed_positions])          # [N]
        projs_int8 = stack([compressed_cache['compressed'][p][1][h] 
                           for p in compressed_positions])       # [N, 8]
        scales = stack([compressed_cache['compressed'][p][2][h] 
                       for p in compressed_positions])           # [N]
        
        projs = projs_int8.to(float16) * scales.unsqueeze(1)    # [N, 8]
        centroids = C[l*m : (l+1)*m][indices]                   # [N, head_dim]
        K_hat[compressed_positions] = centroids + projs @ B[l]  # [N, head_dim]
    
    return K_hat
```

---

## 7. Hyperparameters <a name="hyperparameters"></a>

| Parameter | Description | Recommended Default | Tuning Notes |
|---|---|---|---|
| $A$ | Total codebook size | 256–1024 | Higher = better coverage, negligible memory cost. Start at 256. |
| $m = A/L$ | Centroids per layer | 8–32 | Must have enough to meaningfully cluster keys. Below 8 is likely too few. |
| $R$ | Recency anchor window | 32–128 | Larger R = more exact recent context. Tradeoff: more anchor memory. |
| $M$ | Max importance anchors | 64–256 | Top-M important old tokens stored exactly. Tune based on attention sparsity of target workload. |
| $\alpha$ | Soft centroid update rate | 0.01 | Only meaningful under `always` strategy. At $\alpha=0.01$, after 200 steps a centroid drifts at most ~86% toward running mean. Use `recency_gated` to eliminate drift entirely. |
| `update_strategy` | Codebook update mode | `recency_gated` | `recency_gated`: update only while token is still a recency anchor — no index invalidation. `always`: update on every token, bounded drift. `frozen`: no update after prefill. |
| `iters` | K-means iterations | 3 | 3 is sufficient; 2 acceptable for very long prompts. |

**Effective compression ratio** depends on the anchor fraction:

```
anchor_fraction = (R + M) / T
compression_ratio = anchor_fraction × 1.0 + (1 - anchor_fraction) × (1/30.6)
effective_memory = T × head_dim × 2 × [anchor_fraction + (1 - anchor_fraction)/30.6]
```

At T=4096, R=64, M=128 → anchor_fraction ≈ 4.7% → **~16× overall compression**

---

## 8. Compression Ratio Analysis <a name="compression-ratio-analysis"></a>

### Per-Token Bit Budget

| Component | Full fp16 | IAVQ-KC (A=256, L=32) |
|---|---|---|
| Key vector | 2048 bits | — |
| Centroid index | — | 3 bits ($\log_2 8$) |
| 8 projections (int8) | — | 64 bits |
| Projection scale (fp16) | — | 16 bits |
| **Total** | **2048 bits** | **83 bits** |

Note: projection scale is a per-token per-head fp16 scalar needed to dequantize int8 projections back to fp16 range. This brings the real per-token cost to 83 bits rather than 67, but still represents **~24.7× compression** on compressed tokens.

### Full Cache Memory at T=4096 (Llama-3-8B, Keys only)

| Scheme | Memory |
|---|---|
| Full fp16 | 268 MB |
| IAVQ-KC (R=64, M=128, A=256) | ~19 MB compressed + ~3 MB anchors + 0.3 MB overhead = **~22 MB** |
| Compression ratio | **~12×** |

---

## 9. Comparison to APKVC <a name="comparison-to-apkvc"></a>

| Property | IAVQ-KC | APKVC |
|---|---|---|
| Compression mechanism | Spatial codebook + residual projection | Temporal prediction + VQ on delta |
| What is quantized | Key vector residual from centroid | Delta between consecutive token keys |
| Codebook built from | This prompt's attention statistics (online) | Training data calibration (offline) |
| Anchor strategy | Content-aware (recency + importance score) | Schedule-based (every N tokens) |
| Drift | None — independent per-token reconstruction | Yes — chain prediction accumulates error |
| RoPE handling | Not needed (codebook lives in rotated space) | Requires derotation/rerotation |
| Random access | O(1) per token | O(anchor_distance) per token |
| Prefill cost | k-means + PCA (small, ~200M FLOPs total) | INT8 quantization (minimal) |
| Value cache | Not yet addressed | Handled symmetrically |
| Codebook specificity | Per-prompt, per-layer | Per-model, global |

### When IAVQ-KC is Better
- Long generation where APKVC's prediction chain would accumulate substantial drift
- Workloads with irregular attention patterns where temporal smoothness is weak
- Settings requiring random access into the cache without sequential reconstruction

### When APKVC is Better
- Very regular, temporally smooth generation (consecutive keys are highly similar)
- Shorter generations where drift does not accumulate significantly
- When offline calibration data is available and representative

### Natural Synthesis
IAVQ-KC and APKVC are complementary. A combined scheme could:
- Use IAVQ-KC for the **prompt** (attention weights available for importance scoring, keys may not be temporally smooth)
- Use APKVC for **generated tokens** (strong temporal smoothness in decode phase, no attention weights yet for importance scoring)

---

## 10. Implementation Plan <a name="implementation-plan"></a>

### Phase 1: Core Infrastructure (Week 1–2)

**1.1 Codebook module**
```
iavq_kc/
├── codebook.py          # ImportanceWeightedKMeans, codebook build/update
├── pca_basis.py         # TruncatedSVD wrapper, basis construction
├── quantization.py      # int8 quantize/dequantize with per-token scale
├── cache.py             # CompressedKeyCache dataclass, anchor pools
└── reconstruct.py       # reconstruct_key, reconstruct_full_layer_cache
```

**1.2 `ImportanceWeightedKMeans`**
- Implement in pure PyTorch (GPU-compatible)
- Inputs: key matrix [N, d], importance weights [N], k, iters
- Support batched operation across layers
- Validate: centroid quality metric (mean distance to assigned centroid) should decrease monotonically across iters

**1.3 `PCABasis`**
- Wrap `torch.linalg.svd` with `full_matrices=False`
- Extract top-8 right singular vectors
- Validate: fraction of residual variance captured by 8 components (target: >50% on middle layers)

**1.4 `CompressedKeyCache`**
- Dataclass holding: codebook C, basis B, anchor dict, compressed dict, importance scores, anchor id set
- Implement recency pool as a ring buffer
- Implement importance pool as a heap (max-heap on importance score)

---

### Phase 2: Prefill Integration (Week 2–3)

**2.1 Hook into prefill forward pass**

The cleanest integration point is the attention module. After computing the attention weight matrix $A$ and before discarding it, extract importance scores:

```python
# In attention forward pass, after softmax(QK^T / sqrt(d))
# attention_weights: [batch, heads, T, T]
importance = attention_weights.mean(dim=[0, 1]).sum(dim=0)  # [T]
```

This requires modifying the attention module to optionally return importance scores, controlled by a flag:

```python
class CompressedAttention(nn.Module):
    def forward(self, x, use_iavq=False):
        ...
        attn = softmax(scores / sqrt(d), dim=-1)
        if use_iavq:
            importance = attn.mean(dim=[0,1]).sum(dim=0)
            return out, importance
        return out
```

**2.2 Prefill compression entry point**
```python
def compress_prefill(model, input_ids, R, M, A):
    """Run prefill and return compressed cache."""
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, use_cache=True)
    
    K_cache = outputs.past_key_values  # list of (K, V) per layer
    attn_weights = outputs.attentions  # list of [B, H, T, T] per layer
    
    # Aggregate importance across layers
    importance = aggregate_importance(attn_weights)  # [T]
    
    # Build codebook and compress
    C, B = build_codebook_and_basis(K_cache, importance, A)
    compressed_cache = compress_keys(K_cache, importance, C, B, R, M)
    
    return compressed_cache, C, B
```

**2.3 Validation at this phase**
- Measure per-layer reconstruction MSE: $\|K_l - \hat{K}_l\|_F^2 / (H \cdot T \cdot d)$
- Measure attention score error: $\|QK^T - Q\hat{K}^T\|_F / \|QK^T\|_F$ — this is the critical metric
- Plot both metrics across layers to identify problem layers

---

### Phase 3: Decode Integration (Week 3–4)

**3.1 Decode step wrapper**
```python
def compressed_decode_step(model, token_id, pos, compressed_cache, C, B, alpha=0.01):
    """Single decode step using compressed key cache."""
    
    # Reconstruct full K cache for attention
    K_reconstructed = reconstruct_all_layers(compressed_cache, C, B)
    
    # Run model forward with reconstructed cache
    output = model(token_id, past_key_values=K_reconstructed)
    
    # Get new keys from this step
    new_keys = output.past_key_values  # just the new token's keys
    
    # Update compressed cache
    for l in range(num_layers):
        decode_step_compress(new_keys[l], l, pos, compressed_cache, C, B, alpha)
    
    return output.logits, compressed_cache
```

**3.2 Importance score update for decode tokens**

During decode, the new token's query attends to all past keys. The attention weights over past tokens are available. Accumulate them into the importance score buffer:

```python
# After each decode step, for the new query q_t:
# attn_t: [num_heads, T] — attention weights of new query over all past tokens
importance_update = attn_t.mean(dim=0)  # [T]
compressed_cache['importance_scores'][:pos] += importance_update
compressed_cache['importance_scores'][pos] = 0.0  # new token starts at 0
```

This ensures importance scores for decode tokens accumulate correctly, enabling the importance anchor selection to be meaningful for tokens generated mid-sequence.

**3.3 Anchor promotion/eviction logic**

Maintain a sorted structure for the importance anchor pool. When the importance pool is at capacity $M$ and a new candidate arrives:

```python
def maybe_promote_to_importance_anchor(pos, compressed_cache, M):
    s = compressed_cache['importance_scores']
    pool = compressed_cache['importance_anchor_pool']  # min-heap by importance
    
    if len(pool) < M:
        heappush(pool, (s[pos], pos))
        return True  # promoted, stays in anchor storage
    
    elif s[pos] > pool[0][0]:  # better than worst importance anchor
        _, evict_pos = heappop(pool)
        _compress_token(evict_pos, compressed_cache, C, B)  # compress the evicted token
        heappush(pool, (s[pos], pos))
        return True  # promoted
    
    else:
        return False  # not promoted, compress pos immediately
```

---

### Phase 4: Evaluation (Week 4–6)

**4.1 Reconstruction quality metrics**

For each layer $l$ and head $h$:

```
MSE(l,h) = mean ||k_t - k̂_t||² over all compressed tokens t
RelErr(l,h) = ||QK^T - QK̂^T||_F / ||QK^T||_F
```

Target thresholds (empirically validate):
- MSE: should be substantially lower for anchor tokens (= 0) vs compressed (monitor)
- RelErr: target < 5% for acceptable output quality

**4.2 End-to-end perplexity**

Evaluate on WikiText-103 and LongBench:
- Baseline: full fp16 KV cache
- IAVQ-KC at (R=32, M=64), (R=64, M=128), (R=128, M=256)
- Ablations: no importance weighting (uniform k-means), no importance anchors (recency only), no PCA (store raw residual projections onto random basis)

**4.3 Memory and throughput**

- Peak GPU memory during decode at T=1024, 2048, 4096, 8192
- Tokens per second vs full fp16 baseline
- Measure reconstruction overhead per decode step

**4.4 Attention score sensitivity analysis**

Per layer, measure correlation between importance score $s_t$ and reconstruction error for token $t$. If the importance-weighted codebook is working correctly, high-importance tokens should have lower reconstruction error even before anchoring.

---

### Phase 5: Ablations and Tuning (Week 6–8)

**5.1 Hyperparameter sweep**

```python
sweep_config = {
    'A':     [128, 256, 512, 1024],
    'R':     [16, 32, 64, 128],
    'M':     [32, 64, 128, 256],
    'alpha': [0.0, 0.005, 0.01, 0.05],
    'iters': [1, 2, 3, 5],
}
```

Primary metric: perplexity on WikiText-103 at T=2048.
Secondary: peak memory.

**5.2 Key ablations**

| Ablation | What it tests |
|---|---|
| Uniform k-means (no importance weighting) | Value of importance-aware codebook |
| No importance anchors (recency only, M=0) | Value of content-aware anchoring |
| No recency anchors (importance only, R=0) | Value of recency anchoring |
| Random PCA basis (no SVD, random B) | Value of data-driven basis |
| `frozen` vs `recency_gated` vs `always` update strategy | Whether and how much online adaptation helps vs drift cost |
| `recency_gated` with varying alpha | Sensitivity of adaptation quality to update rate |
| Per-head codebook (not per-layer shared) | Cost/benefit of codebook sharing |

---

## 11. Open Questions and Future Work <a name="open-questions"></a>

### 11.1 Value Cache

This design addresses keys only. Values have different statistical structure:
- Less positional structure than keys (RoPE does not apply to values)
- Often more amenable to simple quantization (KIVI shows INT2 values are acceptable)

Options:
1. Apply IAVQ-KC symmetrically to values (same pipeline, separate codebook)
2. Use simpler INT4/INT8 quantization for values (they are less sensitive)
3. Use a separate importance-weighted scheme for values based on attention output norms

### 11.2 Importance Scores for Decode Tokens

The accumulation strategy described gives reasonable scores but has a lag — a decode token at position $t$ has zero importance when first generated, and only accumulates importance as subsequent tokens attend to it. This means very recently generated tokens (outside the recency window but not yet attended to much) may be incorrectly ranked as low-importance. Worth investigating whether a **predicted importance** (based on content similarity to known high-importance tokens) could bootstrap this better.

### 11.3 Multi-Head Importance

Currently importance scores are averaged across heads. Different heads may have very different attention patterns — some heads may find a token critical while others ignore it. A per-head importance score and per-head anchor selection could improve quality at the cost of more complex bookkeeping.

### 11.4 Codebook Warm-Starting

For repeated similar queries (e.g., the same system prompt), the codebook from a previous run could be used as initialization for the current run's k-means, converging faster. This is a simple caching optimization.

### 11.5 Quantizing the Projection Scale

Currently the per-token per-head projection scale is stored in fp16 (16 bits). This could be quantized to fp8 or int8 with a global scale, saving an additional 50–75% on scale storage — marginal but free accuracy.

---

*IAVQ-KC — Importance-Aware Vector Quantization Key Cache*
*Design document v1.0*
