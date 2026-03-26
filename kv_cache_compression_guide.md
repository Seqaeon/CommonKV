# KV Cache Compression: In-Depth Implementation Reference Guide

**Methods covered:** Palu · MiniCache · CommonKV · ThinKV  
**Purpose:** Deep benchmarking reference — motivation, theory, formulas, pseudocode, and implementation notes.

---

## Table of Contents

1. [Shared Background & Notation](#1-shared-background--notation)
2. [Palu — Low-Rank Projection](#2-palu--low-rank-projection)
3. [MiniCache — Cross-Layer Depth Merging](#3-minicache--cross-layer-depth-merging)
4. [CommonKV — Cross-Layer Parameter Sharing](#4-commonkv--cross-layer-parameter-sharing)
5. [ThinKV — Thought-Adaptive Hybrid Compression](#5-thinkv--thought-adaptive-hybrid-compression)
6. [Method Comparison & Interactions](#6-method-comparison--interactions)
7. [Benchmark Setup Checklist](#7-benchmark-setup-checklist)

---

## 1. Shared Background & Notation

### 1.1 Why KV Cache is the Bottleneck

During autoregressive decoding, the model generates one token at a time. At each step, to compute attention for the new token, the model needs to compare it against **every previously generated token**. Without caching, this would require recomputing the Key and Value projections for all previous tokens from scratch at every step — an O(n²) cost per forward pass.

KV caching solves this by storing the Key and Value tensors for all previous tokens in GPU memory. The cost of each decode step then drops to O(n) in attention computation, but the memory cost grows linearly with sequence length. For a typical 7B-parameter LLM with 32 layers, 32 heads, and a head dimension of 128, a sequence of 32K tokens requires:

```
KV Cache Size = 2 (K+V) × 32 (layers) × 32 (heads) × 128 (d_head) × 32768 (tokens) × 2 (FP16 bytes)
             ≈ 17 GB
```

This exceeds the weight size of the 7B model itself (~14 GB in FP16). At 64K context, the KV cache is essentially the entire memory budget.

The **memory-bandwidth bottleneck** is equally important: during decode, the GPU must load the entire KV cache from HBM for every token generated. At 64K tokens, even ignoring the computation, this means loading ~17 GB of data per decode step, at ~2 TB/s HBM bandwidth — that is ~8.5ms per step purely from memory traffic. Compressing the KV cache directly reduces this load time.

### 1.2 Standard Multi-Head Attention (MHA)

For a new input token `x ∈ R^d` at decode step `t`, each attention head `i` computes:

```
q_i = x · W^q_i          # W^q_i ∈ R^{d × d_h}
k_i = x · W^k_i          # W^k_i ∈ R^{d × d_h}
v_i = x · W^v_i          # W^v_i ∈ R^{d × d_h}
```

The attention score and output for head `i`:

```
p_{t,i} = Softmax( q_i · K_i^T / sqrt(d_h) )    # K_i = all past keys for head i
a_i     = p_{t,i} · V_i                           # weighted sum of past values
```

The final MHA output (concatenating heads, projecting via output matrix W^o):

```
MHA(x) = Σ_i  a_i · W^o_i
```

The **KV cache** stores `K_i` and `V_i` across all previous timesteps for all heads and layers. What each method compresses is exactly this stored material.

### 1.3 Grouped Query Attention (GQA)

Modern LLMs (LLaMA-3, Mistral) use GQA: multiple query heads share a single K/V head pair. If there are `n_q` query heads and `n_kv` K/V heads (with `n_kv < n_q`):

```
Each KV group serves G = n_q / n_kv query heads
KV cache memory is reduced by factor n_kv / n_q versus MHA
```

**Impact on compression methods:** GQA already reduces the KV cache significantly. Palu and CommonKV still apply SVD to the per-KV-head weight matrices. Because `d_kv` is now smaller relative to `d_model`, the SVD rank must be chosen carefully — CommonKV specifically concatenates K and V matrices together to avoid rank deficiency.

### 1.4 SVD Basics and Why It Works for Weight Matrices

SVD (Singular Value Decomposition) exploits the fact that many weight matrices in neural networks are approximately low-rank — meaning most of the "information" in the matrix is captured by a small number of singular vectors.

Given weight matrix `W ∈ R^{m × n}`:

```
W = U Σ V^T
```

where `U ∈ R^{m×m}` and `V ∈ R^{n×n}` are orthogonal matrices, and `Σ` is a diagonal matrix of singular values in descending order `σ_1 ≥ σ_2 ≥ ... ≥ σ_min(m,n) ≥ 0`.

The rank-`r` approximation keeps only the top `r` singular values:

```
W ≈ U_r Σ_r V_r^T = A · B

where:
  A = U_r · sqrt(Σ_r)    ∈ R^{m × r}    (down-projection)
  B = sqrt(Σ_r) · V_r^T  ∈ R^{r × n}    (up-projection)

Reconstruction error:  ||W - AB||_F = sqrt( σ_{r+1}^2 + ... + σ_n^2 )
Storage ratio: (m·r + r·n) / (m·n)  =  r(m+n) / (mn)
```

The key intuition is the **Eckart–Young theorem**: among all rank-`r` matrices, this truncated SVD minimises the Frobenius norm error. If the singular values decay rapidly (which they do in transformer weight matrices), a rank `r << min(m,n)` can capture most of the weight's behaviour.

**Why activation magnitudes matter:** Standard SVD minimises `||W - AB||_F`, but the actual output error is `||x(W - AB)||_F` — weighted by the input activations. Transformer activations often have outlier channels with huge magnitudes (a well-documented phenomenon). SVD-LLM addresses this by scaling columns of `W` by the standard deviation of the corresponding activation channel before decomposing, ensuring that channels with large activations are approximated more carefully.

### 1.5 KV Cache Memory Formula and Compression Targets

```
Memory(KV) = 2 · n_layers · n_kv_heads · d_head · seq_len · bytes_per_element
```

For FP16: `bytes_per_element = 2`. The factor 2 accounts for both K and V.

**Common compression targets across papers:**

| Method | Compression Axis | Reported Gain |
|--------|-----------------|---------------|
| Palu | Hidden dimension (rank) | 50% dim → 1.89× attention speedup |
| MiniCache | Layer depth | Merge L/2 layers → 25% memory → 5× throughput with quant |
| CommonKV | Both dim + depth | 50% ratio → 95%+ performance retained |
| ThinKV | Token count | <5% of FullKV budget with near-lossless accuracy |

### 1.6 Compression Ratio Convention (Important for Benchmarking)

All four papers define "compression ratio" differently. Normalise before comparing:

| Paper | Their Definition | What It Means Concretely |
|-------|-----------------|--------------------------|
| Palu | Fraction of KV hidden-dim removed | 50% = cache `r = d_h/2` channels per token |
| MiniCache | Layers merged + quantization | Merging L/2 + 4-bit → 5.02× total compression |
| CommonKV | `1 − Compressed_KV / Original_KV` | 0.5 = half the KV memory gone |
| ThinKV | Token budget as `%` of FullKV memory | 5% budget = 95% of tokens evicted/quantized |

**Unified formula for comparison:**

```python
def equivalent_memory_fraction(method, params):
    """Returns fraction of original FP16 KV memory used."""
    if method == 'palu':
        # Dim compression × quantization
        return (1 - params['compression']) * (params['bits'] / 16)
    elif method == 'minicache':
        # Layer compression × quantization
        unmerged = params['start_layer'] / params['total_layers']
        merged   = 1 - unmerged
        return (unmerged + merged * 0.5) * (params['bits'] / 16)
    elif method == 'commonkv':
        return 1 - params['compression_ratio']
    elif method == 'thinkv':
        return params['token_budget_fraction'] * (params['avg_bits'] / 16)
```

---

## 2. Palu — Low-Rank Projection

### 2.1 Problem Framing and Core Insight

Existing KV cache compression methods attack two orthogonal axes:

- **Token eviction** (e.g., H2O, SnapKV): remove entire token rows from K/V. These are the "which rows to keep" methods. They cannot help when all tokens matter (long-range dependencies).
- **Quantization** (e.g., KIVI, KVQuant): reduce the bit-width of stored values. This compresses the "precision" of every element but leaves the shape unchanged.

Palu identifies a **third axis**: the **hidden dimension** of each token's KV vector. Even if you keep all tokens at full precision, the fact that K and V vectors live in `R^{d_h}` might be wasteful — the effective rank of the K/V matrix across tokens could be much lower than `d_h`.

The core insight is that `W^k` and `W^v` can be decomposed offline, so at runtime you only ever project the input `x` into a low-dimensional latent space of size `r`, cache that, and reconstruct the full key/value only when needed (and often the reconstruction can be fused into adjacent operations so it's free).

This is fundamentally different from MLA (DeepSeek's Multi-head Latent Attention), which also caches a low-rank representation — but MLA requires pre-training the model with this architecture from scratch. Palu is purely post-training and applies to any existing MHA or GQA model.

### 2.2 Core Formula

Standard projection: `y = x · W`

After SVD decomposition `W ≈ A · B`:

```
h = x · A          # down-project to latent: h ∈ R^r   (CACHED — small)
y = h · B          # up-project on demand:  y ∈ R^{d_h} (RECONSTRUCTED — not cached)
```

**Memory saved per token per layer per head:**
```
Original:   d_h floats stored    (e.g., 128 at FP16 = 256 bytes)
Palu:       r floats stored      (e.g., 64  at FP16 = 128 bytes for 50% compression)
```

The savings compound across all `n_heads × n_layers` cache entries.

**Why this is memory-bandwidth efficient:** During each decode step, only `h` (of size `r`) needs to be loaded from HBM instead of the full key/value vector of size `d_h`. The reconstruction `y = h · B` is compute-bound (matrix multiply with the fixed `B` in registers/SRAM), not memory-bound. Since the bottleneck in decoding is HBM bandwidth, this directly reduces the bottleneck.

### 2.3 Offline SVD Decomposition

```python
def decompose_weight(W, rank):
    """
    W: weight matrix, shape (d_model, d_head)
    rank: target rank r
    Returns A (d_model, r), B (r, d_head)

    The sqrt(Σ) split distributes singular values symmetrically between A and B.
    This matters for quantization compatibility later (both halves have similar scales).
    """
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    # Truncate to rank r (keep top-r singular values/vectors)
    U_r  = U[:, :rank]          # (d_model, rank) — left singular vectors
    S_r  = S[:rank]             # (rank,)          — top-r singular values
    Vt_r = Vt[:rank, :]         # (rank, d_head)   — right singular vectors

    sqrt_S = torch.sqrt(S_r)
    A = U_r * sqrt_S.unsqueeze(0)     # (d_model, rank): each column scaled by sqrt(σ_i)
    B = sqrt_S.unsqueeze(1) * Vt_r    # (rank, d_head):  each row scaled by sqrt(σ_i)
    return A, B
```

**Activation-aware SVD (SVD-LLM variant — what Palu actually uses):**

Standard SVD minimises `||W - AB||_F` (weight reconstruction error), but the true objective is minimising `||xW - xAB||_F` (output reconstruction error). Activation-aware SVD accounts for this:

```python
def activation_aware_svd(W, activation_stats, rank):
    """
    activation_stats: dict with 'std' per input channel,
                      estimated from calibration data (2048 samples, seqlen=1024)

    The idea: scale W so that SVD prioritises columns (input channels) with
    large activations, since errors in those channels matter most to the output.
    """
    scale = activation_stats['std']   # (d_model,) — std of each input channel

    # Scale columns of W by activation std
    # This makes SVD "see" large-activation channels as more important
    W_scaled = W * scale.unsqueeze(1)  # broadcast: scale each row of W

    # Decompose scaled matrix
    U, S, Vt = torch.linalg.svd(W_scaled, full_matrices=False)
    U_r, S_r, Vt_r = U[:, :rank], S[:rank], Vt[:rank, :]

    sqrt_S = torch.sqrt(S_r)
    A_scaled = U_r * sqrt_S.unsqueeze(0)
    B_scaled = sqrt_S.unsqueeze(1) * Vt_r

    # Undo the scaling: A must "undo" the scale before B sees it
    # Since h = x · A and x is unscaled, we need A to incorporate 1/scale
    # The scaling was applied to W = A·B, so W = (A_scaled/scale) · B_scaled
    A = A_scaled / scale.unsqueeze(1)   # (d_model, rank)
    B = B_scaled                         # (rank, d_head) — unchanged
    return A, B
```

**When to use which:** Use plain SVD for a quick baseline. For reproduction of Palu results, use the activation-aware variant — it can recover ~0.5 perplexity points at 50% compression.

### 2.4 Three Decomposition Granularities — Deep Dive

The granularity choice governs how information is shared across attention heads during down-projection. This is the most important design decision in Palu.

**Why M-LRD loses accuracy:** Each head independently computes `h_i = x · A_i`. Since `A_i` is only rank-`r_i` and is fit to `W^k_i` alone, it cannot capture features that span across heads. The SVD of a small matrix (one head's weight) may miss structure that becomes apparent when all heads are considered together. Think of it as overfitting the low-rank approximation to each head in isolation.

**Why J-LRD is too expensive:** The joint latent `h_joint = x · A_joint` is shared by all heads. Reconstructing head `i`'s keys requires `k_i = h_joint · B_joint_i`, where `B_joint_i ∈ R^{r_joint × d_h}` is a slice of the full `B_joint`. If `r_joint = n_heads × r_i` (same total latent size), then each head's reconstruction cost is `r_joint × d_h = n_heads × r_i × d_h` — exactly `n_heads` times the per-head M-LRD cost. For LLaMA-2-7B with 32 heads, this is 32× more FLOPs and memory for reconstruction.

**G-LRD with group_size=4 — the sweet spot:**

```
For group g = {h1, h2, h3, h4}:

  W_g = [W^k_{h1} | W^k_{h2} | W^k_{h3} | W^k_{h4}]   shape: (d_model, 4·d_h)

  SVD: W_g ≈ A_g · B_g
  A_g ∈ R^{d_model × r_g}       (shared projection, captures cross-head features)
  B_g ∈ R^{r_g × 4·d_h}         (head-specific reconstruction, sliced into 4 blocks)

  Latent:       h_g = x · A_g                  (size r_g, shared by 4 heads)
  Reconstruct:  k_{h1} = h_g · B_g[:, 0:d_h]  (each head gets its own slice of B_g)
               k_{h2} = h_g · B_g[:, d_h:2d_h]
               ...
```

**FLOPs comparison (same total latent size across all methods):**

Let total latent size = `r_total`. Then `r_i = r_total / n_heads` for M-LRD, `r_g = r_total / n_groups` for G-LRD, `r_joint = r_total` for J-LRD.

```
M-LRD reconstruction FLOPs:   n_heads × r_i × d_h = r_total × d_h
G-LRD reconstruction FLOPs:   n_groups × r_g × (s × d_h) = r_total × d_h   (same!)
J-LRD reconstruction FLOPs:   n_heads × r_joint × d_h = n_heads × r_total × d_h

Key result: M-LRD and G-LRD have the same reconstruction FLOPs for the same total latent size.
The accuracy gain of G-LRD over M-LRD comes for FREE in terms of FLOPs.
J-LRD is n_heads times more expensive.
```

The memory difference shows up in the reconstruction matrix `B`:

```
M-LRD:  n_heads matrices, each (r_i × d_h)    — small, one per head
G-LRD:  n_groups matrices, each (r_g × s·d_h) — medium, one per group
J-LRD:  1 matrix of size (r_joint × n·d_h)    — huge, n_heads× M-LRD
```

For LLaMA-2-7B with 32 heads at 50% compression: J-LRD's fused reconstruction matrix is 32× larger than M-LRD's equivalent, requiring substantial extra GPU memory.

```python
def split_into_groups(W_all_heads, group_size, n_heads, d_h):
    """
    W_all_heads: list of per-head weight matrices [(d_model, d_h), ...]
    Returns list of group weight matrices [(d_model, group_size * d_h), ...]
    """
    assert n_heads % group_size == 0, "n_heads must be divisible by group_size"
    groups = []
    for g in range(n_heads // group_size):
        start = g * group_size
        end   = start + group_size
        # Concatenate horizontally: (d_model, group_size * d_h)
        W_g = torch.cat(W_all_heads[start:end], dim=1)
        groups.append(W_g)
    return groups
```

### 2.5 Matrix Fusion — The Performance Enabler

Matrix fusion is what converts Palu from a theoretically sound idea into a practically fast one. Without it, each decode step would require an explicit reconstruction `y = h · B` (a matrix multiply), adding overhead that cancels the memory savings.

**Value path fusion (always possible):**

The attention output computation is:

```
a_i = p_i · V_i = p_i · (H^v_i · B^v_i)
```

The final token output is then `a_i · W^o_i`. Substituting:

```
a_i · W^o_i = (p_i · H^v_i · B^v_i) · W^o_i
            = p_i · H^v_i · (B^v_i · W^o_i)     # associativity of matmul
            = p_i · H^v_i · W^o_fused             # W^o_fused precomputed offline
```

This means the computation `H^v → V → output` is replaced by `H^v → output` directly. The reconstruction of `V` is **implicit** — `B^v` is absorbed into `W^o`. At runtime, you never materialise the full value vectors.

**Key path fusion (non-RoPE only):**

The attention score is:

```
score = q_i · K_i^T = q_i · (H^k_i · B^k_i)^T = q_i · (B^k_i)^T · (H^k_i)^T
```

Since `q_i = x · W^q_i`:

```
score = x · W^q_i · (B^k_i)^T · (H^k_i)^T
      = x · (W^q_i · (B^k_i)^T) · (H^k_i)^T
      = x · W^q_fused · (H^k_i)^T               # W^q_fused precomputed offline
```

Now computing the attention score only requires one matmul against the cached low-rank `H^k` — no key reconstruction needed.

**Why RoPE breaks key fusion:**

RoPE (Rotary Position Embedding) applies a position-dependent rotation to each key vector **after** it is computed but **before** it is used in attention. The rotation at position `t` is:

```
k_t = RoPE(x_t · W^k, pos=t) = Rot(t) · (x_t · W^k)
```

where `Rot(t)` is a position-dependent rotation matrix. The key insight is that `Rot(t)` cannot be "pulled out" through `B^k` because `Rot(t)` is different for every token — it depends on position, not just on the weight matrices. So you cannot precompute `W^q_fused` once and use it for all positions.

**Concrete impact at 64K context length:** Without key fusion (RoPE models), Palu must reconstruct each token's key online at each decode step. The paper implements a custom Triton kernel that **fuses** the steps of (1) loading the low-rank `h_k` from HBM, (2) multiplying by `B^k` to get the full key, (3) applying RoPE, and (4) multiplying against the query — all in a single kernel launch. This keeps all intermediate values in SRAM and avoids writing the reconstructed key back to HBM.

```
Without fusion:  load h_k → GEMM(h_k, B_k) → write K to HBM → load K → RoPE → GEMM(q, K^T)
With fusion:     load h_k → GEMM in SRAM → RoPE in SRAM → GEMM in SRAM → write score
```

The bandwidth difference is one extra round-trip of size `seq_len × d_h` per head per layer. At 64K tokens, this is substantial.

### 2.6 Automatic Rank Allocation via Fisher Information

**Why uniform rank allocation is suboptimal:**

Not all K/V projection layers contribute equally to model performance. Early layers tend to capture syntactic/surface features; middle and late layers capture more semantic content. The key and value projections within the same layer also differ — experiments show that value projections are generally more sensitive to compression than key projections. Blindly applying the same rank to all layers wastes capacity in unimportant layers and over-compresses critical ones.

**Fisher information as sensitivity proxy:**

The Fisher information `F(W)` for a weight matrix `W` measures how much the loss changes when `W` is perturbed:

```
F(W) ≈ E[ (∂L/∂W)² ]
```

In practice, this is estimated by squaring the gradients on a small calibration dataset:

```python
def compute_fisher_information(model, calibration_data):
    """
    Estimate Fisher information for each K/V projection weight.
    Uses squared gradient as diagonal Fisher approximation.

    calibration_data: ~2048 samples from Wikitext-2, seqlen=1024
    This is fast — no backprop through the full model needed, just language model loss.
    """
    fisher = {}
    for name, param in model.named_parameters():
        if 'k_proj' in name or 'v_proj' in name:   # only care about K/V weights
            fisher[name] = torch.zeros_like(param.data)

    model.eval()
    for batch in calibration_data:
        outputs = model(**batch)
        loss    = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.data ** 2   # accumulate squared grad

        model.zero_grad()

    # Average over calibration set
    n = len(calibration_data)
    for name in fisher:
        fisher[name] /= n

    return fisher
```

**Rank allocation algorithm:**

```python
def allocate_ranks_fisher(fisher, decomp_targets, global_rank_budget):
    """
    Allocate total rank budget proportionally to Fisher information.

    decomp_targets: list of (layer_id, 'k'/'v', weight_name)
    global_rank_budget: total number of rank units to distribute
                        e.g., if 50% compression and n_targets=64,
                        budget = 64 * d_head * 0.5

    Returns dict mapping (layer_id, kv_type) → assigned_rank
    """
    # Sum Fisher information per decomposition target (K or V per layer)
    importances = {}
    total_importance = 0.0

    for layer_id, kv_type, weight_name in decomp_targets:
        # Fisher info of the full K or V weight matrix for this layer
        F = fisher[weight_name].sum().item()
        importances[(layer_id, kv_type)] = F
        total_importance += F

    # Proportionally allocate ranks
    ranks = {}
    for (layer_id, kv_type), importance in importances.items():
        fraction      = importance / total_importance
        assigned_rank = max(1, int(round(fraction * global_rank_budget)))
        ranks[(layer_id, kv_type)] = assigned_rank

    return ranks
```

**What this produces in practice:**

Experiments on LLaMA-2-7B show that the value projections consistently receive higher ranks than the key projections (roughly 75% vs 25% of the total budget at 50% compression). Within each type, the first half of layers tend to receive higher ranks than the second half. This result emerged purely from data — the Fisher information guided the algorithm to "rediscover" what researchers know intuitively: early-to-middle layers are more informative.

Using rank search vs. uniform rank gives ~1.4 lower perplexity at 70% compression — a significant improvement.

### 2.7 Quantization Compatibility — The Hadamard Transform

**The outlier problem in SVD-compressed representations:**

When you compute `h = x · A` where `A` comes from SVD, the first few dimensions of `h` will have dramatically larger magnitudes than the rest. This is a direct consequence of SVD's construction: the first column of `U_r` (which forms the first column of `A`) corresponds to the largest singular value `σ_1`, so it captures the direction of maximum variance. The activations along this direction are much larger than along subsequent directions (since `σ_1 >> σ_2 >> ...`).

Concretely, if you plot the activation distribution of `h` across its dimensions, you see:

```
Dimension 0: large values (captures σ_1 component)
Dimension 1: smaller values
...
Dimension r-1: very small values

This is not a bell curve — it's heavily skewed.
```

This pattern destroys quantization: a per-tensor or per-group quantization scale set to accommodate the large dimension 0 values will waste bits on dimensions 1 through r-1, creating large quantization error everywhere else.

**The Walsh-Hadamard Transform (WHT) solution:**

The WHT is an orthogonal transformation that "mixes" all dimensions uniformly. Applying it to `h` redistributes the energy evenly across all dimensions, resulting in a distribution much better suited to uniform quantization:

```
H_r: Hadamard matrix of size r × r (r must be a power of 2)
     H_r is orthogonal: H_r · H_r^T = I
     Elements are ±1/sqrt(r)

Transform: h_smooth = h · H_r
```

Since `H_r` is orthogonal, it doesn't lose information — it just rearranges it.

**The offline fusion trick:**

Naively, you'd apply WHT online at every decode step: `h → h·H_r → quantize → dequantize → h·H_r^T → reconstruct`. This adds two matrix multiplications per decode step — unacceptable overhead.

Palu's key insight: since `H_r` is fixed (doesn't depend on the input), you can absorb it into the weight matrices offline:

```
Original:  h = x · A,   y = h · B
           y = x · A · B = x · W

With WHT:  h_smooth = h · H_r = x · A · H_r
           y = h_smooth · H_r^T · B = x · A · H_r · H_r^T · B = x · A · B  ✓ (same output)

But we cache h_smooth = x · (A · H_r), so define:
  A_hat = A · H_r          (new down-projection matrix, computed offline)
  B_hat = H_r^T · B        (new up-projection matrix, computed offline)

Now: h_smooth = x · A_hat   (smooth distribution, cache-friendly for quantization)
     y = h_smooth · B_hat   (reconstruction unchanged)
```

No extra computation at runtime. The Hadamard transform is "baked in" to the weight matrices.

```python
def hadamard_matrix(n):
    """
    Construct n×n normalised Hadamard matrix.
    n must be a power of 2.
    Uses recursive Sylvester construction.
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    if n == 1:
        return torch.tensor([[1.0]])
    H_half = hadamard_matrix(n // 2)
    top    = torch.cat([H_half,  H_half], dim=1)
    bottom = torch.cat([H_half, -H_half], dim=1)
    return torch.cat([top, bottom], dim=0) / math.sqrt(2)

def fuse_hadamard_offline(A, B):
    """
    Fuse Hadamard transform into weight matrices.
    A: (d_model, rank), B: (rank, d_head)
    rank must be a power of 2.
    Returns A_hat, B_hat with same shapes.
    """
    rank = A.shape[1]
    H = hadamard_matrix(rank).to(A.device, A.dtype)

    A_hat = A @ H          # (d_model, rank): each column mixed with others
    B_hat = H.T @ B        # (rank, d_head):  each row mixed with others

    # Verify: A_hat @ B_hat should equal A @ B (within floating point error)
    assert torch.allclose(A_hat @ B_hat, A @ B, atol=1e-5), "Fusion error"

    return A_hat, B_hat
```

**When does this matter most:** At 2-bit quantization, the Hadamard transform provides ~4.17 perplexity improvement on LLaMA-2-7B (from 10.58 to 6.41 on WikiText-2 at 50% rank compression + 2-bit quantization). At 3-bit, the benefit is smaller (~0.22 perplexity). At 4-bit or above, it's negligible.

### 2.8 Full Palu Pseudocode (Production-Ready)

```python
# ─── OFFLINE SETUP ───────────────────────────────────────────────────────────

def palu_setup(model, config):
    """
    config fields:
      group_size        : int, default 4 (G-LRD)
      target_compression: float, default 0.5 (50% of d_head channels removed)
      calibration_data  : DataLoader (Wikitext-2, 2048 samples, seqlen=1024)
      use_hadamard      : bool, default True (for quantization compatibility)
      use_fisher_search : bool, default True
    """
    fisher = compute_fisher_information(model, config['calibration_data'])

    # Identify all K and V projection layers
    decomp_targets = [
        (l, kv, f'model.layers.{l}.self_attn.{kv}_proj.weight')
        for l in range(model.config.num_hidden_layers)
        for kv in ['k', 'v']
    ]

    # Rank allocation
    if config['use_fisher_search']:
        d_head = model.config.head_dim
        n_targets = len(decomp_targets)
        global_budget = int(n_targets * d_head * (1 - config['target_compression']))
        ranks = allocate_ranks_fisher(fisher, decomp_targets, global_budget)
    else:
        # Uniform rank for all targets
        r = int(model.config.head_dim * (1 - config['target_compression']))
        ranks = {(l, kv): r for l, kv, _ in decomp_targets}

    # Decompose and fuse
    palu_weights = {}
    for layer_id in range(model.config.num_hidden_layers):
        for kv in ['k', 'v']:
            W_full = get_full_kv_weight(model, layer_id, kv)
            # W_full: (d_model, d_head * n_kv_heads) for MHA
            #         (d_model, d_head * n_kv_groups) for GQA

            n_kv_heads = model.config.num_key_value_heads
            d_h = model.config.head_dim
            group_size = config['group_size']
            n_groups = n_kv_heads // group_size

            A_list, B_list = [], []
            for g in range(n_groups):
                # Extract group's weight: (d_model, group_size * d_h)
                W_g = W_full[:, g*group_size*d_h : (g+1)*group_size*d_h]

                r = ranks[(layer_id, kv)]
                A_g, B_g = activation_aware_svd(
                    W_g, fisher_activation_stats(model, layer_id), r
                )

                # Fuse Hadamard for quantization compatibility
                if config['use_hadamard']:
                    # Round r up to power of 2 if needed
                    r_rounded = next_power_of_2(r)
                    if r_rounded != r:
                        A_g = F.pad(A_g, (0, r_rounded - r))
                        B_g = F.pad(B_g, (r_rounded - r, 0))
                    A_g, B_g = fuse_hadamard_offline(A_g, B_g)

                A_list.append(A_g)
                B_list.append(B_g)

            palu_weights[(layer_id, kv, 'A')] = A_list  # list of (d_model, r) per group
            palu_weights[(layer_id, kv, 'B')] = B_list  # list of (r, group_size*d_h) per group

        # Offline matrix fusion for value path
        for g in range(n_groups):
            B_v_g = palu_weights[(layer_id, 'v', 'B')][g]   # (r, group_size * d_h)
            W_o_g = get_output_proj_slice(model, layer_id, g, group_size, d_h)
            # W_o_g: (group_size * d_h, d_model) — the portion of W^o for this group
            W_o_fused = B_v_g @ W_o_g   # (r, d_model) — no explicit V reconstruction needed
            palu_weights[(layer_id, 'v_o_fused', g)] = W_o_fused

    return palu_weights


# ─── PREFILL ─────────────────────────────────────────────────────────────────

def palu_prefill(x, layer_id, palu_weights, model_config):
    """
    x: (batch, seq_len, d_model) — hidden states
    Returns latent K and V caches: lists of tensors, one per group.
    """
    n_groups = model_config.num_key_value_heads // model_config.group_size
    latent_k = []
    latent_v = []

    for g in range(n_groups):
        A_k = palu_weights[(layer_id, 'k', 'A')][g]   # (d_model, rank)
        A_v = palu_weights[(layer_id, 'v', 'A')][g]

        h_k = x @ A_k   # (batch, seq_len, rank)  ← WHAT GETS CACHED
        h_v = x @ A_v   # (batch, seq_len, rank)  ← WHAT GETS CACHED

        latent_k.append(h_k)
        latent_v.append(h_v)

    return latent_k, latent_v   # each is a list of (B, S, rank) tensors


# ─── DECODE STEP ─────────────────────────────────────────────────────────────

def palu_decode_step(x_new, latent_k_cache, latent_v_cache,
                     layer_id, palu_weights, model_config,
                     position_ids, use_rope=True):
    """
    x_new: (batch, 1, d_model) — new token hidden state
    position_ids: (batch, 1) — position of the new token

    The cached latent_k_cache and latent_v_cache are EXTENDED in-place.
    """
    n_groups  = model_config.num_key_value_heads // model_config.group_size
    d_h       = model_config.head_dim
    group_size = model_config.group_size

    # ── 1. Extend latent caches with new token ────────────────────────────────
    for g in range(n_groups):
        A_k = palu_weights[(layer_id, 'k', 'A')][g]
        A_v = palu_weights[(layer_id, 'v', 'A')][g]

        h_k_new = x_new @ A_k   # (batch, 1, rank)
        h_v_new = x_new @ A_v

        # Append along sequence dimension
        latent_k_cache[g] = torch.cat([latent_k_cache[g], h_k_new], dim=1)
        latent_v_cache[g] = torch.cat([latent_v_cache[g], h_v_new], dim=1)

    # ── 2. Compute query (standard, no compression) ───────────────────────────
    q = x_new @ model.layers[layer_id].W_q   # (batch, 1, n_q_heads * d_h)
    # Apply RoPE to query if needed
    if use_rope:
        q = apply_rope(q, position_ids)

    # ── 3. Compute attention scores (per group) ───────────────────────────────
    all_group_outputs = []

    for g in range(n_groups):
        B_k = palu_weights[(layer_id, 'k', 'B')][g]   # (rank, group_size * d_h)
        H_k = latent_k_cache[g]                         # (batch, seq_len, rank)

        # Slice query for this group
        q_g = q[:, :, g*group_size*d_h : (g+1)*group_size*d_h]
        # Reshape to (batch, group_size, 1, d_h) for multi-head attention
        q_g = q_g.view(batch, group_size, 1, d_h)

        if use_rope:
            # RoPE case: reconstruct full keys online, apply RoPE, then attend
            # This is the expensive case — Palu uses a fused Triton kernel here
            # that keeps K in SRAM without writing back to HBM.
            #
            # Fused kernel steps (all in SRAM):
            # 1. Load tile of H_k from HBM (size: tile × rank)
            # 2. Multiply by B_k in SRAM → K_tile (tile × group_size × d_h)
            # 3. Apply RoPE position embeddings to K_tile
            # 4. Multiply q_g by K_tile^T to get partial attention scores
            # 5. Accumulate into score array (online softmax)
            # → Only H_k crosses HBM boundary (not the reconstructed K)
            scores_g = triton_fused_reconstruct_rope_attention(H_k, B_k, q_g, position_ids)
        else:
            # Non-RoPE: use fused query (B^k absorbed into W^q offline)
            # W_q_fused[g] = W_q[:, g*group_size*d_h:...] @ B_k.T   (precomputed)
            W_q_fused_g = palu_weights[(layer_id, 'q_fused', g)]     # (d_model, rank)
            q_fused_g   = x_new @ W_q_fused_g                        # (batch, 1, rank)
            # Attend directly against latent: no key reconstruction!
            scores_g    = q_fused_g @ H_k.transpose(-1, -2)          # (batch, 1, seq_len)

        scores_g = scores_g / math.sqrt(d_h)
        attn_g   = F.softmax(scores_g, dim=-1)   # (batch, group_size, 1, seq_len)

        # ── 4. Value computation (fused — no V reconstruction needed) ────────
        H_v = latent_v_cache[g]                             # (batch, seq_len, rank)
        W_o_fused = palu_weights[(layer_id, 'v_o_fused', g)]  # (rank, d_model)

        # attn_g: (batch, group_size, 1, seq_len)
        # H_v:    (batch, seq_len, rank)
        # Compute weighted sum over sequence: (batch, group_size, 1, rank)
        weighted_v = torch.einsum('bgs,bsr->bgr', attn_g.squeeze(-2), H_v)
        # Project to output: (batch, group_size, d_model)
        out_g = weighted_v @ W_o_fused

        all_group_outputs.append(out_g.sum(dim=1))   # sum over group heads

    # Sum contributions from all groups
    return sum(all_group_outputs)   # (batch, d_model)
```

### 2.9 Key Hyperparameters and Ablation Insights

| Parameter | Default | Effect of Changing |
|-----------|---------|-------------------|
| `group_size` | 4 | Larger → better accuracy, higher reconstruction cost. 32 = J-LRD (too expensive) |
| `target_compression` | 0.5 | 0.3 is "safe" (< 1% accuracy drop); 0.7 causes visible degradation |
| `calibration_samples` | 2048 | More samples → better Fisher estimate; diminishing returns beyond 4096 |
| `key_compression` | ~0.75 | Keys tolerate more compression (rank search finds this automatically) |
| `value_compression` | ~0.25 | Values need more capacity; fused into W^o (cheaper to reconstruct) |
| `use_fisher_search` | True | Removes ~2 perplexity at 70% compression vs. uniform allocation |
| `use_hadamard` | True | Critical for 2-bit quantization; marginal benefit at 4-bit+ |

---

## 3. MiniCache — Cross-Layer Depth Merging

### 3.1 Problem Framing and Core Insight

MiniCache attacks a dimension of KV cache redundancy that all other methods (Palu, quantization, token eviction) completely ignore: **the depth dimension**. Palu compresses the hidden dimension; token eviction compresses the sequence dimension; quantization compresses the bit precision. MiniCache asks: do we really need separate KV caches for every layer?

The empirical observation driving MiniCache is that in the middle-to-deep portions of LLMs (roughly from layer `L/2` to `L`), the key cache of layer `l` and the key cache of layer `l-1` are **highly similar** when measured at the same token positions. Specifically:

```
cosine_similarity(K^l[token_i], K^{l-1}[token_i]) > 0.85
for most tokens in layers l >= L/2
```

This is much higher than similarity across non-adjacent layers or across shallow layers, which can be near zero. The pattern holds consistently across model families (LLaMA, Mistral, Mixtral, Phi).

**Why does this happen?** Transformers use residual connections: the hidden state entering layer `l` is `x^l = x^{l-1} + f(x^{l-1})`, where `f` is the transformer block's computation. When `||f(x^{l-1})|| << ||x^{l-1}||` (small residual update, common in deep layers), `x^l ≈ x^{l-1}`, and consequently `K^l ≈ K^{l-1}`. Deeper layers tend to have smaller residual updates because they're refining an already-good representation.

**What MiniCache does with this:** Instead of storing `K^l` and `K^{l-1}` separately, it merges them into a single shared representation. When layer `l` or `l-1` needs their keys during decoding, it reconstructs an approximation from the shared state.

### 3.2 Why Simple Averaging Fails

The most obvious merge function is arithmetic averaging:

```
K_shared[i] = (K^l[i] + K^{l-1}[i]) / 2
```

This performs surprisingly well as a baseline — better than many might expect — because the vectors are similar enough that their average is close to both. However, it has a systematic flaw: **it ignores the magnitude (norm) difference between the two vectors**.

In LLMs, key vectors can have significantly different norms across layers, even when their directions are similar. Averaging collapses both vectors to the same intermediate norm, which corrupts the attention score magnitudes and thus the attention weight distribution. The attention softmax is sensitive to these magnitudes via the `1/sqrt(d_h)` scaling factor.

This is analogous to the problem that motivates weight normalization in neural network training: decoupling the direction and magnitude of parameter vectors allows better optimization. MiniCache applies the same principle to KV cache merging.

### 3.3 SLERP Merge — Full Derivation

SLERP (Spherical Linear intERPolation) is a method for interpolating between two vectors on the unit sphere. It was originally developed for animating rotations in 3D graphics and has since found applications in model merging.

**Setup:** Given two unit vectors `u = x/||x||` and `v = y/||y||`, the angle between them is `Ω = arccos(u · v)`. SLERP traces the shortest arc on the unit sphere from `u` to `v`:

```
SLERP(u, v, t) = [sin((1-t)Ω) / sin(Ω)] · u + [sin(t·Ω) / sin(Ω)] · v
```

At `t=0`: returns `u`. At `t=1`: returns `v`. At `t=0.5`: midpoint on the arc.

**MiniCache application:** For a pair of token vectors `x^l, x^{l-1} ∈ R^{d_h}` at the same position:

```
Step 1 — Compute the angle:
  Ω^{l,l-1} = arccos( (x^l / ||x^l||) · (x^{l-1} / ||x^{l-1}||) )

Step 2 — Compute the directional component via SLERP:
  e^{l,l-1} = [sin((1-t)Ω) / sin(Ω)] · (x^{l-1} / ||x^{l-1}||)
            + [sin(t·Ω)    / sin(Ω)] · (x^l     / ||x^l||)

  (e is a unit vector pointing in the "merged" direction)

Step 3 — Store magnitudes separately:
  ||x^l||, ||x^{l-1}||   (one scalar per token per layer, not per dimension)
```

**Why `t = 0.6` (leaning toward the deeper layer `l`):**

The paper's ablation shows that `t=0.6` consistently outperforms `t=0.5` (pure average) across datasets. The interpretation is that the deeper layer `l` carries more "refined" information that better serves decoding. The norm ratio statistics confirm this: the relative magnitude `||x^l|| / ||x^{l-1}||` is typically around 0.6, meaning the SLERP parameter `t` should mirror the relative weight of the deeper layer's magnitude.

**Edge cases to handle:**

```python
def slerp_safe(u, v, t=0.6, eps=1e-8):
    """
    Safe SLERP that handles near-parallel vectors (Ω ≈ 0).
    When Ω is very small, sin(Ω) ≈ 0 and SLERP is numerically unstable.
    Fall back to linear interpolation in that case.
    """
    cos_angle = (u * v).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    Omega     = torch.acos(cos_angle)    # (B, S, 1)
    sin_Omega = torch.sin(Omega)

    # Mask for near-parallel vectors
    nearly_parallel = (sin_Omega.abs() < eps).squeeze(-1)

    # SLERP coefficients
    coeff_v = torch.sin((1 - t) * Omega) / sin_Omega.clamp(min=eps)
    coeff_u = torch.sin(t       * Omega) / sin_Omega.clamp(min=eps)

    e_slerp = coeff_v * v + coeff_u * u

    # Linear interpolation fallback for near-parallel
    e_linear = (1 - t) * v + t * u

    # Select based on mask
    e = torch.where(nearly_parallel.unsqueeze(-1), e_linear, e_slerp)

    # Renormalise (SLERP should produce unit vector, but floating point drift occurs)
    e = e / e.norm(dim=-1, keepdim=True).clamp(min=eps)

    return e
```

### 3.4 Merged Cache Layout and Restoration

**What is stored per merged layer pair `(l, l-1)` per token:**

```
Component          Shape              FP16 memory per token
─────────────────────────────────────────────────────────────
e   (direction)    (d_h,)             d_h × 2 bytes     [shared by both layers]
||x^l||            (1,)               2 bytes
||x^{l-1}||        (1,)               2 bytes
Ω   (angle)        (1,)               2 bytes            [optional, needed for some metrics]

Total per token: d_h × 2 + 6 bytes   (vs. d_h × 4 bytes for two separate caches)
Overhead: 6 extra bytes per token (scalar norm values)
```

For unmergeable tokens (retained separately): `2 × d_h × 2` bytes (full copies for both layers).

**Restoration at decode time:**

```python
def restore_kv(C, layer_role):
    """
    C: compressed cache dict
    layer_role: 'deep' for layer l, 'shallow' for layer l-1
    """
    e     = C['direction']           # (B, S, d_h) unit vector
    norm  = C[f'norm_{layer_role}']  # (B, S, 1) scalar
    R     = C[f'retain_{layer_role}']# (B, n_retain, d_h) outlier tokens
    I     = C['retain_indices']      # (n_retain,) positions of retained tokens

    # Scale direction by this layer's original norm
    x_hat = e * norm   # (B, S, d_h)

    # Restore outlier tokens at their original positions
    x_hat[:, I, :] = R

    return x_hat
```

### 3.5 Unmergeable Token Retention — Why It Matters

Experiments show that without retention (`γ=0`), performance drops significantly even at low merge rates. The reason is that certain tokens have strongly layer-specific key representations — their keys in layer `l` point in a meaningfully different direction than in layer `l-1`.

These tend to be semantically important tokens: special tokens like `<BOS>`, punctuation, and tokens that serve as "attention sink" positions (heavily attended by many other tokens). Merging these creates a corrupted representation that misleads attention patterns.

**Angular distance as a sensitivity proxy:**

```
d(x^l, x^{l-1}) = (1/π) · arccos( (x^l · x^{l-1}) / (||x^l|| · ||x^{l-1}||) )
```

This is in the range `[0, 1]`. Near 0 means very similar (safe to merge); near 1 means near-opposite (definitely don't merge).

**Retention threshold selection:**

```python
def find_retention_indices(X_l, X_prev, gamma=0.05):
    """
    X_l, X_prev: (batch, seq_len, d_head) — key or value tensors

    Returns indices of tokens to retain (NOT merge).
    gamma=0.05 retains the top 5% most dissimilar tokens.
    """
    # Compute cosine similarity per token position
    cos = F.cosine_similarity(X_l, X_prev, dim=-1)   # (B, S)
    # Angular distance
    d = (1.0 / math.pi) * torch.acos(cos.clamp(-1+1e-6, 1-1e-6))

    # Per-batch retention (threshold computed per sequence)
    B, S = d.shape
    indices_to_retain = []
    for b in range(B):
        d_b    = d[b]       # (S,)
        d_min  = d_b.min()
        d_max  = d_b.max()
        thresh = d_min + (d_max - d_min) * gamma
        retain_mask = (d_b > thresh)
        indices_to_retain.append(retain_mask.nonzero(as_tuple=True)[0])

    return indices_to_retain
```

**Memory overhead of retention:**

With `γ=0.05` and typical sequence lengths, approximately 5% of tokens are retained separately for each of the two layers. This adds `2 × 0.05 × seq_len × d_h × 2` bytes per head per layer pair — small compared to the savings from merging the other 95%.

### 3.6 Full MiniCache Pseudocode

```python
# ─── PREFILL ─────────────────────────────────────────────────────────────────

class MiniCacheLayer:
    """Manages the compressed KV cache for one merged pair of layers (l, l-1)."""

    def __init__(self, t=0.6, gamma=0.05):
        self.t     = t
        self.gamma = gamma
        self.cache = None   # filled during prefill

    def compress(self, K_l, V_l, K_prev, V_prev):
        """
        K_l, V_l:         (batch, seq_len, d_head) — deeper layer
        K_prev, V_prev:   (batch, seq_len, d_head) — shallower layer

        Stores compressed representation; returns nothing (modifies self.cache).
        """
        # Compress keys
        self.cache = {
            'K': self._merge_pair(K_l, K_prev),
            'V': self._merge_pair(V_l, V_prev),
        }

    def _merge_pair(self, X_deep, X_shallow):
        """Merge one pair of (key or value) tensors."""
        norm_deep    = X_deep.norm(dim=-1, keepdim=True)
        norm_shallow = X_shallow.norm(dim=-1, keepdim=True)
        eps = 1e-8

        u = X_deep    / norm_deep.clamp(min=eps)    # unit vector, deep layer
        v = X_shallow / norm_shallow.clamp(min=eps) # unit vector, shallow layer

        # SLERP merge
        e = slerp_safe(u, v, t=self.t)

        # Find and retain unmergeable tokens
        retain_idx = find_retention_indices(X_deep, X_shallow, self.gamma)

        return {
            'e':            e,            # (B, S, d_h) — merged direction
            'norm_deep':    norm_deep,    # (B, S, 1)
            'norm_shallow': norm_shallow, # (B, S, 1)
            'retain_deep':    X_deep[:, retain_idx, :],   # kept verbatim
            'retain_shallow': X_shallow[:, retain_idx, :],
            'retain_idx':     retain_idx,
        }

    def restore(self, layer_role='deep'):
        """
        Restore approximate KV tensors for a given layer role.
        layer_role: 'deep' (layer l) or 'shallow' (layer l-1)
        """
        result = {}
        for kv in ['K', 'V']:
            C = self.cache[kv]
            norm = C[f'norm_{layer_role}']
            e    = C['e']
            eps  = 1e-8

            # Rescale direction by this layer's norm
            x_hat = e * (norm / e.norm(dim=-1, keepdim=True).clamp(min=eps))

            # Restore retained tokens
            I = C['retain_idx']
            if len(I) > 0:
                x_hat[:, I, :] = C[f'retain_{layer_role}']

            result[kv] = x_hat

        return result['K'], result['V']

    def append_new_token(self, k_deep, v_deep, k_shallow, v_shallow):
        """Extend the cache with a new token pair (from decode step)."""
        for kv, x_d, x_s in [('K', k_deep, k_shallow), ('V', v_deep, v_shallow)]:
            C     = self.cache[kv]
            norm_d = x_d.norm(dim=-1, keepdim=True)
            norm_s = x_s.norm(dim=-1, keepdim=True)
            eps    = 1e-8

            e_new = slerp_safe(
                x_d / norm_d.clamp(min=eps),
                x_s / norm_s.clamp(min=eps),
                t=self.t
            )

            # Append to the cache (no retention check for single new token — rare outlier)
            C['e']            = torch.cat([C['e'],            e_new],  dim=1)
            C['norm_deep']    = torch.cat([C['norm_deep'],    norm_d], dim=1)
            C['norm_shallow'] = torch.cat([C['norm_shallow'], norm_s], dim=1)


def minicache_prefill(model, input_ids, start_layer=None, t=0.6, gamma=0.05):
    """
    Run full model prefill with MiniCache compression from start_layer.
    Returns: kv_full (layers 0..S-1) and kv_mini (merged pairs, layers S..L-1)
    """
    S = start_layer or (model.config.num_hidden_layers // 2)

    kv_full = {}   # Standard KV cache for shallow layers
    kv_mini = {}   # MiniCache compressed pairs for deep layers
    temp_kv = {}   # Temporary storage for odd-indexed layers (to be merged)

    hidden = model.embed(input_ids)

    for l in range(model.config.num_hidden_layers):
        # Run the layer (using appropriate cache)
        K_l, V_l, hidden = model.layers[l].forward_with_kv(hidden, past_kv=None)

        if l < S:
            # Shallow layers: standard caching
            kv_full[l] = (K_l, V_l)
        elif l % 2 == 0:
            # Even deep layer: store temporarily, will be merged with next layer
            temp_kv[l] = (K_l, V_l)
        else:
            # Odd deep layer: merge with previous even layer
            K_prev, V_prev = temp_kv[l - 1]
            mc = MiniCacheLayer(t=t, gamma=gamma)
            mc.compress(K_l, V_l, K_prev, V_prev)
            kv_mini[l] = mc   # mc serves both layers l and l-1
            del temp_kv[l - 1]

    return kv_full, kv_mini, S


# ─── DECODE STEP ─────────────────────────────────────────────────────────────

def minicache_decode_step(model, token, hidden, kv_full, kv_mini, S,
                          decode_step_count):
    """
    Single decode step with MiniCache.
    decode_step_count: used to track which layer of each pair we're on.
    """
    for l in range(model.config.num_hidden_layers):
        if l < S:
            # Shallow layer: standard attention with full cache
            K_full, V_full = kv_full[l]
            K_new, V_new, hidden = model.layers[l].forward_single_token(
                hidden, K_full, V_full
            )
            kv_full[l] = (torch.cat([K_full, K_new], dim=1),
                          torch.cat([V_full, V_new], dim=1))

        elif l % 2 == 1:
            # This is the "merged" layer — serves both l and l-1
            mc = kv_mini[l]

            # Determine role: even step uses "deep" (l), odd uses "shallow" (l-1)
            # In practice, the model always uses both layers in sequence,
            # so we restore for the current layer then restore for the previous in the same step.

            # Restore for layer l-1 (shallow)
            K_shallow, V_shallow = mc.restore(layer_role='shallow')
            K_new_s, V_new_s, hidden_l_minus_1 = model.layers[l-1].forward_single_token(
                hidden, K_shallow, V_shallow
            )

            # Restore for layer l (deep)
            K_deep, V_deep = mc.restore(layer_role='deep')
            K_new_d, V_new_d, hidden = model.layers[l].forward_single_token(
                hidden_l_minus_1, K_deep, V_deep
            )

            # Merge new tokens and extend cache
            mc.append_new_token(K_new_d, V_new_d, K_new_s, V_new_s)

    return hidden
```

### 3.7 Key Hyperparameters and Ablation Insights

| Parameter | Default | Effect of Changing |
|-----------|---------|-------------------|
| `S` (start layer) | `L // 2` | Earlier start → more compression but accuracy loss in shallow layers where similarity is low |
| `t` (SLERP param) | `0.6` | Values 0.5–0.7 work; `t=0.5` (average) noticeably worse; `t>0.7` degrades |
| `γ` (retention threshold) | `0.05` | `γ=0` (no retention) causes significant drop; `γ=0.1` marginal improvement over 0.05 |
| Quantization | 4-bit KIVI | MiniCache is orthogonal to quantization; combining gives 5.02× compression |
| LLM size | Larger is better | LLaMA-3-70B shows near-zero drop when merging 87.5% of layers |

---

## 4. CommonKV — Cross-Layer Parameter Sharing

### 4.1 Problem Framing: Why Raw KV Sharing Fails

MiniCache's core challenge is that adjacent KV caches are similar but not similar enough for direct merging at high compression rates (>20%). CommonKV diagnoses the root cause more precisely: the **weight matrices** `W^k_l` and `W^k_{l+1}` are highly dissimilar across layers, even though their inputs `x^l` and `x^{l+1}` (the hidden states) are very similar due to residual connections.

To make this concrete: if `x^l ≈ x^{l+1}` (similar hidden states) but `W^k_l ≠ W^k_{l+1}` (different weight matrices), then `K^l = x^l · W^k_l` and `K^{l+1} = x^{l+1} · W^k_{l+1}` will be dissimilar despite the similar inputs. The weight matrix is the source of divergence.

CommonKV's insight is: **fix the weight matrix dissimilarity** rather than trying to merge the resulting dissimilar caches. If layers `l` and `l+1` share the same projection matrix `A_g` (the down-projection part), then their latent caches `h^l = x^l · A_g` and `h^{l+1} = x^{l+1} · A_g` will be similar because their inputs are similar. These latent caches are then easy to merge.

This is a fundamentally different philosophy from MiniCache: instead of cleverly interpolating between dissimilar vectors, CommonKV makes the vectors similar in the first place by sharing part of the projection machinery.

### 4.2 The Cross-Layer Concatenated SVD — Step by Step

**Goal:** Find a shared matrix `A_g` and per-layer matrices `B^l_k, B^l_v` such that:

```
W^l_k  ≈ A_g · B^l_k     for all l in group g
W^l_v  ≈ A_g · B^l_v     for all l in group g
```

**Why concatenation?** If you decompose each layer separately, you get different left singular vectors `U^l`, and there's no reason for them to be the same — this is just standard Palu/ASVD. To force a shared `A_g`, you must decompose the layers **jointly**.

The key mathematical insight is that SVD of a concatenated matrix gives a shared left factor:

```
W_concat = [W^l_k | W^l_v | W^{l+1}_k | W^{l+1}_v | ...]   # horizontal concat
         ≈ A_g · [B^l_k | B^l_v | B^{l+1}_k | B^{l+1}_v]
           ↑              ↑
           shared         per-layer slices
```

Why? Because `U_r` in the SVD of `W_concat` is the best rank-`r` left-subspace for the entire concatenated matrix. Each column of `A_g = U_r · sqrt(Σ_r)` captures a direction in `d_model`-dimensional space that is relevant across all the concatenated weight matrices. The per-layer `B^l_k` slices then tell us "how to mix the shared directions to reconstruct layer `l`'s key projection."

```python
def cross_layer_svd(model, layer_group, rank, d_h):
    """
    layer_group: list of layer indices in the group (e.g., [4, 5, 6, 7])
    rank: SVD rank for the shared component
    d_h: per-head dimension (after accounting for GQA)

    Returns:
      A_g:  (d_model, rank) — shared down-projection
      B_dict: {(l, 'k'): (rank, d_h_kv), (l, 'v'): (rank, d_h_kv), ...}
    """
    # Collect all K and V weight matrices in the group
    W_list = []
    for l in layer_group:
        W_k = model.layers[l].self_attn.k_proj.weight.T  # (d_model, d_kv)
        W_v = model.layers[l].self_attn.v_proj.weight.T  # (d_model, d_kv)
        W_list.extend([W_k, W_v])

    # Concatenate horizontally: (d_model, 2 * group_size * d_kv)
    W_concat = torch.cat(W_list, dim=1)
    d_model, d_concat = W_concat.shape

    # Joint SVD
    # Note: for large matrices, use randomised SVD for efficiency
    if d_model > 4096:
        U, S, Vt = torch.svd_lowrank(W_concat, q=rank + 10)  # oversampled
        U, S, Vt = U[:, :rank], S[:rank], Vt[:, :rank].T
    else:
        U, S, Vt = torch.linalg.svd(W_concat, full_matrices=False)
        U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    sqrt_S = S.sqrt()
    A_g = U * sqrt_S.unsqueeze(0)         # (d_model, rank) — shared

    # Full B_g: (rank, 2 * group_size * d_kv)
    B_g = sqrt_S.unsqueeze(1) * Vt        # (rank, d_concat)

    # Slice B_g into per-layer, per-{K,V} components
    B_dict = {}
    d_kv = W_list[0].shape[1]
    for i, l in enumerate(layer_group):
        B_dict[(l, 'k')] = B_g[:, (2*i)*d_kv     : (2*i+1)*d_kv]
        B_dict[(l, 'v')] = B_g[:, (2*i+1)*d_kv   : (2*i+2)*d_kv]

    return A_g, B_dict
```

**GQA-specific consideration:** In GQA models, `d_kv = n_kv_heads × d_head` is smaller than `d_model` by a factor of `n_q_heads / n_kv_heads`. For LLaMA-3-8B with 8 KV heads and 32 query heads, `d_kv = 8 × 128 = 1024` while `d_model = 4096`. The concatenated matrix has shape `(4096, 2 × group_size × 1024)`. At group_size=4, this is `(4096, 8192)`, a "landscape" matrix. The effective rank of this matrix may be limited, requiring careful rank selection. CommonKV concatenates both K and V within each layer precisely to increase the effective column count and make the matrix more "square."

### 4.3 Latent Cache Construction and Cross-Layer Merging

**Why the latent cache is more mergeable:**

Standard K cache: `K^l = x^l · W^k_l` — different projections of similar inputs → dissimilar

CommonKV latent: `h^l = x^l · A_g` — **same projection** of similar inputs → similar

Since `x^l ≈ x^{l+1}` (residual connections) and `A_g` is the same for both layers, `h^l ≈ h^{l+1}`. Figure 2 in the paper shows that the cosine similarity of `h^l` across adjacent layers is dramatically higher than that of the raw K/V caches.

```python
def prefill_and_build_latent_cache(model, input_ids, shared_weights, layer_groups):
    """
    Returns latent_cache[l] = h^l = x^l @ A_g  for each layer l.
    """
    latent_cache = {}
    hidden = model.embed(input_ids)

    for l in range(model.config.num_hidden_layers):
        g_idx = get_group_index(l, layer_groups)
        A_g   = shared_weights[g_idx]

        # Run the full attention layer to get hidden state (for residual etc.)
        # But compute the latent cache from the layer's input, not output
        x_l = hidden.clone()   # input to layer l (before layer computation)

        # Layer computation (for residual connection)
        hidden = model.layers[l].forward(hidden, past_kv=None)

        # Compute latent: project the layer's input with shared A_g
        h_l = x_l @ A_g   # (batch, seq_len, rank)
        latent_cache[l] = h_l

    return latent_cache
```

**Fisher-weighted merge:**

After collecting all `h^l` for layers in group `g`, merge them:

```python
def merge_group_latents(latent_cache, layer_group, fisher_scores):
    """
    Merge latent caches for a group using Fisher information weights.

    fisher_scores: {l: F(W^l_k) + F(W^l_v) for l in group}
    Higher Fisher score → this layer's information is more important → higher weight
    """
    hs     = [latent_cache[l] for l in layer_group]         # list of (B, S, rank)
    scores = [fisher_scores[l] for l in layer_group]        # list of scalars

    weights = torch.tensor(scores, dtype=torch.float32)
    weights = weights / weights.sum()                        # normalise to sum to 1

    # Weighted average: C is a convex combination of the latent caches
    merged = sum(w * h for w, h in zip(weights.tolist(), hs))  # (B, S, rank)

    return merged
```

**When NOT to merge (adaptive budget allocation):**

Not all layer groups are equally similar. Shallow layers (near layer 0) have low KV cache similarity because the residual updates are larger — the model is still building up the representation. Merging these groups would cause significant accuracy loss.

CommonKV computes a per-group cosine similarity score and only merges the groups above a threshold:

```python
def compute_group_similarity(latent_cache, layer_group):
    """
    Estimate mergeability by computing cosine similarity between
    the FIRST and LAST layers in the group.
    If the group is internally similar, first≈last and the group is safe to merge.
    """
    h_first = latent_cache[layer_group[0]]    # (B, S, rank)
    h_last  = latent_cache[layer_group[-1]]   # (B, S, rank)

    # Average over batch and sequence
    sim = F.cosine_similarity(
        h_first.reshape(-1, h_first.shape[-1]),
        h_last.reshape(-1, h_last.shape[-1]),
        dim=-1
    ).mean().item()

    return sim

def adaptive_merge_decision(latent_cache, layer_groups, target_compression):
    """
    Decide which groups to merge to meet the target compression ratio.

    target_compression: e.g., 0.5 = reduce KV memory by 50%
    Since each merged group of size s saves (s-1)/s of that group's memory,
    we need to merge enough groups to reach the target.
    """
    sim_scores = {
        g_idx: compute_group_similarity(latent_cache, group)
        for g_idx, group in enumerate(layer_groups)
    }

    # Sort by similarity (merge most similar first — lowest accuracy impact)
    sorted_groups = sorted(sim_scores.items(), key=lambda x: -x[1])

    # Greedily merge until target compression is met
    total_layers   = sum(len(g) for g in layer_groups)
    merged_layers  = 0
    groups_to_merge = set()

    for g_idx, sim in sorted_groups:
        group_size = len(layer_groups[g_idx])
        # Merging this group saves (group_size - 1) / total_layers of KV memory
        savings = (group_size - 1) / total_layers
        if (merged_layers + group_size - 1) / total_layers <= target_compression:
            groups_to_merge.add(g_idx)
            merged_layers += group_size - 1

    return groups_to_merge, sim_scores
```

### 4.4 Inference: Decode Step with CommonKV

```python
def commonkv_decode_step(x_new, layer_id, shared_weights, B_dict,
                          merged_cache, unmerged_cache, groups_to_merge,
                          layer_groups, pos_ids, W_o_fused):
    """
    x_new: (batch, 1, d_model)
    """
    g_idx = get_group_index(layer_id, layer_groups)
    A_g   = shared_weights[g_idx]

    # Project new token to latent space
    h_new = x_new @ A_g   # (batch, 1, rank)

    # Determine which cache to extend
    if g_idx in groups_to_merge:
        cache = merged_cache[g_idx]
    else:
        cache = unmerged_cache[layer_id]

    cache = torch.cat([cache, h_new], dim=1)   # extend along sequence

    if g_idx in groups_to_merge:
        merged_cache[g_idx] = cache
    else:
        unmerged_cache[layer_id] = cache

    H = cache   # (batch, seq_len, rank)

    # Reconstruct keys with RoPE for attention score computation
    B_k = B_dict[(layer_id, 'k')]            # (rank, d_kv)
    K   = H @ B_k                            # (batch, seq_len, d_kv)

    # Pre-compute shared RoPE (same for all layers in group — optimization)
    rope_cos, rope_sin = get_shared_rope(pos_ids, layer_id, g_idx)
    K_rope = apply_rope(K, rope_cos, rope_sin)

    # Query (standard)
    W_q  = get_query_weight(layer_id)
    q    = x_new @ W_q
    q    = apply_rope(q, rope_cos, rope_sin)

    # Attention scores
    d_head = q.shape[-1] // model.config.num_key_value_heads
    scores = q @ K_rope.transpose(-1, -2) / math.sqrt(d_head)
    attn   = F.softmax(scores, dim=-1)

    # Value output (fused: B^v already absorbed into W^o)
    # W_o_fused[(layer_id)] = B_v @ W_o  (computed offline)
    out = attn @ H @ W_o_fused[layer_id]   # (batch, 1, d_model)

    return out
```

### 4.5 Key Hyperparameters and Ablation Insights

| Parameter | Default | Effect of Changing |
|-----------|---------|-------------------|
| `group_size` | 4 | Larger groups → more potential compression but less flexibility in budget allocation |
| `SVD rank` | `0.7 × d_hidden` | For 0.3 and 0.5 compression. `0.6 × d_hidden` for 0.6 compression |
| Target compression | 0.3–0.6 | Beyond 0.6, all methods degrade significantly |
| Merge strategy | Fisher-weighted | Simple averaging is 1–2 points worse on LongBench |
| Static vs. dynamic | Dynamic | Dynamic allocation adds ~5% prefill overhead but saves 3–5 points at 0.5 compression |

---

## 5. ThinKV — Thought-Adaptive Hybrid Compression

### 5.1 Problem Framing: Why Standard Methods Fail on Reasoning Models

LRMs (Large Reasoning Models) generate chains of thought that are fundamentally different from standard LLM outputs in two key ways:

**1. Length:** A single AIME problem might require 9,000 tokens of reasoning. For a GPT-OSS-20B model with batch size 32, this creates a KV cache of ~50 GB — larger than the model weights. Standard methods that were designed for 512-token outputs simply weren't validated at this scale.

**2. Structure:** The CoT output is not a uniform stream of tokens. It alternates between phases: systematic reasoning (thinking through a problem), calculations or code execution, and meta-cognitive transitions ("Wait, that's wrong. Let me reconsider."). These phases have very different importance for the final answer and very different attention patterns.

**Why quantization alone fails for LRMs:** Applying uniform quantization (e.g., KIVI's 2-bit) to all KV tokens degrades accuracy by reducing the precision of tokens mid-reasoning. Worse, ThinKV observes an unexpected side effect: quantization **inflates generation length** by up to 5× — the model, uncertain about its own compressed reasoning context, generates more verification steps. This erases the memory savings (more tokens generated = larger final KV cache) while simultaneously degrading accuracy.

**Why eviction alone fails at high compression:** Standard eviction schemes (H2O, R-KV) select which tokens to keep based on recency or attention score patterns. At extreme compression (<5% budget), these heuristics fail because they operate token-by-token without understanding that entire reasoning segments might be safely abandoned after a trajectory change.

**ThinKV's answer:** Combine both, but make both thought-aware.

### 5.2 The Three Thought Types — Empirical Basis

The tri-modal attention sparsity distribution is the empirical foundation of the entire ThinKV system. Understanding it is essential.

**Attention sparsity definition:** At decode step `t`, layer `l`:

```
sparsity(l, t) = fraction of past tokens with attention score < 1% of row-max
               = |{i : a_{ti} < 0.01 × max_j(a_{tj})}| / t
```

Sparse attention = the model is attending to very few past tokens (focused).
Dense attention  = the model is attending broadly to many past tokens (scanning).

**Why three modes emerge:**

```
Transition thoughts (T) — highest sparsity (~70-80%):
  "Wait... Actually... Hmm..."
  The model is reassessing. It briefly checks a few key anchors (what was the problem?
  what did I just conclude?) but doesn't need deep context from recent reasoning steps.
  This high sparsity means the KV cache of recent thoughts contributes little.
  → Safe to compress aggressively.

Reasoning thoughts (R) — medium sparsity (~40-60%):
  "Let me think about this step by step... So if X, then Y..."
  The model is building a logical chain. It references its recent reasoning but
  also pulls in relevant earlier context (problem statement, established facts).
  → Moderate compression.

Execution thoughts (E) — lowest sparsity (~10-30%):
  "Calculate: (1 - 25/(36*a^2)) / (1 - 25/(36*a^2))..."
  "for i in range(n): result += array[i]..."
  The model is in a tight loop: doing arithmetic or generating code where every
  preceding token in the current execution block is directly relevant.
  Dense attention because you can't skip steps in a calculation.
  → Least compression.
```

This hierarchy is not hardcoded — it emerges from the model's behavior. The calibration process discovers these thresholds automatically from attention statistics, using kernel density estimation to find the natural modes in the distribution.

### 5.3 Thought Decomposition — The Calibration Algorithm

```python
def calibrate_thresholds(model, calibration_prompts, n_thoughts=3, n_cal_layers=4):
    """
    Full offline calibration to find sparsity thresholds that separate the
    three thought types.

    Returns:
      L_star:  list of n_cal_layers layer indices (best for classification)
      theta:   list of n_thoughts-1 sparsity thresholds
    """
    from scipy.stats import gaussian_kde
    from scipy.signal import argrelmin, argrelmax
    import numpy as np

    # Step 1: Collect sparsity measurements across all layers and decode steps
    sparsity_data = defaultdict(lambda: defaultdict(list))
    # sparsity_data[prompt_idx][layer_idx] = [sparsity at step 0, step 1, ...]

    for p_idx, prompt in enumerate(calibration_prompts):
        print(f"Calibrating on prompt {p_idx+1}/{len(calibration_prompts)}")
        tokens = model.generate(prompt, max_length=32768, return_attention=True)

        for step, (token, attn) in enumerate(zip(tokens, attention_at_each_step)):
            for l in range(model.config.num_hidden_layers):
                attn_l    = attn[l]   # (n_heads, 1, step) attention scores for new token
                # Average over heads, then compute sparsity
                attn_avg  = attn_l.mean(dim=0).squeeze(0)   # (step,) — scores over past tokens
                row_max   = attn_avg.max()
                threshold = 0.01 * row_max
                sparsity  = (attn_avg < threshold).float().mean().item()
                sparsity_data[p_idx][l].append(sparsity)

    # Step 2: For each layer, fit KDE and check if tri-modal distribution exists
    layer_quality   = {}   # layer_idx → (n_modes, thresholds)
    all_thresholds  = defaultdict(list)

    for l in range(model.config.num_hidden_layers):
        # Gather all sparsity values for this layer across all prompts
        all_vals = []
        for p_idx in range(len(calibration_prompts)):
            all_vals.extend(sparsity_data[p_idx][l])

        if len(all_vals) < 100:
            continue

        # Fit KDE
        kde = gaussian_kde(all_vals, bw_method='scott')
        xs  = np.linspace(0, 1, 1000)
        ys  = kde(xs)

        # Find modes (local maxima)
        peaks   = argrelmax(ys, order=30)[0]
        valleys = argrelmin(ys, order=30)[0]

        # Check if we have (at least) n_thoughts modes
        if len(peaks) >= n_thoughts:
            # Find valleys between the first n_thoughts peaks
            relevant_peaks   = sorted(peaks)[:n_thoughts]
            relevant_valleys = [v for v in valleys
                                if xs[relevant_peaks[0]] < xs[v] < xs[relevant_peaks[-1]]]

            if len(relevant_valleys) >= n_thoughts - 1:
                thresholds = sorted([xs[v] for v in relevant_valleys[:n_thoughts-1]])
                layer_quality[l] = {
                    'n_modes': len(peaks),
                    'thresholds': thresholds,
                    'quality': ys[peaks].min()   # minimum peak height = weakest mode strength
                }
                for i, t in enumerate(thresholds):
                    all_thresholds[i].append(t)

    # Step 3: Select the best n_cal_layers layers
    # Rank by: (1) has exactly n_thoughts modes, (2) strong peak separation
    valid_layers = sorted(
        [(l, info) for l, info in layer_quality.items()
         if info['n_modes'] == n_thoughts],
        key=lambda x: -x[1]['quality']
    )
    L_star = [l for l, _ in valid_layers[:n_cal_layers]]

    # Step 4: Average thresholds across selected layers and all prompts
    theta = []
    for i in range(n_thoughts - 1):
        theta.append(float(np.mean(all_thresholds[i])))

    return sorted(L_star), sorted(theta)
```

### 5.4 Thought Classification at Decode Time

```python
class ThoughtClassifier:
    """
    Lightweight online thought classifier using pre-calibrated thresholds.
    Only updates every tau=128 steps to minimise overhead.
    """

    def __init__(self, model, L_star, theta, tau=128):
        self.L_star    = L_star          # calibrated layer indices
        self.theta     = theta           # [theta_low, theta_high]
        self.tau       = tau             # refresh interval (tokens)
        self.current   = 2              # start as Reasoning
        self.prev      = 2
        self.step      = 0

    def update(self, attention_maps):
        """
        attention_maps: dict {layer_idx: attention_tensor}
        Called once per generated token.
        Returns current thought type (only changes every tau steps).
        """
        self.step += 1

        if self.step % self.tau != 0:
            return self.current   # no change

        # Compute average sparsity over calibrated layers
        sparsities = []
        for l in self.L_star:
            attn = attention_maps[l]   # (n_heads, 1, seq_len)
            attn_avg = attn.mean(dim=0).squeeze(0)   # (seq_len,)
            row_max  = attn_avg.max()
            if row_max < 1e-8:
                sparsities.append(1.0)   # degenerate case
                continue
            sp = (attn_avg < 0.01 * row_max).float().mean().item()
            sparsities.append(sp)

        avg_sp = float(sum(sparsities) / len(sparsities))

        # Classify: theta[0] < theta[1] are the two thresholds
        self.prev = self.current
        if avg_sp > self.theta[1]:
            self.current = 0   # Transition (T) — highest sparsity
        elif avg_sp > self.theta[0]:
            self.current = 2   # Reasoning (R) — medium sparsity
        else:
            self.current = 1   # Execution (E) — lowest sparsity

        return self.current

    @property
    def is_trajectory_change(self):
        """True if we just entered a Transition thought from a non-Transition."""
        return self.prev != 0 and self.current == 0
```

### 5.5 TBQ — Thought-Adaptive Quantization: Implementation Details

**Data format details:**

```
FP8 E4M3  (8-bit):  1 sign, 4 exponent, 3 mantissa — wide dynamic range
  Used for Reasoning (R) tokens — most sensitive to compression
  Scale: per-tensor FP32 scalar (simple and fast)

NVFP4     (4-bit):  1 sign, 2 exponent, 1 mantissa — limited range, needs group scaling
  Used for Execution (E) and (optionally) Reasoning tokens
  Scale: per-group FP8 E4M3 (group_size=16) — one FP8 value per 16 elements

Ternary   (2-bit):  {-1, 0, +1} with scale — extreme compression
  Used for Transition (T) tokens — least sensitive
  Scale: per-group FP8 E4M3 (group_size=16)
  Storage: 2 tokens packed into 4 bits (two 2-bit values per byte)
  Note: one of the 4 possible 2-bit codes is redundant (-0 → maps to 0)
```

```python
class AdaptiveQuantBuffer:
    """
    Buffer that accumulates tokens until group_size is reached, then quantizes.
    The key invariant: all tokens in a group belong to the same thought type.
    If thought type changes mid-group, flush and start a new group.
    """

    def __init__(self, group_size=16):
        self.group_size   = group_size
        self.buf_k        = []    # list of (1, d_head) tensors
        self.buf_v        = []
        self.buf_thought  = []   # thought types in current buffer
        self.quantized_K  = []   # accumulated quantized groups
        self.quantized_V  = []

    def push(self, k_new, v_new, thought_type):
        """Add one token's KV to the buffer. Returns list of completed quantized groups."""
        completed = []

        # Check if thought type matches buffer
        if self.buf_thought and self.buf_thought[-1] != thought_type:
            # Thought type changed — flush buffer regardless of size
            if self.buf_k:
                completed.extend(self._flush_partial(self.buf_thought[-1]))

        self.buf_k.append(k_new)
        self.buf_v.append(v_new)
        self.buf_thought.append(thought_type)

        # Check if buffer is full
        if len(self.buf_k) == self.group_size:
            completed.extend(self._flush_full(thought_type))

        return completed

    def _flush_full(self, thought_type):
        """Quantize a full group of group_size tokens."""
        K_group = torch.cat(self.buf_k, dim=1)   # (batch, group_size, d_head)
        V_group = torch.cat(self.buf_v, dim=1)

        K_q, V_q = self._quantize(K_group, V_group, thought_type)

        self.buf_k.clear()
        self.buf_v.clear()
        self.buf_thought.clear()

        return [(K_q, V_q, thought_type)]

    def _flush_partial(self, thought_type):
        """Flush a partial buffer (thought type change). Store in FP16 until next group."""
        # Option 1: Pad with zeros to group_size and quantize (wastes some precision)
        # Option 2: Keep in FP16 as a "residual" (simpler, slight memory overhead)
        # ThinKV uses option 2 for simplicity
        K_partial = torch.cat(self.buf_k, dim=1)
        V_partial = torch.cat(self.buf_v, dim=1)
        self.buf_k.clear()
        self.buf_v.clear()
        self.buf_thought.clear()
        # Store as FP16 (marked as partial)
        return [('fp16', K_partial, V_partial, thought_type)]

    def _quantize(self, K, V, thought_type):
        """Quantize K, V based on thought type."""
        if thought_type == 2:   # Reasoning → 8-bit FP8
            return quantize_fp8(K), quantize_fp8(V)
        elif thought_type == 1: # Execution → 4-bit NVFP4
            return quantize_nvfp4(K, self.group_size), quantize_nvfp4(V, self.group_size)
        else:                   # Transition → 2-bit Ternary
            return quantize_ternary(K, self.group_size), quantize_ternary(V, self.group_size)

def quantize_ternary(X, group_size):
    """
    Ternary quantization: {-1, 0, +1} with per-group FP8 scale.
    2 bits per element.
    """
    B, S, D = X.shape
    assert S % group_size == 0

    X_grouped = X.reshape(B, S // group_size, group_size, D)

    # Per-group scale: max absolute value
    scale = X_grouped.abs().amax(dim=(-2, -1), keepdim=True)   # (B, S//g, 1, 1)
    scale = scale.clamp(min=1e-8).to(torch.float8_e4m3fn)

    X_norm = X_grouped / scale.float()

    # Threshold at 0.5/scale for ternary (round to {-1, 0, 1})
    X_ternary = X_norm.sign() * (X_norm.abs() >= 0.5).float()

    # Pack two ternary values into one 4-bit slot
    # Mapping: -1 → 0b01, 0 → 0b10, 1 → 0b11 (avoids -0 ambiguity)
    code = (X_ternary + 2).to(torch.uint8)   # {1, 2, 3}

    # Pack pairs: high 2 bits = element 2i, low 2 bits = element 2i+1
    code_even = code[..., 0::2]   # (..., D//2)
    code_odd  = code[..., 1::2]   # (..., D//2)
    packed = (code_even << 2) | code_odd    # 4 bits per pair

    return {'packed': packed, 'scale': scale, 'shape': X.shape}
```

### 5.6 TBE — Thought-Adaptive Eviction: The Retention Schedule

**The key insight behind the retention schedule `R = {64, 32, 16, 8, 4}`:**

Every time a Transition thought is encountered, it signals that the model has decided to change direction. Prior reasoning segments become progressively less relevant to the eventual answer. The schedule encodes this: the first time a segment is selected for eviction (first transition after it was created), keep 64 tokens. Second time (second transition encountered), keep 32. And so on down to a minimum of 4 tokens.

**Why keep a minimum of 4 tokens and never evict completely?**

The paper demonstrates (§5.3, Figure 10a) that setting `min_R = 0` (complete eviction) causes the model to enter an **infinite reasoning loop** — it keeps revisiting the same dead-end approach because it has no memory of having tried and abandoned it. The 4 retained tokens serve as a "breadcrumb" that tells the model "this direction has been explored." These minimum tokens are chosen by K-means clustering on the segment's key embeddings, selecting the most representative positions.

```python
class TBEEvictionManager:
    """
    Manages thought-based eviction across all segments in the KV cache.
    """

    RETENTION_SCHEDULE = [64, 32, 16, 8, 4]
    MIN_RETENTION      = 4

    def __init__(self, cache_budget):
        self.budget   = cache_budget
        self.segments = []   # list of SegmentState

    class SegmentState:
        def __init__(self, thought_type, start_step, token_indices):
            self.thought_type    = thought_type    # 0=T, 1=E, 2=R
            self.start_step      = start_step      # when this segment started
            self.token_indices   = list(token_indices)  # which tokens in the KV cache
            self.eviction_count  = 0               # how many times evicted

        @property
        def size(self):
            return len(self.token_indices)

    def on_new_segment(self, thought_type, step):
        """Start tracking a new thought segment."""
        seg = self.SegmentState(thought_type, step, [])
        self.segments.append(seg)

    def add_token_to_current_segment(self, token_cache_idx):
        """Register a token as belonging to the current segment."""
        if self.segments:
            self.segments[-1].token_indices.append(token_cache_idx)

    def on_transition_thought(self, key_embeddings, step):
        """
        Called when a Transition thought is detected.
        Progressively evict ALL preceding segments.
        Returns list of (segment_idx, evicted_indices) for the CT kernel to process.
        """
        eviction_instructions = []

        for seg in self.segments[:-1]:   # all segments except the brand-new current one
            n          = seg.eviction_count
            target_k   = self.RETENTION_SCHEDULE[n] if n < len(self.RETENTION_SCHEDULE) \
                         else self.MIN_RETENTION

            if seg.size > target_k:
                # Select representative tokens via K-means on post-RoPE keys
                tokens_to_keep   = self._kmeans_select(
                    seg.token_indices, key_embeddings, k=target_k
                )
                tokens_to_evict  = [i for i in seg.token_indices if i not in set(tokens_to_keep)]

                seg.token_indices = tokens_to_keep
                seg.eviction_count += 1

                eviction_instructions.append((seg, tokens_to_evict))

        return eviction_instructions

    def on_budget_exceeded(self, key_embeddings):
        """
        Fallback eviction when budget is exceeded but no transition occurred.
        Find the oldest, least important segment and evict it.
        """
        # Find eligible segments (not at minimum retention)
        eligible = [seg for seg in self.segments[:-1]   # exclude current
                    if seg.size > self.MIN_RETENTION]

        if not eligible:
            return []

        # Priority: Transition (0) < Execution (1) < Reasoning (2)
        # Among same type, prefer older segments
        target = min(eligible, key=lambda s: (s.thought_type, -s.start_step))

        n        = target.eviction_count
        target_k = self.RETENTION_SCHEDULE[n] if n < len(self.RETENTION_SCHEDULE) \
                   else self.MIN_RETENTION

        if target.size > target_k:
            kept     = self._kmeans_select(target.token_indices, key_embeddings, k=target_k)
            evicted  = [i for i in target.token_indices if i not in set(kept)]
            target.token_indices = kept
            target.eviction_count += 1
            return [(target, evicted)]

        return []

    def _kmeans_select(self, token_indices, key_embeddings, k):
        """
        Select k representative token indices from a segment using K-means.

        key_embeddings: (seq_len, d_head) — post-RoPE key embeddings
        Clusters the keys of the segment's tokens, picks the closest to each centroid.

        Important: Only cluster within the segment (not the whole sequence).
        This avoids RoPE drift issues (keys within 128 tokens have minimal drift).
        """
        if len(token_indices) <= k:
            return token_indices

        keys = key_embeddings[token_indices]   # (segment_len, d_head)

        # K-means (GPU-accelerated)
        centroids, assignments = gpu_kmeans(keys, k=k, n_iter=10)

        # For each cluster, find the closest actual token (not the centroid itself)
        selected = []
        for c_idx in range(k):
            cluster_mask    = (assignments == c_idx)
            if cluster_mask.any():
                cluster_keys = keys[cluster_mask]
                centroid     = centroids[c_idx]
                dists        = ((cluster_keys - centroid) ** 2).sum(dim=-1)
                best_in_cluster = cluster_mask.nonzero()[dists.argmin()].item()
                selected.append(token_indices[best_in_cluster])

        return selected
```

### 5.7 Continuous Thinking (CT) — The Memory Manager

**Why gather-based compaction is catastrophic for throughput:**

When tokens are evicted from arbitrary positions in the KV cache, the remaining tokens are scattered (fragmented). Attention kernels expect contiguous memory. The standard solution — gather-based compaction — physically moves surviving tokens to fill gaps.

For a batch of 256 sequences, each evicting ~1000 tokens per decode step: the gather kernel must move `256 × n_layers × n_heads × 1000 × d_head` elements = tens of GB of HBM writes per step. This saturates the memory bus and causes 37× TPOT slowdown (Table 4 in the paper).

**CT's in-place reuse approach:**

Instead of moving data, CT uses bitmask-driven slot management:

```
Physical memory block (size=8):
  Slot 0: [token A, type=R]  eviction_mask=0 (active)
  Slot 1: [token B, type=R]  eviction_mask=0 (active)
  Slot 2: [token C, type=R]  eviction_mask=1 (evicted — slot available)
  Slot 3: [token D, type=R]  eviction_mask=0 (active)
  ...

When a new Reasoning token arrives:
  CT finds slot 2 in an existing R-type block.
  Writes new token directly into slot 2.
  Sets eviction_mask[2] = 0 (now active again).
  Records new token's position in segment_masks.
```

The attention kernel computes attention over all non-evicted slots. Since attention is **permutation-invariant** (softmax over any ordering of KV pairs gives the same output), the physical ordering of tokens in memory doesn't need to change — only the masking matters.

**Why thought-aware paging matters:**

CT only reuses slots within blocks of the **same thought type**. This prevents mixing quantization formats: a slot evicted from a Reasoning (FP8) block is only reused for a new Reasoning token. Execution (NVFP4) tokens go into Execution blocks. This ensures the dequantization kernels always process uniform data.

### 5.8 Key Hyperparameters and Ablation Insights

| Parameter | Default | Effect |
|-----------|---------|--------|
| `|T|` (thought types) | 3 | 2 or fewer → no Transition detected → eviction degrades to budget-only (worse) |
| `|L*|` (calibration layers) | 4 | More layers → noisier average; fewer → less signal. 4 is optimal |
| `τ` (refresh interval) | 128 | Shorter → better accuracy but higher overhead. 128 is the knee of the curve |
| `min_R` | 4 | 0 causes infinite loops; 8 is slightly more stable but costs memory |
| Precision R | 4-bit | Using 8-bit for R tokens provides marginal accuracy gain (<0.2% on AIME) |
| Precision E | 4-bit | Using 8-bit for E tokens provides no measurable gain |
| Precision T | 2-bit | Can't go lower (1-bit ternary doesn't exist in hardware) |
| Block size | 8 | 16 causes fragmentation overhead; 4 causes excessive block allocation |
| Max gen length | 32K | Papers cap here for fair comparison; longer sequences stress all methods more |

---

## 6. Method Comparison & Interactions

### 6.1 Compression Axis Summary

| Method | What It Compresses | What It Leaves Alone |
|--------|-------------------|---------------------|
| Palu | Hidden dimension (per-channel) | Token count, bit precision |
| MiniCache | Layer depth (merge pairs) | Token count, hidden dim, bit precision |
| CommonKV | Both hidden dim + layer depth | Token count, bit precision |
| ThinKV | Token count (eviction) + bit precision (quant) | Hidden dimension |

**These axes are orthogonal.** This means any combination can in principle be stacked:

```
Palu + MiniCache: low-rank latent caches, merged across layers
  → Each merged pair stores one low-rank h instead of two full K/V matrices

Palu + Quantization (what Palu already does): low-rank latent, then quantize h
  → The Hadamard trick makes this work well

CommonKV + SnapKV/Quantization (what CommonKV experiments with): shared latent + eviction + quant
  → 98% compression reported at minimal performance loss

ThinKV + Palu: thought-adaptive eviction + low-rank projection
  → Theoretically valid; not benchmarked in papers
```

### 6.2 When to Use Which

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Standard LLM, long input context | Palu or CommonKV | Eviction risks removing important context tokens |
| Standard LLM, moderate context, throughput focus | MiniCache + quantization | Simple, large compression factor |
| Reasoning model (chain-of-thought) | ThinKV | Reasoning structure is exploitable; eviction of thought segments is principled |
| Need to add compression to existing model | Palu (training-free, no model change) | Only offline SVD required |
| Highest compression target (>90%) | CommonKV + quantization | Stacks multiple orthogonal axes |

### 6.3 Failure Mode Analysis

| Method | Common Failure Mode | How to Diagnose | Fix |
|--------|-------------------|-----------------|-----|
| Palu | Perplexity spike at high compression | Ppl diverges from baseline at same rank | Reduce compression or use rank search |
| Palu | RoPE kernel latency overhead | TPOT increases vs. non-RoPE models | Use tiled Triton kernel; profile reconstruction FLOPs |
| MiniCache | Drop in tasks requiring cross-layer feature diversity | Low accuracy on multi-step reasoning | Increase `S` (start merging later) or reduce `γ` |
| MiniCache | SLERP NaN/Inf | Near-parallel vectors (cos≈1) → sin(Ω)≈0 | Add epsilon clamp on cos before arccos |
| CommonKV | GQA compression rate lower than MHA | Rank deficiency in small W_k/W_v | Concat K+V; reduce SVD rank ratio |
| CommonKV | Prefill latency spike | Online SVD (xKV-style) is slow | Use offline SVD (CommonKV's approach); 6× faster |
| ThinKV | Model enters reasoning loop | Transition thoughts are always evicted completely | Set `min_R >= 4` |
| ThinKV | Thought classification noise | τ too small (too frequent updates) | Use τ=128; average over L*=4 layers |
| ThinKV | Low throughput at small batch | Sequential gather fallback | Ensure CT kernel is active; check block_size=8 |

---

## 7. Benchmark Setup Checklist

### 7.1 Model Variants to Test

| Method | Tested on (from papers) |
|--------|------------------------|
| Palu | Llama-2-7B, Llama-3-8B-Instruct, Mistral-7B, LongChat-7B |
| MiniCache | LLaMA-2-7B/13B, LLaMA-3-8B/70B, Mistral-7B/Instruct, Mixtral-8x7B, Phi-3-Mini |
| CommonKV | Llama3.1-8B-Instruct, Mistral-v0.2-7B-Instruct |
| ThinKV | DeepSeek-R1-Distill-Llama-8B/70B, GPT-OSS-20B, AceReason-14B |

### 7.2 Evaluation Benchmarks

| Benchmark | Metrics | Used by |
|-----------|---------|---------|
| WikiText-2 (seqlen=2048/4096) | Perplexity ↓ | Palu |
| C4 | Perplexity ↓ | Palu |
| LM-Eval 6 zero-shot tasks | Avg. accuracy ↑ | Palu |
| LongBench (16 tasks) | Avg. score ↑ | Palu, CommonKV, MiniCache |
| Ruler | Avg. score ↑ | CommonKV |
| AIME 2024+2025 (30 prompts) | pass@1 ↑ | ThinKV |
| LiveCodeBench | pass@1 ↑ | ThinKV |
| MATH-500 | pass@1 ↑ | ThinKV |
| GSM8K | Exact Match ↑ | MiniCache, ThinKV |
| ShareGPT (throughput) | tokens/sec ↑ | MiniCache |

### 7.3 Compression Ratio Normalisation

```python
def equivalent_memory_fraction(method, params):
    """
    Returns the fraction of ORIGINAL FP16 KV memory used.
    Use this to compare across methods on equal footing.
    """
    if method == 'palu':
        # (1 - rank_compression) × (quant_bits / 16)
        return (1.0 - params['compression']) * (params.get('bits', 16) / 16.0)

    elif method == 'minicache':
        # Merging from S=L/2: layers 0..S-1 full, layers S..L-1 halved
        L          = params['total_layers']
        S          = params.get('start_layer', L // 2)
        # Merged portion: each pair uses 1 cache instead of 2 (plus small overhead)
        merged_fraction   = (L - S) / L        # fraction of layers that are merged
        unmerged_fraction = S / L
        layer_ratio = unmerged_fraction + merged_fraction * 0.55   # 0.55 accounts for overhead
        return layer_ratio * (params.get('bits', 16) / 16.0)

    elif method == 'commonkv':
        # 1 - compression_ratio (directly reported as memory reduction)
        return 1.0 - params['compression_ratio']

    elif method == 'thinkv':
        # token_budget_fraction × avg_bits / 16
        avg_bits = params.get('avg_bits', 3.5)   # typically 3.2–3.8 from paper
        return params['token_budget_fraction'] * (avg_bits / 16.0)
```

### 7.4 Latency Measurement Protocol

All papers measure on a **single GPU** (Palu: RTX 4090; others: A100 80GB):

```python
def measure_decode_latency(model, input_ids, n_runs=100, warmup=10,
                           measure_tpot=True):
    """
    Measure Time Per Output Token (TPOT) — the key metric for decode speed.
    """
    # Warmup (critical — first runs include kernel compilation overhead)
    for _ in range(warmup):
        _ = model.generate(input_ids, max_new_tokens=1)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model.generate(input_ids, max_new_tokens=1)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))   # milliseconds

    return {
        'tpot_ms_mean': float(np.mean(times)),
        'tpot_ms_std':  float(np.std(times)),
        'tpot_ms_p95':  float(np.percentile(times, 95)),
    }

def measure_throughput(model, workload, max_batch_size=512):
    """
    Throughput = total tokens generated / wall-clock time.
    Find max batch size that fits in memory, then measure tokens/sec.
    """
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if batch_size > max_batch_size:
            break
        try:
            torch.cuda.empty_cache()
            input_ids = workload.sample_batch(batch_size)
            t0 = time.time()
            outputs = model.generate(input_ids, max_new_tokens=512)
            t1 = time.time()
            total_tokens = outputs.shape[0] * outputs.shape[1]
            print(f"Batch {batch_size}: {total_tokens/(t1-t0):.0f} tok/s")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at batch_size={batch_size}")
            break
```

### 7.5 Implementation Notes & Gotchas

| Issue | Method | Root Cause | Solution |
|-------|--------|-----------|----------|
| RoPE blocks key matrix fusion | Palu | Position-dependent rotation can't be pre-fused | Use custom Triton kernel: reconstruct + RoPE + GEMV in one pass |
| SVD outliers destroy quantization | Palu | Large σ₁ component dominates h dimensions | Fuse Walsh-Hadamard offline; zero runtime cost |
| GQA rank deficiency | CommonKV | d_kv << d_model → thin concatenated matrix | Concatenate K+V per layer; sets matrix shape to (d_model, 2×group_size×d_kv) |
| SLERP numerical instability | MiniCache | sin(Ω) → 0 when vectors are near-parallel | Clamp cos(Ω) to [-1+ε, 1-ε]; fall back to linear avg when |sin(Ω)| < 1e-8 |
| Gather compaction kills throughput | ThinKV | Evicting non-contiguous tokens → memory fragmentation | CT kernel: soft-evict + in-place reuse by thought type |
| Quantization inflates generation | ThinKV | Quantized context is noisier → model generates more verification | Use TBQ+TBE hybrid; eviction removes tokens to counteract length inflation |
| Thought classification noise | ThinKV | Single layer is unreliable | Average over L*=4 calibrated layers; refresh every τ=128 tokens |
| Shallow layer merging degrades | MiniCache, CommonKV | Low cosine similarity in shallow layers | MiniCache: start from S=L/2. CommonKV: adaptive budget skips low-similarity groups |
| Compression ratio accounting | All | Decode tokens often excluded from reported ratios | Always report ratio including both prefill AND decode KV memory |
| Memory not actually freed | All | Python/PyTorch keeps GPU memory allocated | Call `torch.cuda.empty_cache()` after deletion; verify with `nvidia-smi` |

### 7.6 Minimal Reproducibility Configuration

```yaml
# Shared baseline config for all four methods
model:          llama3.1-8b-instruct    # use same backbone for comparability
weight_dtype:   fp16
query_dtype:    fp16
gpu:            single A100 80GB (or RTX 4090 for latency tests matching Palu)
context_len:    8192                    # LongBench default
flash_attn:     disabled                # for apple-to-apple (FlashAttn changes timing)
batch_size:     1                       # for TPOT; report max batch for throughput
calib_data:     wikitext-2              # 2048 samples, seqlen=1024
n_runs:         100                     # latency runs
warmup:         10

# Palu-specific
palu:
  group_size:            4              # G-LRD; try 1 (M-LRD) and 32 (J-LRD) for ablation
  compression_rates:     [0.3, 0.5]    # 0.7 for aggressive test
  quantization:          [none, 3bit, 2bit]
  use_fisher_search:     true
  use_hadamard:          true           # always true when combining with quant
  svd_variant:           svd_llm        # activation-aware

# MiniCache-specific
minicache:
  start_layer:   L_half                 # L // 2
  slerp_t:       0.6
  gamma:         0.05                   # retention threshold
  quantization:  kivi_4bit              # optional stacking for 5× compression

# CommonKV-specific
commonkv:
  group_size:         4
  svd_rank_ratio:     0.7               # for compression 0.3 and 0.5
  svd_rank_ratio_hi:  0.6               # for compression 0.6
  compression_ratios: [0.3, 0.5, 0.6]
  merge_strategy:     fisher_weighted

# ThinKV-specific
thinkv:
  n_thoughts:          3                # R, E, T
  n_cal_layers:        4                # L* size
  refresh_tau:         128              # tokens between reclassification
  group_size:          16               # quantization group
  retention_schedule:  [64, 32, 16, 8, 4]
  min_retention:       4
  token_budgets:       [64, 128, 256, 512, 1024, 2048, 4096]
  max_gen_len:         32768
  ct_block_size:       8
  precision_R:         nvfp4            # 4-bit; use fp8 for stricter accuracy mode
  precision_E:         nvfp4
  precision_T:         ternary          # 2-bit
  model_type:          lrm              # use lrm for reasoning models, llm for standard
```

---

*Guide compiled from: Palu (arXiv:2407.21118v2), MiniCache (arXiv:2405.14366v2), CommonKV (arXiv:2508.16134v1), ThinKV (arXiv:2510.01290v1).*
