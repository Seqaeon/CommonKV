# APKVC Implementation Guide (Revised Direction)

## 0. Context and Goal

The goal is to establish APKVC as a **decode-time KV cache codec** that outperforms
flat quantization methods (specifically KIVI) in long-generation regimes by exploiting
temporal correlation between adjacent token KV states.

The publishable claim is:
> "KV cache states during decode exhibit strong temporal correlation. Exploiting this
> structure via predictive coding yields better compression per bit than
> position-agnostic quantization at equivalent quality."

This guide covers:
1. Critical fixes to the existing implementation
2. Hybrid prefill (int8) + decode (APKVC) design
3. The Long-Decode Compression Benchmark (LDCB)
4. Comparison methodology against KIVI

---

## 1. Critical Fixes (Do These First)

These must be implemented before any benchmarking. The current high distortion
(median ~6) is almost certainly caused by one or more of these.

### Fix 1 — Anchor-only sanity check

Before doing anything else, set `max_anchor_interval = 1` so every token is an
anchor and no residuals are ever stored. Distortion should be ~0 and scores should
match FullKV exactly. If they do not, the bug is in the reconstruction path, not
the compression logic. Fix reconstruction before proceeding.

### Fix 2 — Per-head scale normalization (highest impact)

KV residuals vary wildly in magnitude across layers and heads. A codebook trained
on the global distribution is a poor match for any individual head. Without
normalization, the nearest-codeword search is nearly meaningless.

```python
def compress_residual(delta, codebooks, n_codebooks):
    # delta: [B, H, D]
    scale = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, H, 1]
    delta_normed = delta / scale
    codes = additive_quantize(delta_normed, codebooks[:n_codebooks])
    return codes, scale  # store scale alongside codes

def decompress_residual(codes, scale, codebooks):
    # scale: [B, H, 1]
    delta_normed = additive_decode(codes, codebooks)
    return delta_normed * scale
```

Store `scale` as a single float per head per token alongside the codebook indices.
This adds minimal storage (H floats per token) but dramatically improves codebook
hit quality.

### Fix 3 — Per-layer codebooks

Do not share codebooks across layers. Layer 0 and layer 20 have completely
different KV distributions. Train a separate codebook set per layer.

Storage cost: `n_layers × n_codebooks × codebook_size × D` floats.
For LLaMA-2-7B (32 layers, 4 codebooks, 256 codewords, D=128):
`32 × 4 × 256 × 128 × 2 bytes ≈ 67 MB` — acceptable.

### Fix 4 — Shorten anchor interval during debugging

Use `max_anchor_interval = 4` while validating. Once quality is stable at 4,
push to 8, then 16. At interval=16 with broken codebooks, drift compounds across
16 steps and reconstruction degrades catastrophically.

### Fix 5 — Lazy reconstruction

Do not call `reconstruct_all` on every decode step. This is O(T) and grows with
sequence length.

Instead, maintain a running reconstruction cache:
```python
# After writing a new token entry:
if entry['is_anchor']:
    recon_cache[t] = (K_true.clone(), V_true.clone())
    reset reconstruction chain
else:
    K_prev, V_prev = recon_cache[t-1]
    delta_K = decompress_residual(entry['codes_K'], entry['scale_K'], codebooks_K)
    delta_V = decompress_residual(entry['codes_V'], entry['scale_V'], codebooks_V)
    K = K_prev + delta_K
    V = V_prev + delta_V
    recon_cache[t] = (K, V)
```

This turns reconstruction into O(1) per new token. Only rebuild from scratch if
an anchor is written in the middle (reset the chain from that anchor forward).

---

## 2. Codebook Training (Revised)

### What to collect

Run the target model on ~512 representative prompts from the **same distribution as
your benchmark tasks** (not generic text). For the long-decode benchmark, these
should be continuation/essay prompts with long outputs (~500 tokens each).

For each decode token and each layer, record:
- `delta_K_base`: derotated key residual (after identity prediction, after RoPE derotation)
- `delta_V`: value residual (no derotation)

Normalize each residual by its per-head norm before storing. Train codebooks on
the normalized residuals.

### Training procedure per layer

```python
def train_codebooks_rvq(residuals, n_codebooks, codebook_size):
    # residuals: [N, D] — normalized residuals from calibration
    codebooks = []
    r = residuals.clone()
    for m in range(n_codebooks):
        # k-means on current remainder
        C = kmeans(r, k=codebook_size, n_iter=50)
        codebooks.append(C)
        # subtract best codeword for each residual
        dists = torch.cdist(r, C)
        best = dists.argmin(dim=-1)
        r = r - C[best]
    return codebooks  # list of [codebook_size, D] tensors
```

Save as `codebooks_K_layer{l}.pt` and `codebooks_V_layer{l}.pt` for each layer l.

### Asymmetric K/V settings

| | K | V |
|---|---|---|
| n_codebooks | 4 | 2 |
| codebook_size | 256 | 256 |
| residual_norm_threshold | 1.5× mean calib norm | 3.0× mean calib norm |

---

## 3. Hybrid Prefill + Decode Design

### Why hybrid

- APKVC decode compression on LongBench-style tasks saves ~50 tokens of VRAM
  against an 8000-token prefill — statistically invisible
- Int8 prefill compression is fast (~1x overhead), well-established, not novel
- The contribution is the decode codec; the prefill codec is infrastructure

### Prefill: int8 per-channel quantization

```python
def quantize_prefill_kv(K, V):
    # K, V: [B, H, T, D] — full prefill sequence
    # Per-channel: compute scale over token dimension
    K_scale = K.abs().amax(dim=2, keepdim=True) / 127.0  # [B, H, 1, D]
    V_scale = V.abs().amax(dim=2, keepdim=True) / 127.0
    K_int8 = (K / K_scale).round().clamp(-128, 127).to(torch.int8)
    V_int8 = (V / V_scale).round().clamp(-128, 127).to(torch.int8)
    return K_int8, K_scale, V_int8, V_scale

def dequantize_prefill_kv(K_int8, K_scale, V_int8, V_scale):
    K = K_int8.float() * K_scale
    V = V_int8.float() * V_scale
    return K, V
```

Store `(K_int8, K_scale, V_int8, V_scale)` in the cache for prefill tokens.
At attention time, dequantize on the fly before computing attention.

### Decode: APKVC

All tokens generated after prefill go through the APKVC pipeline described in
sections 1 and 4. The first decode token uses the last prefill token's
dequantized KV as K_prev for the identity predictor.

### Storage layout

```
cache = {
    'prefill': {
        'K_int8':  [B, H, T_pre, D] int8,
        'K_scale': [B, H, 1, D]     fp16,
        'V_int8':  [B, H, T_pre, D] int8,
        'V_scale': [B, H, 1, D]     fp16,
    },
    'decode': [
        # per-token entries, one of:
        {'is_anchor': True,  'K': fp16, 'V': fp16, 'position': t},
        {'is_anchor': False, 'codes_K': [...], 'scale_K': fp16,
                             'codes_V': [...], 'scale_V': fp16, 'position': t},
    ]
}
```

---

## 4. Full APKVC Decode Write Path (With Fixes)

```python
def compress_decode_token(K_true, V_true, t, state, codebooks_K, codebooks_V, config):
    """
    K_true, V_true: [B, H, D] — true KV for this token
    state: running state dict with last_K, last_K2, last_anchor_t, recon_cache, etc.
    """

    if t == 0 or should_reset(t, state, config):
        entry = {
            'is_anchor': True,
            'K': K_true.half(),
            'V': V_true.half(),
            'position': t,
        }
        state['entries'].append(entry)
        state['recon_cache'][t] = (K_true, V_true)
        state['last_anchor_t'] = t
        state['last_distortion'] = 0.0
        _update_rolling(state, K_true, V_true)
        return

    # --- Predict ---
    if config.predictor_type == 'identity' or state.get('last_K2') is None:
        K_hat = state['last_K']
        V_hat = state['last_V']
    else:
        K_hat = config.alpha_K * state['last_K'] + config.beta_K * state['last_K2']
        V_hat = config.alpha_V * state['last_V'] + config.beta_V * state['last_V2']

    delta_K = K_true - K_hat
    delta_V = V_true - V_hat

    # --- Distortion check ---
    distortion = compute_distortion(state, K_true, delta_K, delta_V, config)
    if distortion > config.rd_threshold:
        entry = {
            'is_anchor': True,
            'K': K_true.half(),
            'V': V_true.half(),
            'position': t,
        }
        state['entries'].append(entry)
        state['recon_cache'][t] = (K_true, V_true)
        state['last_anchor_t'] = t
        state['last_distortion'] = 0.0
        _update_rolling(state, K_true, V_true)
        return

    # --- Per-head normalization ---
    scale_K = delta_K.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    scale_V = delta_V.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    delta_K_normed = delta_K / scale_K
    delta_V_normed = delta_V / scale_V

    # --- RoPE derotation (keys only) ---
    if config.use_rope_aware_aq:
        delta_K_base = rope_derotate(delta_K_normed, t)
    else:
        delta_K_base = delta_K_normed

    # --- Additive quantization ---
    codes_K = additive_quantize(delta_K_base, codebooks_K[:config.K_num_codebooks])
    codes_V = additive_quantize(delta_V_normed, codebooks_V[:config.V_num_codebooks])

    entry = {
        'is_anchor': False,
        'codes_K': codes_K,
        'scale_K': scale_K.half(),
        'codes_V': codes_V,
        'scale_V': scale_V.half(),
        'position': t,
    }
    state['entries'].append(entry)

    # --- Lazy reconstruction update ---
    delta_K_recon = additive_decode(codes_K, codebooks_K)
    if config.use_rope_aware_aq:
        delta_K_recon = rope_rotate(delta_K_recon, t)
    delta_K_recon = delta_K_recon * scale_K
    delta_V_recon = additive_decode(codes_V, codebooks_V) * scale_V
    K_recon = K_hat + delta_K_recon
    V_recon = V_hat + delta_V_recon
    state['recon_cache'][t] = (K_recon, V_recon)

    state['last_residual_norm'] = max(delta_K.norm().item(), delta_V.norm().item())
    state['last_distortion'] = distortion
    _update_rolling(state, K_recon, V_recon)  # predict from reconstructed, not true
```

---

## 5. Long-Decode Compression Benchmark (LDCB)

This is the benchmark designed to make the decode compression claim visible.

### Core setup

- Model: LLaMA-2-7B or Mistral-7B-v0.1
- Short fixed prompts (< 128 tokens) so prefill is negligible
- Long forced generation (target: 1000 / 2000 / 4000 tokens)
- Baselines: FullKV, KIVI-int2, KIVI-int4

### Task 1 — Long continuation (smooth temporal correlation)

```python
PROMPTS_CONTINUATION = [
    "Write a detailed technical report on the history of transformer "
    "architectures, covering all major developments from 2017 to the present.",

    "Write a comprehensive essay on the political economy of colonial Nigeria, "
    "examining land tenure, taxation, and labour extraction in detail.",

    "Write a detailed account of how operating system kernels manage memory, "
    "covering paging, virtual address spaces, and page replacement algorithms.",
]
```

This produces slowly varying, topically consistent KV states — exactly where
predictive coding exploits temporal correlation best. KIVI compresses each token
independently and cannot exploit this.

### Task 2 — Structured reasoning (tests anchor/reset)

```python
PROMPTS_REASONING = [
    "Solve the following 15 algebra problems step by step, showing all working:\n"
    + "\n".join([f"{i+1}. {generate_algebra_problem()}" for i in range(15)]),
]
```

Reasoning produces abrupt topic shifts between problems — KV states jump at
problem boundaries. This stresses the anchor/reset mechanism. Expect more
anchors here; compression ratio will be lower than Task 1.

### Task 3 — Multi-turn simulation

```python
def simulate_multiturn(model, n_turns=15, user_msg_tokens=10, response_tokens=100):
    """
    Simulate a conversation. Measure VRAM at each turn boundary.
    Returns list of (turn_number, vram_gb, response_quality) tuples.
    """
    cache = initialize_cache()
    for turn in range(n_turns):
        user_msg = generate_user_message(turn)
        response = model.generate(user_msg, cache=cache, max_new_tokens=response_tokens)
        vram = torch.cuda.max_memory_allocated() / 1e9
        yield turn, vram, evaluate_response(response)
        cache = update_cache(cache, user_msg, response)
```

### Metrics to collect per task

```python
@dataclass
class BenchmarkMetrics:
    compression_ratio: float   # compressed_bytes / fullkv_bytes
    perplexity: float          # of generated tokens under FullKV model
    rouge_l: float             # ROUGE-L vs FullKV generation (as reference)
    peak_vram_gb: float        # torch.cuda.max_memory_allocated() / 1e9
    tokens_per_second: float   # generated_tokens / wall_clock_seconds
    distortion_mean: float     # mean attention distortion across all layers/steps
    distortion_p95: float      # 95th percentile distortion (catches spikes)
    anchor_rate: float         # fraction of decode tokens stored as anchors
```

### Key plots to generate

**Plot 1 — Compression ratio vs generation length**
```
x: tokens generated (checkpoints at 250, 500, 1000, 2000, 4000)
y: compression_ratio
lines: FullKV=1.0, KIVI-int4, KIVI-int2, APKVC-identity, APKVC-linear
```
APKVC's ratio should improve as generation gets longer because more tokens
participate in residual chains. KIVI's ratio is flat. This is the key plot.

**Plot 2 — Pareto frontier (quality vs compression)**
```
x: compression_ratio (lower = more compressed)
y: perplexity or ROUGE-L
points: each method at multiple configurations
```
Any APKVC point that dominates KIVI points (lower perplexity at same or better
compression ratio) is the publishable result.

**Plot 3 — VRAM over turns (Task 3)**
```
x: conversation turn number (1..15)
y: VRAM in GB
lines: FullKV, KIVI-int4, APKVC
```

### Ablation table

Run these variants and report all metrics:

| Variant | predictor | rope_aq | scale_norm | n_codebooks_K |
|---|---|---|---|---|
| FullKV | — | — | — | — |
| KIVI-int4 | — | — | — | — |
| KIVI-int2 | — | — | — | — |
| APKVC-anchor-only | none | off | off | 0 |
| APKVC-identity-noscale | identity | off | off | 4 |
| APKVC-identity | identity | on | on | 4 |
| APKVC-linear | linear | on | on | 4 |
| APKVC-full | linear | on | on | 8 |

The anchor-only row validates reconstruction correctness (should match FullKV).
The identity-noscale row isolates the effect of per-head normalization.

---

## 6. Distortion Metric Implementation

Use key-dot proxy for layers 0..L//2, sampled attention output for L//2..L.

```python
def compute_distortion(state, K_true, delta_K, delta_V, config):
    layer = state['layer_idx']
    L = state['n_layers']

    if layer < L // 2 or config.rd_metric == 'key_dot':
        # Key dot-product proxy
        Q = state.get('last_query')
        if Q is None:
            return 0.0
        scale = K_true.shape[-1] ** 0.5
        # K_compressed approximation: K_hat (prediction) — delta cancels out if small
        K_hat = state['last_K']
        scores_true = torch.einsum('bhd,bhtd->bht', Q, K_true.unsqueeze(2)) / scale
        scores_comp = torch.einsum('bhd,bhtd->bht', Q, K_hat.unsqueeze(2)) / scale
        return (scores_true - scores_comp).abs().max().item()

    else:
        # Sampled attention output proxy (subset of heads)
        n = config.rd_sample_heads
        Q = state.get('last_query')
        if Q is None:
            return 0.0
        K_all_true = state['K_all_true'][:, :n]   # [B, n, T, D]
        V_all_true = state['V_all_true'][:, :n]
        K_hat = state['last_K']
        # Approximate K_comp as K_hat for distortion estimation
        def attn(q, k, v):
            w = torch.softmax(
                torch.einsum('bhd,bhtd->bht', q, k) / k.shape[-1]**0.5, dim=-1)
            return torch.einsum('bht,bhtd->bhd', w, v)
        out_true = attn(Q[:, :n], K_all_true, V_all_true)
        out_comp = attn(Q[:, :n],
                        torch.cat([K_all_true[:, :, :-1], K_hat[:, :n].unsqueeze(2)], dim=2),
                        V_all_true)
        return (out_true - out_comp).norm(dim=-1).mean().item()
```

### Calibrating rd_threshold

```python
def calibrate_rd_threshold(model, calibration_prompts, target_fraction=0.5):
    """
    Run model uncompressed. Record natural token-to-token attention output
    variance. Set rd_threshold to target_fraction × that variance.
    """
    variances = []
    for prompt in calibration_prompts:
        outputs = model(prompt, output_attentions=True, output_hidden_states=True)
        # measure step-to-step attn output variation
        for t in range(1, len(outputs)):
            diff = (outputs[t] - outputs[t-1]).norm(dim=-1).mean().item()
            variances.append(diff)
    return target_fraction * np.median(variances)
```

---

## 7. Competitor: KIVI

KIVI is the primary baseline. Key facts:

- Quantizes K per-channel in int2, V per-token in int2
- Keeps a residual buffer of the last 32 tokens in fp16 (handles recency bias)
- Treats every token independently — no temporal structure exploited
- This is the structural gap APKVC targets

To install and use:
```bash
pip install kivi-cache
```

Or implement directly:
```python
def kivi_quantize_K(K, bits=2):
    # K: [B, H, T, D]
    # Per-channel: scale over token dimension
    scale = K.abs().amax(dim=2, keepdim=True)
    levels = 2 ** bits - 1
    K_q = (K / scale * (levels / 2)).round().clamp(-levels//2, levels//2)
    return K_q.to(torch.int8), scale

def kivi_quantize_V(V, bits=2):
    # V: [B, H, T, D]
    # Per-token: scale over feature dimension
    scale = V.abs().amax(dim=3, keepdim=True)
    levels = 2 ** bits - 1
    V_q = (V / scale * (levels / 2)).round().clamp(-levels//2, levels//2)
    return V_q.to(torch.int8), scale
```

Keep last 32 tokens of K and V in fp16 regardless (the residual buffer).

---

## 8. Recommended Config Defaults

```python
@dataclass
class APKVCConfig:
    # Predictor
    predictor_type: str        = 'identity'   # start here; upgrade to 'linear' later
    alpha_K: float             = 1.5
    beta_K: float              = -0.5
    alpha_V: float             = 1.5
    beta_V: float              = -0.5

    # AQ
    use_rope_aware_aq: bool    = True
    use_per_head_scale: bool   = True          # Fix 2 — must be True
    K_num_codebooks: int       = 4
    V_num_codebooks: int       = 2
    codebook_size: int         = 256
    per_layer_codebooks: bool  = True          # Fix 3 — must be True

    # Reset policy
    max_anchor_interval: int           = 8    # start at 4, increase after validation
    residual_norm_threshold_K: float   = 1.5  # × mean calibration norm
    residual_norm_threshold_V: float   = 3.0

    # Distortion
    rd_metric: str             = 'key_dot'
    rd_threshold: float        = 0.05         # calibrate per model
    rd_sample_heads: int       = 4

    # Prefill
    prefill_bits: int          = 8            # int8 quantization for prefill
```

---

## 9. Build Order

### Stage 1 — Validate reconstruction (do not skip)
- Implement anchor-only mode (`max_anchor_interval = 1`)
- Scores must match FullKV exactly
- Distortion must be ~0

### Stage 2 — Add residual compression without AQ
- Identity predictor
- Scalar quantization of residuals (simple round to N bits)
- Per-head scale normalization
- Fixed anchor interval = 4
- Validate: distortion should be low, compression ratio > 1

### Stage 3 — Replace scalar quantization with RoPE-aware AQ
- Train per-layer codebooks on calibration data
- Implement derotate → normalize → quantize → decode → re-rotate → denormalize
- Validate roundtrip with M large enough to approximate identity

### Stage 4 — Add distortion-gated reset
- Implement `should_reset` with combined policy
- Implement `compute_distortion` with key-dot proxy
- Calibrate `rd_threshold` on calibration prompts

### Stage 5 — Add int8 prefill + run LDCB
- Implement per-channel int8 prefill quantization
- Run full benchmark against KIVI-int2, KIVI-int4, FullKV
- Generate the three key plots

### Stage 6 — Upgrade predictor and tune
- Swap identity for linear predictor
- Tune `max_anchor_interval` and `rd_threshold` for best Pareto point
- Add layer-wise policy (aggressive early layers, conservative late layers)
