# Attention-Aware Predictive KV Compression

This document is an implementation-ready specification for a KV-cache compression codec that combines:

1. **Predictive / delta coding**
2. **RoPE-aware additive quantization (AQ)**
3. **Attention-aware rate-distortion control**
4. **Anchor / keyframe resets**

The goal is to compress KV cache state during autoregressive inference with minimal accuracy loss and minimal runtime overhead.

---

## 1) What the method is trying to do

Instead of storing a full key/value vector for every token, store:

- occasional **anchors** (full KV states)
- **predicted residuals** for the tokens in between
- residuals compressed with **RoPE-aware AQ**
- reset to a new anchor when distortion becomes too large

This treats the KV cache like a predictable time series with occasional abrupt changes.

---

## 2) Where the method sits in the inference pipeline

### Prefill stage
The prompt is processed normally to build an initial KV representation.

### Decode stage
Each new token is handled as follows:

1. Predict the next KV state from recent history
2. Compute the residual between the true KV and the predicted KV
3. Compress the residual
4. Store the compressed residual unless a reset is triggered
5. If a reset is triggered, store the full KV as a new anchor

Important distinction:
- **Prefill** can be left uncompressed initially for simplicity.
- **Decode** is the main target for compression.

---

## 3) Tensor shapes and notation

For a transformer with:
- L layers
- H heads
- D head dimension
- batch size B
- token index t

Per layer ℓ and token t:
- K[ℓ, t] has shape [B, H, D]
- V[ℓ, t] has shape [B, H, D]

The compressed cache is stored per layer.

---

## 4) Core idea: anchor + predictive residuals

For each layer and token:

### Anchor token
Store the full KV state:
- K_full
- V_full

### Non-anchor token
Predict the next state:
- K_hat
- V_hat

Compute residuals:
- delta_K = K_true - K_hat
- delta_V = V_true - V_hat

Then compress the residuals instead of the raw KV.

---

## 5) Predictor choice: what is most practical with low overhead?

This is the most important implementation decision.

### Recommended default: identity predictor
Use the previous token as the prediction:
- K_hat[t] = K[t-1]
- V_hat[t] = V[t-1]

Why this is the best default:
- zero learned parameters
- essentially no extra compute
- very easy to implement
- often surprisingly effective because adjacent KV states are correlated
- makes residuals small enough to help compression without adding latency

This should be the first implementation to test.

### Second-best option: cheap linear predictor
Use a tiny linear extrapolation from the last one or two tokens:
- K_hat[t] = a * K[t-1] + b * K[t-2]
- V_hat[t] = a_v * V[t-1] + b_v * V[t-2]

Why it is practical:
- still O(D)
- no MLP
- can be shared across heads or layers
- usually gives better residuals than the identity predictor

This is the best “upgrade” if identity prediction is not enough.

### Not recommended by default: per-head MLP predictor
Avoid unless the simpler predictors fail.

Why:
- adds noticeable compute during inference
- increases implementation complexity
- can erase the benefit of compression
- is hard to justify unless compression gains are very large

### Practical recommendation
Use this order:
1. identity predictor
2. cheap linear predictor
3. only then consider anything heavier

### Suggested configuration knob
predictor_type ∈ {"identity", "linear"}

---

## 6) Why prediction helps compression

Raw KV values are often moderately structured but still fairly high entropy.

Residuals after prediction tend to be:
- smaller in magnitude
- more centered around zero
- more sparse or clustered
- easier to quantize or codebook-compress

This is the key reason predictive coding can make the downstream compression much more effective.

---

## 7) RoPE-aware additive quantization

After prediction, compress the residual with additive quantization.

### Residual representation
Approximate each residual vector as the sum of a small number of codewords:
- residual ≈ sum of codebook entries

### Why additive quantization
AQ is a good match because residuals are already low-entropy and codebooks can capture recurring patterns.

### Why RoPE-aware
Keys are position-rotated by RoPE, so the same semantic direction appears rotated at different positions.

A RoPE-aware codebook should:
- reuse a shared base codebook
- apply the RoPE rotation corresponding to the token position at runtime
- avoid wasting separate codebooks for each position

This improves codebook reuse and makes the compressor more position-consistent.

### Practical coding strategy
For each residual vector:
1. initialize the residual as r
2. for each codebook m:
   - choose the codeword nearest to the current residual
   - subtract that codeword from the residual
3. store only the selected codeword indices

### Decoding
To reconstruct the residual:
- sum the selected codewords
- apply the relevant RoPE transform if needed

### Suggested configuration knobs
- num_codebooks
- codebook_size
- rope_aware = true/false
- shared_codebook_across_layers = true/false
- separate_kv_codebooks = true/false

---

## 8) K and V should probably be treated differently

This is important.

### Keys (K)
- are directly used in attention scoring
- are more sensitive to distortion
- should usually get higher precision
- may need fewer compression bits or fewer codebook errors

### Values (V)
- are mixed after attention weights are computed
- can often tolerate more compression than keys
- can usually be compressed more aggressively

### Recommended default
Use a stricter compression budget for K than for V.

Suggested options:
- separate predictors for K and V
- separate codebooks for K and V
- separate distortion thresholds for K and V

---

## 9) Attention-aware rate-distortion control

The compression should not be judged only by reconstruction error of KV vectors.

What matters is whether the compressed cache changes the attention output too much.

### Practical principle
Compress more when the compressed representation does not materially change attention behavior.

### Distortion signals to use
Use one or more of the following:

1. **Key-dot-product proxy**
   - compare query-key scores before and after compression

2. **Attention-output proxy**
   - compare attention outputs directly on sampled queries

3. **Layer-specific proxy**
   - use a cheaper proxy in early layers and a stricter one in later layers if desired

### Recommended default
Use a cheap proxy first:
- compare query-key score changes or a sampled attention-output estimate

### Config knobs
- rd_metric = {"key_dot", "attention_output", "sampled_attention_output"}
- rd_threshold
- rd_sample_size

---

## 10) Reset / anchor policy

Residual coding needs resets to prevent drift.

### Reset conditions
Store a new anchor when any of these are true:

1. Residual magnitude is too large
2. Attention distortion is too large
3. Too many tokens have passed since the last anchor
4. The predicted residual becomes too complex to compress efficiently

### Recommended reset rule
Use a combined policy:
- hard maximum anchor interval
- residual magnitude threshold
- attention-distortion threshold

### Why resets matter
Without resets:
- prediction error accumulates
- quantization error accumulates
- reconstruction can drift
- attention quality can degrade rapidly

### Suggested config knobs
- max_anchor_interval
- residual_norm_threshold
- rd_threshold
- max_residual_sparsity

---

## 11) Storage format

Each token entry per layer should store one of two states:

### Anchor entry
- is_anchor = true
- store full K and V (or full compressed anchor representation if you want a second-stage codec)

### Residual entry
- is_anchor = false
- store:
  - prediction mode used
  - AQ code indices for delta_K
  - AQ code indices for delta_V
  - optional scale / zero-point metadata if needed

### Recommended metadata per token
- token position
- anchor flag
- predictor type used
- codebook indices
- optional per-token bit budget
- optional residual norm or distortion score

---

## 12) Reconstruction path

At attention time, reconstruction should be done efficiently.

### Reconstruction steps
For each needed token:
1. find the nearest anchor before it
2. reconstruct forward token-by-token using cached residuals
3. cache intermediate reconstructions locally for reuse during that forward pass
4. compute attention normally on reconstructed K and V

### Important implementation detail
Do not repeatedly reconstruct the same token multiple times inside one attention call.
Cache reconstructed values temporarily.

---

## 13) Implementation strategy: build it in stages

### Stage 1: baseline predictive residual codec
Implement:
- identity predictor
- anchor/reset logic
- simple quantization of residuals

This gives a minimal proof of concept with low engineering risk.

### Stage 2: add RoPE-aware additive quantization
Replace simple quantization with AQ codebooks.

### Stage 3: add attention-aware RD control
Use a distortion proxy to decide:
- how many codebooks to use
- whether to store a residual or reset to anchor
- whether K and V need different bit budgets

### Stage 4: optional refinements
- linear predictor
- separate K/V policies
- layer-wise compression budgets
- head-wise budgets
- shared vs separate codebooks

---

## 14) Parameterization checklist

These should be exposed as tunable arguments:

predictor_type:
- identity
- linear

use_rope_aware_aq:
- true/false

num_codebooks:
- integer

codebook_size:
- integer

separate_kv_policies:
- true/false

rd_metric:
- key_dot
- attention_output
- sampled_attention_output

rd_threshold:
- float

max_anchor_interval:
- integer

residual_norm_threshold:
- float

layerwise_budget_schedule:
- optional array or function

headwise_budget_schedule:
- optional array or function

shared_codebook_across_layers:
- true/false

store_anchor_in_fp16:
- true/false

---

## 15) Pseudocode

```text
for each generated token t:
  for each layer l:
    if should_reset(layer=l, token=t):
        store_full_anchor(K_true, V_true)
        continue

    if predictor_type == "identity":
        K_hat = previous_K
        V_hat = previous_V
    else if predictor_type == "linear":
        K_hat = a*K_prev + b*K_prev2
        V_hat = a_v*V_prev + b_v*V_prev2

    delta_K = K_true - K_hat
    delta_V = V_true - V_hat

    if attention_distortion_too_large(K_true, V_true, delta_K, delta_V):
        store_full_anchor(K_true, V_true)
    else:
        codes_K = rope_aware_additive_quantize(delta_K)
        codes_V = rope_aware_additive_quantize(delta_V)
        store_residual(codes_K, codes_V)
```

---

## 16) Complexity expectations

### Predictor cost
- identity: essentially zero
- linear: very cheap
- MLP: too expensive unless proven worthwhile

### AQ cost
- encoding: moderate
- decoding: moderate but manageable if codebooks are small and well implemented

### Main performance goal
The added compression logic must be much cheaper than the memory bandwidth saved.

A good rule:
- if the predictor or coder becomes more expensive than the memory traffic saved, the design is not worth it.

---

## 17) What success should look like

A successful implementation should show:
- lower KV memory footprint
- minimal perplexity or task metric degradation
- small runtime overhead
- stable behavior over long sequences
- better compression on values than on keys
- better compression at later layers or predictable regions

---

## 18) Known weak points / failure modes

1. Predictor overhead becomes too large
2. Residuals do not become small enough to help compression
3. K distortion harms attention too much
4. Reset intervals are too long and drift accumulates
5. AQ codebooks are not expressive enough
6. Reconstructing from anchors becomes too slow

---

## 19) Recommended default design

If forced to choose one practical configuration:

- predictor_type = identity
- use_rope_aware_aq = true
- separate_kv_policies = true
- rd_metric = sampled_attention_output
- anchor reset = combined threshold + max interval
- codebooks shared across layers only if memory is tight and quality remains acceptable
- use stronger compression for V than K

This is the safest starting point and the best balance between novelty and feasibility.



---

## 1. What This Does

For each decode-step token, instead of storing the full KV vector:

1. **Predict** the KV from recent history (cheap).
2. **Compute the residual** between the true KV and the prediction.
3. **Compress the residual** using RoPE-aware additive quantization.
4. **Store only the codebook indices.**

If the residual is too large or attention distortion exceeds a threshold, store
a full **anchor** (uncompressed KV) instead and restart the prediction chain.

---

## 2. Notation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `L, H, D` | scalars | layers, heads, head dimension |
| `K[t], V[t]` | `[B, H, D]` | true KV at token t |
| `K̂[t], V̂[t]` | `[B, H, D]` | predicted KV |
| `δK, δV` | `[B, H, D]` | residuals = true − predicted |
| `R(θ_t)` | rotation | RoPE rotation at position t |
| `M` | scalar | number of additive codebooks |
| `S` | scalar | codewords per codebook |
| `C_m` | `[S, D]` | m-th codebook |

---

## 3. Predictor

### Identity predictor (default — implement this first)

```python
K_hat = K[t-1]
V_hat = V[t-1]
```

Zero parameters, zero extra compute. If `t` is the first token after an anchor,
use the anchor itself as the prediction.

### Linear predictor (upgrade if needed)

```python
K_hat = alpha_K * K[t-1] + beta_K * K[t-2]
V_hat = alpha_V * V[t-1] + beta_V * V[t-2]
```

`alpha` and `beta` are scalars (not matrices). Safe default: `alpha=1.5, beta=-0.5`
(linear extrapolation). Enforce `|alpha| + |beta| <= 1` to prevent divergence.
Fall back to identity if `t-2` is before the nearest anchor.

Do not implement an MLP predictor. The latency cost will erase the compression
benefit.

---

## 4. RoPE-Aware Additive Quantization

### Why RoPE matters

RoPE rotates key vectors by position: `K_rope[t] = R(θ_t) @ K_base`. This means
the same semantic direction appears at a different angle at every position.
Codebooks trained in rotated space cannot be reused across positions.

Fix: derotate the key residual before quantizing, then re-rotate after decoding.
Values are never RoPE-rotated — do not derotate V.

### Derotation and re-rotation

RoPE operates on pairs of dimensions `(2i, 2i+1)` with angle `t * base^(-2i/D)`:

```python
def rope_derotate(x, position, base=10000):
    # x: [B, H, D]
    D = x.shape[-1]
    i = torch.arange(0, D // 2, dtype=torch.float32, device=x.device)
    theta = base ** (-2 * i / D)
    angle = position * theta          # [D//2]
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    # Inverse rotation: negate sin
    out = torch.empty_like(x)
    out[..., 0::2] =  x_even * cos + x_odd * sin
    out[..., 1::2] = -x_even * sin + x_odd * cos
    return out

def rope_rotate(x, position, base=10000):
    # Forward rotation
    D = x.shape[-1]
    i = torch.arange(0, D // 2, dtype=torch.float32, device=x.device)
    theta = base ** (-2 * i / D)
    angle = position * theta
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out
```

### Additive quantization — encoding

Approximate the residual as the sum of M codewords, one from each codebook.
Only the indices are stored.

```python
def additive_quantize(residual, codebooks):
    # residual:  [B, H, D]
    # codebooks: list of M tensors, each [S, D]
    # returns:   list of M tensors, each [B, H] (int indices)
    r = residual.clone()
    indices = []
    for C in codebooks:               # C: [S, D]
        # dist: [B, H, S]
        dist = torch.cdist(r, C.unsqueeze(0).expand(r.shape[0], -1, -1))
        best = dist.argmin(dim=-1)    # [B, H]
        r = r - C[best]              # subtract chosen codeword
        indices.append(best)
    return indices
```

### Additive quantization — decoding

```python
def additive_decode(indices, codebooks):
    # indices:   list of M tensors, each [B, H]
    # codebooks: list of M tensors, each [S, D]
    # returns:   [B, H, D]
    out = torch.zeros(*indices[0].shape, codebooks[0].shape[-1],
                      device=indices[0].device)
    for idx, C in zip(indices, codebooks):
        out += C[idx]
    return out
```

### Codebook training (offline, before running experiments)

Run the target model on ~512 representative prompts. Collect all derotated key
residuals (`delta_K_base`) and raw value residuals (`delta_V`) across all layers.
Train M codebooks using Residual Vector Quantization (RVQ): fit codebook 1 on raw
residuals, codebook 2 on what's left after subtracting codebook 1's best match,
and so on. Save as model-specific artifacts loaded at inference time.

Use separate codebooks for K and V — their distributions differ substantially.
Sharing codebooks across layers is acceptable if memory is tight, but validate
per-layer quality first.

---

## 5. Attention-Aware Distortion Metric

Gate compression on attention output error, not KV reconstruction error.

### Proxy 1: Key dot-product change (default)

```python
def key_dot_distortion(Q, K_true, K_compressed):
    # Q: [B, H, D], K: [B, H, T, D]
    scale = K_true.shape[-1] ** 0.5
    scores_true = torch.einsum('bhd,bhtd->bht', Q, K_true) / scale
    scores_comp = torch.einsum('bhd,bhtd->bht', Q, K_compressed) / scale
    return (scores_true - scores_comp).abs().max().item()
```

### Proxy 2: Sampled attention output (use for deep layers)

```python
def sampled_attention_distortion(Q, K_true, V_true, K_comp, V_comp, n_heads=4):
    def attn(q, k, v):
        scale = q.shape[-1] ** 0.5
        w = torch.softmax(torch.einsum('bhd,bhtd->bht', q, k) / scale, dim=-1)
        return torch.einsum('bht,bhtd->bhd', w, v)
    out_true = attn(Q[:, :n_heads], K_true[:, :n_heads], V_true[:, :n_heads])
    out_comp = attn(Q[:, :n_heads], K_comp[:, :n_heads], V_comp[:, :n_heads])
    return (out_true - out_comp).norm(dim=-1).mean().item()
```

Recommended split: use `key_dot` for layers `0` to `L//2`, use
`sampled_attention_output` for layers `L//2` to `L`.

To set `rd_threshold`: run the model uncompressed, record the natural
token-to-token variance in attention outputs, set threshold to ~0.5× that value.

---

## 6. Anchor / Reset Policy

```python
def should_reset(t, state, config):
    if (t - state.last_anchor_t) >= config.max_anchor_interval:
        return True
    if state.last_residual_norm > config.residual_norm_threshold:
        return True
    if state.last_distortion > config.rd_threshold:
        return True
    return False
```

Token `t=0` is always an anchor. Recommended starting defaults:

| Parameter | Default |
|---|---|
| `max_anchor_interval` | 16 |
| `residual_norm_threshold` | 2.0× mean calibration residual norm |
| `rd_threshold` | 0.05 |

---

## 7. Asymmetric K / V Policy

| | K | V |
|---|---|---|
| `num_codebooks` (M) | 4 | 2 |
| `residual_norm_threshold` | 1.5× | 3.0× |

Keys are used for dot-product scoring and are more sensitive. Values are mixed
after softmax and tolerate more compression.

---

## 8. Write Path

```python
def compress_new_token(K_true, V_true, t, state, codebooks_K, codebooks_V, config):

    if t == 0 or should_reset(t, state, config):
        state.entries.append({
            'is_anchor': True,
            'K': K_true.half(), 'V': V_true.half(), 'position': t
        })
        state.last_anchor_t = t
        state.last_distortion = 0.0
        _update_rolling(state, K_true, V_true)
        return

    # Predict
    if config.predictor_type == 'identity' or state.last_K2 is None:
        K_hat, V_hat = state.last_K, state.last_V
    else:
        K_hat = config.alpha_K * state.last_K + config.beta_K * state.last_K2
        V_hat = config.alpha_V * state.last_V + config.beta_V * state.last_V2

    delta_K = K_true - K_hat
    delta_V = V_true - V_hat

    # Distortion check before committing to residual storage
    distortion = compute_distortion(state, K_true, V_true, K_hat, V_hat,
                                    delta_K, delta_V, config)
    if distortion > config.rd_threshold:
        state.entries.append({
            'is_anchor': True,
            'K': K_true.half(), 'V': V_true.half(), 'position': t
        })
        state.last_anchor_t = t
        state.last_distortion = 0.0
        _update_rolling(state, K_true, V_true)
        return

    # RoPE derotation (keys only)
    delta_K_base = rope_derotate(delta_K, t) if config.use_rope_aware_aq else delta_K

    # Additive quantization
    codes_K = additive_quantize(delta_K_base, codebooks_K[:config.K_num_codebooks])
    codes_V = additive_quantize(delta_V,       codebooks_V[:config.V_num_codebooks])

    state.entries.append({
        'is_anchor': False,
        'codes_K': codes_K, 'codes_V': codes_V,
        'position': t, 'predictor_type': config.predictor_type,
    })
    state.last_residual_norm = max(delta_K.norm().item(), delta_V.norm().item())
    state.last_distortion = distortion
    _update_rolling(state, K_true, V_true)


def _update_rolling(state, K, V):
    state.last_K2 = state.last_K
    state.last_V2 = state.last_V
    state.last_K = K
    state.last_V = V
```

---

## 9. Read Path (Reconstruction)

Single forward scan from each anchor. Call once per layer per decode step.

```python
def reconstruct_all(state, codebooks_K, codebooks_V, config):
    T = len(state.entries)
    B, H, D = state.last_K.shape
    K_all = torch.empty(B, H, T, D, device=state.last_K.device)
    V_all = torch.empty(B, H, T, D, device=state.last_K.device)
    K_prev = K_prev2 = V_prev = V_prev2 = None

    for i, entry in enumerate(state.entries):
        t = entry['position']
        if entry['is_anchor']:
            K = entry['K'].float()
            V = entry['V'].float()
        else:
            if entry['predictor_type'] == 'identity' or K_prev2 is None:
                K_hat, V_hat = K_prev, V_prev
            else:
                K_hat = config.alpha_K * K_prev + config.beta_K * K_prev2
                V_hat = config.alpha_V * V_prev + config.beta_V * V_prev2

            delta_K_base = additive_decode(entry['codes_K'], codebooks_K)
            delta_V      = additive_decode(entry['codes_V'], codebooks_V)
            delta_K = rope_rotate(delta_K_base, t) if config.use_rope_aware_aq \
                      else delta_K_base
            K = K_hat + delta_K
            V = V_hat + delta_V

        K_all[:, :, i, :] = K
        V_all[:, :, i, :] = V
        K_prev2, V_prev2 = K_prev, V_prev
        K_prev,  V_prev  = K, V

    return K_all, V_all
```

Do not persist the reconstructed tensors between decode steps — rebuild from
compressed entries each time. Do not call `reconstruct_all` more than once per
layer per step.

---

## 10. Config

```python
@dataclass
class APKVCConfig:
    predictor_type: str  = 'identity'
    alpha_K: float       = 1.5
    beta_K: float        = -0.5
    alpha_V: float       = 1.5
    beta_V: float        = -0.5

    use_rope_aware_aq: bool  = True
    K_num_codebooks: int     = 4
    V_num_codebooks: int     = 2
    codebook_size: int       = 256
    separate_kv_codebooks: bool = True

    rd_metric: str       = 'key_dot'
    rd_threshold: float  = 0.05
    rd_sample_heads: int = 4

    max_anchor_interval: int         = 16
    residual_norm_threshold_K: float = 1.5
    residual_norm_threshold_V: float = 3.0
```

---

## 11. Build Order

**Stage 1:** Identity predictor + fixed anchor interval + scalar quantization on
residuals (no AQ, no RoPE awareness, no distortion gating). Verify the
predict → residual → quantize → reconstruct loop is correct end-to-end.

**Stage 2:** Replace scalar quantization with RoPE-aware AQ. Train codebooks
offline. Verify `derotate → quantize → decode → re-rotate` roundtrips correctly
with no compression (use M large enough to approximate identity).

**Stage 3:** Replace fixed anchor interval with `should_reset` using the
distortion proxy. Start with `key_dot`.

**Stage 4:** Asymmetric K/V policies, linear predictor, `sampled_attention_output`
proxy, layer-wise budget schedule.

---

## 20) Final summary

The method should be implemented as a **KV-cache codec**:

- store anchors occasionally
- predict intermediate KV states cheaply
- compress the residuals with RoPE-aware additive quantization
- decide reset/compression strength using attention-aware distortion rather than raw vector error

The most practical predictor is the **identity predictor**, followed by a **cheap linear predictor**. Avoid an MLP predictor unless a prototype shows a very clear gain that justifies the extra latency.