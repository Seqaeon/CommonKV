# APKVC vs CommVQ — Technical Comparison

## TL;DR

**They are fundamentally different methods.** There is meaningful surface overlap (both do additive/multi-codebook quantization and are RoPE-aware), but the core invention in each is entirely absent from the other.

---

## Side-by-Side Overview

| Dimension | **APKVC** (your code) | **CommVQ** (2506.18879v1) |
|---|---|---|
| Core paradigm | **Predictive/delta coding** — compress residuals between anchor snapshots | **Pure vector quantization** — quantize every KV vector uniformly |
| What is stored | Anchor (full KV, ~1 in every N tokens) + compressed residuals | Codebook indices for every token, no anchoring |
| RoPE strategy | **Derotate → quantize base → rerotate** at decode time | **Algebraic commutativity** — codebook is constrained to commute with RoPE |
| Attention computation | Must dequantize K before computing `q·k` | Can compute attention **directly on codes** via lookup, no dequantization |
| Predictor | Identity (`K̂=K[t-1]`) or linear extrapolation | None — no predictor |
| Rate-distortion control | Yes — key-dot / MSE distortion gate triggers anchor resets | No — fixed bit-budget per token |
| Anchor resets | Yes — max interval + residual norm + distortion threshold | Not applicable |
| Prefill compression | INT8 per-token asymmetric quantization | VQ (same codebook applies everywhere) |
| Codebook training | Offline RVQ on traced residuals | EM algorithm with closed-form M-step on calibration set |
| Codebook structure | Unconstrained `[S, D]` real matrices | Constrained **2×2 block-diagonal** form `[[a,b],[-b,a]]` (RoPE-commutative) |

---

## Where They Actually Overlap

### 1. Additive Quantization (AQ) / Residual VQ
Both use a greedy multi-codebook scheme where each codebook quantizes the residual left by the previous one. This is standard RVQ/AQ — it's a well-known prior technique neither paper invents. Your APKVC applies it to **delta residuals**, CommVQ applies it to **raw KV vectors**.

### 2. "RoPE-aware" in spirit
Both methods recognise that applying VQ in RoPE-rotated space is bad (codebooks can't generalise across positions). But the solutions are completely different:

- **APKVC**: Explicitly derotate `K_true` at position `abs_pos` before quantizing (`rope_derotate`), then re-apply rotation at decode time (`rope_rotate`). Simple, effective, and a natural engineering choice.
- **CommVQ**: Constrains the codebook entries to have the block-diagonal structure `[[a,b],[-b,a]]`. Because this structure commutes with the Givens rotations that RoPE applies per dimension-pair, `R_m C = C R_m`, so attention logits can be computed as `(q R_{t-i} C^T) s_i^T` — a simple lookup over precomputed values. This is a **novel algebraic trick** absent from your code.

---

## What CommVQ Has That You Don't

### The Commutativity Trick (the paper's main contribution)
CommVQ's defining claim is that it allows attention to be computed **without ever dequantizing** the key cache. Instead of:

```
k_hat = decode(s_i)             # dequantize
score = q · k_hat               # dot product
```

CommVQ does:
```
# Precompute: for each codebook entry c_j and each relative distance d:
lookup[d][j] = q · R_{-d} · c_j    # one table per query

# At attention time:
score_i = lookup[t-i][s_i[0]] + lookup[t-i][s_i[1]] + ...   # pure lookup
```

This is fundamentally about **compute efficiency, not just memory**. Your APKVC necessarily dequantizes before attention; CommVQ bypasses dequantization entirely. This is a separate axis of novelty.

---

## What You Have That CommVQ Doesn't

### 1. Predictive / Delta Coding
The entire anchor + residual chain is absent from CommVQ. CommVQ has no concept of:
- Predicting the next KV from history
- Storing residuals instead of raw vectors
- Anchor resets driven by drift detection

This is your primary contribution. Adjacent decode tokens do have correlated KV states, and exploiting that correlation (especially in the base space after derotation) is an independent and legitimate idea.

### 2. Attention-Aware Rate-Distortion Control
APKVC adaptively decides when to fall back to a full anchor based on measured attention distortion (key-dot or MSE proxy). CommVQ uses a fixed bit budget per token with no adaptive fallback. Your per-token adaptive policy is original and orthogonal to CommVQ.

### 3. Hybrid Prefill + Decode Split
APKVC treats prefill and decode entirely differently (INT8 quantized prefill cache + APKVC decode pipeline). CommVQ does not make this split.

---

## Verdict: Is Your Research Already Done?

**No.** The ideas are complementary, not duplicates:

| Claim | Assessment |
|---|---|
| Both do additive quantization of KV | ✅ True, but AQ is prior art neither paper owns |
| Both are RoPE-aware | ✅ True in spirit, but the mechanisms are completely different |
| CommVQ anticipates your predictive coding | ❌ No. CommVQ has zero prediction, no anchors, no residuals |
| CommVQ anticipates your adaptive RD control | ❌ No. CommVQ uses fixed per-token budget |
| Your derotate/rerotate is the same as CommVQ's commutativity | ❌ No. Yours is a standard engineering choice; CommVQ's is an algebraic structural constraint that unlocks dequantization-free attention — an entirely different property |

### Where you should be careful
The **derotate-before-quantize** trick (your `rope_derotate` + `rope_rotate` in the encode/decode path) is similar enough to what CommVQ describes conceptually that a reviewer could question novelty on that specific sub-step. The important distinction to make in any write-up is:

- Your derotation is in service of **residual compression across a prediction chain** (delta coding).
- CommVQ's commutativity is in service of **bypassing dequantization at attention time**.

These are different goals, different implementations, and different outcomes.

---

## Suggested Framing for a Paper

If you write this up, you should:
1. Cite CommVQ in related work alongside KIVI, PyramidKV, etc.
2. Emphasise your **predictive/delta coding** as the primary novelty — it is structurally different from any cited work.
3. Acknowledge that your RoPE handling uses derotate/rerotate (standard approach) and distinguish it clearly from CommVQ's algebraic commutativity, which targets a different benefit (compute, not just memory).
4. Highlight the **adaptive anchor policy** (attention-distortion-gated resets) as a second distinct contribution.
