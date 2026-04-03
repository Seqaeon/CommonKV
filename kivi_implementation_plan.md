# KIVI Implementation Plan: INT2 & INT4 KV Cache Quantization

---

## 1. Quantization Primitives

The foundation is asymmetric round-to-nearest quantization. For a tensor X:

```
zero_point (z) = min(X)
scale (s)      = (max(X) - min(X)) / (2^B - 1)
quantized      = round((X - z) / s)
dequantized    = quantized * s + z
```

You need two variants of this:

- **Per-token**: compute `z` and `s` independently for each row (token dimension). Each token gets its own scale/zero-point pair.
- **Per-channel**: compute `z` and `s` independently for each column (channel dimension). Each channel gets its own scale/zero-point pair.

Group-wise quantization is a refinement — instead of a single scale for the entire token/channel, you divide it into chunks of size G (default 32) and compute a scale per chunk. This tightens the quantization range significantly.

---

## 2. Asymmetric Treatment of Key vs Value Cache

This is the central insight of KIVI. Key and value caches have structurally different distributions:

- **Key cache** has a few "massive outlier channels" — certain fixed channel indices always carry disproportionately large magnitudes. Per-channel quantization confines error to those channels without contaminating others.
- **Value cache** has no clear outlier structure, but because the attention output is a *weighted sum over tokens* (and attention is sparse), per-token quantization isolates error per token, so unimportant tokens' quantization errors don't corrupt important ones.

Therefore:
- Keys → **group-wise quantization along the channel dimension**
- Values → **group-wise quantization along the token dimension**

---

## 3. The Streaming Problem and the Grouped/Residual Split

Auto-regressive decoding is the hard part. Tokens arrive one at a time, but per-channel key quantization requires seeing multiple tokens at once (you need a full group of G tokens to compute a channel-wise scale). You solve this with a two-part cache:

```
Total Key Cache = [Grouped Part | Residual Part]
                   (quantized)    (full precision)
```

**Grouped part** (`XKg`): All tokens that can form complete groups of size G. Stored in INT2/INT4.

**Residual part** (`XKr`): The overflow tokens that don't yet fill a complete group. Stored in FP16. Max size is R (residual length, default 128).

When `XKr` accumulates R tokens, you quantize the entire residual and flush it into the grouped part, then reset `XKr` to empty.

Value cache uses the same split, but the motivation is slightly different — you keep the most recent R tokens in full precision as a "sliding window" of local context, which is especially important for hard tasks like mathematical reasoning (GSM8K).

---

## 4. Prefill Phase Logic

During prefill, the full prompt is available upfront, so you quantize in one shot:

```
1. Compute XK = X * WK,  XV = X * WV  (standard attention projections)

2. For KEY cache:
   - Split: XKg = XK[:, :l-r],  XKr = XK[:, l-r:]
     where r = l % R  (r tokens go to residual)
   - Quantize XKg per-channel with group size G → Q(XKg)
   - Store: Q(XKg), XKr

3. For VALUE cache:
   - Split: XVg = XV[:l-R, :],  XVr = XV[l-R:, :]
   - Quantize XVg per-token with group size G → Q(XVg)
   - Store: Q(XVg), XVr

4. Return the FULL PRECISION XK, XV to the current layer
   (the quantized cache is only for future decoding steps)
```

> **Key subtlety in step 4:** the current prefill forward pass uses full precision. Only subsequent decoding reloads from the quantized cache.

---

## 5. Decoding Phase Logic

Each new token generates `tK` and `tV`. The logic per step:

**Key cache update:**
```
1. Append tK to XKr
2. If len(XKr) == R:
   - Quantize XKr per-channel → Q(XKr)
   - Concatenate Q(XKr) onto Q(XKg) along token dim
   - Reset XKr = empty
```

**Value cache update:**
```
1. Append tV to XVr
2. If len(XVr) > R:
   - Take the oldest token(s) from XVr
   - Quantize them per-token → Q(XV'r)
   - Concatenate onto Q(XVg)
   - Trim XVr to last R tokens
```

**Attention computation (mixed precision):**
```
Ag = tQ @ dequantize(Q(XKg)).T    ← fused Q_MatMul kernel
Ar = tQ @ XKr.T                   ← full precision matmul

A  = softmax(concat([Ag, Ar], dim=token))

A_grouped  = A[:, :-R]
A_residual = A[:, -R:]

tO = A_grouped @ dequantize(Q(XVg)) + A_residual @ XVr
```

The dequantization should be fused with the matrix multiplication at the tile level (via a custom CUDA/Triton kernel) to avoid materializing the full FP16 tensor in memory — that would defeat the purpose.

---

## 6. The Q_MatMul Kernel (Fused Dequant + MatMul)

The efficiency gain relies on never fully dequantizing. The idea is:

1. Load INT2/INT4 tiles from HBM into SRAM
2. Dequantize within the SRAM tile (cheap, no extra memory traffic)
3. Multiply immediately
4. Accumulate in FP32, write result

This is implemented in Triton for the group-wise quantization kernel, and in CUDA for the fused dequant-matmul. The scale/zero-point tensors are small (one per group) and can stay in registers or shared memory during the tile computation.

---

## 7. Hyperparameters and Their Effects

| Parameter | Default | Effect |
|---|---|---|
| **Group size G** | 32 | Smaller = better accuracy, more scale overhead. G=128 degrades notably. |
| **Residual length R** | 128 | Must be divisible by G. Larger R = better accuracy on hard tasks (GSM8K), slightly less memory savings. R=32 gives higher throughput with modest accuracy trade-off. |
| **Bit width B** | 2 or 4 | INT4 is safe for all models. INT2 works well for Llama/Mistral but Falcon (multi-query attention) needs INT4 due to its already-compressed KV cache. |

---

## 8. INT2 vs INT4 Decision Logic

```
if model uses multi-query attention (e.g. Falcon):
    use KIVI-4 (INT4)
else:  # standard or grouped-query attention (Llama, Mistral)
    use KIVI-2 (INT2)
    # expect <2% accuracy drop on CoQA, TruthfulQA, GSM8K
```

---

## 9. Memory Layout

For INT2, pack 4 values per byte. For INT4, pack 2 values per byte. Store the quantized tensor plus two auxiliary tensors:

- `scales`: shape `[num_groups]` in FP16
- `zero_points`: shape `[num_groups]` in FP16 (or INT8 for extra savings)

For **per-channel key quantization** with group size G:
```
num_groups = num_channels / G
```

For **per-token value quantization** with group size G:
```
num_groups = num_tokens * (hidden_dim / G)
```

---

## 10. Compatibility Notes

- **Orthogonal to weight quantization** (AWQ, GPTQ, etc.) — both can be applied simultaneously.
- **Orthogonal to token eviction** methods (H2O, Scissorhands) — the two approaches can be combined.
- Plugs into standard HuggingFace attention by replacing the KV cache store/load calls, with **no retraining or fine-tuning required**.
