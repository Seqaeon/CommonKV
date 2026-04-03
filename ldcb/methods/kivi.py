"""
KIVI KV-cache compression — axes faithful to the official implementation.

From quant/new_pack.py:
  quant_and_pack_kcache : input [B,H,T,D], groups of G along TOKEN axis (dim=-2).
                          Scale shape [B,H,T//G,1,D].  Per-channel outliers are
                          isolated because each channel (D position) contributes
                          one value per group → channels with persistent large
                          magnitudes get their own per-group range.
  quant_and_pack_vcache : input [B,H,T,D], groups of G along HEAD-DIM axis (dim=-1).
                          Scale shape [B,H,T,D//G,1].  Per-token: each token's
                          channels are independently grouped.

Flush policy (from llama_kivi.py):
  Key  : accumulate in fp16 residual until residual == R tokens, then flush
         ALL R tokens to quantised store at once and reset residual to empty.
  Value: append each new token to fp16 residual; when residual > R, evict the
         single oldest token to quantised store, keeping last R in fp16.
"""

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from .base import KVCacheMethod, CacheState
from ldcb.utils import get_kv_iterator


# ---------------------------------------------------------------------------
# Pure-PyTorch equivalents of quant_and_pack_kcache / quant_and_pack_vcache
# ---------------------------------------------------------------------------

def _quant_K(K: torch.Tensor, group_size: int, bits: int):
    """
    Group-wise asymmetric quantisation along the TOKEN axis.

    K       : [B, H, T, D]   (T must be divisible by group_size)
    Returns
      K_q   : [B, H, T, D]   uint8  (unpacked; ideally bit-packed for memory)
      scale : [B, H, T//G, D]  fp16
      mn    : [B, H, T//G, D]  fp16
    """
    B, H, T, D = K.shape
    G = group_size
    assert T % G == 0, f"_quant_K: T={T} not divisible by G={G}"
    levels = (1 << bits) - 1

    # [B, H, T//G, G, D]
    Kg = K.float().view(B, H, T // G, G, D)
    mn = Kg.amin(dim=-2)                                    # [B, H, T//G, D]
    mx = Kg.amax(dim=-2)
    scale = (mx - mn).clamp(min=1e-5) / levels             # [B, H, T//G, D]

    Kg_q = ((Kg - mn.unsqueeze(-2)) / scale.unsqueeze(-2)).round_().clamp_(0, levels)
    K_q  = Kg_q.to(torch.uint8).view(B, H, T, D)
    return K_q, scale.half(), mn.half()


def _dequant_K(K_q: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
               group_size: int) -> torch.Tensor:
    """
    K_q   : [B, H, T, D]        uint8
    scale : [B, H, T//G, D]     fp16
    mn    : [B, H, T//G, D]     fp16
    Returns fp16 [B, H, T, D].
    """
    B, H, T, D = K_q.shape
    G = group_size
    Kg_q = K_q.float().view(B, H, T // G, G, D)           # [B, H, T//G, G, D]
    out  = Kg_q * scale.float().unsqueeze(-2) + mn.float().unsqueeze(-2)
    return out.view(B, H, T, D).half()


def _quant_V(V: torch.Tensor, group_size: int, bits: int):
    """
    Group-wise asymmetric quantisation along the HEAD-DIM axis.

    V can be a single token [B,H,1,D] or multiple [B,H,T,D].
    D must be divisible by group_size.

    Returns
      V_q   : [B, H, T, D]        uint8
      scale : [B, H, T, D//G]     fp16
      mn    : [B, H, T, D//G]     fp16
    """
    B, H, T, D = V.shape
    G = group_size
    assert D % G == 0, f"_quant_V: D={D} not divisible by G={G}"
    levels = (1 << bits) - 1

    # [B, H, T, D//G, G]
    Vg = V.float().view(B, H, T, D // G, G)
    mn = Vg.amin(dim=-1)                                    # [B, H, T, D//G]
    mx = Vg.amax(dim=-1)
    scale = (mx - mn).clamp(min=1e-5) / levels

    Vg_q = ((Vg - mn.unsqueeze(-1)) / scale.unsqueeze(-1)).round_().clamp_(0, levels)
    V_q  = Vg_q.to(torch.uint8).view(B, H, T, D)
    return V_q, scale.half(), mn.half()


def _dequant_V(V_q: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
               group_size: int) -> torch.Tensor:
    """
    V_q   : [B, H, T, D]        uint8
    scale : [B, H, T, D//G]     fp16
    mn    : [B, H, T, D//G]     fp16
    Returns fp16 [B, H, T, D].
    """
    B, H, T, D = V_q.shape
    G = group_size
    Vg_q = V_q.float().view(B, H, T, D // G, G)           # [B, H, T, D//G, G]
    out  = Vg_q * scale.float().unsqueeze(-1) + mn.float().unsqueeze(-1)
    return out.view(B, H, T, D).half()


# ---------------------------------------------------------------------------
# KIVI method
# ---------------------------------------------------------------------------

class KIVIMethod(KVCacheMethod):

    def __init__(self, bits: int = 4, group_size: int = 32,
                 residual_length: int = 128,
                 cpu_offload_quant: bool = False):
        assert bits in (2, 4, 8)
        self.bits   = bits
        self.G      = group_size
        self.R      = residual_length
        self.cpu    = cpu_offload_quant
        self.name   = f"KIVI-int{bits}"

    # ------------------------------------------------------------------
    # Byte estimate (theoretical bit-packing, matching original convention)
    # ------------------------------------------------------------------

    def _cache_bytes(self, n_tokens, n_layers, n_heads, head_dim) -> int:
        G, R   = self.G, self.R
        T_q    = max(0, n_tokens - R)
        T_r    = min(n_tokens, R)

        # Key quantised: values (bits/8 B each) + scale [T_q//G, D] fp16 + mn same
        K_val  = T_q * head_dim * (self.bits / 8)
        K_meta = (T_q // G) * head_dim * 2 * 2          # scale + mn, fp16

        # Value quantised: values + scale [T_q, D//G] fp16 + mn same
        V_val  = T_q * head_dim * (self.bits / 8)
        V_meta = T_q * (head_dim // G) * 2 * 2

        # Residual window (K + V, fp16)
        res    = 2 * T_r * head_dim * 2

        per_head = K_val + K_meta + V_val + V_meta + res
        return int(per_head * n_layers * n_heads)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps):
        inputs        = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers      = model.config.num_hidden_layers
        n_heads       = getattr(model.config, "num_key_value_heads",
                                model.config.num_attention_heads)
        head_dim      = model.config.hidden_size // model.config.num_attention_heads
        G, R          = self.G, self.R

        generated_ids = inputs.input_ids
        snapshots     = []
        chk_iter      = iter(checkpoint_steps)
        cur_chk       = next(chk_iter, None)
        tok_gen       = 0

        # ---- Prefill ----
        with torch.no_grad():
            outputs = model(generated_ids, use_cache=True)

        # Build per-layer quantised cache.
        # State per layer: [K_q, K_scale, K_mn, V_q, V_scale, V_mn, K_res, V_res]
        #   K_q    : [B,H,T_kq,D]    uint8 or None
        #   K_scale: [B,H,T_kq//G,D] fp16  or None
        #   K_mn   : [B,H,T_kq//G,D] fp16  or None
        #   V_q    : [B,H,T_vq,D]    uint8 or None
        #   V_scale: [B,H,T_vq,D//G] fp16  or None
        #   V_mn   : [B,H,T_vq,D//G] fp16  or None
        #   K_res  : [B,H,T_kr,D]    fp16  (0 <= T_kr < R)
        #   V_res  : [B,H,T_vr,D]    fp16  (0 <= T_vr <= R)
        quant_cache = []
        for _, (layer_K, layer_V) in get_kv_iterator(outputs.past_key_values):
            T = layer_K.shape[2]

            # ---- Keys ----
            # Keep the tail (T % R) tokens as fp16 residual; quantise the rest
            # in chunks of exactly R.  If T < R, everything is residual.
            T_kq = (T // R) * R          # tokens that fill complete R-groups
            K_old = layer_K[:, :, :T_kq, :].to(torch.float16)
            K_res = layer_K[:, :, T_kq:, :].to(torch.float16)   # 0..R-1 tokens

            if T_kq > 0:
                # Quantise in one shot (T_kq is a multiple of R which is a multiple of G)
                assert T_kq % G == 0
                K_q, K_sc, K_mn = _quant_K(K_old, G, self.bits)
                if self.cpu:
                    K_q, K_sc, K_mn = K_q.cpu(), K_sc.cpu(), K_mn.cpu()
            else:
                K_q = K_sc = K_mn = None

            # ---- Values ----
            # Keep last R tokens as fp16 residual; quantise earlier tokens
            # one-by-one (but we can do it in one batch here at prefill).
            T_vq = max(0, T - R)
            V_old = layer_V[:, :, :T_vq, :].to(torch.float16)
            V_res = layer_V[:, :, T_vq:, :].to(torch.float16)   # last min(T,R) tokens

            if T_vq > 0:
                assert head_dim % G == 0, \
                    f"head_dim={head_dim} not divisible by group_size={G}"
                V_q, V_sc, V_mn = _quant_V(V_old, G, self.bits)
                if self.cpu:
                    V_q, V_sc, V_mn = V_q.cpu(), V_sc.cpu(), V_mn.cpu()
            else:
                V_q = V_sc = V_mn = None

            quant_cache.append([K_q, K_sc, K_mn, V_q, V_sc, V_mn, K_res, V_res])

        next_token    = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        tok_gen      += 1
        del outputs

        # ---- Decode loop ----
        with torch.no_grad():
            while tok_gen < max_new_tokens:

                # Reconstruct full fp16 KV for attention
                past_kv = DynamicCache()
                for i, (K_q, K_sc, K_mn, V_q, V_sc, V_mn, K_res, V_res) in enumerate(quant_cache):
                    dev = K_res.device

                    if K_q is not None:
                        kq = K_q.to(dev) if self.cpu else K_q
                        ks = K_sc.to(dev) if self.cpu else K_sc
                        km = K_mn.to(dev) if self.cpu else K_mn
                        K_fp = torch.cat([_dequant_K(kq, ks, km, G).to(model.dtype),
                                          K_res.to(model.dtype)], dim=2)
                        if self.cpu:
                            del kq, ks, km
                    else:
                        K_fp = K_res.to(model.dtype)

                    if V_q is not None:
                        vq = V_q.to(dev) if self.cpu else V_q
                        vs = V_sc.to(dev) if self.cpu else V_sc
                        vm = V_mn.to(dev) if self.cpu else V_mn
                        V_fp = torch.cat([_dequant_V(vq, vs, vm, G).to(model.dtype),
                                          V_res.to(model.dtype)], dim=2)
                        if self.cpu:
                            del vq, vs, vm
                    else:
                        V_fp = V_res.to(model.dtype)

                    past_kv.update(K_fp, V_fp, i)

                outputs = model(next_token, past_key_values=past_kv, use_cache=True)

                new_kvs = [
                    (K[:, :, -1:, :].half().clone(), V[:, :, -1:, :].half().clone())
                    for _, (K, V) in get_kv_iterator(outputs.past_key_values)
                ]
                next_token    = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del outputs, past_kv
                torch.cuda.empty_cache()

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tok_gen      += 1

                # Update quantised cache
                new_cache = []
                for i, (K_q, K_sc, K_mn, V_q, V_sc, V_mn, K_res, V_res) in enumerate(quant_cache):
                    nK, nV = new_kvs[i]

                    # --- Key residual (flush when full = R tokens) ---
                    K_res = torch.cat([K_res, nK], dim=2)
                    if K_res.shape[2] == R:
                        # Flush all R tokens at once
                        pK_q, pK_sc, pK_mn = _quant_K(K_res, G, self.bits)
                        if self.cpu:
                            # Move to CPU and concatenate on CPU to avoid large GPU transfers
                            pk_q_cpu, pk_sc_cpu, pk_mn_cpu = pK_q.cpu(), pK_sc.cpu(), pK_mn.cpu()
                            if K_q is None:
                                K_q, K_sc, K_mn = pk_q_cpu, pk_sc_cpu, pk_mn_cpu
                            else:
                                K_q  = torch.cat([K_q,  pk_q_cpu],  dim=2)
                                K_sc = torch.cat([K_sc, pk_sc_cpu], dim=2)
                                K_mn = torch.cat([K_mn, pk_mn_cpu], dim=2)
                        else:
                            # Standard GPU path
                            if K_q is None:
                                K_q, K_sc, K_mn = pK_q, pK_sc, pK_mn
                            else:
                                K_q  = torch.cat([K_q,  pK_q],  dim=2)
                                K_sc = torch.cat([K_sc, pK_sc], dim=2)
                                K_mn = torch.cat([K_mn, pK_mn], dim=2)
                        K_res = K_res[:, :, :0, :]   # reset to empty

                    # --- Value residual (flush one token at a time) ---
                    V_res = torch.cat([V_res, nV], dim=2)
                    if V_res.shape[2] > R:
                        # Evict oldest single token
                        evict = V_res[:, :, :1, :].contiguous()
                        V_res = V_res[:, :, 1:, :]
                        pV_q, pV_sc, pV_mn = _quant_V(evict, G, self.bits)
                        if self.cpu:
                            # Move to CPU and concatenate on CPU
                            pv_q_cpu, pv_sc_cpu, pv_mn_cpu = pV_q.cpu(), pV_sc.cpu(), pV_mn.cpu()
                            if V_q is None:
                                V_q, V_sc, V_mn = pv_q_cpu, pv_sc_cpu, pv_mn_cpu
                            else:
                                V_q  = torch.cat([V_q,  pv_q_cpu],  dim=2)
                                V_sc = torch.cat([V_sc, pv_sc_cpu], dim=2)
                                V_mn = torch.cat([V_mn, pv_mn_cpu], dim=2)
                        else:
                            # Standard GPU path
                            if V_q is None:
                                V_q, V_sc, V_mn = pV_q, pV_sc, pV_mn
                            else:
                                V_q  = torch.cat([V_q,  pV_q],  dim=2)
                                V_sc = torch.cat([V_sc, pV_sc], dim=2)
                                V_mn = torch.cat([V_mn, pV_mn], dim=2)

                    new_cache.append([K_q, K_sc, K_mn, V_q, V_sc, V_mn, K_res, V_res])

                quant_cache = new_cache

                if tok_gen == cur_chk:
                    T        = generated_ids.shape[1]
                    compr    = self._cache_bytes(T, n_layers, n_heads, head_dim)
                    full_kv  = T * n_layers * n_heads * head_dim * 2 * 2
                    snapshots.append(CacheState(compressed_bytes=compr, fullkv_bytes=full_kv))
                    cur_chk  = next(chk_iter, None)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        T       = generated_ids.shape[1]
        compr   = self._cache_bytes(T, n_layers, n_heads, head_dim)
        full_kv = T * n_layers * n_heads * head_dim * 2 * 2
        text    = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        final   = CacheState(compressed_bytes=compr, fullkv_bytes=full_kv)

        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final)

        return text, snapshots, final
