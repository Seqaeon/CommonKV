"""
KIVI KV-cache compression — aligned with official quant/new_pack.py math.

Key/Value quantization axes (from quant/new_pack.py):
  quant_and_pack_kcache : groups of G along TOKEN axis (dim=-2).
  quant_and_pack_vcache : groups of G along HEAD-DIM axis (dim=-1).

Values are bit-packed into int32 (matching the reference) so byte
estimates are accurate.  Dequantization is pure PyTorch so no Triton
dependency is needed.

Flush policy (from models/llama_kivi.py):
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
# Triton-accelerated bit-packing (from KIVI/quant/new_pack.py)
# Falls back to pure-PyTorch when Triton is unavailable or op is on CPU.
# ---------------------------------------------------------------------------

try:
    import sys as _sys
    import os as _os
    _kivi_quant_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "KIVI", "quant"
    )
    if _kivi_quant_dir not in _sys.path:
        _sys.path.insert(0, _kivi_quant_dir)
    from new_pack import (
        quant_and_pack_kcache as _triton_quant_K,
        quant_and_pack_vcache as _triton_quant_V,
        unpack_and_dequant_kcache as _triton_dequant_K,
        unpack_and_dequant_vcache as _triton_dequant_V,
    )
    _HAS_TRITON_KIVI = True
    print("[KIVI] Using Triton-accelerated kernels from KIVI/quant/new_pack.py")
except Exception as _e:
    _HAS_TRITON_KIVI = False
    print(f"[KIVI] Triton kernels unavailable ({_e}), falling back to pure-PyTorch bit-packing.")

def _pack_tensor(data: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    """Pack integer tensor along pack_dim into int32 words (pure PyTorch)."""
    assert bits in (2, 4, 8)
    shape = data.shape
    feat_per_int = 32 // bits
    assert shape[pack_dim] % feat_per_int == 0
    packed_shape = shape[:pack_dim] + (shape[pack_dim] // feat_per_int,) + shape[pack_dim + 1:]
    data_int = data.to(torch.int32)
    code = torch.zeros(packed_shape, dtype=torch.int32, device=data.device)
    for j in range(feat_per_int):
        src_idx = [slice(None)] * len(shape)
        src_idx[pack_dim] = slice(j, shape[pack_dim], feat_per_int)
        dst_idx = [slice(None)] * len(packed_shape)
        dst_idx[pack_dim] = slice(None)
        code[tuple(dst_idx)] |= (data_int[tuple(src_idx)] << (bits * j))
    return code


def _unpack_tensor(code: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    """Unpack int32 tensor to int16 values (pure PyTorch)."""
    assert bits in (2, 4, 8)
    shape = code.shape
    feat_per_int = 32 // bits
    new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim + 1:]
    out = torch.zeros(new_shape, dtype=torch.int16, device=code.device)
    mask = (0xFF >> (8 - bits))
    for j in range(feat_per_int):
        dst_idx = [slice(None)] * len(new_shape)
        dst_idx[pack_dim] = slice(j, new_shape[pack_dim], feat_per_int)
        out[tuple(dst_idx)] = ((code >> (bits * j)) & mask).to(torch.int16)
    return out


# ---------------------------------------------------------------------------
# K quantisation: groups of G tokens along the TOKEN axis (dim=-2)
# ---------------------------------------------------------------------------

def _quant_K(K: torch.Tensor, group_size: int, bits: int):
    """
    Group-wise asymmetric quantisation along the TOKEN axis.
    K      : [B, H, T, D]   (T must be divisible by group_size)
    Returns
      K_code : [B, H, T//feat_per_int, D]  int32  (bit-packed)
      scale  : [B, H, T//G, 1, D]          fp32
      mn     : [B, H, T//G, 1, D]          fp32
    Prefers Triton kernel (3-5x faster); falls back to pure-PyTorch on CPU.
    """
    B, H, T, D = K.shape
    G = group_size
    assert T % G == 0, f"_quant_K: T={T} not divisible by G={G}"
    levels = (1 << bits) - 1

    if _HAS_TRITON_KIVI and K.is_cuda:
        # Triton path: expects [B, H, T, D] fp16/bf16
        # Returns code [B, H, T//fpi, D], scale [B,H,T//G,D], mn [B,H,T//G,D]
        # Note: Triton convention has scale/mn trailing D not leading, reshape to match internal fmt
        try:
            code, sc, mn = _triton_quant_K(K.half(), G, bits)
            # Triton returns scale/mn as [B, H, T//G, D]; we need [B, H, T//G, 1, D]
            return code, sc.unsqueeze(-2).float(), mn.unsqueeze(-2).float()
        except Exception:
            pass  # fall through to PyTorch path

    # Pure-PyTorch fallback
    Kg = K.float().view(B, H, T // G, G, D)            # [B, H, T//G, G, D]
    mn = Kg.amin(dim=-2, keepdim=True)                  # [B, H, T//G, 1, D]
    mx = Kg.amax(dim=-2, keepdim=True)
    scale = (mx - mn).clamp(min=1e-5) / levels         # [B, H, T//G, 1, D]

    Kg_q = ((Kg - mn) / scale).round_().clamp_(0, levels).to(torch.int32)
    K_int = Kg_q.view(B, H, T, D)                      # [B, H, T, D] int32

    feat_per_int = 32 // bits
    assert T % feat_per_int == 0
    K_code = _pack_tensor(K_int, bits, pack_dim=2)      # [B, H, T//fpi, D]
    return K_code, scale, mn


def _dequant_K(K_code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
               group_size: int, bits: int, T: int) -> torch.Tensor:
    """
    K_code : [B, H, T//feat_per_int, D]  int32
    scale  : [B, H, T//G, 1, D]          fp32
    mn     : [B, H, T//G, 1, D]          fp32
    Returns fp16 [B, H, T, D].
    """
    B, H, _, D = K_code.shape
    G = group_size
    K_int = _unpack_tensor(K_code, bits, pack_dim=2).to(torch.float32)  # [B, H, T, D]
    Kg = K_int.view(B, H, T // G, G, D)
    out = Kg * scale + mn
    return out.view(B, H, T, D)#.half()


# ---------------------------------------------------------------------------
# V quantisation: groups of G elements along the HEAD-DIM axis (dim=-1)
# ---------------------------------------------------------------------------

def _quant_V(V: torch.Tensor, group_size: int, bits: int):
    """
    Group-wise asymmetric quantisation along the HEAD-DIM axis.
    V      : [B, H, T, D]   (D must be divisible by group_size)
    Returns
      V_code : [B, H, T, D//feat_per_int]  int32
      scale  : [B, H, T, D//G, 1]          fp32
      mn     : [B, H, T, D//G, 1]          fp32
    Prefers Triton kernel; falls back to pure-PyTorch on CPU.
    """
    B, H, T, D = V.shape
    G = group_size
    assert D % G == 0, f"_quant_V: D={D} not divisible by G={G}"
    levels = (1 << bits) - 1

    if _HAS_TRITON_KIVI and V.is_cuda:
        try:
            code, sc, mn = _triton_quant_V(V.half(), G, bits)
            # Triton returns scale/mn as [B, H, T, D//G]; we need [B, H, T, D//G, 1]
            return code, sc.unsqueeze(-1).float(), mn.unsqueeze(-1).float()
        except Exception:
            pass

    # Pure-PyTorch fallback
    Vg = V.float().view(B, H, T, D // G, G)            # [B, H, T, D//G, G]
    mn = Vg.amin(dim=-1, keepdim=True)                  # [B, H, T, D//G, 1]
    mx = Vg.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-5) / levels

    Vg_q = ((Vg - mn) / scale).round_().clamp_(0, levels).to(torch.int32)
    V_int = Vg_q.view(B, H, T, D)                      # [B, H, T, D] int32

    feat_per_int = 32 // bits
    assert D % feat_per_int == 0
    V_code = _pack_tensor(V_int, bits, pack_dim=3)      # [B, H, T, D//fpi]
    return V_code, scale, mn


def _dequant_V(V_code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
               group_size: int, bits: int, D: int) -> torch.Tensor:
    """
    V_code : [B, H, T, D//feat_per_int]  int32
    scale  : [B, H, T, D//G, 1]          fp32
    mn     : [B, H, T, D//G, 1]          fp32
    Returns fp16 [B, H, T, D].
    """
    B, H, T, _ = V_code.shape
    G = group_size
    V_int = _unpack_tensor(V_code, bits, pack_dim=3).to(torch.float32)   # [B, H, T, D]
    Vg = V_int.view(B, H, T, D // G, G)
    out = Vg * scale + mn
    return out.view(B, H, T, D)#.half()


# ---------------------------------------------------------------------------
# KIVI method
# ---------------------------------------------------------------------------

class KIVIMethod(KVCacheMethod):

    def __init__(self, bits: int = 4, group_size: int = 32,
                 residual_length: int = 128,
                 cpu_offload_quant: bool = False):
        assert bits in (2, 4, 8)
        self.bits = bits
        self.G = group_size
        self.R = residual_length
        self.cpu = cpu_offload_quant
        self.name = f"KIVI-int{bits}"

    # ------------------------------------------------------------------
    # Byte estimate — uses actual int32 bit-packed size
    # ------------------------------------------------------------------

    def _cache_bytes(self, n_tokens, n_layers, n_heads, head_dim) -> int:
        G, R, bits = self.G, self.R, self.bits
        feat_per_int = 32 // bits
        T_q = max(0, n_tokens - R)
        T_r = min(n_tokens, R)

        # Key quantised (bit-packed along token dim): T_q * D / feat_per_int * 4B per int32
        K_val  = (T_q // feat_per_int) * head_dim * 4 if T_q > 0 else 0
        K_meta = (T_q // G) * head_dim * 4 * 2          # scale + mn, fp32 (stored as fp32 internally)

        # Value quantised (bit-packed along head-dim): T_q * (D / feat_per_int) * 4B
        V_val  = T_q * (head_dim // feat_per_int) * 4 if T_q > 0 else 0
        V_meta = T_q * (head_dim // G) * 4 * 2

        # Residual window (K + V, fp16)
        res = 2 * T_r * head_dim * 2

        per_head = K_val + K_meta + V_val + V_meta + res
        return int(per_head * n_layers * n_heads)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def prefill(self, model, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        mdtype = next(model.parameters()).dtype
        G, R = self.G, self.R

        with torch.no_grad():
            outputs = model(inputs.input_ids, use_cache=True)

        quant_cache = []
        for _, (layer_K, layer_V) in get_kv_iterator(outputs.past_key_values):
            T = layer_K.shape[2]
            D = layer_K.shape[3]

            # ---- Keys ----
            T_kq = (T // R) * R
            K_old = layer_K[:, :, :T_kq, :].to(dtype=torch.float32)
            K_res = layer_K[:, :, T_kq:, :].to(dtype=mdtype)

            if T_kq > 0:
                assert T_kq % G == 0
                K_code, K_sc, K_mn = _quant_K(K_old, G, self.bits)
                if self.cpu:
                    K_code, K_sc, K_mn = K_code.cpu(), K_sc.cpu(), K_mn.cpu()
            else:
                K_code = K_sc = K_mn = None

            # ---- Values ----
            T_vq = max(0, T - R)
            V_old = layer_V[:, :, :T_vq, :].to(dtype=torch.float32)
            V_res = layer_V[:, :, T_vq:, :].to(dtype=mdtype)

            if T_vq > 0:
                assert D % G == 0
                V_code, V_sc, V_mn = _quant_V(V_old, G, self.bits)
                if self.cpu:
                    V_code, V_sc, V_mn = V_code.cpu(), V_sc.cpu(), V_mn.cpu()
            else:
                V_code = V_sc = V_mn = None

            quant_cache.append([K_code, K_sc, K_mn, T_kq,
                                 V_code, V_sc, V_mn, D,
                                 K_res, V_res])
        return quant_cache

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps, cached_state=None):
        inputs   = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers = model.config.num_hidden_layers
        n_heads  = getattr(model.config, "num_key_value_heads",
                           model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        mdtype   = next(model.parameters()).dtype   # use actual model dtype (fp16 OR bf16)
        G, R     = self.G, self.R

        generated_ids = inputs.input_ids
        snapshots = []
        chk_iter  = iter(checkpoint_steps)
        cur_chk   = next(chk_iter, None)
        tok_gen   = 0
        
        if cached_state is not None:
            quant_cache = cached_state
            # Prefill the new part of the prompt starting from cached_state
            with torch.no_grad():
                past_kv = DynamicCache()
                for i, (K_code, K_sc, K_mn, K_T_q,
                         V_code, V_sc, V_mn, V_D,
                         K_res, V_res) in enumerate(quant_cache):
                    dev = K_res.device
                    if K_code is not None:
                        K_fp = torch.cat([_dequant_K(K_code.to(dev), K_sc.to(dev), K_mn.to(dev), G, self.bits, K_T_q).to(mdtype), K_res], dim=2)
                    else:
                        K_fp = K_res
                    if V_code is not None:
                        V_fp = torch.cat([_dequant_V(V_code.to(dev), V_sc.to(dev), V_mn.to(dev), G, self.bits, V_D).to(mdtype), V_res], dim=2)
                    else:
                        V_fp = V_res
                    past_kv.update(K_fp, V_fp, i)

                outputs = model(generated_ids, past_key_values=past_kv, use_cache=True)
            
            # Incorporate new tokens into quant_cache
            new_pkvs = list(get_kv_iterator(outputs.past_key_values))
            for i, (_, (layer_K, layer_V)) in enumerate(new_pkvs):
                T = layer_K.shape[2]
                D = layer_K.shape[3]
                
                # Reserving the logic for simplicity: just update the résiduels for now
                # In a real KIVI resumption, we'd need to re-quantize if we cross R.
                # Since 'prompt' in GSM8K after few-shot is small, we mostly just append to residuals.
                # But to be safe, we should probably just re-run the quant logic on the whole new layer_K/layer_V
                
                # ---- Keys ----
                T_kq = (T // R) * R
                K_old = layer_K[:, :, :T_kq, :].to(dtype=torch.float32)
                K_res = layer_K[:, :, T_kq:, :].to(dtype=mdtype)
                if T_kq > 0:
                    K_code, K_sc, K_mn = _quant_K(K_old, G, self.bits)
                    if self.cpu: K_code, K_sc, K_mn = K_code.cpu(), K_sc.cpu(), K_mn.cpu()
                else:
                    K_code = K_sc = K_mn = None

                # ---- Values ----
                T_vq = max(0, T - R)
                V_old = layer_V[:, :, :T_vq, :].to(dtype=torch.float32)
                V_res = layer_V[:, :, T_vq:, :].to(dtype=mdtype)
                if T_vq > 0:
                    V_code, V_sc, V_mn = _quant_V(V_old, G, self.bits)
                    if self.cpu: V_code, V_sc, V_mn = V_code.cpu(), V_sc.cpu(), V_mn.cpu()
                else:
                    V_code = V_sc = V_mn = None

                quant_cache[i] = [K_code, K_sc, K_mn, T_kq, V_code, V_sc, V_mn, D, K_res, V_res]

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            tok_gen = 1
            del outputs, past_kv
        else:
            # ---- Normal Prefill ----
            with torch.no_grad():
                outputs = model(generated_ids, use_cache=True)
            
            quant_cache = []
            for _, (layer_K, layer_V) in get_kv_iterator(outputs.past_key_values):
                T, D = layer_K.shape[2], layer_K.shape[3]
                T_kq = (T // R) * R
                K_old, K_res = layer_K[:, :, :T_kq, :].float(), layer_K[:, :, T_kq:, :].to(mdtype)
                K_code, K_sc, K_mn = _quant_K(K_old, G, self.bits) if T_kq > 0 else (None, None, None)
                if self.cpu and K_code is not None: K_code, K_sc, K_mn = K_code.cpu(), K_sc.cpu(), K_mn.cpu()

                T_vq = max(0, T - R)
                V_old, V_res = layer_V[:, :, :T_vq, :].float(), layer_V[:, :, T_vq:, :].to(mdtype)
                V_code, V_sc, V_mn = _quant_V(V_old, G, self.bits) if T_vq > 0 else (None, None, None)
                if self.cpu and V_code is not None: V_code, V_sc, V_mn = V_code.cpu(), V_sc.cpu(), V_mn.cpu()
                
                quant_cache.append([K_code, K_sc, K_mn, T_kq, V_code, V_sc, V_mn, D, K_res, V_res])

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            tok_gen = 1
            del outputs

        # ---- Decode loop ----
        with torch.no_grad():
            while tok_gen < max_new_tokens:
                # Reconstruct full KV for attention
                past_kv = DynamicCache()
                for i, (K_code, K_sc, K_mn, K_T_q, V_code, V_sc, V_mn, V_D, K_res, V_res) in enumerate(quant_cache):
                    dev = K_res.device
                    if K_code is not None:
                        kc, ks, km = (K_code.to(dev), K_sc.to(dev), K_mn.to(dev)) if self.cpu else (K_code, K_sc, K_mn)
                        K_fp = torch.cat([_dequant_K(kc, ks, km, G, self.bits, K_T_q).to(mdtype), K_res], dim=2)
                    else: K_fp = K_res
                    if V_code is not None:
                        vc, vs, vm = (V_code.to(dev), V_sc.to(dev), V_mn.to(dev)) if self.cpu else (V_code, V_sc, V_mn)
                        V_fp = torch.cat([_dequant_V(vc, vs, vm, G, self.bits, V_D).to(mdtype), V_res], dim=2)
                    else: V_fp = V_res
                    past_kv.update(K_fp, V_fp, i)

                outputs = model(next_token, past_key_values=past_kv, use_cache=True)
                new_kvs = [(K[:, :, -1:, :].to(mdtype).clone(), V[:, :, -1:, :].to(mdtype).clone())
                            for _, (K, V) in get_kv_iterator(outputs.past_key_values)]
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del outputs, past_kv
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tok_gen += 1

                for i, state in enumerate(quant_cache):
                    K_code, K_sc, K_mn, K_T_q, V_code, V_sc, V_mn, V_D, K_res, V_res = state
                    nK, nV = new_kvs[i]
                    K_res = torch.cat([K_res, nK], dim=2)
                    if K_res.shape[2] == R:
                        pK_code, pK_sc, pK_mn = _quant_K(K_res.float(), G, self.bits)
                        if self.cpu: pK_code, pK_sc, pK_mn = pK_code.cpu(), pK_sc.cpu(), pK_mn.cpu()
                        if K_code is None: K_code, K_sc, K_mn, K_T_q = pK_code, pK_sc, pK_mn, R
                        else:
                            K_code = torch.cat([K_code, pK_code], dim=2)
                            K_sc = torch.cat([K_sc, pK_sc], dim=2)
                            K_mn = torch.cat([K_mn, pK_mn], dim=2)
                            K_T_q += R
                        K_res = K_res[:, :, :0, :]

                    V_res = torch.cat([V_res, nV], dim=2)
                    if V_res.shape[2] > R:
                        evict = V_res[:, :, :1, :].contiguous()
                        V_res = V_res[:, :, 1:, :]
                        pV_code, pV_sc, pV_mn = _quant_V(evict.float(), G, self.bits)
                        if self.cpu: pV_code, pV_sc, pV_mn = pV_code.cpu(), pV_sc.cpu(), pV_mn.cpu()
                        if V_code is None: V_code, V_sc, V_mn = pV_code, pV_sc, pV_mn
                        else:
                            V_code = torch.cat([V_code, pV_code], dim=2)
                            V_sc = torch.cat([V_sc, pV_sc], dim=2)
                            V_mn = torch.cat([V_mn, pV_mn], dim=2)
                    quant_cache[i] = [K_code, K_sc, K_mn, K_T_q, V_code, V_sc, V_mn, V_D, K_res, V_res]

                if tok_gen == cur_chk:
                    T = generated_ids.shape[1]
                    compr = self._cache_bytes(T, n_layers, n_heads, head_dim)
                    full_kv = T * n_layers * n_heads * head_dim * 2 * 2
                    snapshots.append(CacheState(compressed_bytes=compr, fullkv_bytes=full_kv))
                    cur_chk = next(chk_iter, None)
                if next_token.item() == tokenizer.eos_token_id: break

        T = generated_ids.shape[1]
        compr = self._cache_bytes(T, n_layers, n_heads, head_dim)
        full_kv = T * n_layers * n_heads * head_dim * 2 * 2
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        final = CacheState(compressed_bytes=compr, fullkv_bytes=full_kv)
        while len(snapshots) < len(checkpoint_steps): snapshots.append(final)
        return text, snapshots, final

