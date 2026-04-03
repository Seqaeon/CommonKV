import torch
from transformers.cache_utils import DynamicCache
from .base import KVCacheMethod, CacheState
from ldcb.utils import get_kv_iterator


class KIVIMethod(KVCacheMethod):
    """
    KIVI KV-cache quantization — paper-correct implementation.

    Key design points (from KIVI §3):
    ─────────────────────────────────
    Key cache:   group-wise asymmetric quantization along the **channel** (head-dim)
                 axis, group size G.  Per-channel quantization isolates channel
                 outliers so they don't inflate the scale for all tokens.

    Value cache: group-wise asymmetric quantization along the **token** axis,
                 group size G.  Per-token quantization isolates unimportant tokens'
                 error from attending tokens.

    Residual window: the most-recent R tokens are kept in fp16.  When the window
    fills to exactly R tokens (`len == R`) the full window is flushed to the
    quantised store and the window is reset to empty.

    Each token is quantised exactly once when it graduates from the residual
    window — no re-quantisation, no error accumulation.

    Storage layout (per layer)
    ──────────────────────────
    K quantised part:  (K_q, K_scale, K_zero)
        K_q    : [B, H, T_q, D_padded]  uint8   (group-packed along channel)
        K_scale: [B, H, T_q, D//G]      float16
        K_zero : [B, H, T_q, D//G]      float16

    V quantised part:  (V_q, V_scale, V_zero)
        V_q    : [B, H, T_q_g, G]       uint8   (group-packed along token)
                 where T_q_g = T_q // G (number of complete token groups)
        V_scale: [B, H, T_q_g, D]       float16
        V_zero : [B, H, T_q_g, D]       float16

    K/V residual: [B, H, T_r, D] float16  (T_r <= R)

    Note on V layout: quantizing along the token axis in groups of G means we
    always flush G tokens at once from the residual window → effective R must
    equal exactly G (or a multiple). We default R = G = 32 and flush in one shot.
    """

    def __init__(self, bits: int = 4, group_size: int = 32,
                 residual_length: int = 128,
                 cpu_offload_quant: bool = False):
        assert bits in (2, 4, 8)
        self.bits = bits
        self.G = group_size
        self.R = residual_length
        self.cpu_offload_quant = cpu_offload_quant
        self.name = f"KIVI-int{bits}"

    # ------------------------------------------------------------------
    # Per-channel-group quantization for Keys
    # ------------------------------------------------------------------

    def _quantize_K(self, K: torch.Tensor):
        """
        Group-wise asymmetric quantization along the channel (head-dim) axis.

        K : [B, H, T, D]
        Returns
        -------
        K_q    : [B, H, T, D]       uint8
        K_scale: [B, H, T, D//G]    float16  (one scale per group per token)
        K_zero : [B, H, T, D//G]    float16
        """
        B, H, T, D = K.shape
        G = self.G
        levels = (1 << self.bits) - 1   # e.g. 15 for int4, 3 for int2

        # Pad D to a multiple of G if necessary
        pad = (G - D % G) % G
        if pad:
            K = torch.nn.functional.pad(K, (0, pad))
        D_pad = K.shape[-1]  # D + pad

        # Reshape to expose groups: [B, H, T, D_pad//G, G]
        Kg = K.float().view(B, H, T, D_pad // G, G)

        K_min = Kg.amin(dim=-1)            # [B, H, T, D_pad//G]
        K_max = Kg.amax(dim=-1)
        K_scale = (K_max - K_min).clamp(min=1e-5) / levels   # [B, H, T, D_pad//G]
        K_zero  = K_min                                        # [B, H, T, D_pad//G]

        # Quantise
        Kg_q = ((Kg - K_min.unsqueeze(-1)) / K_scale.unsqueeze(-1)).round().clamp(0, levels)
        K_q  = Kg_q.to(torch.uint8).view(B, H, T, D_pad)    # [B, H, T, D_pad]

        # Trim padding back
        if pad:
            K_q = K_q[..., :D]

        return K_q.contiguous(), K_scale.to(torch.float16), K_zero.to(torch.float16)

    def _dequantize_K(self, K_q: torch.Tensor,
                      K_scale: torch.Tensor,
                      K_zero:  torch.Tensor,
                      orig_D:  int) -> torch.Tensor:
        """
        K_q    : [B, H, T, D]       uint8
        K_scale: [B, H, T, D//G]    float16
        K_zero : [B, H, T, D//G]    float16
        Returns dequantised float16 [B, H, T, orig_D]
        """
        B, H, T, D = K_q.shape
        G = self.G
        pad = (G - orig_D % G) % G
        D_pad = orig_D + pad

        # Pad quantised tensor if needed
        if pad:
            K_q = torch.nn.functional.pad(K_q, (0, pad))

        Kg_q = K_q.float().view(B, H, T, D_pad // G, G)  # [B, H, T, n_groups, G]
        K_out = Kg_q * K_scale.float().unsqueeze(-1) + K_zero.float().unsqueeze(-1)
        K_out = K_out.view(B, H, T, D_pad)

        if pad:
            K_out = K_out[..., :orig_D]
        return K_out.to(torch.float16)

    # ------------------------------------------------------------------
    # Per-token-group quantization for Values
    # ------------------------------------------------------------------

    def _quantize_V(self, V: torch.Tensor):
        """
        Group-wise asymmetric quantization along the token axis.
        Expects T to be a multiple of G (only called on complete groups).

        V : [B, H, T, D]  (T must be divisible by G)
        Returns
        -------
        V_q    : [B, H, T, D]       uint8
        V_scale: [B, H, T//G, D]    float16
        V_zero : [B, H, T//G, D]    float16
        """
        B, H, T, D = V.shape
        G = self.G
        assert T % G == 0, f"V quantization called on T={T} not divisible by G={G}"
        levels = (1 << self.bits) - 1

        # Reshape to expose token groups: [B, H, T//G, G, D]
        Vg = V.float().view(B, H, T // G, G, D)

        V_min   = Vg.amin(dim=-2)    # [B, H, T//G, D]
        V_max   = Vg.amax(dim=-2)
        V_scale = (V_max - V_min).clamp(min=1e-5) / levels
        V_zero  = V_min

        Vg_q    = ((Vg - V_min.unsqueeze(-2)) / V_scale.unsqueeze(-2)).round().clamp(0, levels)
        V_q     = Vg_q.to(torch.uint8).view(B, H, T, D)     # [B, H, T, D]

        return V_q.contiguous(), V_scale.to(torch.float16), V_zero.to(torch.float16)

    def _dequantize_V(self, V_q: torch.Tensor,
                      V_scale: torch.Tensor,
                      V_zero:  torch.Tensor) -> torch.Tensor:
        """
        V_q    : [B, H, T, D]       uint8
        V_scale: [B, H, T//G, D]    float16
        V_zero : [B, H, T//G, D]    float16
        Returns dequantised float16 [B, H, T, D]
        """
        B, H, T, D = V_q.shape
        G = self.G
        Vg_q = V_q.float().view(B, H, T // G, G, D)  # [B, H, T//G, G, D]
        V_out = Vg_q * V_scale.float().unsqueeze(-2) + V_zero.float().unsqueeze(-2)
        return V_out.view(B, H, T, D).to(torch.float16)

    # ------------------------------------------------------------------
    # Helpers: flush residual window → quantised store
    # ------------------------------------------------------------------

    def _flush_K_residual(self, K_res, K_q, K_scale, K_zero):
        """Quantize K_res and append to quantised store. Returns updated store tensors."""
        pK_q, pK_s, pK_z = self._quantize_K(K_res)
        if self.cpu_offload_quant:
            pK_q, pK_s, pK_z = pK_q.cpu(), pK_s.cpu(), pK_z.cpu()
        if K_q is None:
            return pK_q, pK_s, pK_z
        return (torch.cat([K_q, pK_q], dim=2),
                torch.cat([K_scale, pK_s], dim=2),
                torch.cat([K_zero, pK_z], dim=2))

    def _flush_V_residual(self, V_flush, V_q, V_scale, V_zero):
        """Quantize a complete G-token block of V and append to quantised store."""
        pV_q, pV_s, pV_z = self._quantize_V(V_flush)
        if self.cpu_offload_quant:
            pV_q, pV_s, pV_z = pV_q.cpu(), pV_s.cpu(), pV_z.cpu()
        if V_q is None:
            return pV_q, pV_s, pV_z
        return (torch.cat([V_q, pV_q], dim=2),
                torch.cat([V_scale, pV_s], dim=2),
                torch.cat([V_zero, pV_z], dim=2))

    # ------------------------------------------------------------------
    # Compression byte estimate
    # ------------------------------------------------------------------

    def _cache_bytes(self, n_tokens: int, n_layers: int,
                     n_heads: int, head_dim: int) -> int:
        G = self.G
        R = self.R
        T_q = max(0, n_tokens - R)
        T_r  = min(n_tokens, R)

        # Per-group count along D for K, along T for V
        n_groups_K = max(1, (head_dim + G - 1) // G)
        n_groups_V = max(1, T_q // G)

        # K quantised: uint8 values + fp16 scale + fp16 zero  (per token, per group)
        K_quant   = T_q * head_dim * (self.bits / 8)
        K_meta    = T_q * n_groups_K * 2 * 2          # scale + zero, fp16

        # V quantised: uint8 values + fp16 scale + fp16 zero  (per group-token, per channel)
        V_T_q = (T_q // G) * G   # round down to complete groups
        V_quant = V_T_q * head_dim * (self.bits / 8)
        V_meta  = (T_q // G) * head_dim * 2 * 2       # scale + zero, fp16

        # Residual: K + V in fp16
        residual = 2 * T_r * head_dim * 2             # K and V

        # Per layer, per head
        total_per_head = K_quant + K_meta + V_quant + V_meta + residual
        return int(total_per_head * n_layers * n_heads)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps):
        inputs     = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers   = model.config.num_hidden_layers
        n_heads    = getattr(model.config, "num_key_value_heads",
                             model.config.num_attention_heads)
        head_dim   = model.config.hidden_size // model.config.num_attention_heads

        generated_ids = inputs.input_ids
        snapshots     = []
        next_chk      = iter(checkpoint_steps)
        cur_chk       = next(next_chk, None)
        tokens_gen    = 0

        # ----------------------------------------------------------------
        # Prefill — run model, get full fp16 KV
        # ----------------------------------------------------------------
        with torch.no_grad():
            outputs = model(generated_ids, use_cache=True)

        # Build quantised cache.
        # K policy: last R tokens → fp16 residual; earlier → quantise per-channel-group
        # V policy: keep fp16 residual until we have >= G tokens per group, then flush
        #
        # quant_cache per layer: [K_q, K_scale, K_zero, V_q, V_scale, V_zero, K_res, V_res]
        #   K_q/V_q         : [B,H,T_q,D] uint8  or None
        #   K_scale/K_zero  : [B,H,T_q,D//G] fp16 or None
        #   V_scale/V_zero  : [B,H,T_q//G,D] fp16 or None
        #   K_res/V_res     : [B,H,T_r,D] fp16
        quant_cache = []
        for _, (layer_K, layer_V) in get_kv_iterator(outputs.past_key_values):
            T = layer_K.shape[2]
            R = self.R
            G = self.G

            # ---- Keys ----
            split_K = max(T - R, 0)
            K_old   = layer_K[:, :, :split_K, :].to(torch.float16)
            K_res   = layer_K[:, :, split_K:, :].to(torch.float16)

            if split_K > 0:
                K_q, K_scale, K_zero = self._quantize_K(K_old)
                if self.cpu_offload_quant:
                    K_q, K_scale, K_zero = K_q.cpu(), K_scale.cpu(), K_zero.cpu()
            else:
                K_q = K_scale = K_zero = None

            # ---- Values ----
            # Quantise complete groups of G tokens; remainder → residual
            n_groups_V  = T // G
            split_V     = n_groups_V * G                # tokens that form complete groups
            V_old       = layer_V[:, :, :split_V, :].to(torch.float16)
            V_res       = layer_V[:, :, split_V:, :].to(torch.float16)  # < G tokens, fp16

            if split_V > 0:
                V_q, V_scale, V_zero = self._quantize_V(V_old)
                if self.cpu_offload_quant:
                    V_q, V_scale, V_zero = V_q.cpu(), V_scale.cpu(), V_zero.cpu()
            else:
                V_q = V_scale = V_zero = None

            quant_cache.append([
                K_q, K_scale, K_zero,
                V_q, V_scale, V_zero,
                K_res, V_res,
            ])

        next_token     = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids  = torch.cat([generated_ids, next_token], dim=1)
        tokens_gen    += 1
        del outputs

        # ----------------------------------------------------------------
        # Decode loop
        # ----------------------------------------------------------------
        with torch.no_grad():
            while tokens_gen < max_new_tokens:

                orig_D   = head_dim
                past_kv_deq = DynamicCache()

                for i, (K_q, K_scale, K_zero,
                        V_q, V_scale, V_zero,
                        K_res, V_res) in enumerate(quant_cache):

                    dev = K_res.device

                    # --- Reconstruct K ---
                    if K_q is not None:
                        kq = K_q.to(dev)   if self.cpu_offload_quant else K_q
                        ks = K_scale.to(dev) if self.cpu_offload_quant else K_scale
                        kz = K_zero.to(dev)  if self.cpu_offload_quant else K_zero
                        K_fp = torch.cat([
                            self._dequantize_K(kq, ks, kz, orig_D).to(model.dtype),
                            K_res.to(model.dtype),
                        ], dim=2)
                        if self.cpu_offload_quant:
                            del kq, ks, kz
                    else:
                        K_fp = K_res.to(model.dtype)

                    # --- Reconstruct V ---
                    if V_q is not None:
                        vq = V_q.to(dev)    if self.cpu_offload_quant else V_q
                        vs = V_scale.to(dev) if self.cpu_offload_quant else V_scale
                        vz = V_zero.to(dev)  if self.cpu_offload_quant else V_zero
                        V_fp = torch.cat([
                            self._dequantize_V(vq, vs, vz).to(model.dtype),
                            V_res.to(model.dtype),
                        ], dim=2)
                        if self.cpu_offload_quant:
                            del vq, vs, vz
                    else:
                        V_fp = V_res.to(model.dtype)

                    past_kv_deq.update(K_fp, V_fp, i)

                outputs    = model(next_token, past_key_values=past_kv_deq, use_cache=True)

                # Extract only the new token's KV (last position)
                new_tok_kvs = [
                    (K[:, :, -1:, :].to(torch.float16).clone(),
                     V[:, :, -1:, :].to(torch.float16).clone())
                    for _, (K, V) in get_kv_iterator(outputs.past_key_values)
                ]
                next_token    = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del outputs, past_kv_deq
                torch.cuda.empty_cache()

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_gen   += 1

                # Update cache: append new token, flush residuals when ready
                new_quant_cache = []
                G = self.G
                R = self.R

                for i, (K_q, K_scale, K_zero,
                        V_q, V_scale, V_zero,
                        K_res, V_res) in enumerate(quant_cache):

                    new_K, new_V = new_tok_kvs[i]

                    # --- Key residual ---
                    K_res = torch.cat([K_res, new_K], dim=2)
                    if K_res.shape[2] >= R:
                        # Flush oldest (K_res.shape[2] - R + 1) tokens to quantised store
                        # to keep exactly R-1 tokens in window after flush
                        n_flush = K_res.shape[2] - R + 1
                        K_flush = K_res[:, :, :n_flush, :]
                        K_res   = K_res[:, :, n_flush:, :]
                        K_q, K_scale, K_zero = self._flush_K_residual(
                            K_flush, K_q, K_scale, K_zero)

                    # --- Value residual ---
                    V_res = torch.cat([V_res, new_V], dim=2)
                    # Flush complete G-token groups
                    while V_res.shape[2] >= G:
                        V_flush = V_res[:, :, :G, :]
                        V_res   = V_res[:, :, G:, :]
                        V_q, V_scale, V_zero = self._flush_V_residual(
                            V_flush, V_q, V_scale, V_zero)

                    new_quant_cache.append([
                        K_q, K_scale, K_zero,
                        V_q, V_scale, V_zero,
                        K_res, V_res,
                    ])

                quant_cache = new_quant_cache

                # Record snapshot
                if tokens_gen == cur_chk:
                    T = generated_ids.shape[1]
                    compressed = self._cache_bytes(T, n_layers, n_heads, head_dim)
                    fullkv     = T * n_layers * n_heads * head_dim * 2 * 2
                    snapshots.append(CacheState(compressed_bytes=compressed,
                                                fullkv_bytes=fullkv))
                    cur_chk = next(next_chk, None)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        # ----------------------------------------------------------------
        # Final state
        # ----------------------------------------------------------------
        T          = generated_ids.shape[1]
        compressed = self._cache_bytes(T, n_layers, n_heads, head_dim)
        fullkv     = T * n_layers * n_heads * head_dim * 2 * 2
        gen_text   = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        final_state = CacheState(compressed_bytes=compressed, fullkv_bytes=fullkv)

        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final_state)

        return gen_text, snapshots, final_state
