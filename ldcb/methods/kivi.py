import torch
from transformers.cache_utils import DynamicCache
from .base import KVCacheMethod, CacheState
from ldcb.utils import get_kv_iterator


class KIVIMethod(KVCacheMethod):
    """
    KIVI-style KV-cache quantization with correct incremental quantization.

    Core invariant: each token's KV is quantized **exactly once** — when it
    graduates from the fp16 residual window into the permanent quantized store.
    No token is ever re-quantized, so quantization error does not accumulate
    over decode steps.

    The most-recent `residual_buffer` tokens are kept in fp16 to preserve
    attention quality on the immediately preceding context (KIVI paper §3.2).

    Scale convention: per-token (amax over head_dim), stored as fp16 alongside
    each int8 token block.  This allows independent quantization of individual
    tokens and lossless concatenation of their scales — a requirement for
    true incremental quantization.  The original KIVI paper uses per-channel-K
    scale, which is incompatible with incremental append because the global
    channel max changes with each new token and would require re-quantizing all
    prior tokens to stay consistent.
    """

    def __init__(self, bits: int = 4, residual_buffer: int = 32,
                 cpu_offload_quant: bool = False):
        assert bits in (2, 4, 8)
        self.bits = bits
        self.residual_buffer = residual_buffer
        self.cpu_offload_quant = cpu_offload_quant
        self.name = f"KIVI-int{bits}"

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------

    def _quantize(self, X: torch.Tensor):
        """
        Per-token quantize X of shape [B, H, T, D].

        Returns
        -------
        X_q   : [B, H, T, D]  int8
        scale : [B, H, T, 1]  float16

        Each token (position t) gets its own scale = amax over the head-dim
        axis so that tokens can be quantized independently and concatenated
        without touching earlier scales.
        """
        levels = 2 ** self.bits
        scale = X.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)  # [B,H,T,1]
        X_q = (
            X.float() / scale * (levels / 2 - 1)
        ).round().clamp(-(levels // 2), levels // 2 - 1)
        return X_q.to(torch.int8), scale.to(torch.float16)

    def _dequantize(self, X_q: torch.Tensor, scale: torch.Tensor):
        """Inverse of _quantize.  Returns float32."""
        levels = 2 ** self.bits
        return X_q.float() * scale.float() / (levels / 2 - 1)

    # ------------------------------------------------------------------
    # Byte-count for compression ratio
    # ------------------------------------------------------------------

    def _cache_bytes(self, n_tokens: int, n_layers: int, n_heads: int,
                     head_dim: int) -> int:
        T_q = max(0, n_tokens - self.residual_buffer)
        T_r = min(n_tokens, self.residual_buffer)
        # K + V quantized values: bits/8 bytes each (theoretical bit-packing)
        quant_values = 2 * T_q * n_layers * n_heads * head_dim * (self.bits / 8)
        # Per-token fp16 scales: [B, H, T_q, 1] → 2 bytes per entry
        quant_scales  = 2 * T_q * n_layers * n_heads * 2
        # Residual window K + V in fp16
        residual       = 2 * T_r * n_layers * n_heads * head_dim * 2
        return int(quant_values + quant_scales + residual)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers = model.config.num_hidden_layers
        n_heads  = getattr(model.config, "num_key_value_heads",
                           model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        generated_ids = inputs.input_ids
        snapshots = []
        next_checkpoint = iter(checkpoint_steps)
        current_checkpoint = next(next_checkpoint, None)
        tokens_generated = 0

        # ----------------------------------------------------------------
        # Prefill
        # ----------------------------------------------------------------
        with torch.no_grad():
            outputs = model(generated_ids, use_cache=True)

        # Partition prefill KV into:
        #   - quantized block  (all tokens except the last residual_buffer)
        #   - fp16 residual window (most-recent residual_buffer tokens)
        #
        # Cache state per layer: [K_q, K_scale, V_q, V_scale, K_res, V_res]
        #   K_q, V_q         : [B, H, T_q, D]  int8
        #   K_scale, V_scale : [B, H, T_q, 1]  float16  (per-token scale)
        #   K_res, V_res     : [B, H, T_r, D]  float16
        quant_cache = []
        for _, (layer_K, layer_V) in get_kv_iterator(outputs.past_key_values):
            T = layer_K.shape[2]
            split = max(T - self.residual_buffer, 0)
            K_old = layer_K[:, :, :split, :]
            V_old = layer_V[:, :, :split, :]
            K_res = layer_K[:, :, split:, :].to(torch.float16)
            V_res = layer_V[:, :, split:, :].to(torch.float16)

            if split > 0:
                K_q, K_scale = self._quantize(K_old)
                V_q, V_scale = self._quantize(V_old)
                if self.cpu_offload_quant:
                    K_q, K_scale = K_q.cpu(), K_scale.cpu()
                    V_q, V_scale = V_q.cpu(), V_scale.cpu()
            else:
                K_q = V_q = K_scale = V_scale = None

            quant_cache.append([K_q, K_scale, V_q, V_scale, K_res, V_res])

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        tokens_generated += 1
        del outputs

        # ----------------------------------------------------------------
        # Decode loop — one token at a time
        # ----------------------------------------------------------------
        with torch.no_grad():
            while tokens_generated < max_new_tokens:

                # 1. Reconstruct full fp16 KV for the attention forward pass.
                past_kv_deq = DynamicCache()
                for i, (K_q, K_scale, V_q, V_scale, K_res, V_res) in enumerate(quant_cache):
                    if K_q is not None:
                        dev = K_res.device
                        K_q_d = K_q.to(dev) if self.cpu_offload_quant else K_q
                        K_s_d = K_scale.to(dev) if self.cpu_offload_quant else K_scale
                        V_q_d = V_q.to(dev) if self.cpu_offload_quant else V_q
                        V_s_d = V_scale.to(dev) if self.cpu_offload_quant else V_scale
                        K_fp = torch.cat([
                            self._dequantize(K_q_d, K_s_d).to(model.dtype),
                            K_res.to(model.dtype),
                        ], dim=2)
                        V_fp = torch.cat([
                            self._dequantize(V_q_d, V_s_d).to(model.dtype),
                            V_res.to(model.dtype),
                        ], dim=2)
                        if self.cpu_offload_quant:
                            del K_q_d, K_s_d, V_q_d, V_s_d
                    else:
                        K_fp = K_res.to(model.dtype)
                        V_fp = V_res.to(model.dtype)
                    past_kv_deq.update(K_fp, V_fp, i)

                # 2. Forward pass.
                outputs = model(next_token, past_key_values=past_kv_deq, use_cache=True)

                # 3. Extract ONLY the new token's KV (last position).
                #    Clone immediately so we can free the full output tensors.
                new_tok_kvs = [
                    (
                        K[:, :, -1:, :].to(torch.float16).clone(),
                        V[:, :, -1:, :].to(torch.float16).clone(),
                    )
                    for _, (K, V) in get_kv_iterator(outputs.past_key_values)
                ]
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del outputs, past_kv_deq
                torch.cuda.empty_cache()

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1

                # 4. Update quantized cache.
                #
                #    Append new token to the residual window.
                #    When the residual window exceeds `residual_buffer`, evict
                #    the oldest token into the permanent quantized store —
                #    quantizing it ONCE and never touching it again.
                new_quant_cache = []
                for i, (K_q, K_scale, V_q, V_scale, K_res, V_res) in enumerate(quant_cache):
                    new_K, new_V = new_tok_kvs[i]

                    # Grow residual window
                    K_res = torch.cat([K_res, new_K], dim=2)
                    V_res = torch.cat([V_res, new_V], dim=2)

                    # Promote oldest residual token to quantized cache if full
                    if K_res.shape[2] > self.residual_buffer:
                        promo_K = K_res[:, :, :1, :]   # oldest token in window
                        promo_V = V_res[:, :, :1, :]

                        # Quantize this token exactly once
                        pK_q, pK_scale = self._quantize(promo_K)
                        pV_q, pV_scale = self._quantize(promo_V)
                        if self.cpu_offload_quant:
                            pK_q, pK_scale = pK_q.cpu(), pK_scale.cpu()
                            pV_q, pV_scale = pV_q.cpu(), pV_scale.cpu()

                        if K_q is not None:
                            K_q     = torch.cat([K_q,     pK_q],     dim=2)
                            K_scale = torch.cat([K_scale, pK_scale], dim=2)
                            V_q     = torch.cat([V_q,     pV_q],     dim=2)
                            V_scale = torch.cat([V_scale, pV_scale], dim=2)
                        else:
                            K_q, K_scale = pK_q, pK_scale
                            V_q, V_scale = pV_q, pV_scale

                        # Trim promoted token from residual window
                        K_res = K_res[:, :, 1:, :]
                        V_res = V_res[:, :, 1:, :]

                    new_quant_cache.append([K_q, K_scale, V_q, V_scale, K_res, V_res])

                quant_cache = new_quant_cache

                # 5. Record snapshot at checkpoint
                if tokens_generated == current_checkpoint:
                    T = generated_ids.shape[1]
                    compressed = self._cache_bytes(T, n_layers, n_heads, head_dim)
                    fullkv = T * n_layers * n_heads * head_dim * 2 * 2
                    snapshots.append(CacheState(compressed_bytes=compressed,
                                                fullkv_bytes=fullkv))
                    current_checkpoint = next(next_checkpoint, None)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        # ----------------------------------------------------------------
        # Final state
        # ----------------------------------------------------------------
        T = generated_ids.shape[1]
        compressed = self._cache_bytes(T, n_layers, n_heads, head_dim)
        fullkv = T * n_layers * n_heads * head_dim * 2 * 2
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        final_state = CacheState(compressed_bytes=compressed, fullkv_bytes=fullkv)

        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final_state)

        return generated_text, snapshots, final_state
