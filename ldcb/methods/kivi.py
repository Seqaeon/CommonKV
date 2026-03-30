import torch
from transformers.cache_utils import DynamicCache
from .base import KVCacheMethod, CacheState
from ldcb.utils import get_kv_iterator

class KIVIMethod(KVCacheMethod):

    def __init__(self, bits: int = 4, residual_buffer: int = 32):
        assert bits in (2, 4, 8)
        self.bits = bits
        self.residual_buffer = residual_buffer  # last N tokens kept in fp16
        self.name = f"KIVI-int{bits}"

    def _quantize_K(self, K):
        levels = 2 ** self.bits
        scale = K.abs().amax(dim=2, keepdim=True).clamp(min=1e-6)
        K_q = (K / scale * (levels / 2 - 1)).round().clamp(-(levels//2), levels//2 - 1)
        return K_q.to(torch.int8), scale

    def _quantize_V(self, V):
        levels = 2 ** self.bits
        scale = V.abs().amax(dim=3, keepdim=True).clamp(min=1e-6)
        V_q = (V / scale * (levels / 2 - 1)).round().clamp(-(levels//2), levels//2 - 1)
        return V_q.to(torch.int8), scale

    def _dequantize(self, X_q, scale, bits):
        levels = 2 ** bits
        return X_q.float() * scale / (levels / 2 - 1)

    def _cache_bytes(self, n_tokens, n_layers, n_heads, head_dim):
        quant_bytes = 2 * n_tokens * n_layers * n_heads * head_dim * (self.bits / 8)
        scale_bytes_K = n_layers * n_heads * head_dim * 2          
        scale_bytes_V = n_tokens * n_layers * n_heads * 2          
        residual_bytes = min(n_tokens, self.residual_buffer) * n_layers * n_heads * head_dim * 2 * 2
        return int(quant_bytes + scale_bytes_K + scale_bytes_V + residual_bytes)

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers = model.config.num_hidden_layers
        n_heads  = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        generated_ids = inputs.input_ids
        snapshots = []
        next_checkpoint = iter(checkpoint_steps)
        current_checkpoint = next(next_checkpoint, None)
        tokens_generated = 0
        
        with torch.no_grad():
            outputs = model(generated_ids, use_cache=True)
            past_kv = outputs.past_key_values  
            
            quant_cache = []
            for _, (layer_K, layer_V) in get_kv_iterator(past_kv):
                K_q, K_scale = self._quantize_K(layer_K)
                V_q, V_scale = self._quantize_V(layer_V)
                quant_cache.append((K_q, K_scale, V_q, V_scale))

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            tokens_generated += 1

            while tokens_generated < max_new_tokens:
                past_kv_tuple = tuple(
                    (self._dequantize(K_q, K_scale, self.bits).to(model.dtype),
                     self._dequantize(V_q, V_scale, self.bits).to(model.dtype))
                    for K_q, K_scale, V_q, V_scale in quant_cache
                )
                past_kv_deq = DynamicCache.from_legacy_cache(past_kv_tuple)

                outputs = model(next_token, past_key_values=past_kv_deq, use_cache=True)
                new_kv_full = outputs.past_key_values
                
                quant_cache = []
                for _, (layer_K, layer_V) in get_kv_iterator(new_kv_full):
                    K_q, K_scale = self._quantize_K(layer_K)
                    V_q, V_scale = self._quantize_V(layer_V)
                    quant_cache.append((K_q, K_scale, V_q, V_scale))

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1

                if tokens_generated == current_checkpoint:
                    T = generated_ids.shape[1]
                    compressed = self._cache_bytes(T, n_layers, n_heads, head_dim)
                    fullkv = T * n_layers * n_heads * head_dim * 2 * 2
                    snapshots.append(CacheState(compressed_bytes=compressed, fullkv_bytes=fullkv))
                    current_checkpoint = next(next_checkpoint, None)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        T = generated_ids.shape[1]
        compressed = self._cache_bytes(T, n_layers, n_heads, head_dim)
        fullkv = T * n_layers * n_heads * head_dim * 2 * 2
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        final_state = CacheState(compressed_bytes=compressed, fullkv_bytes=fullkv)

        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final_state)

        return generated_text, snapshots, final_state
