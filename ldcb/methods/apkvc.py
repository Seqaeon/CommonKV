import torch
from .base import KVCacheMethod, CacheState
from commonkv.apkvc_cache import HybridAPKVCCache

class APKVCMethod(KVCacheMethod):
    def __init__(self, predictor_type="identity", **kwargs):
        self.apkvc_kwargs = {
            "predictor_type": predictor_type,
            "max_anchor_interval": 4,
            "rd_threshold": 0.05,
            "compress_K": True,
            "compress_V": True,
            "K_num_codebooks": 4,
            "V_num_codebooks": 2,
            "use_scale_normalization": True,
            "use_rope_aware_aq": True,
            "per_layer_codebooks": True,
            "prefill_compression": "int8",  # int8 | vq | fp16
            "codebook_structure": "unconstrained",  # unconstrained | rope_commutative_2x2
            "enable_code_attention_lookup": False,
        }
        self.apkvc_kwargs.update(kwargs)
        self.name = f"APKVC-{predictor_type}"

    def _estimate_bytes(self, cache: HybridAPKVCCache):
        total_bytes = 0
        n_layers = len(cache.decode_states)
        
        for i in range(n_layers):
            # 1. Prefill (INT8 + Per-channel scale)
            K_int8 = cache.prefill_K_int8[i]
            if K_int8 is not None:
                # K_int8, V_int8: [B, H, T, D]
                # K_scale, V_scale: [B, H, 1, D]
                T_pre = K_int8.shape[-2]
                H = K_int8.shape[1]
                D = K_int8.shape[-1]
                total_bytes += 2 * (T_pre * H * D * 1) # INT8 K and V
                total_bytes += 2 * (H * D * 2) # FP16 scales
            elif cache.prefill_K_codes[i] is not None:
                # VQ code-only prefill storage (1 byte/code assumption).
                k_codes = cache.prefill_K_codes[i]
                v_codes = cache.prefill_V_codes[i]
                total_bytes += sum(c.numel() for c in k_codes)  # K codes
                total_bytes += sum(c.numel() for c in v_codes)  # V codes
            
            # 2. Decoding (Anchors or AQ)
            state = cache.decode_states[i]
            H = cache.model_config.num_key_value_heads if hasattr(cache.model_config, "num_key_value_heads") else cache.model_config.num_attention_heads
            D = cache.model_config.hidden_size // cache.model_config.num_attention_heads
            
            for entry in state['entries']:
                if entry.get('is_anchor', False):
                    # Full FP16
                    total_bytes += 2 * (H * D * 2)
                else:
                    # AQ codes (assume 8-bit)
                    n_code_K = self.apkvc_kwargs.get("K_num_codebooks", 0)
                    n_code_V = self.apkvc_kwargs.get("V_num_codebooks", 0)
                    total_bytes += H * (n_code_K + n_code_V) # 1 byte per code
                    # Scales if normalized
                    if entry.get('scale_K') is not None:
                        total_bytes += 2 * (H * 2) # FP16 scale per head
        return int(total_bytes)

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps):
        # Instantiate HybridAPKVCCache
        cache = HybridAPKVCCache(model.config, apkvc_kwargs=self.apkvc_kwargs)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated_ids = inputs.input_ids
        
        n_layers = model.config.num_hidden_layers
        n_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        
        snapshots = []
        next_checkpoint = iter(checkpoint_steps)
        current_checkpoint = next(next_checkpoint, None)
        tokens_generated = 0
        
        # In APKVC, we use model.generate with our cache passed in past_key_values
        # But to capture snapshots at EXACT token counts, we'll step manually.
        
        with torch.no_grad():
            # Initial forward pass (Prefill)
            # Standard transformers forward pass with cache as past_key_values
            # Note: llama_model.py was patched to use cache_kwargs['query_states']
            outputs = model(generated_ids, past_key_values=cache, use_cache=True)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            tokens_generated += 1
            
            while tokens_generated < max_new_tokens:
                # Decoding pass
                # We need to pass query_states in cache_kwargs. 
                # Our llama_model.py patch expects 'query_states' in cache_kwargs if past_key_value is present.
                # However, model.forward() doesn't usually take cache_kwargs directly.
                # DynamicCache.update() is called by the model.
                
                outputs = model(next_token, past_key_values=cache, use_cache=True)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1
                
                if tokens_generated == current_checkpoint:
                    T = generated_ids.shape[1]
                    compressed = self._estimate_bytes(cache)
                    fullkv = T * n_layers * n_heads * head_dim * 2 * 2
                    
                    # Compute anchor counts
                    anchors = 0
                    residuals = 0
                    distortions = []
                    for s in cache.decode_states:
                        for e in s['entries']:
                            if e.get('is_anchor', False): anchors += 1
                            else: residuals += 1
                        distortions.extend(s['cluster'].distortion_history)
                    anchor_positions = [
                        e.get("position", -1)
                        for e in cache.decode_states[0]['entries']
                        if e.get("is_anchor", False)
                    ] if cache.decode_states else []
                        
                    snapshots.append(CacheState(
                        compressed_bytes=compressed, 
                        fullkv_bytes=fullkv,
                        anchor_count=anchors,
                        residual_count=residuals,
                        distortions=distortions,
                        anchor_positions=anchor_positions,
                    ))
                    current_checkpoint = next(next_checkpoint, None)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        T = generated_ids.shape[1]
        compressed = self._estimate_bytes(cache)
        fullkv = T * n_layers * n_heads * head_dim * 2 * 2
        
        anchors = 0
        residuals = 0
        distortions = []
        for s in cache.decode_states:
            for e in s['entries']:
                if e.get('is_anchor', False): anchors += 1
                else: residuals += 1
            distortions.extend(s['cluster'].distortion_history)
        anchor_positions = [
            e.get("position", -1)
            for e in cache.decode_states[0]['entries']
            if e.get("is_anchor", False)
        ] if cache.decode_states else []
            
        final_state = CacheState(
            compressed_bytes=compressed, 
            fullkv_bytes=fullkv,
            anchor_count=anchors,
            residual_count=residuals,
            distortions=distortions,
            anchor_positions=anchor_positions,
        )
        
        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final_state)
            
        return generated_text, snapshots, final_state
