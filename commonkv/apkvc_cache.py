import torch
from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple, Union

class HybridAPKVCCache(DynamicCache):
    """
    A hybridized KV cache designed to physically save VRAM by compressing 
    the Prompt prefill down to INT8 per-channel quantization, and managing
    autoregressive Decode using APKVC codebooks.
    """
    def __init__(self, model_config, apkvc_kwargs=None):
        super().__init__()
        # Pre-allocate caches
        self.prefill_K_int8 = []
        self.prefill_K_scale = []
        self.prefill_V_int8 = []
        self.prefill_V_scale = []
        
        self.decode_states = []
        
        self.apkvc_kwargs = apkvc_kwargs if apkvc_kwargs is not None else {}
        self.model_config = model_config
        self._seen_tokens = 0
        # Diagnostic option: store prefill in fp16 instead of INT8.
        # When True, anchor-only should produce ROUGE-L = 1.0 vs FullKV,
        # confirming the decode path is clean. If it still doesn't, the bug
        # is in the decode path. If it does, INT8 prefill is the culprit.
        self.fp16_prefill = self.apkvc_kwargs.get("fp16_prefill", False)
        
    def _init_layer(self, layer_idx):
        from attention_aware_predictive_kv import AttentionAwarePredictiveKVCluster
        while len(self.decode_states) <= layer_idx:
            # Inject layer-specific config
            layer_kwargs = dict(self.apkvc_kwargs)
            layer_kwargs['layer_idx'] = len(self.decode_states)
            cluster = AttentionAwarePredictiveKVCluster(**layer_kwargs)
            
            # Hybrid states dictionary strictly for the decoding pipeline (replaces HF full fp16 tracking)
            state = {
                'cluster': cluster,
                'entries': [],            # Only holds {'is_anchor', 'K'/'V' or 'codes_K'/'codes_V'}
                'recon_cache': {},        # Key=position, Value=(K, V) tuple rebuilt from history
                'last_anchor_t': -1,
                'last_distortion': 0.0,
                'last_K': None,
                'last_V': None,
                'last_K2': None,
                'last_V2': None,
                'decode_K_cat': None,
                'decode_V_cat': None,
            }
            self.decode_states.append(state)
            self.prefill_K_int8.append(None)
            self.prefill_K_scale.append(None)
            self.prefill_V_int8.append(None)
            self.prefill_V_scale.append(None)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.decode_states) <= layer_idx:
            return 0
        
        prefill_len = 0
        if self.prefill_K_int8[layer_idx] is not None:
            prefill_len = self.prefill_K_int8[layer_idx].shape[-2]
            
        decode_len = len(self.decode_states[layer_idx]['entries'])
        return prefill_len + decode_len

    def quantize_prefill_kv(self, K: torch.Tensor, V: torch.Tensor):
        # Per-channel: scale over token dimension (dim=2)
        K_scale = K.abs().amax(dim=2, keepdim=True) / 127.0
        V_scale = V.abs().amax(dim=2, keepdim=True) / 127.0
        # Prevent zero division
        K_scale = K_scale.clamp(min=1e-5)
        V_scale = V_scale.clamp(min=1e-5)
        
        K_int8 = (K / K_scale).round().clamp(-128, 127).to(torch.int8)
        V_int8 = (V / V_scale).round().clamp(-128, 127).to(torch.int8)
        return K_int8, K_scale.half(), V_int8, V_scale.half()

    def dequantize_prefill_kv(self, layer_idx: int):
        K_raw   = self.prefill_K_int8[layer_idx]
        K_scale = self.prefill_K_scale[layer_idx]
        V_raw   = self.prefill_V_int8[layer_idx]
        V_scale = self.prefill_V_scale[layer_idx]
        if K_scale is None:
            # fp16_prefill mode — raw tensors are already fp16
            return K_raw, V_raw
        K = K_raw.half() * K_scale
        V = V_raw.half() * V_scale
        return K, V

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self._init_layer(layer_idx)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        is_prefill = key_states.shape[-2] > 1
        
        if is_prefill:
            if self.fp16_prefill:
                # Store fp16 directly — decode steps will use original values,
                # making anchor-only identical to FullKV for diagnostic purposes.
                self.prefill_K_int8[layer_idx]  = key_states.half()
                self.prefill_K_scale[layer_idx] = None  # sentinel: fp16 mode
                self.prefill_V_int8[layer_idx]  = value_states.half()
                self.prefill_V_scale[layer_idx] = None
            else:
                # 1. Quantize and save prefill arrays in physical memory
                K_int8, K_scale, V_int8, V_scale = self.quantize_prefill_kv(key_states, value_states)
                self.prefill_K_int8[layer_idx]  = K_int8
                self.prefill_K_scale[layer_idx] = K_scale
                self.prefill_V_int8[layer_idx]  = V_int8
                self.prefill_V_scale[layer_idx] = V_scale
            
            # Instantiate prediction states so decoded tokens can securely predict
            state = self.decode_states[layer_idx]
            cluster = state['cluster']
            Q = cache_kwargs.get("query_states", None)
            
            # The cluster manages codes statically, pass identity to bootstrap
            state['last_K'] = key_states[:, :, -1:, :]
            state['last_V'] = value_states[:, :, -1:, :]
            
            # Return full tensors for this instantaneous attention calculation
            # PyTorch immediately discards this temporary fp16 array!
            return key_states, value_states
            
        else:
            # 2. Decode Sequence Processing
            state = self.decode_states[layer_idx]
            cluster = state['cluster']
            Q = cache_kwargs.get("query_states", None)
            
            t = len(state['entries'])  # decode-local step (used for anchor interval logic)
            # Absolute sequence position: needed for correct RoPE derotation/rotation.
            # K_true already has RoPE applied at prefill_len + t; derotating at t
            # alone would corrupt the base vector and therefore all reconstructed K.
            prefill_len = (
                self.prefill_K_int8[layer_idx].shape[-2]
                if self.prefill_K_int8[layer_idx] is not None else 0
            )
            abs_pos = prefill_len + t
            
            # We defer the actual codebook heavy lifting completely into the cluster module
            # Let it write its mathematical outputs (Anchors & Dictionaries) securely into `state`
            K_recon, V_recon = cluster.compress_decode_token(key_states, value_states, Q, t, state, abs_pos=abs_pos)
            
            # Finally, rebuild the complete sequence tensor to feed to scaled-dot-product attention.
            K_prefill, V_prefill = self.dequantize_prefill_kv(layer_idx)
            
            # Gather decode history from recon_cache
            # Because PyTorch requires contiguous arrays, we must materialize them
            # O(T) materialization is required here as HF past_key_value inherently requires the entire history!
            if state['decode_K_cat'] is None:
                state['decode_K_cat'] = K_recon
                state['decode_V_cat'] = V_recon
            else:
                state['decode_K_cat'] = torch.cat([state['decode_K_cat'], K_recon], dim=-2)
                state['decode_V_cat'] = torch.cat([state['decode_V_cat'], V_recon], dim=-2)
            K_decode = state['decode_K_cat']
            V_decode = state['decode_V_cat']
            
            final_K = torch.cat([K_prefill, K_decode], dim=-2)
            final_V = torch.cat([V_prefill, V_decode], dim=-2)
            
            return final_K, final_V
