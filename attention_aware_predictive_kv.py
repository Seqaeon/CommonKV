import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pyramidkv.pyramidkv_utils import BaseCluster

@dataclass
class APKVCConfig:
    predictor_type: str  = 'identity' # 'identity' or 'linear'
    alpha_K: float       = 1.5
    beta_K: float        = -0.5
    alpha_V: float       = 1.5
    beta_V: float        = -0.5

    use_rope_aware_aq: bool  = True
    K_num_codebooks: int     = 4
    V_num_codebooks: int     = 2
    codebook_size: int       = 256
    
    rd_metric: str       = 'key_dot' # 'key_dot' or 'sampled_attention_output'
    rd_threshold: float  = 0.05
    rd_sample_heads: int = 4

    max_anchor_interval: int         = 16
    residual_norm_threshold_K: float = 1.5
    residual_norm_threshold_V: float = 3.0

def rope_rotate(x: torch.Tensor, position: int, base: float = 10000.0) -> torch.Tensor:
    """Apply forward RoPE rotation."""
    D = x.shape[-1]
    device = x.device
    i = torch.arange(0, D // 2, dtype=torch.float32, device=device)
    theta = base ** (-2 * i / D)
    angle = position * theta
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out

def rope_derotate(x: torch.Tensor, position: int, base: float = 10000.0) -> torch.Tensor:
    """Apply inverse RoPE rotation to get base vectors."""
    D = x.shape[-1]
    device = x.device
    i = torch.arange(0, D // 2, dtype=torch.float32, device=device)
    theta = base ** (-2 * i / D)
    angle = position * theta
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    out = torch.empty_like(x)
    # Inverse rotation: negate sin
    out[..., 0::2] = x_even * cos + x_odd * sin
    out[..., 1::2] = -x_even * sin + x_odd * cos
    return out

class AttentionAwarePredictiveKVCluster(BaseCluster):
    """
    Attention-Aware Predictive KV Compression (APKVC)
    
    Implements:
    - Anchor-based predictive coding (deltas)
    - RoPE-aware Additive Quantization (AQ)
    - Attention-informed reset policy
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        # Extract config from kwargs or use defaults
        self.apkvc_config = APKVCConfig(
            predictor_type=kwargs.get("predictor_type", 'identity'),
            rd_threshold=kwargs.get("rd_threshold", 0.05),
            max_anchor_interval=kwargs.get("max_anchor_interval", 16),
            K_num_codebooks=kwargs.get("K_num_codebooks", 4),
            V_num_codebooks=kwargs.get("V_num_codebooks", 2)
        )
        
        self.head_dim = None
        self.codebooks_K = nn.ParameterList()
        self.codebooks_V = nn.ParameterList()
        
        # Internal state for tracking anchors and residuals
        self.entries = [] # List of dicts representing cached tokens
        self.last_anchor_t = -1
        self.last_distortion = 0.0
        self.last_residual_norm = 0.0
        
        # For prediction
        self.last_K = None
        self.last_V = None
        self.last_K2 = None
        self.last_V2 = None
        
        self.initialized = False

    def _init_lazy(self, head_dim, device):
        """Lazy initialization of codebooks once head_dim is known."""
        self.head_dim = head_dim
        # Initialize codebooks randomly for now (Calibration usually fits these)
        # Using a fixed seed for consistency in prototype
        torch.manual_seed(42)
        
        # Create M codebooks for K
        for _ in range(self.apkvc_config.K_num_codebooks):
            cb = nn.Parameter(torch.randn(self.apkvc_config.codebook_size, head_dim, device=device) * 0.01)
            self.codebooks_K.append(cb)
            
        # Create M codebooks for V
        for _ in range(self.apkvc_config.V_num_codebooks):
            cb = nn.Parameter(torch.randn(self.apkvc_config.codebook_size, head_dim, device=device) * 0.01)
            self.codebooks_V.append(cb)
            
        self.initialized = True

    def additive_quantize(self, residual, codebooks):
        """Encode residual into indices."""
        # residual: [B, H, D]
        # codebooks: List[Parameter]
        r = residual.clone()
        indices = []
        for C in codebooks:
            # C: [S, D]
            # unsqueeze for broadcasting: [B, H, 1, D] vs [1, 1, S, D]
            # We use cdist-like logic: (A-B)^2 = A^2 + B^2 - 2AB
            # [B, H, S]
            dist = torch.norm(r.unsqueeze(-2) - C, dim=-1)
            best = dist.argmin(dim=-1) # [B, H]
            indices.append(best)
            # Subtract chosen codeword
            # C[best] shape: [B, H, D]
            batch_indices = torch.arange(r.shape[0], device=r.device).view(-1, 1)
            head_indices = torch.arange(r.shape[1], device=r.device).view(1, -1)
            r = r - C[best]
        return indices

    def additive_decode(self, indices, codebooks):
        """Reconstruct residual from indices."""
        out = torch.zeros(indices[0].shape[0], indices[0].shape[1], self.head_dim, device=indices[0].device)
        for idx, C in zip(indices, codebooks):
            out += C[idx]
        return out

    def compute_distortion(self, Q, K_true, K_reconstructed):
        """Compute key-dot distortion proxy."""
        # Q: [B, H, 1, D]
        # K_true, K_reconstructed: [B, H, 1, D]
        scale = math.sqrt(self.head_dim)
        scores_true = torch.einsum('bhqd,bhkd->bhqk', Q, K_true) / scale
        scores_comp = torch.einsum('bhqd,bhkd->bhqk', Q, K_reconstructed) / scale
        return (scores_true - scores_comp).abs().max().item()

    def _update_rolling(self, K, V):
        self.last_K2 = self.last_K
        self.last_V2 = self.last_V
        self.last_K = K
        self.last_V = V

    def should_reset(self, t):
        if (t - self.last_anchor_t) >= self.apkvc_config.max_anchor_interval:
            return True
        if self.last_distortion > self.apkvc_config.rd_threshold:
            return True
        # Could add residual norm check here too
        return False

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """
        Intersects the forward pass to perform compression.
        
        Args:
            key_states (torch.Tensor): [B, H, S, D]
            query_states (torch.Tensor): [B, H, Q, D]
            value_states (torch.Tensor): [B, H, S, D]
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        device = query_states.device
        
        if not self.initialized:
            self._init_lazy(head_dim, device)

        # We assume t is the current sequence position
        # In this repo's Custom implementation, update_kv is called with:
        # - Full prefix during prefill (S > 1)
        # - Single token during decoding (S = 1)
        
        t = len(self.entries)
        
        # If it's a batch/prefill block (S > 1), we treat the whole block as an anchor for simplicity
        # or we could iterate. The user spec says "Prefill can be left uncompressed initially".
        if key_states.shape[-2] > 1:
            # Anchor the whole block
            for i in range(key_states.shape[-2]):
                self.entries.append({'is_anchor': True, 'position': t + i})
            self.last_anchor_t = t + key_states.shape[-2] - 1
            self._update_rolling(key_states[:, :, -1:], value_states[:, :, -1:])
            return key_states, value_states

        # Decoding loop (single token)
        K_true = key_states # [B, H, 1, D]
        V_true = value_states # [B, H, 1, D]
        Q = query_states # [B, H, 1, D]
        
        if t == 0 or self.should_reset(t):
            self.entries.append({'is_anchor': True, 'position': t})
            self.last_anchor_t = t
            self.last_distortion = 0.0
            self._update_rolling(K_true, V_true)
            return K_true, V_true

        # 1. Predict
        if self.apkvc_config.predictor_type == 'identity' or self.last_K2 is None:
            K_hat, V_hat = self.last_K, self.last_V
        else:
            K_hat = self.apkvc_config.alpha_K * self.last_K + self.apkvc_config.beta_K * self.last_K2
            V_hat = self.apkvc_config.alpha_V * self.last_V + self.apkvc_config.beta_V * self.last_V2

        delta_K = K_true - K_hat
        delta_V = V_true - V_hat

        # 2. RoPE derotation (keys only)
        if self.apkvc_config.use_rope_aware_aq:
            delta_K_base = rope_derotate(delta_K, t)
        else:
            delta_K_base = delta_K

        # 3. Additive quantization
        codes_K = self.additive_quantize(delta_K_base.squeeze(2), self.codebooks_K)
        codes_V = self.additive_quantize(delta_V.squeeze(2), self.codebooks_V)

        # 4. Reconstruct for distortion check and return value
        recon_delta_K_base = self.additive_decode(codes_K, self.codebooks_K).unsqueeze(2)
        recon_delta_V = self.additive_decode(codes_V, self.codebooks_V).unsqueeze(2)
        
        if self.apkvc_config.use_rope_aware_aq:
            recon_delta_K = rope_rotate(recon_delta_K_base, t)
        else:
            recon_delta_K = recon_delta_K_base
            
        K_recon = K_hat + recon_delta_K
        V_recon = V_hat + recon_delta_V

        # 5. Distortion check
        distortion = self.compute_distortion(Q, K_true, K_recon)
        self.last_distortion = distortion
        
        if distortion > self.apkvc_config.rd_threshold:
            # Distortion too high -> Trigger Reset (Anchor)
            self.entries.append({'is_anchor': True, 'position': t})
            self.last_anchor_t = t
            self.last_distortion = 0.0
            self._update_rolling(K_true, V_true)
            return K_true, V_true
        else:
            # Commit compression
            self.entries.append({
                'is_anchor': False,
                'codes_K': codes_K,
                'codes_V': codes_V,
                'position': t,
                'predictor_type': self.apkvc_config.predictor_type
            })
            self._update_rolling(K_recon, V_recon)
            return K_recon, V_recon
