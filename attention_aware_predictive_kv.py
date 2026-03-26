import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pyramidkv.pyramidkv_utils import BaseCluster

@dataclass
class APKVCConfig:
    predictor_type: str  = 'linear' # 'identity' or 'linear'
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
    cos = torch.cos(angle).to(x.dtype)
    sin = torch.sin(angle).to(x.dtype)
    
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
    cos = torch.cos(angle).to(x.dtype)
    sin = torch.sin(angle).to(x.dtype)
    
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

    def _init_lazy(self, head_dim, device, dtype):
        """Lazy initialization of codebooks once head_dim is known."""
        self.head_dim = head_dim
        # Initialize codebooks randomly for now (Calibration usually fits these)
        # Using a local generator avoids perturbing the global RNG state.
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        
        # Create M codebooks for K
        for _ in range(self.apkvc_config.K_num_codebooks):
            cb = nn.Parameter(
                torch.randn(
                    self.apkvc_config.codebook_size,
                    head_dim,
                    device=device,
                    dtype=dtype,
                    generator=gen,
                ) * 0.01
            )
            self.codebooks_K.append(cb)
            
        # Create M codebooks for V
        for _ in range(self.apkvc_config.V_num_codebooks):
            cb = nn.Parameter(
                torch.randn(
                    self.apkvc_config.codebook_size,
                    head_dim,
                    device=device,
                    dtype=dtype,
                    generator=gen,
                ) * 0.01
            )
            self.codebooks_V.append(cb)
            
        self.initialized = True

    def additive_quantize(self, residual, codebooks):
        """Vectorized Additive Quantization encoding."""
        # residual: [B, H, D]
        # codebooks: List[Parameter]
        device = residual.device
        dtype = residual.dtype
        r = residual.clone()
        indices = []
        
        for C in codebooks:
            # C: [S, D]
            # Use cdist for fast distance calculation: [B, H, S]
            # torch.cdist(x1, x2) computes p-norm distance
            # Here it's [B, H, D] vs [S, D]
            B, H, D = r.shape
            r_flat = r.view(B * H, D)
            C_flat = C # [S, D]
            
            # [B*H, S]
            dists = torch.cdist(r_flat.unsqueeze(1), C_flat.unsqueeze(0)).squeeze(1)
            best = dists.argmin(dim=-1) # [B*H]
            best = best.view(B, H)
            indices.append(best)
            
            # Subtract chosen codewords
            # C[best] -> [B, H, D]
            r = r - C[best]
            
        return indices

    def additive_decode(self, indices, codebooks):
        """Reconstruct residual from indices."""
        device = codebooks[0].device
        dtype = codebooks[0].dtype
        B, H = indices[0].shape
        out = torch.zeros(B, H, self.head_dim, device=device, dtype=dtype)
        for idx, C in zip(indices, codebooks):
            out += C[idx]
        return out

    def _update_rolling(self, K, V):
        """Update last two anchors for linear trajectory prediction."""
        # Shift old states
        self.last_K2 = self.last_K
        self.last_V2 = self.last_V
        # Store new anchors
        self.last_K = K.clone()
        self.last_V = V.clone()

    def should_reset(self, t):
        """Check if distance-based or interval-based reset is needed."""
        interval_reset = (t - self.last_anchor_t) >= self.apkvc_config.max_anchor_interval
        return interval_reset

    def compute_distortion(self, Q, K_true, K_reconstructed):
        """Compute key-dot distortion proxy."""
        # Q: [B, H, 1, D]
        # K_true, K_reconstructed: [B, H, 1, D]
        scale = math.sqrt(self.head_dim)
        # Faster dot product: [B, H, 1, 1]
        score_true = torch.sum(Q * K_true, dim=-1, keepdim=True) / scale
        score_comp = torch.sum(Q * K_reconstructed, dim=-1, keepdim=True) / scale
        # Max absolute discrepancy across batch/heads
        return torch.max(torch.abs(score_true - score_comp)).item()

    def _residual_norm_exceeds(self, delta_K, delta_V):
        """Residual magnitude gate used by the anchor reset policy."""
        k_norm = delta_K.norm(dim=-1).mean().item()
        v_norm = delta_V.norm(dim=-1).mean().item()
        self.last_residual_norm = max(k_norm, v_norm)
        return (
            k_norm > self.apkvc_config.residual_norm_threshold_K
            or v_norm > self.apkvc_config.residual_norm_threshold_V
        )

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """
        APKVC Update Step.
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        device = query_states.device
        dtype = query_states.dtype
        
        if not self.initialized:
            self._init_lazy(head_dim, device, dtype)

        t = len(self.entries)
        
        # 1. Prefill / Multi-token fallback
        if key_states.shape[-2] > 1:
            for i in range(key_states.shape[-2]):
                self.entries.append({'is_anchor': True, 'position': t + i})
            self.last_anchor_t = t + key_states.shape[-2] - 1
            self._update_rolling(key_states[:, :, -1:], value_states[:, :, -1:])
            return key_states, value_states

        # 2. Decoding Step (Single token)
        K_true = key_states
        V_true = value_states
        Q = query_states
        
        # Check for anchor reset
        if t == 0 or self.should_reset(t):
            self.entries.append({'is_anchor': True, 'position': t})
            self.last_anchor_t = t
            self.last_distortion = 0.0
            self._update_rolling(K_true, V_true)
            return K_true, V_true

        # 3. Predict K_hat and V_hat
        if self.apkvc_config.predictor_type == 'linear' and self.last_K2 is not None:
            # 2nd order extrapolation
            K_hat = self.apkvc_config.alpha_K * self.last_K + self.apkvc_config.beta_K * self.last_K2
            V_hat = self.apkvc_config.alpha_V * self.last_V + self.apkvc_config.beta_V * self.last_V2
        else:
            # Identity (last step)
            K_hat, V_hat = self.last_K.clone(), self.last_V.clone()

        # 4. Compressing residuals
        delta_K = K_true - K_hat
        delta_V = V_true - V_hat
        residual_too_large = self._residual_norm_exceeds(delta_K, delta_V)
        if residual_too_large:
            self.entries.append({'is_anchor': True, 'position': t})
            self.last_anchor_t = t
            self.last_distortion = 0.0
            self._update_rolling(K_true, V_true)
            return K_true, V_true
        
        if self.apkvc_config.use_rope_aware_aq:
            delta_K_base = rope_derotate(delta_K, t)
        else:
            delta_K_base = delta_K

        codes_K = self.additive_quantize(delta_K_base.squeeze(2), self.codebooks_K)
        codes_V = self.additive_quantize(delta_V.squeeze(2), self.codebooks_V)

        # 5. Reconstruction for local feedback
        recon_delta_K_base = self.additive_decode(codes_K, self.codebooks_K).unsqueeze(2)
        recon_delta_V = self.additive_decode(codes_V, self.codebooks_V).unsqueeze(2)
        
        if self.apkvc_config.use_rope_aware_aq:
            recon_delta_K = rope_rotate(recon_delta_K_base, t)
        else:
            recon_delta_K = recon_delta_K_base
            
        K_recon = K_hat + recon_delta_K
        V_recon = V_hat + recon_delta_V

        # 6. Attention-Aware distortion check
        distortion = self.compute_distortion(Q, K_true, K_recon)
        self.last_distortion = distortion
        
        if distortion > self.apkvc_config.rd_threshold:
            # Quality too low -> Downgrade to Anchor
            self.entries.append({'is_anchor': True, 'position': t})
            self.last_anchor_t = t
            self.last_distortion = 0.0
            self._update_rolling(K_true, V_true)
            return K_true, V_true
        else:
            # Success -> Store compressed indices
            self.entries.append({
                'is_anchor': False,
                'codes_K': codes_K,
                'codes_V': codes_V,
                'position': t
            })
            self._update_rolling(K_recon, V_recon)
            return K_recon, V_recon
