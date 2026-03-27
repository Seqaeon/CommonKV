import torch
import torch.nn as nn
import math
import os
import atexit
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
    calibration_path: Optional[str] = None
    trace_output_path: Optional[str] = None
    trace_max_samples: int = 400000
    trace_chunk_size: int = 0

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
    
    _trace_registered = False
    _trace_output_path = None
    _trace_max_samples = 400000
    _trace_chunk_size = 0
    _trace_delta_k: List[torch.Tensor] = []
    _trace_delta_v: List[torch.Tensor] = []
    _reported_calibration_loads = set()
    _trace_part_idx = 0

    def __init__(self, **kwargs):
        super().__init__()
        # Extract config from kwargs or use defaults
        self.apkvc_config = APKVCConfig(
            predictor_type=kwargs.get("predictor_type", 'identity'),
            rd_threshold=kwargs.get("rd_threshold", 0.05),
            max_anchor_interval=kwargs.get("max_anchor_interval", 16),
            K_num_codebooks=kwargs.get("K_num_codebooks", 4),
            V_num_codebooks=kwargs.get("V_num_codebooks", 2),
            calibration_path=kwargs.get("apkvc_calibration_path", kwargs.get("calibration_path", None)),
            trace_output_path=kwargs.get("apkvc_trace_output_path", kwargs.get("trace_output_path", None)),
            trace_max_samples=int(kwargs.get("apkvc_trace_max_samples", kwargs.get("trace_max_samples", 400000))),
            trace_chunk_size=int(kwargs.get("apkvc_trace_chunk_size", kwargs.get("trace_chunk_size", 0))),
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
        self._setup_trace_dump()

    @classmethod
    def _save_trace_payload(cls, out_path, delta_k, delta_v):
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "delta_K_base": delta_k.cpu().half(),
                "delta_V": delta_v.cpu().half(),
                "metadata": {
                    "num_samples": int(min(delta_k.shape[0], delta_v.shape[0])),
                    "head_dim": int(delta_k.shape[-1]),
                },
            },
            out_path,
        )
        print(f"[APKVC] dumped trace residuals -> {out_path}")

    @classmethod
    def _dump_traces_at_exit(cls):
        if not cls._trace_output_path:
            return
        if len(cls._trace_delta_k) == 0 or len(cls._trace_delta_v) == 0:
            return
        try:
            delta_k = torch.cat(cls._trace_delta_k, dim=0)[: cls._trace_max_samples].contiguous()
            delta_v = torch.cat(cls._trace_delta_v, dim=0)[: cls._trace_max_samples].contiguous()
            if cls._trace_chunk_size > 0:
                out_path = f"{cls._trace_output_path}.part{cls._trace_part_idx:04d}.pt"
                cls._trace_part_idx += 1
            else:
                out_path = cls._trace_output_path
            cls._save_trace_payload(out_path, delta_k, delta_v)
        except Exception as e:
            print(f"[APKVC][WARN] failed to dump traces: {e}")

    def _setup_trace_dump(self):
        path = self.apkvc_config.trace_output_path
        if not path:
            return
        AttentionAwarePredictiveKVCluster._trace_output_path = path
        AttentionAwarePredictiveKVCluster._trace_max_samples = max(
            1, int(self.apkvc_config.trace_max_samples)
        )
        AttentionAwarePredictiveKVCluster._trace_chunk_size = max(
            0, int(self.apkvc_config.trace_chunk_size)
        )
        if not AttentionAwarePredictiveKVCluster._trace_registered:
            atexit.register(AttentionAwarePredictiveKVCluster._dump_traces_at_exit)
            AttentionAwarePredictiveKVCluster._trace_registered = True

    def _append_trace_samples(self, delta_K_base, delta_V):
        if not self.apkvc_config.trace_output_path:
            return
        k = delta_K_base.reshape(-1, delta_K_base.shape[-1]).detach().to("cpu", dtype=torch.float32)
        v = delta_V.reshape(-1, delta_V.shape[-1]).detach().to("cpu", dtype=torch.float32)
        if k.numel() == 0 or v.numel() == 0:
            return
        AttentionAwarePredictiveKVCluster._trace_delta_k.append(k)
        AttentionAwarePredictiveKVCluster._trace_delta_v.append(v)
        total = sum(x.shape[0] for x in AttentionAwarePredictiveKVCluster._trace_delta_k)
        if total > AttentionAwarePredictiveKVCluster._trace_max_samples:
            # keep only recent chunks under budget
            while (
                len(AttentionAwarePredictiveKVCluster._trace_delta_k) > 1
                and total > AttentionAwarePredictiveKVCluster._trace_max_samples
            ):
                dropped = AttentionAwarePredictiveKVCluster._trace_delta_k.pop(0).shape[0]
                AttentionAwarePredictiveKVCluster._trace_delta_v.pop(0)
                total -= dropped
        chunk_size = AttentionAwarePredictiveKVCluster._trace_chunk_size
        if chunk_size > 0 and total >= chunk_size:
            try:
                delta_k = torch.cat(AttentionAwarePredictiveKVCluster._trace_delta_k, dim=0).contiguous()
                delta_v = torch.cat(AttentionAwarePredictiveKVCluster._trace_delta_v, dim=0).contiguous()
                out_path = f"{AttentionAwarePredictiveKVCluster._trace_output_path}.part{AttentionAwarePredictiveKVCluster._trace_part_idx:04d}.pt"
                AttentionAwarePredictiveKVCluster._trace_part_idx += 1
                AttentionAwarePredictiveKVCluster._save_trace_payload(out_path, delta_k, delta_v)
                AttentionAwarePredictiveKVCluster._trace_delta_k = []
                AttentionAwarePredictiveKVCluster._trace_delta_v = []
            except Exception as e:
                print(f"[APKVC][WARN] failed to flush trace chunk: {e}")

    def _try_load_calibrated_codebooks(self, head_dim, device, dtype):
        path = self.apkvc_config.calibration_path
        if not path:
            return False
        if not os.path.isfile(path):
            print(f"[APKVC][WARN] calibration file not found: {path}. Falling back to random codebooks.")
            return False
        try:
            payload = torch.load(path, map_location="cpu")
            k_cbs = payload["K_codebooks"]
            v_cbs = payload["V_codebooks"]
            if isinstance(k_cbs, list):
                k_cbs = torch.stack(k_cbs, dim=0)
            if isinstance(v_cbs, list):
                v_cbs = torch.stack(v_cbs, dim=0)
            # expected: [num_codebooks, codebook_size, head_dim]
            if k_cbs.dim() != 3 or v_cbs.dim() != 3:
                raise ValueError("calibrated codebooks must be rank-3 tensors")
            if k_cbs.shape[-1] != head_dim or v_cbs.shape[-1] != head_dim:
                raise ValueError(
                    f"head_dim mismatch: expected {head_dim}, got K={k_cbs.shape[-1]}, V={v_cbs.shape[-1]}"
                )
            if k_cbs.shape[0] != self.apkvc_config.K_num_codebooks or v_cbs.shape[0] != self.apkvc_config.V_num_codebooks:
                raise ValueError(
                    "num_codebooks mismatch between config and calibration file "
                    f"(K expected={self.apkvc_config.K_num_codebooks}, file={k_cbs.shape[0]}; "
                    f"V expected={self.apkvc_config.V_num_codebooks}, file={v_cbs.shape[0]})"
                )
            self.codebooks_K = nn.ParameterList(
                [nn.Parameter(k_cbs[i].to(device=device, dtype=dtype).contiguous()) for i in range(k_cbs.shape[0])]
            )
            self.codebooks_V = nn.ParameterList(
                [nn.Parameter(v_cbs[i].to(device=device, dtype=dtype).contiguous()) for i in range(v_cbs.shape[0])]
            )
            if path not in AttentionAwarePredictiveKVCluster._reported_calibration_loads:
                print(f"[APKVC] Loaded calibrated codebooks from: {path}")
                AttentionAwarePredictiveKVCluster._reported_calibration_loads.add(path)
            return True
        except Exception as e:
            print(f"[APKVC][WARN] failed to load calibration file '{path}': {e}. Using random codebooks.")
            return False

    def _init_lazy(self, head_dim, device, dtype):
        """Lazy initialization of codebooks once head_dim is known."""
        self.head_dim = head_dim
        if self._try_load_calibrated_codebooks(head_dim, device, dtype):
            self.initialized = True
            return
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
            if self.apkvc_config.trace_output_path and key_states.shape[-2] > 1:
                prefill_delta_k = key_states[:, :, 1:, :] - key_states[:, :, :-1, :]
                prefill_delta_v = value_states[:, :, 1:, :] - value_states[:, :, :-1, :]
                # For prefill traces we keep deltas in current space; decode-time traces remain RoPE-base.
                self._append_trace_samples(prefill_delta_k, prefill_delta_v)
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
        self._append_trace_samples(delta_K_base, delta_V)

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
