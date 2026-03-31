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
    predictor_type: str  = 'attention' # 'identity', 'linear', 'attention', or 'none'
    per_layer_codebooks: bool = False
    compress_K: bool = True
    compress_V: bool = True
    use_scale_normalization: bool = True
    
    attention_window_size: int = 64
    alpha_K: float       = 1.5
    beta_K: float        = -0.5
    alpha_V: float       = 1.5
    beta_V: float        = -0.5

    use_rope_aware_aq: bool  = True
    K_num_codebooks: int     = 4
    V_num_codebooks: int     = 2
    codebook_size: int       = 256
    
    rd_metric: str       = 'sampled_attention_output' # 'key_dot' or 'sampled_attention_output'
    rd_threshold: float  = 0.1
    rd_sample_heads: int = 4

    max_anchor_interval: int         = 16
    linear_window_size: int          = 2
    residual_norm_threshold_K: float = 1000.0
    residual_norm_threshold_V: float = 1000.0
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
    _trace_delta_k: Dict[int, List[torch.Tensor]] = {}
    _trace_delta_v: Dict[int, List[torch.Tensor]] = {}
    _reported_calibration_loads = set()
    _trace_part_idx = 0

    def __init__(self, **kwargs):
        super().__init__()
        self.layer_idx = kwargs.get('layer_idx', 0)
        def _get(name, default=None):
            return kwargs.get(name, kwargs.get(f"apkvc_{name}", default))
        def _get_bool(name, default=False):
            val = _get(name, default)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("1", "true", "yes", "y", "on")
            return bool(val)
        # Extract config from kwargs or use defaults
        self.apkvc_config = APKVCConfig(
            predictor_type=_get("predictor_type", 'identity'),
            per_layer_codebooks=_get_bool("per_layer_codebooks", False),
            compress_K=_get_bool("compress_K", True),
            compress_V=_get_bool("compress_V", True),
            use_scale_normalization=_get_bool("use_scale_normalization", True),
            attention_window_size=_get("attention_window_size", 16),
            rd_threshold=_get("rd_threshold", 0.05),
            max_anchor_interval=_get("max_anchor_interval", 16),
            K_num_codebooks=_get("K_num_codebooks", 4),
            V_num_codebooks=_get("V_num_codebooks", 2),
            calibration_path=_get("calibration_path", None),
            trace_output_path=_get("trace_output_path", None),
            trace_max_samples=int(_get("trace_max_samples", 400000)),
            trace_chunk_size=int(_get("trace_chunk_size", 0)),
        )
        
        self.head_dim = None
        self.codebooks_K = nn.ParameterList()
        self.codebooks_V = nn.ParameterList()
        
        # Internal state for tracking anchors and residuals
        self.entries = [] # List of dicts representing cached tokens
        self.last_anchor_t = -1
        self.last_distortion = 0.0
        self.distortion_history = [] # History of all distortion values
        self.last_residual_norm = 0.0
        
        # For prediction
        self.last_K = None
        self.last_V = None
        self.last_K2 = None
        self.last_V2 = None
        self.anchor_window_K = [] # Rolling buffer for linear predictor
        self.anchor_window_V = []
        self.anchor_buffer_K = None
        self.anchor_buffer_V = None
        
        self.initialized = False
        self._setup_trace_dump()

    @classmethod
    def _save_trace_payload(cls, out_path, delta_k_dict, delta_v_dict):
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        # Determine head_dim from the first available tensor
        head_dim = 128
        for k_tensor in delta_k_dict.values():
            if k_tensor.shape[0] > 0:
                head_dim = int(k_tensor.shape[-1])
                break
                
        torch.save(
            {
                "delta_K_base": {k: v for k, v in delta_k_dict.items()},
                "delta_V": {k: v for k, v in delta_v_dict.items()},
                "metadata": {
                    "num_layers": len(delta_k_dict),
                    "head_dim": head_dim,
                },
            },
            out_path,
        )
        print(f"[APKVC] dumped trace residuals (layers: {len(delta_k_dict)}) -> {out_path}")

    @classmethod
    def _dump_traces_at_exit(cls):
        if not cls._trace_output_path:
            return
        if len(cls._trace_delta_k) == 0 or len(cls._trace_delta_v) == 0:
            return
        try:
            k_dict, v_dict = {}, {}
            for l_idx in list(cls._trace_delta_k.keys()):
                k_list = cls._trace_delta_k[l_idx]
                v_list = cls._trace_delta_v[l_idx]
                if len(k_list) == 0: continue
                # Free memory before cat by clearing the class dictionary
                cls._trace_delta_k[l_idx] = [] 
                cls._trace_delta_v[l_idx] = []
                k_dict[l_idx] = torch.cat(k_list, dim=0)[: cls._trace_max_samples].contiguous()
                v_dict[l_idx] = torch.cat(v_list, dim=0)[: cls._trace_max_samples].contiguous()
            
            if len(k_dict) == 0: return

            if cls._trace_chunk_size > 0:
                out_path = f"{cls._trace_output_path}.part{cls._trace_part_idx:04d}.pt"
                cls._trace_part_idx += 1
            else:
                out_path = cls._trace_output_path
            cls._save_trace_payload(out_path, k_dict, v_dict)
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
        k = delta_K_base.reshape(-1, delta_K_base.shape[-1]).detach().to("cpu", dtype=torch.float16)
        v = delta_V.reshape(-1, delta_V.shape[-1]).detach().to("cpu", dtype=torch.float16)
        if k.numel() == 0 or v.numel() == 0:
            return
            
        l_idx = self.layer_idx
        if l_idx not in AttentionAwarePredictiveKVCluster._trace_delta_k:
            AttentionAwarePredictiveKVCluster._trace_delta_k[l_idx] = []
            AttentionAwarePredictiveKVCluster._trace_delta_v[l_idx] = []
            
        AttentionAwarePredictiveKVCluster._trace_delta_k[l_idx].append(k)
        AttentionAwarePredictiveKVCluster._trace_delta_v[l_idx].append(v)
        
        total = sum(x.shape[0] for x in AttentionAwarePredictiveKVCluster._trace_delta_k[l_idx])
        if total > AttentionAwarePredictiveKVCluster._trace_max_samples:
            # keep only recent chunks under budget
            while (
                len(AttentionAwarePredictiveKVCluster._trace_delta_k[l_idx]) > 1
                and total > AttentionAwarePredictiveKVCluster._trace_max_samples
            ):
                dropped = AttentionAwarePredictiveKVCluster._trace_delta_k[l_idx].pop(0).shape[0]
                AttentionAwarePredictiveKVCluster._trace_delta_v[l_idx].pop(0)
                total -= dropped
                
        chunk_size = AttentionAwarePredictiveKVCluster._trace_chunk_size
        if chunk_size > 0 and total >= chunk_size:
            try:
                # Flush ALL layers if any layer hits chunk size to keep geometry aligned
                k_dict, v_dict = {}, {}
                for idx in AttentionAwarePredictiveKVCluster._trace_delta_k.keys():
                    if len(AttentionAwarePredictiveKVCluster._trace_delta_k[idx]) == 0: continue
                    k_dict[idx] = torch.cat(AttentionAwarePredictiveKVCluster._trace_delta_k[idx], dim=0).contiguous()
                    v_dict[idx] = torch.cat(AttentionAwarePredictiveKVCluster._trace_delta_v[idx], dim=0).contiguous()
                    AttentionAwarePredictiveKVCluster._trace_delta_k[idx] = []
                    AttentionAwarePredictiveKVCluster._trace_delta_v[idx] = []
                out_path = f"{AttentionAwarePredictiveKVCluster._trace_output_path}.part{AttentionAwarePredictiveKVCluster._trace_part_idx:04d}.pt"
                AttentionAwarePredictiveKVCluster._trace_part_idx += 1
                AttentionAwarePredictiveKVCluster._save_trace_payload(out_path, k_dict, v_dict)
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
                
            # If per_layer_codebooks is config'ed, expect [L, M, S, D]
            if self.apkvc_config.per_layer_codebooks:
                if k_cbs.dim() != 4 or v_cbs.dim() != 4:
                    raise ValueError(f"per_layer_codebooks=True but calibration file is rank-{k_cbs.dim()} (expected 4)")
                l_idx = self.layer_idx
                # Fallback to last layer if L doesn't match evaluation layer count
                if l_idx >= k_cbs.shape[0]: l_idx = k_cbs.shape[0] - 1
                k_cbs = k_cbs[l_idx]
                v_cbs = v_cbs[l_idx]
            else:
                if k_cbs.dim() != 3 or v_cbs.dim() != 3:
                    raise ValueError(f"per_layer_codebooks=False but calibration file is rank-{k_cbs.dim()} (expected 3)")

            # expected: [num_codebooks, codebook_size, head_dim]
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
        if len(indices) == 0:
            return torch.zeros(0, 0, self.head_dim)
        if len(codebooks) == 0:
            B, H = indices[0].shape
            return torch.zeros(B, H, self.head_dim, device=indices[0].device, dtype=torch.float32)
        device = codebooks[0].device
        dtype = codebooks[0].dtype
        B, H = indices[0].shape
        out = torch.zeros(B, H, self.head_dim, device=device, dtype=dtype)
        for idx, C in zip(indices, codebooks):
            out += C[idx]
        return out

    def _update_rolling(self, K, V):
        self.last_V2 = self.last_V
        # Store new anchors
        self.last_K = K.clone()
        self.last_V = V.clone()
        
        # New: Rolling window for multi-token linear predictor
        self.anchor_window_K.append(K.clone())
        self.anchor_window_V.append(V.clone())
        if len(self.anchor_window_K) > self.apkvc_config.linear_window_size:
            self.anchor_window_K.pop(0)
            self.anchor_window_V.pop(0)

    def _append_anchor(self, K_base, V):
        """Append to the anchor buffer for attention prediction."""
        if self.anchor_buffer_K is None:
            self.anchor_buffer_K = K_base.clone()
            self.anchor_buffer_V = V.clone()
        else:
            self.anchor_buffer_K = torch.cat([self.anchor_buffer_K, K_base.clone()], dim=-2)
            self.anchor_buffer_V = torch.cat([self.anchor_buffer_V, V.clone()], dim=-2)
            
        W = self.apkvc_config.attention_window_size
        if self.anchor_buffer_K.shape[-2] > W:
            self.anchor_buffer_K = self.anchor_buffer_K[..., -W:, :]
            self.anchor_buffer_V = self.anchor_buffer_V[..., -W:, :]

    def should_reset(self, t):
        """Check if distance-based or interval-based reset is needed."""
        interval_reset = (t - self.last_anchor_t) >= self.apkvc_config.max_anchor_interval
        return interval_reset

    def compute_distortion(self, Q, K_true, K_reconstructed):
        """Compute distortion metric."""
        if Q is None:
            return 0.0
        if self.apkvc_config.rd_metric == 'key_dot':
            scale = math.sqrt(self.head_dim)
            score_true = torch.sum(Q * K_true, dim=-1, keepdim=True) / scale
            score_comp = torch.sum(Q * K_reconstructed, dim=-1, keepdim=True) / scale
            return torch.max(torch.abs(score_true - score_comp)).item()
        else:
            # Robust MSE loss prevents catastrophic null-space hallucination
            return torch.nn.functional.mse_loss(K_true, K_reconstructed).item()

    def _residual_norm_exceeds(self, delta_K, delta_V):
        """Residual magnitude gate used by the anchor reset policy."""
        k_norm = delta_K.norm(dim=-1).mean().item()
        v_norm = delta_V.norm(dim=-1).mean().item()
        self.last_residual_norm = max(k_norm, v_norm)
        return (
            k_norm > self.apkvc_config.residual_norm_threshold_K
            or v_norm > self.apkvc_config.residual_norm_threshold_V
        )

    def _update_rolling(self, state, K_recon, V_recon):
        """Maintains the immediate N past anchors inside the state object."""
        state['last_K2'] = state['last_K']
        state['last_K'] = K_recon.clone()
        state['last_V2'] = state['last_V']
        state['last_V'] = V_recon.clone()
        
        # Keep window size N
        W = self.apkvc_config.linear_window_size
        if 'anchor_window_K' not in state:
            state['anchor_window_K'] = []
            state['anchor_window_V'] = []
            
        state['anchor_window_K'].append(state['last_K'])
        state['anchor_window_V'].append(state['last_V'])
        if len(state['anchor_window_K']) > W:
            state['anchor_window_K'].pop(0)
            state['anchor_window_V'].pop(0)

    def _should_reset_state(self, t, state):
        interval = self.apkvc_config.max_anchor_interval
        if interval > 0 and (t - state['last_anchor_t']) >= interval:
            return True
        return False

    def compress_decode_token(self, key_states, value_states, Q, t, state, abs_pos=None):
        """
        K_true, V_true: [B, H, 1, D] — true KV for this token
        Q: [B, H, 1, D] - true Q for this token
        t: decode-local step counter (0, 1, 2...) — used for anchor interval logic
        abs_pos: absolute sequence position (prefill_len + t) — used for RoPE
                 derotation/rotation.  If None, falls back to t (incorrect for
                 sequences with a non-zero prefill, kept for backward compat).
        state: running state dict safely isolated by HybridAPKVCCache
        """
        K_true = key_states
        V_true = value_states
        rope_pos = abs_pos if abs_pos is not None else t  # correct absolute position
        if not self.initialized:
            self._init_lazy(
                head_dim=K_true.shape[-1],
                device=K_true.device,
                dtype=K_true.dtype,
            )
        
        if self.apkvc_config.use_rope_aware_aq:
            K_true_base = rope_derotate(K_true, rope_pos)  # was: t — wrong for non-zero prefill
        else:
            K_true_base = K_true
            
        # 1. Anchor Reset Check
        if t == 0 or self._should_reset_state(t, state):
            state['entries'].append({
                'is_anchor': True,
                'K': K_true.half(),
                'V': V_true.half(),
                'position': t
            })
            state['recon_cache'][t] = (K_true, V_true)
            state['last_anchor_t'] = t
            state['last_distortion'] = 0.0
            self.distortion_history.append(0.0)
            self._update_rolling(state, K_true_base, V_true)
            return K_true, V_true

        # 2. Predict KV via state trajectories
        if self.apkvc_config.predictor_type == 'linear' and len(state.get('anchor_window_K', [])) >= 2:
            win_K = state['anchor_window_K']
            win_V = state['anchor_window_V']
            if self.apkvc_config.linear_window_size == 2 or len(win_K) == 2:
                K_hat_base = self.apkvc_config.alpha_K * win_K[-1] + self.apkvc_config.beta_K * win_K[-2]
                V_hat = self.apkvc_config.alpha_V * win_V[-1] + self.apkvc_config.beta_V * win_V[-2]
            else:
                deltas_k = [(win_K[i] - win_K[i-1]) for i in range(1, len(win_K))]
                avg_delta_k = torch.stack(deltas_k).mean(dim=0)
                K_hat_base = win_K[-1] + 0.5 * avg_delta_k
                deltas_v = [(win_V[i] - win_V[i-1]) for i in range(1, len(win_V))]
                avg_delta_v = torch.stack(deltas_v).mean(dim=0)
                V_hat = win_V[-1] + 0.5 * avg_delta_v
        elif self.apkvc_config.predictor_type == 'attention' and state.get('anchor_buffer_K') is not None:
             # Legacy attention path disabled for clean implementation to match guide
             K_hat_base = state['last_K']
             V_hat = state['last_V']
        elif self.apkvc_config.predictor_type == 'none':
            K_hat_base = torch.zeros_like(K_true_base)
            V_hat = torch.zeros_like(V_true)
        else:
            # Identity predictor fallback (highest performance/stability)
            K_hat_base = state.get('last_K', torch.zeros_like(K_true_base))
            V_hat = state.get('last_V', torch.zeros_like(V_true))

        delta_K_base = K_true_base - K_hat_base
        delta_V = V_true - V_hat
        
        # 3. Scale Normalization (Fixed #2)
        scale_K, scale_V = None, None
        if self.apkvc_config.use_scale_normalization:
            scale_K = delta_K_base.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            delta_K_normed = delta_K_base / scale_K
            scale_V = delta_V.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            delta_V_normed = delta_V / scale_V
        else:
            delta_K_normed = delta_K_base
            delta_V_normed = delta_V

        residual_too_large = self._residual_norm_exceeds(delta_K_normed, delta_V_normed)
        if residual_too_large:
            state['entries'].append({'is_anchor': True, 'position': t})
            self.distortion_history.append(self.last_residual_norm)
            state['recon_cache'][t] = (K_true, V_true)
            state['last_anchor_t'] = t
            state['last_distortion'] = 0.0
            self._update_rolling(state, K_true_base, V_true)
            return K_true, V_true
            
        # We trace the NORMED residuals for Codebook RVQ calibration
        self._append_trace_samples(delta_K_normed, delta_V_normed)
        
        # 4. Additive Quantization (AQ)
        if self.apkvc_config.compress_K:
            codes_K = self.additive_quantize(delta_K_normed.squeeze(2), self.codebooks_K)
            recon_delta_K_normed = self.additive_decode(codes_K, self.codebooks_K).unsqueeze(2)
        else:
            codes_K = None
            recon_delta_K_normed = delta_K_normed
            
        if self.apkvc_config.compress_V:
            codes_V = self.additive_quantize(delta_V_normed.squeeze(2), self.codebooks_V)
            recon_delta_V_normed = self.additive_decode(codes_V, self.codebooks_V).unsqueeze(2)
        else:
            codes_V = None
            recon_delta_V_normed = delta_V_normed
            
        # 5. Reconstruction
        if self.apkvc_config.use_scale_normalization:
            recon_delta_K_base = recon_delta_K_normed * scale_K
            recon_delta_V = recon_delta_V_normed * scale_V
        else:
            recon_delta_K_base = recon_delta_K_normed
            recon_delta_V = recon_delta_V_normed
            
        K_recon_base = K_hat_base + recon_delta_K_base
        if self.apkvc_config.use_rope_aware_aq:
            K_recon = rope_rotate(K_recon_base, rope_pos)  # was: t — wrong for non-zero prefill
        else:
            K_recon = K_recon_base
        V_recon = V_hat + recon_delta_V
        
        # 6. Attention Distortion Fallback Check
        distortion = self.compute_distortion(Q, K_true, K_recon)
        state['last_distortion'] = distortion
        self.distortion_history.append(distortion)
        
        if distortion > self.apkvc_config.rd_threshold:
            state['entries'].append({'is_anchor': True, 'position': t})
            state['recon_cache'][t] = (K_true, V_true)
            state['last_anchor_t'] = t
            state['last_distortion'] = 0.0
            self._update_rolling(state, K_true_base, V_true)
            return K_true, V_true
        else:
            state['entries'].append({
                'is_anchor': False,
                'codes_K': codes_K,
                'codes_V': codes_V,
                'scale_K': scale_K.squeeze(2).half() if scale_K is not None else None,
                'scale_V': scale_V.squeeze(2).half() if scale_V is not None else None,
                'position': t
            })
            state['recon_cache'][t] = (K_recon, V_recon)
            self._update_rolling(state, K_recon_base, V_recon)
            return K_recon, V_recon
