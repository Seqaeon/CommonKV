"""
IAVQ-KC: Importance-Aware Vector Quantization Key Cache Compression
Design reference: IAVQ_KC_design.md

Compression strategy:
  - Prefill keys:  importance-weighted k-means codebook (built online from this
                   prompt's own attention weights) + PCA residual projection
                   (8 int8 scalars + 1 fp16 scale per token per head).
  - Prefill values: INT8 per-token asymmetric quantization.
  - Decode keys:   new tokens enter as recency anchors (full fp16).
                   When a token ages out of the recency window it is either
                   promoted to the importance pool or compressed.
  - Decode values: INT8 per-token asymmetric quantization.

No offline calibration required. The codebook is built online from each
prompt's own attention statistics during the prefill forward pass.
"""

import heapq
import gc
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from .base import KVCacheMethod, CacheState
from ldcb.utils import get_kv_iterator


# ---------------------------------------------------------------------------
# INT8 value quantization helpers (per-token asymmetric)
# ---------------------------------------------------------------------------

def _quant_int8_per_token(V: torch.Tensor):
    """
    Per-token asymmetric uint8 quantization.
    V: [B, H, T, D]
    Returns (V_uint8 [B,H,T,D], scale [B,H,T,1] fp16, zero [B,H,T,1] fp16)
    """
    V_min = V.float().amin(dim=-1, keepdim=True)
    V_max = V.float().amax(dim=-1, keepdim=True)
    scale = (V_max - V_min).clamp(min=1e-5) / 255.0
    V_uint8 = ((V.float() - V_min) / scale).round().clamp(0, 255).to(torch.uint8)
    return V_uint8, scale.half(), V_min.half()


def _dequant_int8_per_token(V_uint8, scale, zero, dtype):
    """Reconstruct fp tensor from uint8 + scale + zero-point."""
    return (V_uint8.float() * scale.float() + zero.float()).to(dtype)


# ---------------------------------------------------------------------------
# Importance-weighted k-means
# ---------------------------------------------------------------------------

def _importance_weighted_kmeans(
    X: torch.Tensor, weights: torch.Tensor, k: int, iters: int = 3
) -> torch.Tensor:
    """
    Importance-biased k-means on GPU/CPU.
    X:       [N, D]
    weights: [N] float — attention column sums (importance scores)
    Returns: centroids [k, D]
    """
    N, D = X.shape
    weights = weights.float().clamp(min=1e-8)

    if N <= k:
        # Not enough distinct points — pad by repeating
        reps = (k + N - 1) // N
        X = X.repeat(reps, 1)[:k]
        weights = weights.repeat(reps)[:k]
        N = k

    probs = weights / weights.sum()
    centroid_ids = torch.multinomial(probs, k, replacement=False)
    centroids = X[centroid_ids].clone().float()

    for _ in range(iters):
        # E-step: assign each point to nearest centroid
        # Use chunked cdist to avoid OOM on large T
        chunk = 4096
        assignments = torch.empty(N, dtype=torch.long, device=X.device)
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            d2 = torch.cdist(X[s:e].float(), centroids)  # [B, k]
            assignments[s:e] = d2.argmin(dim=1)

        # M-step: importance-weighted centroid update
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask = assignments == c
            if mask.sum() == 0:
                new_centroids[c] = centroids[c]
                continue
            w = weights[mask]
            new_centroids[c] = (X[mask].float() * w.unsqueeze(1)).sum(0) / w.sum()

        centroids = new_centroids

    return centroids.to(X.dtype)


# ---------------------------------------------------------------------------
# Per-layer compressed state
# ---------------------------------------------------------------------------

class _LayerState:
    """All compressed key/value state for one transformer layer."""

    def __init__(
        self,
        codebook: torch.Tensor,    # [m, D] fp16
        pca_basis: torch.Tensor,   # [P, D] fp16
        dtype: torch.dtype,
        P: int = 8,
    ):
        self.codebook = codebook   # [m, D]
        self.pca_basis = pca_basis # [P, D]
        self.dtype = dtype
        self.P = P
        self.m = codebook.shape[0]

        # Anchors: {position → [H, D]} full fp16
        self.anchor_keys: dict[int, torch.Tensor] = {}
        self.anchor_ids: set[int] = set()

        # Compressed: {position → (idx [H] uint8, proj_int8 [H,P] int8, scale [H] fp16)}
        self.compressed: dict[int, tuple] = {}

        # Value cache: list of per-token slices stored as (uint8, scale, zero)
        # Each element corresponds to one token; index = insertion order.
        # _val_pos[i] = absolute position of the i-th stored value token.
        self._val_uint8: list[torch.Tensor] = []   # each [B, H, 1, D]
        self._val_scale: list[torch.Tensor] = []
        self._val_zero:  list[torch.Tensor] = []
        self._val_pos:   list[int] = []             # absolute positions

        # Importance tracking
        self.importance_scores: torch.Tensor | None = None  # [T_total] float32
        # min-heap: (score, pos) — min at top so we can pop worst importance anchor
        self._importance_heap: list = []

    # ---- Codebook compression ----

    def assign_and_compress(self, k: torch.Tensor, pos: int) -> None:
        """
        Compress key at `pos` into (centroid_idx, projected_residual_int8, scale).
        k: [H, D]
        """
        H, D = k.shape
        C = self.codebook.float()   # [m, D]
        B = self.pca_basis.float()  # [P, D]

        # Nearest centroid per head
        dists = torch.cdist(k.float(), C)    # [H, m]
        idx = dists.argmin(dim=1)            # [H]

        # Residual from centroid
        centroids_h = C[idx]                 # [H, D]
        residual = k.float() - centroids_h   # [H, D]

        # Project onto PCA basis: [H, P]
        proj = residual @ B.T                # [H, P]

        # Per-head int8 quantization of projections
        scale = proj.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)  # [H, 1]
        proj_int8 = (proj / scale).clamp(-128, 127).to(torch.int8)    # [H, P]

        self.compressed[pos] = (
            idx.to(torch.uint8),          # [H]
            proj_int8,                    # [H, P]
            scale.squeeze(1).half(),      # [H]
        )

    def reconstruct_key(self, pos: int, H: int, D: int) -> torch.Tensor:
        """Reconstruct key at pos → [H, D] in self.dtype."""
        if pos in self.anchor_ids:
            return self.anchor_keys[pos].to(self.dtype)

        idx, proj_int8, scale = self.compressed[pos]
        # Dequantize projections: [H, P]
        proj = proj_int8.float() * scale.float().unsqueeze(1)           # [H, P]
        # Reconstruct: centroid + proj @ B
        centroids_h = self.codebook[idx.long()].float()                  # [H, D]
        k_hat = centroids_h + proj @ self.pca_basis.float()              # [H, D]
        return k_hat.to(self.dtype)

    def reconstruct_full_keys(self, H: int, D: int, device) -> torch.Tensor:
        """Reconstruct all stored keys → [1, H, T, D]."""
        all_pos = sorted(list(self.anchor_ids) + list(self.compressed.keys()))
        T = len(all_pos)
        K = torch.zeros(T, H, D, dtype=self.dtype, device=device)
        for i, pos in enumerate(all_pos):
            K[i] = self.reconstruct_key(pos, H, D)
        return K.permute(1, 0, 2).unsqueeze(0)  # [1, H, T, D]

    def reconstruct_full_values(self) -> torch.Tensor:
        """Reconstruct all stored values → [B, H, T, D]."""
        return torch.cat(
            [_dequant_int8_per_token(u, s, z, self.dtype)
             for u, s, z in zip(self._val_uint8, self._val_scale, self._val_zero)],
            dim=2,
        )

    def add_value(self, v: torch.Tensor, pos: int) -> None:
        """v: [B, H, 1, D] — append one token's value (compressed to INT8)."""
        u, s, z = _quant_int8_per_token(v)
        self._val_uint8.append(u)
        self._val_scale.append(s)
        self._val_zero.append(z)
        self._val_pos.append(pos)

    def total_tokens(self) -> int:
        return len(self.anchor_ids) + len(self.compressed)

    # ---- Importance pool ----

    def push_importance(self, pos: int, score: float) -> None:
        heapq.heappush(self._importance_heap, (score, pos))

    def pop_min_importance(self) -> tuple:
        return heapq.heappop(self._importance_heap)  # (score, pos)

    def min_importance_score(self) -> float:
        return self._importance_heap[0][0] if self._importance_heap else -float("inf")

    def importance_pool_size(self) -> int:
        return len(self._importance_heap)


# ---------------------------------------------------------------------------
# IAVQ-KC Method
# ---------------------------------------------------------------------------

class IAVQKCMethod(KVCacheMethod):
    """
    Importance-Aware Vector Quantization Key Cache (IAVQ-KC).

    Keys are compressed via an online, importance-weighted codebook built from
    each prompt's own prefill attention statistics — no offline calibration
    needed.  Values use simple INT8 per-token asymmetric quantization.

    Anchor strategy:
      - last R generated tokens are always stored in full fp16 (recency anchors)
      - top-M old tokens by accumulated attention mass are kept in full fp16
        (importance anchors)
    """

    def __init__(
        self,
        codebook_size: int = 256,
        recency_window: int = 64,
        importance_anchors: int = 128,
        kmeans_iters: int = 3,
        pca_components: int = 8,
        update_strategy: str = "recency_gated",  # recency_gated | frozen | always
        centroid_update_alpha: float = 0.01,
        update_importance_during_decode: bool = False,
    ):
        self.A = codebook_size
        self.R = recency_window
        self.M = importance_anchors
        self.kmeans_iters = kmeans_iters
        self.P = pca_components
        self.update_strategy = update_strategy
        self.alpha = centroid_update_alpha
        self.update_importance = update_importance_during_decode
        self.name = "IAVQ-KC"

    # ------------------------------------------------------------------
    # Byte estimator
    # ------------------------------------------------------------------

    def _cache_bytes(
        self, layer_states: list, n_layers: int, n_heads: int, head_dim: int
    ) -> int:
        total = 0
        for state in layer_states:
            n_anchors = len(state.anchor_ids)
            n_comp = len(state.compressed)
            n_vals = len(state._val_uint8)

            # Keys: anchor fp16 [H, D] + compressed (1B idx + P B proj + 2B scale) per head
            total += n_anchors * n_heads * head_dim * 2
            total += n_comp * n_heads * (1 + self.P + 2)

            # Values: uint8 [B,H,1,D] + scale/zero [B,H,1,1] fp16 each
            total += n_vals * n_heads * head_dim * 1   # uint8 data
            total += n_vals * n_heads * 1 * 4          # scale + zero (fp16 each)

        return total

    # ------------------------------------------------------------------
    # Prefill: build layer state
    # ------------------------------------------------------------------

    def _build_layer_state(
        self,
        K_l: torch.Tensor,        # [B, H, T, D]
        V_l: torch.Tensor,        # [B, H, T, D]
        importance: torch.Tensor, # [T] float32
        layer_idx: int,
        num_layers: int,
        dtype: torch.dtype,
    ) -> _LayerState:
        B, H, T, D = K_l.shape
        m = max(4, self.A // max(num_layers, 1))    # centroids per layer

        # ---- Importance-weighted codebook ----
        # Pool keys across heads: [H*T, D]
        K_pool = K_l[0].permute(1, 0, 2).reshape(H * T, D)
        # Each position has the same importance across all heads
        w_pool = importance.unsqueeze(0).expand(H, -1).reshape(H * T)

        codebook = _importance_weighted_kmeans(
            K_pool, w_pool, k=m, iters=self.kmeans_iters
        ).to(dtype)

        # ---- PCA basis from prefill residuals ----
        dists = torch.cdist(K_pool.float(), codebook.float())  # [H*T, m]
        assign = dists.argmin(dim=1)                            # [H*T]
        residuals = K_pool.float() - codebook[assign].float()  # [H*T, D]

        try:
            _, _, Vt = torch.linalg.svd(residuals, full_matrices=False)
            pca_basis = Vt[: self.P].to(dtype)                 # [P, D]
        except Exception:
            pca_basis = F.normalize(
                torch.randn(self.P, D, device=K_l.device, dtype=dtype), dim=1
            )

        state = _LayerState(codebook=codebook, pca_basis=pca_basis,
                            dtype=dtype, P=self.P)

        # ---- Determine anchor set ----
        recency_start = max(0, T - self.R)
        recency_set = set(range(recency_start, T))

        older = list(range(0, recency_start))
        if older:
            imp_older = importance[older]
            topk_k = min(self.M, len(older))
            topk_vals, topk_rel_idx = imp_older.topk(topk_k)
            importance_set = {older[i] for i in topk_rel_idx.tolist()}
            for i in topk_rel_idx.tolist():
                state.push_importance(older[i], imp_older[i].item())
        else:
            importance_set = set()

        # ---- Compress / anchor each prefill token ----
        for t in range(T):
            k_t = K_l[0, :, t, :]     # [H, D]
            if t in importance_set or t in recency_set:
                state.anchor_keys[t] = k_t
                state.anchor_ids.add(t)
            else:
                state.assign_and_compress(k_t, t)

        # ---- Store prefill values (INT8 quantized per token) ----
        for t in range(T):
            state.add_value(V_l[:, :, t : t + 1, :], pos=t)

        state.importance_scores = importance.clone()
        return state

    # ------------------------------------------------------------------
    # Decode: anchor aging
    # ------------------------------------------------------------------

    def _age_out_token(self, state: _LayerState, aging_pos: int) -> None:
        """Transition aging_pos out of the recency window."""
        if aging_pos not in state.anchor_ids:
            return

        imp_score = (
            state.importance_scores[aging_pos].item()
            if (state.importance_scores is not None
                and aging_pos < state.importance_scores.shape[0])
            else 0.0
        )

        if state.importance_pool_size() < self.M:
            # Pool not full — promote directly (stays in anchor_ids)
            state.push_importance(aging_pos, imp_score)

        elif imp_score > state.min_importance_score():
            # Better than worst importance anchor — evict worst, promote this
            _, evict_pos = state.pop_min_importance()
            if evict_pos in state.anchor_ids:
                k_evict = state.anchor_keys.pop(evict_pos)
                state.anchor_ids.discard(evict_pos)
                state.assign_and_compress(k_evict, evict_pos)
            state.push_importance(aging_pos, imp_score)
            # aging_pos stays in anchor_ids

        else:
            # Not important enough — compress it
            k_aging = state.anchor_keys.pop(aging_pos)
            state.anchor_ids.discard(aging_pos)
            state.assign_and_compress(k_aging, aging_pos)

    def _soft_centroid_update(
        self, state: _LayerState, k_new: torch.Tensor
    ) -> None:
        """Exponential moving average update of nearest centroid. k_new: [H, D]"""
        if self.update_strategy == "frozen":
            return
        dists = torch.cdist(k_new.float(), state.codebook.float())  # [H, m]
        idx = dists.argmin(dim=1)                                    # [H]
        a = self.alpha
        for h in range(k_new.shape[0]):
            state.codebook[idx[h]] = (
                (1 - a) * state.codebook[idx[h]] + a * k_new[h]
            ).to(state.dtype)

    # ------------------------------------------------------------------
    # Main generate
    # ------------------------------------------------------------------

    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        checkpoint_steps: list,
    ):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        n_layers = model.config.num_hidden_layers
        n_heads = getattr(
            model.config, "num_key_value_heads", model.config.num_attention_heads
        )
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        T_prefill = inputs.input_ids.shape[1]

        # ----------------------------------------------------------------
        # Prefill pass — output_attentions=True to get importance scores
        # ----------------------------------------------------------------
        with torch.no_grad():
            prefill_out = model(
                inputs.input_ids,
                use_cache=True,
                output_attentions=True,
            )

        # Aggregate importance across layers and heads → [T_prefill]
        importance = torch.zeros(T_prefill, device=device, dtype=torch.float32)
        if prefill_out.attentions is not None:
            for attn in prefill_out.attentions:
                # attn: [B, H, T, T] → column sums averaged over heads/batch
                importance += attn[0].float().mean(dim=0).sum(dim=0)
            importance = importance / max(len(prefill_out.attentions), 1)
        importance = importance.clamp(min=0)

        # Build per-layer states
        layer_states: list[_LayerState] = []
        for l, (K_l, V_l) in get_kv_iterator(prefill_out.past_key_values):
            # K_l, V_l: [B, H, T, D]
            state = self._build_layer_state(
                K_l, V_l, importance, l, n_layers, dtype
            )
            layer_states.append(state)

        # First token from prefill logits
        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([inputs.input_ids, next_token], dim=1)
        tok_gen = 1
        del prefill_out
        gc.collect()

        snapshots = []
        chk_iter = iter(checkpoint_steps)
        cur_chk = next(chk_iter, None)

        # ----------------------------------------------------------------
        # Decode loop
        # ----------------------------------------------------------------
        with torch.no_grad():
            while tok_gen < max_new_tokens:

                # Absolute position of the *most recently stored* key token.
                # After prefill: T_prefill-1 + tok_gen-1 = T_prefill + tok_gen - 2
                # (the new token being added this step lands at T_prefill + tok_gen - 1)
                abs_pos = T_prefill + tok_gen - 1

                # Reconstruct full KV cache for attention
                hf_cache = DynamicCache()
                for i, state in enumerate(layer_states):
                    K_full = state.reconstruct_full_keys(n_heads, head_dim, device)
                    V_full = state.reconstruct_full_values()
                    hf_cache.update(K_full, V_full, i)

                out = model(
                    next_token,
                    past_key_values=hf_cache,
                    use_cache=True,
                    output_attentions=self.update_importance,
                )

                # Extract new K/V (last position from output cache)
                new_kvs = [
                    (K[:, :, -1:, :].clone(), V[:, :, -1:, :].clone())
                    for _, (K, V) in get_kv_iterator(out.past_key_values)
                ]

                # Optional importance score update from decode attention
                if self.update_importance and out.attentions is not None:
                    T_so_far = abs_pos + 1
                    for l, attn in enumerate(out.attentions):
                        # attn: [B, H, 1, T_so_far]
                        upd = attn[0, :, 0, :T_so_far].float().mean(dim=0)
                        sc = layer_states[l].importance_scores
                        if sc is not None:
                            if sc.shape[0] < T_so_far:
                                # Extend
                                sc = torch.cat(
                                    [sc, torch.zeros(T_so_far - sc.shape[0],
                                                     device=device)]
                                )
                                layer_states[l].importance_scores = sc
                            sc[:T_so_far] += upd

                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del out, hf_cache
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tok_gen += 1

                # Update each layer's state
                for l, state in enumerate(layer_states):
                    nK, nV = new_kvs[l]
                    k_new = nK[0, :, 0, :]    # [H, D]

                    # Store as recency anchor
                    state.anchor_keys[abs_pos] = k_new
                    state.anchor_ids.add(abs_pos)

                    # Soft centroid update while token is a recency anchor
                    if self.update_strategy in ("recency_gated", "always"):
                        self._soft_centroid_update(state, k_new)

                    # Age out the token that just fell off the recency window
                    aging_pos = abs_pos - self.R
                    if aging_pos >= 0:
                        self._age_out_token(state, aging_pos)

                    # Store new value token
                    state.add_value(nV, pos=abs_pos)

                # Checkpoint snapshot
                if tok_gen == cur_chk:
                    T_total = generated_ids.shape[1]
                    compr = self._cache_bytes(layer_states, n_layers, n_heads, head_dim)
                    full_kv = T_total * n_layers * n_heads * head_dim * 2 * 2
                    n_anch = sum(len(s.anchor_ids) for s in layer_states)
                    n_comp = sum(len(s.compressed) for s in layer_states)
                    snapshots.append(CacheState(
                        compressed_bytes=compr,
                        fullkv_bytes=full_kv,
                        anchor_count=n_anch,
                        residual_count=n_comp,
                    ))
                    cur_chk = next(chk_iter, None)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        T_total = generated_ids.shape[1]
        compr = self._cache_bytes(layer_states, n_layers, n_heads, head_dim)
        full_kv = T_total * n_layers * n_heads * head_dim * 2 * 2
        n_anch = sum(len(s.anchor_ids) for s in layer_states)
        n_comp = sum(len(s.compressed) for s in layer_states)
        final = CacheState(
            compressed_bytes=compr,
            fullkv_bytes=full_kv,
            anchor_count=n_anch,
            residual_count=n_comp,
        )

        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final)

        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return text, snapshots, final
