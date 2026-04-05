"""
CommVQ KV-cache compression — benchmark wrapper.

Uses pre-trained RVQ codebooks from:
    senfu/Llama-3.1-8B-Instruct-CommVQ-{bits}bit-codebook   (keys)

Values use INT8 per-token quantization because the CommVQ value compressor
is a trained neural network contained in the finetuned model checkpoint
(senfu/Llama-3.1-8B-Instruct-CommVQ-{bits}bit) and cannot be extracted
without loading that checkpoint separately.  INT8 is a strong standalone
baseline (equivalent to KIVI-int8 on values) and keeps comparisons fair.

Key encoding follows the CommVQ paper exactly:
  1. Reshape keys from [B, H, T, D] → [B*T, G, D_g] (pair-interleaved, per group)
  2. L2-normalise per token
  3. Residual VQ: greedy nearest-centre assignment, subtract, repeat
  4. Store codes (uint16) + prescale (fp16)

Key decoding:
  1. Look up codewords, sum across residual stages
  2. Rescale, unflatten back to [B, H, T, D]

Requires: huggingface_hub, einops
"""

import os
import glob
import sys

import torch
from transformers.cache_utils import DynamicCache

from .base import KVCacheMethod, CacheState
from ldcb.utils import get_kv_iterator


# ---------------------------------------------------------------------------
# CommVQ key encoder — self-contained, no hardcoded DIR
# ---------------------------------------------------------------------------

class _CommVQKeyEncoder:
    """Loads and applies CommVQ key codebooks for one layer."""

    def __init__(self, local_dir: str, layer_idx: int, bits: int, config):
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.bits = bits

        # Each .pt file corresponds to one "group" (a subset of KV heads)
        pattern = os.path.join(local_dir, f"{str(layer_idx).zfill(3)}_*.pt")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No CommVQ codebook files found for layer {layer_idx} in {local_dir}.\n"
                f"Pattern: {pattern}"
            )

        # self.codebook: [R, G, K, D_g]
        # R = num residual stages, G = num groups, K = codebook size, D_g = dim per group
        codebook_list = []
        for f in files:
            data = torch.load(f, map_location="cpu")["steps"]
            # Each step has "clustering_centers": [K, D_g]
            cc = torch.cat([s["clustering_centers"].unsqueeze(0) for s in data])  # [R, K, D_g]
            codebook_list.append(cc.unsqueeze(1))  # [R, 1, K, D_g]

        self.codebook = torch.cat(codebook_list, dim=1)   # [R, G, K, D_g]
        self.R = self.codebook.shape[0]
        self.G = self.codebook.shape[1]

    def _preprocess(self, K: torch.Tensor):
        """
        K: [B, H_kv, T, D]   standard attention format
        Returns
          tensor   : [(B*T), G, D_g]   pair-interleaved, normalised
          prescale : [(B*T), 1]
        """
        try:
            import einops
        except ImportError:
            raise ImportError("einops is required for CommVQ. Install with: pip install einops")

        B, H, T, D = K.shape
        # Transpose to [B, T, H, D] then flatten H and D → [B, T, H*D]
        x = K.transpose(1, 2).reshape(B, T, H * D)
        # CommVQ pair-interleaving: rearrange dim pairs for RoPE commutativity
        # [B, T, H, D] → [B, T, H, 2, D//2] → transpose → [B, T, H, D//2, 2] → flatten
        x = K.transpose(1, 2).reshape(B, T, H, 2, D // 2).transpose(3, 4).flatten(2)
        # → [B, T, H * D] (same size, different layout — pairs are adjacent)

        # L2 normalise per token
        prescale = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)  # [B, T, 1]
        x = x / prescale

        # Flatten batch and time, split into groups
        x_bt = x.reshape(B * T, self.G, -1)   # [(B*T), G, D_g]
        ps_bt = prescale.reshape(B * T, 1)     # [(B*T), 1]
        return x_bt, ps_bt

    def encode(self, K: torch.Tensor):
        """
        K: [B, H_kv, T, D]
        Returns
          codes    : [R, G, B*T]  uint16
          prescale : [B*T, 1]     fp16
        """
        B, H, T, D = K.shape
        tensor, prescale = self._preprocess(K)   # [(B*T), G, D_g]
        BT = tensor.shape[0]
        dev = K.device

        cb = self.codebook.to(dev).float()       # [R, G, K, D_g]
        codes = torch.zeros(self.R, self.G, BT, dtype=torch.int32, device=dev)

        residual = tensor.float()
        for r in range(self.R):
            cb_r = cb[r]   # [G, K, D_g]
            # Distance: [G, B*T, K]
            dists = torch.cdist(
                residual.permute(1, 0, 2),       # [G, B*T, D_g]
                cb_r                             # [G, K, D_g]
            )
            assign = dists.argmin(dim=-1)        # [G, B*T]
            codes[r] = assign

            # Subtract chosen codewords
            chosen = torch.gather(
                cb_r,
                dim=1,
                index=assign.unsqueeze(-1).expand(-1, -1, cb_r.shape[-1])
            )   # [G, B*T, D_g]
            residual -= chosen.permute(1, 0, 2)  # [(B*T), G, D_g]

        return codes.to(torch.int32), prescale.half()

    def decode(self, codes: torch.Tensor, prescale: torch.Tensor,
               B: int, T: int, device) -> torch.Tensor:
        """
        codes    : [R, G, B*T]  int32
        prescale : [B*T, 1]     fp16
        Returns  : [B, H_kv, T, D]
        """
        H, D = self.num_key_value_heads, self.head_dim
        cb = self.codebook.to(device).float()   # [R, G, K, D_g]

        # Look up and sum across residual stages
        # codes: [R, G, BT] → gather → [R, G, BT, D_g]
        codes_l = codes.long()
        out = torch.zeros(self.G, B * T, cb.shape[-1], device=device, dtype=torch.float32)
        for r in range(self.R):
            idx = codes_l[r]  # [G, BT]
            chosen = torch.gather(
                cb[r],   # [G, K, D_g]
                dim=1,
                index=idx.unsqueeze(-1).expand(-1, -1, cb.shape[-1])
            )   # [G, BT, D_g]
            out += chosen

        # out: [G, BT, D_g] → [(BT), G*D_g] → rescale
        out = out.permute(1, 0, 2).flatten(1)   # [(BT), G*D_g] = [(BT), H*D]
        out = out * prescale.float().to(device)  # [(BT), H*D]

        # Undo pair-interleaving: [BT, H, D//2, 2] → [BT, H, 2, D//2] → [BT, H, D]
        out = out.reshape(B * T, H, D // 2, 2).transpose(2, 3).reshape(B * T, H, D)
        # → [B, T, H, D] → [B, H, T, D]
        out = out.reshape(B, T, H, D).transpose(1, 2).contiguous()
        return out.half()


# ---------------------------------------------------------------------------
# INT8 value quantisation (per-token, asymmetric) — same as APKVC prefill
# ---------------------------------------------------------------------------

def _quant_V_int8(V: torch.Tensor):
    """V: [B, H, T, D] → uint8, scale [B,H,T,1], zero [B,H,T,1]"""
    mn = V.float().amin(dim=-1, keepdim=True)
    mx = V.float().amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-5) / 255.0
    V_u8 = ((V.float() - mn) / scale).round().clamp(0, 255).to(torch.uint8)
    return V_u8, scale.half(), mn.half()


def _dequant_V_int8(V_u8, scale, mn):
    return (V_u8.float() * scale.float() + mn.float()).half()


# ---------------------------------------------------------------------------
# CommVQMethod
# ---------------------------------------------------------------------------

class CommVQMethod(KVCacheMethod):
    """
    CommVQ key compression with INT8 value compression for the LDCB benchmark.

    Parameters
    ----------
    bits : int
        Key quantisation bits — 1 or 2 (selects codebook repo accordingly).
    residual_length : int
        Number of recent tokens kept uncompressed in fp16 (same concept as KIVI).
    cpu_offload : bool
        Offload compressed key codes to CPU to save GPU VRAM.
    hf_cache_dir : str | None
        Optional cache directory for huggingface_hub downloads.
    """

    def __init__(self, bits: int = 2, residual_length: int = 128,
                 cpu_offload: bool = False, hf_cache_dir: str | None = None):
        assert bits in (1, 2), "CommVQ only supports 1-bit and 2-bit key compression"
        self.bits = bits
        self.R = residual_length
        self.cpu = cpu_offload
        self.hf_cache_dir = hf_cache_dir
        self.name = f"CommVQ-{bits}bit"
        self._encoders: list[_CommVQKeyEncoder] | None = None

    def _ensure_encoders(self, model):
        if self._encoders is not None:
            return
        from huggingface_hub import snapshot_download
        repo_id = f"senfu/Llama-3.1-8B-Instruct-CommVQ-{self.bits}bit-codebook"
        print(f"[CommVQ] Downloading codebooks: {repo_id} ...")
        local_dir = snapshot_download(repo_id, cache_dir=self.hf_cache_dir)
        print(f"[CommVQ] Codebooks at: {local_dir}")

        n_layers = model.config.num_hidden_layers
        self._encoders = [
            _CommVQKeyEncoder(local_dir, layer_idx=i, bits=self.bits, config=model.config)
            for i in range(n_layers)
        ]
        print(f"[CommVQ] Loaded codebooks for {n_layers} layers.")

    def _cache_bytes(self, n_tokens, n_layers, n_kv_heads, head_dim) -> int:
        R_bits = self._encoders[0].R if self._encoders else 8  # residual stages
        T_q = max(0, n_tokens - self.R)
        T_r = min(n_tokens, self.R)

        # Keys compressed: codes uint16 (2 bytes) per residual stage per group
        #   shape: [R_stages, G=H_kv, T_q] where each entry is uint16
        key_codes = R_bits * n_kv_heads * T_q * 2  # 2 bytes per uint16 code
        key_prescale = T_q * 2                       # fp16 prescale per token
        # Value compressed: uint8 + scale (fp16) + zero (fp16)
        val_codes = n_kv_heads * T_q * head_dim * 1
        val_meta  = n_kv_heads * T_q * 2 * 2

        # Residual (fp16 K + V)
        res = 2 * n_kv_heads * T_r * head_dim * 2

        per_layer = key_codes + key_prescale + val_codes + val_meta + res
        return int(per_layer * n_layers)

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps):
        self._ensure_encoders(model)

        inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers  = model.config.num_hidden_layers
        n_kv_heads = model.config.num_key_value_heads
        head_dim  = model.config.hidden_size // model.config.num_attention_heads
        mdtype    = next(model.parameters()).dtype
        R         = self.R

        generated_ids = inputs.input_ids
        snapshots = []
        chk_iter  = iter(checkpoint_steps)
        cur_chk   = next(chk_iter, None)
        tok_gen   = 0

        # ---- Prefill ----
        with torch.no_grad():
            outputs = model(generated_ids, use_cache=True)

        # Build per-layer compressed cache.
        # State per layer:
        #   K_codes  : [R_stages, G, T_q]  int32   (quantised block, cpu if offload)
        #   K_prescale : [T_q, 1]          fp16
        #   K_res    : [B, H_kv, T_r, D]   fp16    (recent uncompressed)
        #   V_u8     : [B, H_kv, T_q, D]   uint8
        #   V_scale  : [B, H_kv, T_q, 1]   fp16
        #   V_mn     : [B, H_kv, T_q, 1]   fp16
        #   V_res    : [B, H_kv, T_r, D]   fp16
        #   T_q      : int
        compr_cache = []
        for i, (_, (layer_K, layer_V)) in enumerate(get_kv_iterator(outputs.past_key_values)):
            T = layer_K.shape[2]
            T_q = (T // R) * R   # how many tokens to compress (multiple of R)

            K_old = layer_K[:, :, :T_q, :].to(dtype=torch.float32)
            K_res = layer_K[:, :, T_q:, :].to(dtype=mdtype)
            V_old = layer_V[:, :, :T_q, :].to(dtype=torch.float32)
            V_res = layer_V[:, :, T_q:, :].to(dtype=mdtype)

            if T_q > 0:
                K_codes, K_ps  = self._encoders[i].encode(K_old.to(mdtype))
                V_u8, V_sc, V_mn = _quant_V_int8(V_old)
                if self.cpu:
                    K_codes = K_codes.cpu(); K_ps = K_ps.cpu()
                    V_u8 = V_u8.cpu(); V_sc = V_sc.cpu(); V_mn = V_mn.cpu()
            else:
                K_codes = K_ps = None
                V_u8 = V_sc = V_mn = None

            compr_cache.append([K_codes, K_ps, T_q, K_res,
                                 V_u8, V_sc, V_mn, V_res])

        next_token    = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        tok_gen += 1
        del outputs

        # ---- Decode loop ----
        with torch.no_grad():
            while tok_gen < max_new_tokens:
                dev = next_token.device
                past_kv = DynamicCache()
                for i, (K_codes, K_ps, T_q, K_res, V_u8, V_sc, V_mn, V_res) in enumerate(compr_cache):
                    B = K_res.shape[0]

                    if K_codes is not None:
                        kc = K_codes.to(dev) if self.cpu else K_codes
                        kp = K_ps.to(dev)    if self.cpu else K_ps
                        K_decomp = self._encoders[i].decode(kc, kp, B, T_q, dev)
                        K_fp = torch.cat([K_decomp.to(mdtype), K_res.to(mdtype)], dim=2)
                        if self.cpu: del kc, kp
                    else:
                        K_fp = K_res.to(mdtype)

                    if V_u8 is not None:
                        vc = V_u8.to(dev) if self.cpu else V_u8
                        vs = V_sc.to(dev) if self.cpu else V_sc
                        vm = V_mn.to(dev) if self.cpu else V_mn
                        V_decomp = _dequant_V_int8(vc, vs, vm)
                        V_fp = torch.cat([V_decomp.to(mdtype), V_res.to(mdtype)], dim=2)
                        if self.cpu: del vc, vs, vm
                    else:
                        V_fp = V_res.to(mdtype)

                    past_kv.update(K_fp, V_fp, i)

                outputs = model(next_token, past_key_values=past_kv, use_cache=True)
                new_kvs = [
                    (K[:, :, -1:, :].to(mdtype).clone(),
                     V[:, :, -1:, :].to(mdtype).clone())
                    for _, (K, V) in get_kv_iterator(outputs.past_key_values)
                ]
                next_token    = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del outputs, past_kv
                torch.cuda.empty_cache()

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tok_gen += 1

                # Update cache: same flush policy as KIVI
                new_cache = []
                for i, (K_codes, K_ps, T_q, K_res, V_u8, V_sc, V_mn, V_res) in enumerate(compr_cache):
                    nK, nV = new_kvs[i]
                    B = nK.shape[0]

                    # Keys: accumulate in residual, flush whole block when full
                    K_res = torch.cat([K_res, nK], dim=2)
                    if K_res.shape[2] == R:
                        nc, np_ = self._encoders[i].encode(K_res.to(torch.float32))
                        if self.cpu: nc = nc.cpu(); np_ = np_.cpu()
                        new_T_q = T_q + R
                        if K_codes is None:
                            K_codes, K_ps = nc, np_
                        else:
                            K_codes = torch.cat([K_codes, nc], dim=2)
                            # prescale: [T_q, 1] → cat along dim 0
                            K_ps = torch.cat([
                                K_ps.to(nc.device) if self.cpu else K_ps,
                                np_
                            ], dim=0)
                        T_q = new_T_q
                        K_res = K_res[:, :, :0, :]

                    # Values: flush oldest when > R
                    V_res = torch.cat([V_res, nV], dim=2)
                    if V_res.shape[2] > R:
                        evict = V_res[:, :, :1, :].float()
                        V_res = V_res[:, :, 1:, :]
                        eu8, esc, emn = _quant_V_int8(evict)
                        if self.cpu: eu8 = eu8.cpu(); esc = esc.cpu(); emn = emn.cpu()
                        if V_u8 is None:
                            V_u8, V_sc, V_mn = eu8, esc, emn
                        else:
                            V_u8 = torch.cat([V_u8, eu8], dim=2)
                            V_sc = torch.cat([V_sc, esc], dim=2)
                            V_mn = torch.cat([V_mn, emn], dim=2)

                    new_cache.append([K_codes, K_ps, T_q, K_res, V_u8, V_sc, V_mn, V_res])

                compr_cache = new_cache

                if tok_gen == cur_chk:
                    T_total = generated_ids.shape[1]
                    compr   = self._cache_bytes(T_total, n_layers, n_kv_heads, head_dim)
                    full_kv = T_total * n_layers * n_kv_heads * head_dim * 2 * 2
                    snapshots.append(CacheState(compressed_bytes=compr, fullkv_bytes=full_kv))
                    cur_chk = next(chk_iter, None)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        T_total = generated_ids.shape[1]
        compr   = self._cache_bytes(T_total, n_layers, n_kv_heads, head_dim)
        full_kv = T_total * n_layers * n_kv_heads * head_dim * 2 * 2
        text    = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        final   = CacheState(compressed_bytes=compr, fullkv_bytes=full_kv)

        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final)

        return text, snapshots, final
