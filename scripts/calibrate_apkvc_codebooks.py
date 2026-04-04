import argparse
import os
from typing import List

import torch


def _flatten_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    return x.reshape(-1, x.shape[-1])


def _load_trace_tensors(paths: List[str], key: str, max_samples: int) -> dict:
    """Loads traces and returns a dict mapping layer_idx -> [N, D] tensor."""
    dict_out = {}
    for p in paths:
        payload = torch.load(p, map_location="cpu")
        if key not in payload:
            raise KeyError(f"Missing key '{key}' in trace file: {p}")
            
        val = payload[key]
        if isinstance(val, dict):
            # New format: {layer_idx: tensor}
            for l_idx, t in val.items():
                if l_idx not in dict_out: dict_out[l_idx] = []
                dict_out[l_idx].append(_flatten_last_dim(t).float())
        else:
            # Legacy monolithic format
            if 0 not in dict_out: dict_out[0] = []
            dict_out[0].append(_flatten_last_dim(val).float())
            
    for l_idx in dict_out:
        X = torch.cat(dict_out[l_idx], dim=0)
        if X.shape[0] > max_samples:
            idx = torch.randperm(X.shape[0])[:max_samples]
            X = X[idx]
        dict_out[l_idx] = X
    return dict_out


def _argmin_dist_chunked(X: torch.Tensor, centers: torch.Tensor, chunk_size: int = 8192):
    """
    Memory-safe nearest-center assignment.
    Avoids materializing [N, K, D] tensors.
    """
    N = X.shape[0]
    assignments = torch.empty(N, dtype=torch.long, device=X.device)
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        x = X[s:e]  # [B, D]
        # squared L2 distance via ||x-c||^2 = ||x||^2 + ||c||^2 - 2x·c
        x2 = (x * x).sum(dim=1, keepdim=True)                # [B, 1]
        c2 = (centers * centers).sum(dim=1).unsqueeze(0)     # [1, K]
        xc = x @ centers.t()                                  # [B, K]
        d2 = x2 + c2 - 2.0 * xc                               # [B, K]
        assignments[s:e] = d2.argmin(dim=1)
    return assignments


def _kmeans(X: torch.Tensor, k: int, iters: int = 25, chunk_size: int = 8192) -> torch.Tensor:
    # X: [N, D]
    N = X.shape[0]
    if N == 0:
        raise ValueError("No samples available for k-means calibration.")
    if N < k:
        reps = (k + N - 1) // N
        X = X.repeat(reps, 1)
        N = X.shape[0]
    perm = torch.randperm(N)[:k]
    centers = X[perm].clone()

    for _ in range(iters):
        assign = _argmin_dist_chunked(X, centers, chunk_size=chunk_size)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(assign, minlength=k)
        new_centers.index_add_(0, assign, X)
        nonzero = counts > 0
        new_centers[nonzero] = new_centers[nonzero] / counts[nonzero].unsqueeze(1)
        if (~nonzero).any():
            # re-seed empty clusters from random points
            num_empty = int((~nonzero).sum().item())
            ridx = torch.randint(0, N, (num_empty,), device=X.device)
            new_centers[~nonzero] = X[ridx]
        centers = new_centers
    return centers


def _project_rope_commutative_2x2(centers: torch.Tensor) -> torch.Tensor:
    if centers.shape[-1] % 2 != 0:
        raise ValueError("rope_commutative_2x2 requires even feature dimension")
    return centers.reshape(centers.shape[0], -1, 2).reshape_as(centers)


def _em_closed_form(
    X: torch.Tensor,
    k: int,
    iters: int = 25,
    chunk_size: int = 8192,
    temperature: float = 1.0,
    commutative_2x2: bool = False,
) -> torch.Tensor:
    """
    Soft-EM where M-step is the closed-form weighted mean.
    """
    N = X.shape[0]
    if N < k:
        reps = (k + N - 1) // N
        X = X.repeat(reps, 1)
        N = X.shape[0]
    centers = X[torch.randperm(N)[:k]].clone()
    for _ in range(iters):
        chunks = []
        weights = []
        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)
            x = X[s:e]
            d2 = torch.cdist(x, centers) ** 2
            w = torch.softmax(-d2 / max(1e-6, temperature), dim=-1)
            chunks.append(x)
            weights.append(w)
        w_all = torch.cat(weights, dim=0)
        x_all = torch.cat(chunks, dim=0)
        denom = w_all.sum(dim=0, keepdim=True).t().clamp(min=1e-8)
        new_centers = (w_all.t() @ x_all) / denom
        if commutative_2x2:
            new_centers = _project_rope_commutative_2x2(new_centers)
        centers = new_centers
    return centers


def _residual_kmeans(
    X: torch.Tensor,
    num_codebooks: int,
    codebook_size: int,
    iters: int = 25,
    chunk_size: int = 8192,
    trainer: str = "kmeans",
    commutative_2x2: bool = False,
) -> torch.Tensor:
    residual = X.clone()
    codebooks = []
    for _ in range(num_codebooks):
        if trainer == "em_closed_form":
            C = _em_closed_form(
                residual,
                codebook_size,
                iters=iters,
                chunk_size=chunk_size,
                commutative_2x2=commutative_2x2,
            )
        else:
            C = _kmeans(residual, codebook_size, iters=iters, chunk_size=chunk_size)
            if commutative_2x2:
                C = _project_rope_commutative_2x2(C)
        codebooks.append(C)
        idx = _argmin_dist_chunked(residual, C, chunk_size=chunk_size)
        residual = residual - C[idx]
    return torch.stack(codebooks, dim=0)


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate APKVC AQ codebooks from offline residual traces")
    p.add_argument("--trace_paths", type=str, nargs="+", required=True, help=".pt files containing delta_K_base and delta_V tensors")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--K_num_codebooks", type=int, default=4)
    p.add_argument("--V_num_codebooks", type=int, default=2)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--max_samples", type=int, default=120000)
    p.add_argument("--chunk_size", type=int, default=8192, help="Chunk size for memory-safe distance assignment")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for k-means calibration")
    p.add_argument("--per_layer_codebooks", action="store_true", help="Train completely independent AQ codebooks per layer")
    p.add_argument("--trainer", type=str, default="kmeans", choices=["kmeans", "em_closed_form"], help="Codebook trainer")
    p.add_argument("--codebook_structure", type=str, default="unconstrained", choices=["unconstrained", "rope_commutative_2x2"])
    return p.parse_args()


def main():
    args = parse_args()
    use_commutative = args.codebook_structure == "rope_commutative_2x2"
    for p in args.trace_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    K_dicts = _load_trace_tensors(args.trace_paths, "delta_K_base", args.max_samples)
    V_dicts = _load_trace_tensors(args.trace_paths, "delta_V", args.max_samples)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    if args.per_layer_codebooks:
        print("[APKVC] Training Layer-Specific Codebooks...")
        num_layers = max([0] + list(K_dicts.keys()) + list(V_dicts.keys())) + 1
        
        K_cbs_list, V_cbs_list = [], []
        for l_idx in range(num_layers):
            print(f" -> Layer {l_idx}/{num_layers-1} ...")
            
            # fallback to layer 0 if missing
            kl = K_dicts.get(l_idx, K_dicts.get(0))
            if kl is not None: kl = kl.to(device)
            else: kl = torch.zeros((10, args.codebook_size), device=device) # dummy
                
            vl = V_dicts.get(l_idx, V_dicts.get(0))
            if vl is not None: vl = vl.to(device)
            else: vl = torch.zeros((10, args.codebook_size), device=device)
            
            Ck = _residual_kmeans(
                kl, args.K_num_codebooks, args.codebook_size, args.iters, args.chunk_size,
                trainer=args.trainer, commutative_2x2=use_commutative,
            )
            K_cbs_list.append(Ck)
            Cv = _residual_kmeans(
                vl, args.V_num_codebooks, args.codebook_size, args.iters, args.chunk_size,
                trainer=args.trainer, commutative_2x2=use_commutative,
            )
            V_cbs_list.append(Cv)
            
        K_codebooks = torch.stack(K_cbs_list, dim=0) # [L, M, S, D]
        V_codebooks = torch.stack(V_cbs_list, dim=0)
    else:
        print("[APKVC] Training Unified Monolithic Codebook...")
        K_all = torch.cat(list(K_dicts.values()), dim=0)
        V_all = torch.cat(list(V_dicts.values()), dim=0)
        if K_all.shape[0] > args.max_samples: K_all = K_all[torch.randperm(K_all.shape[0])[:args.max_samples]]
        if V_all.shape[0] > args.max_samples: V_all = V_all[torch.randperm(V_all.shape[0])[:args.max_samples]]
        
        K_all, V_all = K_all.to(device), V_all.to(device)
        K_codebooks = _residual_kmeans(
            K_all, args.K_num_codebooks, args.codebook_size, args.iters, args.chunk_size,
            trainer=args.trainer, commutative_2x2=use_commutative,
        )
        V_codebooks = _residual_kmeans(
            V_all, args.V_num_codebooks, args.codebook_size, args.iters, args.chunk_size,
            trainer=args.trainer, commutative_2x2=use_commutative,
        )

    payload = {
        "K_codebooks": K_codebooks.half(),
        "V_codebooks": V_codebooks.half(),
        "metadata": {
            "K_num_codebooks": args.K_num_codebooks,
            "V_num_codebooks": args.V_num_codebooks,
            "codebook_size": args.codebook_size,
            "max_samples": args.max_samples,
            "trace_paths": args.trace_paths,
            "trainer": args.trainer,
            "codebook_structure": args.codebook_structure,
        },
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"Saved calibrated codebooks -> {args.output_path}")


if __name__ == "__main__":
    main()
