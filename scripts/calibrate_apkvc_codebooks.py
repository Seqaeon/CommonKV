import argparse
import os
from typing import List

import torch


def _flatten_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    return x.reshape(-1, x.shape[-1])


def _load_trace_tensors(paths: List[str], key: str, max_samples: int) -> torch.Tensor:
    chunks = []
    for p in paths:
        payload = torch.load(p, map_location="cpu")
        if key not in payload:
            raise KeyError(f"Missing key '{key}' in trace file: {p}")
        t = _flatten_last_dim(payload[key]).float()
        chunks.append(t)
    X = torch.cat(chunks, dim=0)
    if X.shape[0] > max_samples:
        idx = torch.randperm(X.shape[0])[:max_samples]
        X = X[idx]
    return X


def _kmeans(X: torch.Tensor, k: int, iters: int = 25) -> torch.Tensor:
    # X: [N, D]
    N = X.shape[0]
    if N < k:
        reps = (k + N - 1) // N
        X = X.repeat(reps, 1)
        N = X.shape[0]
    perm = torch.randperm(N)[:k]
    centers = X[perm].clone()

    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)  # [N, k]
        assign = d2.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(assign, minlength=k)
        for i in range(k):
            if counts[i] > 0:
                new_centers[i] = X[assign == i].mean(dim=0)
            else:
                new_centers[i] = X[torch.randint(0, N, (1,))]
        centers = new_centers
    return centers


def _residual_kmeans(X: torch.Tensor, num_codebooks: int, codebook_size: int, iters: int = 25) -> torch.Tensor:
    residual = X.clone()
    codebooks = []
    for _ in range(num_codebooks):
        C = _kmeans(residual, codebook_size, iters=iters)
        codebooks.append(C)
        d2 = ((residual[:, None, :] - C[None, :, :]) ** 2).sum(dim=-1)
        idx = d2.argmin(dim=1)
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
    p.add_argument("--max_samples", type=int, default=400000)
    return p.parse_args()


def main():
    args = parse_args()
    for p in args.trace_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    K = _load_trace_tensors(args.trace_paths, "delta_K_base", args.max_samples)
    V = _load_trace_tensors(args.trace_paths, "delta_V", args.max_samples)

    K_codebooks = _residual_kmeans(
        K,
        num_codebooks=args.K_num_codebooks,
        codebook_size=args.codebook_size,
        iters=args.iters,
    )
    V_codebooks = _residual_kmeans(
        V,
        num_codebooks=args.V_num_codebooks,
        codebook_size=args.codebook_size,
        iters=args.iters,
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
        },
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"Saved calibrated codebooks -> {args.output_path}")


if __name__ == "__main__":
    main()
