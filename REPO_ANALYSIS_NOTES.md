# Repo Analysis Notes

This note summarizes how CommonKV benchmarking is wired and flags key implementation caveats discovered during code review.

- LongBench and RULER inference entrypoints: `run_longbench.py`, `run_ruler.py`.
- Attention method dispatch is monkey-patched in `commonkv/monkeypatch.py`.
- Method implementations include:
  - CommonKV SVD-merged model classes (`commonkv/modeling_llama_svd_merge.py`, `commonkv/modeling_mistral_svd_merge.py`)
  - Token/depth/dim compressors (`pyramidkv/pyramidkv_utils.py`)
  - APKVC prototype (`attention_aware_predictive_kv.py`)
- Scoring:
  - LongBench: `eval.py` (expects external `metrics` module)
  - RULER: `eval_ruler.py` (also expects external `metrics` module)

Primary caveat: compression ratio fields saved in result JSON are heuristic estimates per method and are not normalized to a single definition.
