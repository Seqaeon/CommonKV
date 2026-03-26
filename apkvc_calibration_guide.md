# APKVC Calibration Guide (Trace-Based)

`attention_aware_predictive_kv.py` supports loading pre-calibrated AQ codebooks via:
- `--apkvc_calibration_path <path>` in `run_longbench.py` / `run_ruler.py`

This is **not full model finetuning**. It is an offline calibration pass over representative traces.

## 1) Which data should be used for calibration?

Use data that matches your inference distribution:

1. **Long context QA + retrieval (recommended baseline):**
   - A mixed sample from LongBench datasets you care about (e.g., `hotpotqa`, `musique`, `qasper`, `gov_report`).
2. **Needle/retrieval stress:**
   - RULER contexts at target lengths (e.g., 8k/16k/32k).
3. **Production style prompts (best for deployment):**
   - An anonymized sample of real user prompts with similar context length and task mix.

A good practical split is:
- 50% LongBench-like
- 30% RULER-like
- 20% production traces

If production traces are unavailable, use LongBench + RULER only.

## 2) Expected trace format

The calibrator script expects one or more `.pt` files containing:
- `delta_K_base`: tensor with shape `[N, D]` or `[..., D]`
- `delta_V`: tensor with shape `[N, D]` or `[..., D]`

Where:
- `delta_K_base` is key residual in the de-rotated (RoPE base) space
- `delta_V` is value residual

## 3) Build codebooks from traces

```bash
python scripts/calibrate_apkvc_codebooks.py \
  --trace_paths traces/lb_mix.pt traces/ruler_16k.pt \
  --output_path artifacts/apkvc_codebooks.pt \
  --K_num_codebooks 4 \
  --V_num_codebooks 2 \
  --codebook_size 256
```

## 4) Use calibrated codebooks at inference

### LongBench
```bash
python run_longbench.py \
  --method apkvc \
  --model_path <model_path> \
  --apkvc_calibration_path artifacts/apkvc_codebooks.pt
```

### RULER
```bash
python run_ruler.py \
  --method apkvc \
  --model_path <model_path> \
  --apkvc_calibration_path artifacts/apkvc_codebooks.pt
```

## Notes

- If calibration file is missing or invalid, APKVC falls back to random codebooks and prints a warning.
- Keep codebook dimensions aligned with model `head_dim` and with `K_num_codebooks`/`V_num_codebooks` args.
