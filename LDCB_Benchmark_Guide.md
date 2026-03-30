# Long-Decode Compression Benchmark (LDCB) — Implementation Guide

## 0. Purpose

LDCB is designed to make the decode compression advantage of APKVC visible
against flat quantization baselines (KIVI-int2, KIVI-int4). The core insight
it tests is:

> APKVC's compression ratio improves as generation gets longer because
> more tokens participate in residual chains. KIVI's ratio is flat
> because it has no mechanism to exploit inter-token structure.

All three tasks use short prompts (< 128 tokens) so prefill is negligible
and the benchmark isolates decode-time compression behaviour.

---

## 1. Repository Layout

```
ldcb/
├── run_benchmark.py          # main entry point
├── tasks/
│   ├── __init__.py
│   ├── continuation.py       # Task 1
│   ├── reasoning.py          # Task 2
│   └── multiturn.py          # Task 3
├── methods/
│   ├── __init__.py
│   ├── fullkv.py             # baseline
│   ├── kivi.py               # KIVI-int2 and KIVI-int4
│   └── apkvc.py              # APKVC wrapper
├── metrics.py                # all measurement utilities
├── plots.py                  # plot 1, 2, 3 generators
├── config.py                 # dataclasses for all settings
└── results/                  # auto-created, one JSON per run
```

---

## 2. Dependencies

```bash
pip install torch transformers datasets rouge-score numpy matplotlib
```

Model used throughout: `meta-llama/Llama-2-7b-hf` or `mistralai/Mistral-7B-v0.1`.
Both are equivalent for benchmarking purposes. Use whichever you have access to.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "meta-llama/Llama-2-7b-hf"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer
```

---

## 3. Method Interface

Every method must implement this interface. All three tasks call it identically.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple
import torch

@dataclass
class CacheState:
    """Returned by each method after generation. Used to compute metrics."""
    compressed_bytes: int         # total bytes used by cache
    fullkv_bytes: int             # bytes if FullKV had been used
    anchor_count: int = 0         # APKVC only; 0 for others
    residual_count: int = 0       # APKVC only; 0 for others
    distortions: list = field(default_factory=list)  # per-step distortion values


class KVCacheMethod(ABC):

    @abstractmethod
    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        checkpoint_steps: list,      # token counts at which to snapshot metrics
    ) -> Tuple[str, list, CacheState]:
        """
        Returns:
          generated_text: str
          checkpoint_snapshots: list of CacheState, one per checkpoint_step
          final_state: CacheState at end of generation
        """
        pass
```

---

## 4. Task 1 — Long Continuation

Tests smooth temporal correlation. KV states evolve slowly within a coherent
long-form passage. This is where residuals are smallest and APKVC's advantage
over KIVI should be largest.

### Prompts

Use at least 5 prompts to average over variance. Suggested set:

```python
CONTINUATION_PROMPTS = [
    "Write a detailed technical report on the history of transformer "
    "architectures, covering all major developments from 2017 to present.",

    "Write a comprehensive essay on the political economy of colonial West "
    "Africa, examining land tenure, taxation, and labour extraction in depth.",

    "Write a detailed explanation of how operating system kernels manage "
    "memory, covering virtual address spaces, paging, and page replacement.",

    "Write a thorough account of how deep learning changed the field of "
    "natural language processing between 2013 and 2023.",

    "Write a detailed analysis of how monetary policy transmission works "
    "in an economy with a large informal sector.",
]
```

These prompts are chosen to produce topically consistent, slowly drifting text
rather than lists, tables, or abrupt transitions. Avoid prompts that tend to
produce bullet-point responses — those introduce more abrupt KV state shifts.

### Runner

```python
# tasks/continuation.py

CHECKPOINT_STEPS = [250, 500, 1000, 2000, 4000]
MAX_NEW_TOKENS = 4000

def run_continuation(method: KVCacheMethod, model, tokenizer) -> dict:
    all_results = []

    for prompt in CONTINUATION_PROMPTS:
        # Tokenize and verify prompt is short
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        assert input_ids.shape[1] <= 128, \
            f"Prompt too long: {input_ids.shape[1]} tokens. Trim it."

        torch.cuda.reset_peak_memory_stats()
        generated_text, snapshots, final_state = method.generate(
            model, tokenizer, prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            checkpoint_steps=CHECKPOINT_STEPS,
        )

        result = {
            "prompt": prompt[:60] + "...",
            "snapshots": [
                {
                    "tokens_generated": step,
                    "compression_ratio": snap.compressed_bytes / snap.fullkv_bytes,
                    "anchor_rate": snap.anchor_count / max(snap.anchor_count + snap.residual_count, 1),
                }
                for step, snap in zip(CHECKPOINT_STEPS, snapshots)
            ],
            "final_compression_ratio": final_state.compressed_bytes / final_state.fullkv_bytes,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "perplexity": compute_perplexity(model, tokenizer, generated_text),
            "distortion_mean": float(torch.tensor(final_state.distortions).mean())
                               if final_state.distortions else 0.0,
            "distortion_p95": float(torch.tensor(final_state.distortions).quantile(0.95))
                              if final_state.distortions else 0.0,
        }
        all_results.append(result)

    return aggregate_task_results(all_results)
```

### What to expect

On a topically consistent essay, APKVC with identity predictor should produce
residuals that are noticeably smaller in magnitude than raw KV vectors for the
majority of tokens. Anchor rate should be low (< 15% of tokens). If anchor rate
is high (> 40%), the distortion threshold is too tight or codebooks need retraining.

---

## 5. Task 2 — Structured Reasoning

Tests anchor/reset behaviour. Multi-problem reasoning produces abrupt KV state
shifts at problem boundaries — residuals spike, anchors are triggered, chains reset.
The argument to make here is that APKVC handles these gracefully while still
compressing the smooth intra-problem reasoning spans.

### Prompt construction

```python
# tasks/reasoning.py

import random

def make_algebra_problem(seed: int) -> str:
    random.seed(seed)
    ops = [
        lambda: f"Solve for x: {random.randint(2,9)}x + {random.randint(1,20)} = {random.randint(20,80)}",
        lambda: f"Simplify: ({random.randint(2,5)}x + {random.randint(1,8)})^2",
        lambda: f"Factor: x^2 + {random.randint(2,10)}x + {random.randint(1,20)}",
        lambda: f"Find the derivative of f(x) = {random.randint(2,6)}x^{random.randint(2,4)} + {random.randint(1,5)}x",
    ]
    return random.choice(ops)()

def make_reasoning_prompt(n_problems: int = 15, seed: int = 42) -> str:
    problems = [make_algebra_problem(seed + i) for i in range(n_problems)]
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(problems))
    return (
        "Solve the following problems step by step, showing all working clearly.\n\n"
        + numbered
    )

REASONING_PROMPTS = [make_reasoning_prompt(seed=s) for s in [42, 99, 137, 200, 314]]
MAX_NEW_TOKENS = 2000
CHECKPOINT_STEPS = [250, 500, 1000, 1500, 2000]
```

Verify each prompt tokenizes to under 128 tokens. Reduce `n_problems` if needed.

### Additional metric: per-segment anchor rate

For reasoning tasks, also measure anchor rate within segments vs at boundaries.
This requires tagging token positions by which problem they belong to (approximate
by splitting generated text at numbered headers).

```python
def estimate_boundary_vs_interior_anchor_rate(generated_text, anchor_positions):
    """
    Approximate boundary tokens as those within 5 tokens of a newline
    followed by a digit + period (e.g. "3."). Measure anchor rate separately
    for boundary vs interior tokens.
    """
    lines = generated_text.split("\n")
    boundary_tokens = set()
    cursor = 0
    for line in lines:
        if line.strip() and line.strip()[0].isdigit() and "." in line[:3]:
            for offset in range(-5, 15):
                boundary_tokens.add(cursor + offset)
        cursor += len(line) + 1  # +1 for newline

    boundary_anchors = sum(1 for p in anchor_positions if p in boundary_tokens)
    interior_anchors = len(anchor_positions) - boundary_anchors
    n_boundary = len(boundary_tokens)
    n_interior = len(anchor_positions) + len([p for p in range(cursor) if p not in anchor_positions]) - n_boundary

    return {
        "boundary_anchor_rate": boundary_anchors / max(n_boundary, 1),
        "interior_anchor_rate": interior_anchors / max(n_interior, 1),
    }
```

The expected result: boundary anchor rate >> interior anchor rate. This shows the
reset mechanism is responding to genuine KV state discontinuities, not firing randomly.

---

## 6. Task 3 — Multi-turn Simulation

Tests cache growth across conversation turns. Measures VRAM at each turn boundary.
The headline result is: how many turns before OOM on a fixed GPU?

### Turn simulator

```python
# tasks/multiturn.py

USER_MESSAGES = [
    "Tell me about neural networks.",
    "How does backpropagation work?",
    "What is a transformer?",
    "Explain attention mechanisms.",
    "What are residual connections?",
    "How does layer normalization work?",
    "What is the difference between BERT and GPT?",
    "How is fine-tuning done?",
    "What are LoRA and QLoRA?",
    "Explain knowledge distillation.",
    "What is quantization in deep learning?",
    "How does speculative decoding work?",
    "What is a KV cache?",
    "Why is memory a bottleneck in LLM inference?",
    "How do KV cache compression methods work?",
]

RESPONSE_TOKENS_PER_TURN = 100
N_TURNS = 15

def run_multiturn(method: KVCacheMethod, model, tokenizer) -> dict:
    """
    Simulates a conversation by appending each user message + response to
    a growing context. Cache is maintained across turns.
    """
    turn_results = []
    conversation_context = ""

    for turn_idx, user_msg in enumerate(USER_MESSAGES[:N_TURNS]):
        torch.cuda.reset_peak_memory_stats()

        prompt = conversation_context + f"User: {user_msg}\nAssistant:"

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        generated_text, _, state = method.generate(
            model, tokenizer, prompt,
            max_new_tokens=RESPONSE_TOKENS_PER_TURN,
            checkpoint_steps=[RESPONSE_TOKENS_PER_TURN],
        )

        vram_gb = torch.cuda.max_memory_allocated() / 1e9

        turn_results.append({
            "turn": turn_idx + 1,
            "vram_gb": vram_gb,
            "compression_ratio": state.compressed_bytes / state.fullkv_bytes,
            "context_tokens": input_ids.shape[1],
            "generated_tokens": RESPONSE_TOKENS_PER_TURN,
        })

        # Append to conversation
        response = generated_text[len(prompt):]
        conversation_context = prompt + response + "\n"

        print(f"Turn {turn_idx+1:2d} | VRAM: {vram_gb:.2f} GB | "
              f"Compression: {state.compressed_bytes / state.fullkv_bytes:.3f} | "
              f"Context: {input_ids.shape[1]} tokens")

    return {
        "turns": turn_results,
        "oom_turn": next(
            (r["turn"] for r in turn_results if r["vram_gb"] > 22.0),
            None  # None = never OOM'd within 15 turns
        ),
    }
```

The `oom_turn` field is the headline number for the narrative: "APKVC allows N more
turns before hitting 22GB VRAM on a 24GB GPU vs KIVI."

---

## 7. Metrics Module

```python
# metrics.py

import torch
import numpy as np
from rouge_score import rouge_scorer

def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    Compute perplexity of the generated text under the model.
    Lower = model considers the text more probable = better quality.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    max_length = 2048
    stride = 512
    nlls = []

    for i in range(0, input_ids.shape[1], stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.shape[1])
        trg_len = end_loc - i
        chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L between compressed-method output and FullKV output.
    Reference = FullKV generation for the same prompt.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_compression_ratio(compressed_bytes: int, fullkv_bytes: int) -> float:
    return compressed_bytes / fullkv_bytes


def estimate_cache_bytes(n_tokens: int, n_layers: int, n_heads: int,
                          head_dim: int, dtype_bytes: int = 2) -> int:
    """
    Estimate FullKV cache size in bytes.
    dtype_bytes = 2 for fp16.
    """
    # K and V, per layer
    return 2 * n_tokens * n_layers * n_heads * head_dim * dtype_bytes


def aggregate_task_results(results: list) -> dict:
    """Average metrics across prompts within a task."""
    keys = [k for k in results[0] if k != "prompt" and k != "snapshots"]
    aggregated = {}
    for k in keys:
        vals = [r[k] for r in results if isinstance(r[k], (int, float))]
        if vals:
            aggregated[k] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }
    # Aggregate snapshots separately
    if "snapshots" in results[0]:
        n_steps = len(results[0]["snapshots"])
        aggregated["snapshots"] = []
        for step_idx in range(n_steps):
            step_data = [r["snapshots"][step_idx] for r in results]
            aggregated["snapshots"].append({
                "tokens_generated": step_data[0]["tokens_generated"],
                "compression_ratio_mean": float(np.mean([s["compression_ratio"] for s in step_data])),
                "compression_ratio_std":  float(np.std( [s["compression_ratio"] for s in step_data])),
                "anchor_rate_mean":       float(np.mean([s["anchor_rate"] for s in step_data])),
            })
    return aggregated
```

---

## 8. KIVI Implementation

Implement KIVI as the primary baseline. This is the method whose structural gap
APKVC is designed to exploit.

```python
# methods/kivi.py

import torch
from .base import KVCacheMethod, CacheState

class KIVIMethod(KVCacheMethod):

    def __init__(self, bits: int = 4, residual_buffer: int = 32):
        assert bits in (2, 4, 8)
        self.bits = bits
        self.residual_buffer = residual_buffer  # last N tokens kept in fp16
        self.name = f"KIVI-int{bits}"

    def _quantize_K(self, K):
        # K: [B, H, T, D] — per-channel quantization (over token dim)
        levels = 2 ** self.bits
        scale = K.abs().amax(dim=2, keepdim=True).clamp(min=1e-6)
        K_q = (K / scale * (levels / 2 - 1)).round().clamp(-(levels//2), levels//2 - 1)
        return K_q.to(torch.int8), scale

    def _quantize_V(self, V):
        # V: [B, H, T, D] — per-token quantization (over feature dim)
        levels = 2 ** self.bits
        scale = V.abs().amax(dim=3, keepdim=True).clamp(min=1e-6)
        V_q = (V / scale * (levels / 2 - 1)).round().clamp(-(levels//2), levels//2 - 1)
        return V_q.to(torch.int8), scale

    def _dequantize(self, X_q, scale, bits):
        levels = 2 ** bits
        return X_q.float() * scale / (levels / 2 - 1)

    def _cache_bytes(self, n_tokens, n_layers, n_heads, head_dim):
        # Quantized storage: bits/8 bytes per element
        quant_bytes = 2 * n_tokens * n_layers * n_heads * head_dim * (self.bits // 8)
        # Scales: fp16, one per (layer, head, 1, dim) for K; one per (layer, head, token, 1) for V
        scale_bytes_K = n_layers * n_heads * head_dim * 2          # [H, 1, D] fp16
        scale_bytes_V = n_tokens * n_layers * n_heads * 2          # [H, T, 1] fp16
        # Residual buffer: last residual_buffer tokens in fp16
        residual_bytes = min(n_tokens, self.residual_buffer) * n_layers * n_heads * head_dim * 2 * 2
        return quant_bytes + scale_bytes_K + scale_bytes_V + residual_bytes

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers = model.config.num_hidden_layers
        n_heads  = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads

        generated_ids = inputs.input_ids
        snapshots = []
        next_checkpoint = iter(checkpoint_steps)
        current_checkpoint = next(next_checkpoint, None)
        tokens_generated = 0
        distortions = []

        with torch.no_grad():
            # Prefill — run normally, quantize the resulting KV cache
            outputs = model(generated_ids, use_cache=True)
            past_kv = outputs.past_key_values  # list of (K, V) per layer

            # Quantize prefill KV
            quant_cache = []
            for layer_K, layer_V in past_kv:
                K_q, K_scale = self._quantize_K(layer_K)
                V_q, V_scale = self._quantize_V(layer_V)
                quant_cache.append((K_q, K_scale, V_q, V_scale))

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            while tokens_generated < max_new_tokens:
                # Dequantize cache for this step
                past_kv_deq = tuple(
                    (self._dequantize(K_q, K_scale, self.bits),
                     self._dequantize(V_q, V_scale, self.bits))
                    for K_q, K_scale, V_q, V_scale in quant_cache
                )

                outputs = model(next_token, past_key_values=past_kv_deq, use_cache=True)
                new_kv = outputs.past_key_values

                # Re-quantize updated cache
                quant_cache = []
                for layer_K, layer_V in new_kv:
                    K_q, K_scale = self._quantize_K(layer_K)
                    V_q, V_scale = self._quantize_V(layer_V)
                    quant_cache.append((K_q, K_scale, V_q, V_scale))

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1

                if tokens_generated == current_checkpoint:
                    T = generated_ids.shape[1]
                    compressed = self._cache_bytes(T, n_layers, n_heads, head_dim)
                    fullkv     = T * n_layers * n_heads * head_dim * 2 * 2  # fp16 K+V
                    snapshots.append(CacheState(
                        compressed_bytes=compressed,
                        fullkv_bytes=fullkv,
                    ))
                    current_checkpoint = next(next_checkpoint, None)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        T = generated_ids.shape[1]
        compressed = self._cache_bytes(T, n_layers, n_heads, head_dim)
        fullkv     = T * n_layers * n_heads * head_dim * 2 * 2

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        final_state = CacheState(compressed_bytes=compressed, fullkv_bytes=fullkv)

        # Pad snapshots if generation ended early
        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final_state)

        return generated_text, snapshots, final_state
```

---

## 9. Plots Module

```python
# plots.py

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

COLORS = {
    "FullKV":    "#888780",
    "KIVI-int4": "#3B8BD4",
    "KIVI-int2": "#185FA5",
    "APKVC-identity": "#9E6AC4",
    "APKVC-linear":   "#6E3A9A",
}

def plot1_compression_vs_length(results_by_method: dict, task_name: str,
                                 save_path: str = None):
    """
    Plot 1: Compression ratio vs tokens generated.

    results_by_method: {method_name: {"snapshots": [{"tokens_generated": int,
                                                      "compression_ratio_mean": float,
                                                      "compression_ratio_std": float}]}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, results in results_by_method.items():
        snaps = results["snapshots"]
        xs = [s["tokens_generated"] for s in snaps]
        ys = [s["compression_ratio_mean"] for s in snaps]
        stds = [s.get("compression_ratio_std", 0) for s in snaps]
        color = COLORS.get(method_name, "#333")

        ax.plot(xs, ys, label=method_name, color=color, linewidth=2, marker="o", ms=5)
        ax.fill_between(xs,
                         [y - s for y, s in zip(ys, stds)],
                         [y + s for y, s in zip(ys, stds)],
                         alpha=0.12, color=color)

    ax.axhline(y=1.0, color=COLORS["FullKV"], linestyle="--", linewidth=1.2,
               label="FullKV (1.0)")
    ax.set_xlabel("Tokens generated", fontsize=12)
    ax.set_ylabel("Compression ratio  (lower = more compressed)", fontsize=12)
    ax.set_title(f"Compression ratio vs generation length — {task_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # Annotate: APKVC improves over time, KIVI stays flat
    if "APKVC-identity" in results_by_method and "KIVI-int4" in results_by_method:
        ax.annotate("APKVC improves →", xy=(0.62, 0.35), xycoords="axes fraction",
                    fontsize=9, color=COLORS["APKVC-identity"])
        ax.annotate("KIVI stays flat →", xy=(0.62, 0.55), xycoords="axes fraction",
                    fontsize=9, color=COLORS["KIVI-int4"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot2_pareto_frontier(pareto_data: list, save_path: str = None):
    """
    Plot 2: Quality (perplexity or ROUGE-L) vs compression ratio.

    pareto_data: list of dicts:
      {"method": str, "compression_ratio": float, "perplexity": float,
       "rouge_l": float, "config_label": str}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, invert in [
        (axes[0], "perplexity",      "Perplexity (lower = better)", True),
        (axes[1], "rouge_l",         "ROUGE-L (higher = better)",   False),
    ]:
        plotted_methods = set()
        for point in pareto_data:
            m = point["method"]
            color = COLORS.get(m, "#333")
            label = m if m not in plotted_methods else None
            ax.scatter(point["compression_ratio"], point[metric],
                       color=color, s=80, label=label, zorder=3)
            ax.annotate(point.get("config_label", ""),
                        (point["compression_ratio"], point[metric]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color)
            plotted_methods.add(m)

        # Draw Pareto frontier for APKVC points
        apkvc_points = [(p["compression_ratio"], p[metric])
                        for p in pareto_data if "APKVC" in p["method"]]
        if apkvc_points:
            apkvc_points.sort(key=lambda x: x[0])
            # Pareto: no point dominated on both axes
            frontier = []
            best = float("inf") if invert else float("-inf")
            for cr, q in apkvc_points:
                if (invert and q < best) or (not invert and q > best):
                    frontier.append((cr, q))
                    best = q
            if frontier:
                fx, fy = zip(*frontier)
                ax.step(fx, fy, where="post", color=COLORS["APKVC-linear"],
                        linewidth=1.5, linestyle="--", alpha=0.6, label="APKVC Pareto")

        ax.set_xlabel("Compression ratio  (lower = more compressed)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title(ylabel, fontsize=12)

    fig.suptitle("Quality vs compression ratio (Pareto frontier)", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot3_vram_over_turns(turn_results_by_method: dict, gpu_limit_gb: float = 22.0,
                           save_path: str = None):
    """
    Plot 3: VRAM (GB) over conversation turns.

    turn_results_by_method: {method_name: [{"turn": int, "vram_gb": float}]}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, turns in turn_results_by_method.items():
        xs = [t["turn"] for t in turns]
        ys = [t["vram_gb"] for t in turns]
        color = COLORS.get(method_name, "#333")
        ax.plot(xs, ys, label=method_name, color=color, linewidth=2, marker="o", ms=5)

    ax.axhline(y=gpu_limit_gb, color="#E24B4A", linestyle="--", linewidth=1.5,
               label=f"GPU limit ({gpu_limit_gb} GB)")

    ax.set_xlabel("Conversation turn", fontsize=12)
    ax.set_ylabel("Peak VRAM (GB)", fontsize=12)
    ax.set_title("VRAM usage over multi-turn conversation", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
```

---

## 10. Main Entry Point

```python
# run_benchmark.py

import json
import os
import torch
from datetime import datetime

from methods.fullkv import FullKVMethod
from methods.kivi import KIVIMethod
from methods.apkvc import APKVCMethod
from tasks.continuation import run_continuation, CHECKPOINT_STEPS as CONT_STEPS
from tasks.reasoning import run_reasoning
from tasks.multiturn import run_multiturn
from plots import plot1_compression_vs_length, plot2_pareto_frontier, plot3_vram_over_turns

def main():
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, tokenizer = load_model()

    methods = {
        "FullKV":          FullKVMethod(),
        "KIVI-int4":       KIVIMethod(bits=4),
        "KIVI-int2":       KIVIMethod(bits=2),
        "APKVC-identity":  APKVCMethod(predictor_type="identity"),
        "APKVC-linear":    APKVCMethod(predictor_type="linear"),
    }

    all_results = {}

    # ----- Task 1: Continuation -----
    print("=" * 60)
    print("TASK 1: Long continuation")
    print("=" * 60)
    task1_results = {}
    for name, method in methods.items():
        print(f"\nRunning {name}...")
        task1_results[name] = run_continuation(method, model, tokenizer)
    all_results["task1_continuation"] = task1_results

    # ----- Task 2: Reasoning -----
    print("\n" + "=" * 60)
    print("TASK 2: Structured reasoning")
    print("=" * 60)
    task2_results = {}
    for name, method in methods.items():
        print(f"\nRunning {name}...")
        task2_results[name] = run_reasoning(method, model, tokenizer)
    all_results["task2_reasoning"] = task2_results

    # ----- Task 3: Multi-turn -----
    print("\n" + "=" * 60)
    print("TASK 3: Multi-turn simulation")
    print("=" * 60)
    task3_results = {}
    for name, method in methods.items():
        print(f"\nRunning {name}...")
        task3_results[name] = run_multiturn(method, model, tokenizer)
    all_results["task3_multiturn"] = task3_results

    # ----- Save raw results -----
    results_path = f"results/ldcb_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ----- Generate plots -----
    plot1_compression_vs_length(
        {name: r for name, r in task1_results.items()},
        task_name="Continuation",
        save_path=f"results/plot1_continuation_{timestamp}.png",
    )

    # Pareto: collect one point per method from task1 final metrics
    pareto_data = []
    for name, r in task1_results.items():
        pareto_data.append({
            "method": name,
            "compression_ratio": r.get("final_compression_ratio", {}).get("mean", 1.0),
            "perplexity": r.get("perplexity", {}).get("mean", 0.0),
            "rouge_l": r.get("rouge_l", {}).get("mean", 0.0),
            "config_label": name.split("-")[-1],
        })
    plot2_pareto_frontier(pareto_data,
                          save_path=f"results/plot2_pareto_{timestamp}.png")

    plot3_vram_over_turns(
        {name: r["turns"] for name, r in task3_results.items()},
        save_path=f"results/plot3_vram_{timestamp}.png",
    )

    print("Plots saved to results/")

    # ----- Print summary table -----
    print("\n" + "=" * 60)
    print("SUMMARY — Task 1 (Continuation, 4000 tokens)")
    print("=" * 60)
    print(f"{'Method':<20} {'Compression':>12} {'Perplexity':>12} {'ROUGE-L':>10}")
    print("-" * 56)
    for name, r in task1_results.items():
        cr = r.get("final_compression_ratio", {}).get("mean", 0)
        pp = r.get("perplexity", {}).get("mean", 0)
        rl = r.get("rouge_l", {}).get("mean", 0)
        print(f"{name:<20} {cr:>12.3f} {pp:>12.2f} {rl:>10.3f}")


if __name__ == "__main__":
    main()
```

---

## 11. Ablation Table

Run these variants through Task 1 only and report all metrics.
This table isolates the contribution of each APKVC component.

| Variant | predictor | rope_aq | scale_norm | K_codebooks | Expected behaviour |
|---|---|---|---|---|---|
| FullKV | — | — | — | — | Baseline ceiling |
| KIVI-int4 | — | — | — | — | Flat compression ratio |
| KIVI-int2 | — | — | — | — | Better compression, worse quality |
| APKVC-anchor-only | none | off | off | 0 | Must match FullKV. If not, reconstruction bug |
| APKVC-noscale | identity | on | off | 4 | Worse quality than with scale |
| APKVC-noro | identity | off | on | 4 | Slightly worse than rope-aware |
| APKVC-identity | identity | on | on | 4 | Core method |
| APKVC-linear | linear | on | on | 4 | Should improve over identity |
| APKVC-full | linear | on | on | 8 | Best quality / most compute |

Run anchor-only first. If it does not match FullKV, stop and fix reconstruction
before running any other variant.

---

## 12. Expected Results

If the implementation is correct and codebooks are well-trained:

**Plot 1 (compression ratio vs length):**
- KIVI-int4 should sit at a flat ~0.25 regardless of generation length
- APKVC-identity should start near 1.0 (first few tokens are anchors) and
  decrease toward ~0.15–0.25 as residual chains accumulate
- APKVC-linear should be slightly below APKVC-identity

**Plot 2 (Pareto):**
- APKVC should have at least one configuration that sits to the left of and
  below the KIVI-int4 point (better compression at similar quality) OR
  above and to the right (better quality at similar compression)

**Plot 3 (VRAM over turns):**
- FullKV VRAM should grow roughly linearly
- KIVI-int4 should grow at ~25% of FullKV rate
- APKVC should grow slower than KIVI in turns 5+ as residual chains mature

If APKVC's curve in Plot 1 is flat (not improving with length), the residual
chains are being broken too frequently — reduce rd_threshold or increase
max_anchor_interval. If anchor_rate > 50%, codebooks or scale normalization
need attention.
