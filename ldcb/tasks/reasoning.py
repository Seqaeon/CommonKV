import torch
import random
from ..utils import get_total_vram_gb
from ..metrics import aggregate_task_results

MAX_NEW_TOKENS = 2000
CHECKPOINT_STEPS = [250, 500, 1000, 1500, 2000]

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

def estimate_boundary_vs_interior_anchor_rate(generated_text, anchor_positions):
    """
    Approximate boundary tokens as those within 5 tokens of a newline
    followed by a digit + period (e.g. "3."). 
    """
    lines = generated_text.split("\n")
    boundary_tokens = set()
    cursor = 0
    for line in lines:
        if line.strip() and line.strip()[0].isdigit() and "." in line[:3]:
            for offset in range(-5, 15):
                boundary_tokens.add(cursor + offset)
        cursor += len(line) + 1 

    boundary_anchors = sum(1 for p in anchor_positions if p in boundary_tokens)
    interior_anchors = len(anchor_positions) - boundary_anchors
    n_boundary = len(boundary_tokens)
    n_interior = len(anchor_positions) + (cursor - len(anchor_positions)) - n_boundary

    return {
        "boundary_anchor_rate": boundary_anchors / max(n_boundary, 1),
        "interior_anchor_rate": interior_anchors / max(n_interior, 1),
    }

def run_reasoning(method, model, tokenizer, max_new_tokens=None) -> dict:
    all_results = []
    
    # Calculate limits based on model capacity if provided
    limit = max_new_tokens or MAX_NEW_TOKENS
    # Filter and cap checkpoint steps
    active_steps = [s for s in CHECKPOINT_STEPS if s < limit]
    active_steps.append(limit)

    for prompt in REASONING_PROMPTS:
        # Tokenize and verify prompt is short
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        assert input_ids.shape[1] <= 128, \
            f"Prompt too long: {input_ids.shape[1]} tokens. Trim it."

        torch.cuda.reset_peak_memory_stats()
        generated_text, snapshots, final_state = method.generate(
            model, tokenizer, prompt,
            max_new_tokens=limit,
            checkpoint_steps=active_steps,
        )

        result = {
            "prompt": prompt[:60] + "...",
            "snapshots": [
                {
                    "tokens_generated": step,
                    "compression_ratio": snap.compressed_bytes / snap.fullkv_bytes,
                    "anchor_rate": snap.anchor_count / max(snap.anchor_count + snap.residual_count, 1),
                }
                for step, snap in zip(active_steps, snapshots)
            ],
            "final_compression_ratio": final_state.compressed_bytes / final_state.fullkv_bytes,
            "peak_vram_gb": get_total_vram_gb(),
            "perplexity": 0.0, # Reasoning task focuses on logic rather than PPL
        }
        all_results.append(result)

    return aggregate_task_results(all_results)
