"""
ldcb/tasks/gsm8k.py

GSM8K (Grade School Math 8K) benchmark task for LDCB.

Evaluates KV-cache compression methods on multi-step arithmetic reasoning:
  - Downloads openai/gsm8k from HuggingFace on first run (auto-cached).
  - Runs each method's custom generate() loop (compatible with KVCacheMethod).
  - Extracts the final numeric answer and checks exact-match against the gold label.
  - Reports accuracy (%), mean compression ratio, mean tokens-per-second.

Usage:
  python -m ldcb.run_benchmark --tasks gsm8k --gsm8k_shots 8 --gsm8k_steps 200
"""

import re
import time
import torch
from ..metrics import aggregate_task_results


# ---------------------------------------------------------------------------
# 8-shot chain-of-thought examples from the original GSM8K paper (Brown 2020)
# ---------------------------------------------------------------------------

_8SHOT_EXAMPLES = """\
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he gave some to Denny. Now he has 12 lollipops. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
A: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more on Wednesday, he had 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.\
"""


def _build_prompt(question: str, n_shots: int) -> str:
    """Build a few-shot chain-of-thought prompt for a GSM8K question."""
    if n_shots == 0:
        return f"Q: {question}\nA:"

    # Use the first n_shots examples from the 8-shot bank
    examples = _8SHOT_EXAMPLES.split("\n\n")
    few_shot_block = "\n\n".join(examples[:n_shots])
    return f"{few_shot_block}\n\nQ: {question}\nA:"


def _extract_answer(text: str) -> str | None:
    """
    Extract the final numeric answer from model output.
    Handles patterns like:
      "The answer is 42."
      "= 42"
      "42\n"  (last number in text)
    Returns the answer string or None.
    """
    # Prefer explicit "The answer is N" pattern
    m = re.search(r"[Tt]he answer is\s*([+-]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")

    # Fall back to last number found
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def _gold_answer(answer_text: str) -> str:
    """Extract the numeric part from GSM8K's gold answer field (e.g. '#### 72')."""
    m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in the field
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", answer_text)
    return numbers[-1].replace(",", "") if numbers else ""


def run_gsm8k(
    method,
    model,
    tokenizer,
    n_shots: int = 8,
    steps: int = 200,
    max_new_tokens: int = 256,
    split: str = "test",
) -> dict:
    """
    Run GSM8K on a single KVCacheMethod and return aggregated metrics.

    Parameters
    ----------
    method        : KVCacheMethod instance
    model         : loaded HuggingFace model
    tokenizer     : corresponding tokenizer
    n_shots       : few-shot examples to prepend (0–8)
    steps         : number of GSM8K examples to evaluate (-1 = full 1319)
    max_new_tokens: max decode tokens per example
    split         : "test" (default) or "train"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run `pip install datasets` to use the GSM8K task.")

    print(f"  Loading GSM8K ({split}) ...")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if steps != -1:
        ds = ds.select(range(min(steps, len(ds))))

    n_correct = 0
    total = 0
    compression_ratios = []
    latencies = []
    tpss = []

    for example in ds:
        question = example["question"]
        gold = _gold_answer(example["answer"])
        prompt = _build_prompt(question, n_shots=n_shots)

        t0 = time.time()
        try:
            gen_text, _snapshots, final_state = method.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                checkpoint_steps=[max_new_tokens],
            )
        except Exception as e:
            print(f"    [WARN] generate() failed: {e}")
            gen_text = ""
            final_state = None
        elapsed = time.time() - t0

        # Strip prompt prefix if returned in gen_text
        pred_text = gen_text[len(prompt):].strip() if gen_text.startswith(prompt) else gen_text.strip()
        pred_ans = _extract_answer(pred_text)
        correct = pred_ans is not None and pred_ans == gold

        n_correct += int(correct)
        total += 1

        n_pred_tokens = max(len(tokenizer.encode(pred_text)), 1)
        latencies.append(elapsed)
        tpss.append(n_pred_tokens / elapsed if elapsed > 0 else 0.0)

        if final_state is not None and hasattr(final_state, "compression_ratio"):
            compression_ratios.append(final_state.compression_ratio)

        torch.cuda.empty_cache()

    accuracy = round(100.0 * n_correct / max(total, 1), 2)
    mean_cr = round(sum(compression_ratios) / len(compression_ratios), 4) if compression_ratios else 1.0
    mean_lat = round(sum(latencies) / len(latencies), 4) if latencies else 0.0
    mean_tps = round(sum(tpss) / len(tpss), 2) if tpss else 0.0

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": total,
        "mean_compression_ratio": mean_cr,
        "mean_latency_s": mean_lat,
        "mean_tps": mean_tps,
        # Pack into a shape expected by aggregate_task_results for consistency
        "final_compression_ratio": {"mean": mean_cr, "std": 0.0},
        "status": "done",
    }
