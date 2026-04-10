"""
ldcb/tasks/longbench.py

LongBench task runner for methods with a custom generate() loop (e.g. IAVQ-KC).
Writes predictions in the same per-dataset JSON format produced by run_longbench.py
so that eval.py can score all methods in one pass.

Output structure matches run_longbench.py:
  save_dir/<dataset>/<method_name>.json   (one JSON object per line)
  save_dir/<dataset>/<method_name>.jsonl  (same)
"""

import json
import os
import time
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# LongBench prompt templates (mirrors run_longbench.py's model2prompt dict)
# ---------------------------------------------------------------------------

_PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely "
        "as you can, using a single phrase or sentence if possible. If the question cannot be "
        "answered based on the information in the article, write \"unanswerable\". If the "
        "question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Article: {context}\n\n Answer the question based on the above article as concisely "
        "as you can, using a single phrase or sentence if possible. If the question cannot be "
        "answered based on the information in the article, write \"unanswerable\". If the "
        "question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report."
        "\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences."
        "\n\nQuery: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. "
        "Here are some examples of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not "
        "output any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. "
        "The following are some examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there "
        "are after removing duplicates. In other words, how many non-repeating paragraphs are "
        "there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs "
        "after removing duplicates. The output format should only contain the number, "
        "such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which "
        "paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n"
        "{input}\n\nPlease enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

_DATASET_MAX_NEW_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_longbench_iavqkc(
    method,
    model,
    tokenizer,
    datasets: list,
    data_dir: str,
    save_dir: str,
    steps: int = -1,
    model_max_len: int = 4096,
) -> None:
    """
    Run IAVQ-KC (or any KVCacheMethod with a custom generate() loop) on the
    given LongBench datasets and write prediction files compatible with eval.py.

    Parameters
    ----------
    method      : KVCacheMethod instance (IAVQ-KC)
    model       : loaded HuggingFace model
    tokenizer   : corresponding tokenizer
    datasets    : list of LongBench dataset names to evaluate
    data_dir    : folder containing <dataset>.jsonl data files
    save_dir    : folder to write <dataset>/IAVQ-KC.json output files
    steps       : max examples per dataset (-1 = all)
    model_max_len : max prompt length in tokens (hard-truncate to this)
    """
    method_name = getattr(method, "name", "IAVQ-KC")
    device = next(model.parameters()).device

    for dataset in datasets:
        data_file = os.path.join(data_dir, f"{dataset}.jsonl")
        if not os.path.exists(data_file):
            print(f"  [LongBench] Skipping {dataset} — data file not found: {data_file}")
            continue

        # Load examples
        examples = []
        with open(data_file) as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if steps != -1:
            examples = examples[:steps]

        template = _PROMPT.get(dataset)
        if template is None:
            print(f"  [LongBench] Skipping {dataset} — no prompt template defined.")
            continue

        max_new_tokens = _DATASET_MAX_NEW_TOKENS.get(dataset, 64)
        out_dir = os.path.join(save_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f"{method_name}.json")
        jsonl_path = os.path.join(out_dir, f"{method_name}.jsonl")

        predictions = []
        print(f"  [LongBench] Dataset: {dataset} ({len(examples)} examples) ...")

        for ex in examples:
            prompt = template.format(**ex)

            # Hard-truncate prompt to model_max_len tokens
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(token_ids) > model_max_len:
                half = model_max_len // 2
                token_ids = token_ids[:half] + token_ids[-(model_max_len - half):]
                prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

            start = time.time()
            try:
                # IAVQ-KC's generate() expects (model, tokenizer, prompt, max_new_tokens, checkpoint_steps)
                gen_text, _snapshots, final_state = method.generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    checkpoint_steps=[max_new_tokens],
                )
                # Strip the prompt from the beginning of generated text
                if gen_text.startswith(prompt):
                    pred = gen_text[len(prompt):].strip()
                else:
                    # Fallback: take everything after the last newline of the prompt
                    pred = gen_text.strip().split("\n")[-1]
            except Exception as e:
                pred = ""
                print(f"    [WARN] generate() failed on {dataset}: {e}")

            latency = time.time() - start
            n_toks = max(len(tokenizer.encode(pred)), 1)
            tps = n_toks / latency if latency > 0 else 0.0

            record = {
                "prompt":    prompt,
                "input":     ex.get("input", ""),
                "context":   ex.get("context", ""),
                "answers":   ex.get("answers", []),
                "pred":      pred,
                "length":    ex.get("length", len(token_ids)),
                "dataset":   dataset,
                "language":  ex.get("language", "en"),
                "all_classes": ex.get("all_classes"),
                "_id":       ex.get("_id", ""),
                "compression_ratio": (
                    final_state.compression_ratio if hasattr(final_state, "compression_ratio") else 1.0
                ),
                "latency": round(latency, 4),
                "tps":     round(tps, 4),
            }
            predictions.append(record)

            torch.cuda.empty_cache()

        # Write output
        with open(json_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(jsonl_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print(f"    Wrote {len(predictions)} predictions → {json_path}")
