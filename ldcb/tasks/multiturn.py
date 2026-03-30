import torch
from ..utils import get_total_vram_gb

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

def run_multiturn(method, model, tokenizer) -> dict:
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

        vram_gb = get_total_vram_gb()

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
