import torch
from .base import KVCacheMethod, CacheState
from ..metrics import estimate_cache_bytes

class FullKVMethod(KVCacheMethod):
    def __init__(self):
        self.name = "FullKV"

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers = model.config.num_hidden_layers
        n_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        
        generated_ids = inputs.input_ids
        snapshots = []
        next_checkpoint = iter(checkpoint_steps)
        current_checkpoint = next(next_checkpoint, None)
        tokens_generated = 0
        
        # We use a simple loop to capture checkpoints, or model.generate with a callback if possible.
        # For simplicity and consistency with the guide's KIVI implementation, we'll use a loop.
        
        past_key_values = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True) if past_key_values is not None else model(generated_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1
                
                if tokens_generated == current_checkpoint:
                    T = generated_ids.shape[1]
                    bytes_val = estimate_cache_bytes(T, n_layers, n_heads, head_dim)
                    snapshots.append(CacheState(compressed_bytes=bytes_val, fullkv_bytes=bytes_val))
                    current_checkpoint = next(next_checkpoint, None)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        T = generated_ids.shape[1]
        final_bytes = estimate_cache_bytes(T, n_layers, n_heads, head_dim)
        final_state = CacheState(compressed_bytes=final_bytes, fullkv_bytes=final_bytes)
        
        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final_state)
            
        return generated_text, snapshots, final_state
