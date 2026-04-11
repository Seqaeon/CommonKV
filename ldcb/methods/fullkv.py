import torch
from .base import KVCacheMethod, CacheState
from ..metrics import estimate_cache_bytes

class FullKVMethod(KVCacheMethod):
    def __init__(self):
        self.name = "FullKV"

    def prefill(self, model, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(inputs.input_ids, use_cache=True)
        return outputs.past_key_values

    def generate(self, model, tokenizer, prompt, max_new_tokens, checkpoint_steps, cached_state=None):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_layers = model.config.num_hidden_layers
        n_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        
        # If resuming from cached_state, generated_ids starts from the new prompt
        # but the model knows the previous context via past_key_values.
        generated_ids = inputs.input_ids
        snapshots = []
        next_checkpoint = iter(checkpoint_steps)
        current_checkpoint = next(next_checkpoint, None)
        tokens_generated = 0
        
        past_key_values = cached_state
        
        # If starting fresh (no cache), do the first full prefill
        if past_key_values is None:
            with torch.no_grad():
                outputs = model(generated_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated = 1
        else:
            # Resuming from cache, do prefill for the "new" part of the prompt
            with torch.no_grad():
                outputs = model(generated_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated = 1

        with torch.no_grad():
            while tokens_generated < max_new_tokens:
                outputs = model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1
                
                if current_checkpoint is not None and tokens_generated == current_checkpoint:
                    T = generated_ids.shape[1]
                    # If cached, we need to account for the prefix length
                    if cached_state is not None:
                        # cached_state is past_key_values, we can get length from it
                        prefix_len = past_key_values[0][0].shape[2] - tokens_generated
                        T += prefix_len
                    bytes_val = estimate_cache_bytes(T, n_layers, n_heads, head_dim)
                    snapshots.append(CacheState(compressed_bytes=bytes_val, fullkv_bytes=bytes_val))
                    current_checkpoint = next(next_checkpoint, None)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        T = generated_ids.shape[1]
        if cached_state is not None:
            prefix_len = past_key_values[0][0].shape[2] - tokens_generated
            T += prefix_len
            
        final_bytes = estimate_cache_bytes(T, n_layers, n_heads, head_dim)
        final_state = CacheState(compressed_bytes=final_bytes, fullkv_bytes=final_bytes)
        
        while len(snapshots) < len(checkpoint_steps):
            snapshots.append(final_state)
            
        return generated_text, snapshots, final_state

