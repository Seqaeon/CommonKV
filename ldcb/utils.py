import torch

def get_kv_iterator(past_kv):
    """
    Safely yields (layer_idx, (K, V)) regardless of the underlying cache format.
    Handles DynamicCache, legacy nested tuples, and legacy flat lists.
    """
    if past_kv is None:
        return

    # Case 1: DynamicCache or similar instance with key_cache/value_cache attributes
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for i, (k, v) in enumerate(zip(past_kv.key_cache, past_kv.value_cache)):
            yield i, (k, v)
        return

    # Case 2: Legacy "Flat" Format: ([k0, k1, k2...], [v0, v1, v2...])
    # Identified by being a 2-element list/tuple where each element is a list of tensors.
    if isinstance(past_kv, (list, tuple)) and len(past_kv) == 2:
        if isinstance(past_kv[0], (list, tuple)) and len(past_kv[0]) > 0:
            if isinstance(past_kv[0][0], torch.Tensor):
                for i, (k, v) in enumerate(zip(past_kv[0], past_kv[1])):
                    yield i, (k, v)
                return

    # Case 3: Standard Nested format: ((k0, v0), (k1, v1), ...) 
    # or Custom format with metadata: ((k0, v0, meta0), (k1, v1, meta1)...)
    for i, layer_kv in enumerate(past_kv):
        if isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2:
            # Explicitly take only the first 2 elements (K and V)
            yield i, (layer_kv[0], layer_kv[1])
        else:
            # Last-resort fallback for very unusual single-tensor-per-layer formats
            yield i, layer_kv

def get_total_vram_gb():
    """Returns the sum of peak memory allocated across all CUDA devices."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
