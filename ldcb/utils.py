import torch

def get_kv_iterator(past_kv):
    """
    Safely yields (layer_idx, (K, V)) regardless of the underlying cache format.
    Handles DynamicCache, legacy tuples, and custom cache objects.
    """
    if past_kv is None:
        return []

    # If it's a DynamicCache-like object (e.g. from Transformers 4.36+)
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        # We zip the internal list of tensors
        return enumerate(zip(past_kv.key_cache, past_kv.value_cache))

    # If it's the legacy tuple structure: ((K0, V0), (K1, V1), ...)
    # This also handles lists of these pairs.
    return enumerate(past_kv)

def get_total_vram_gb():
    """Returns the sum of peak memory allocated across all CUDA devices."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
