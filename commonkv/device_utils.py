import torch
import random
import numpy as np

def get_device():
    """Detect and return the best available accelerator device."""
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except (ImportError, RuntimeError):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available(): # Support Intel XPU
            return torch.device("xpu")
        return torch.device("cpu")

def seed_everything(seed: int):
    """universally seed all available backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = get_device()
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device.type == "xla":
        # Seeding for XLA is usually handled via standard torch.manual_seed
        # but some specific XLA operations might need xm.set_rng_state if used.
        pass

def cleanup_memory():
    """Safely clear accelerator caches."""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xla":
        # XLA handles memory differently; usually no explicit empty_cache equivalent 
        # is needed in the same way as CUDA.
        pass
    import gc
    gc.collect()
