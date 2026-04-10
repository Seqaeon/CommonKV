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
    anchor_positions: list = field(default_factory=list)  # optional token positions for anchor diagnostics

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 1.0
        return self.fullkv_bytes / self.compressed_bytes


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
