from __future__ import annotations
from dataclasses import dataclass
import numpy as np
Array = np.ndarray
@dataclass(frozen=True)
class RandomTrimSpec:
    threshold_label: str
    threshold_quantile: float
    threshold_value: float
    n_keep: int
def matched_random_keep_mask(rng: np.random.Generator, size: int, n_keep: int) -> Array:
    n_keep = max(0, min(int(n_keep), size))
    mask = np.zeros(size, dtype=bool)
    if n_keep == 0:
        return mask
    chosen = rng.choice(size, size=n_keep, replace=False)
    mask[chosen] = True
    return mask
