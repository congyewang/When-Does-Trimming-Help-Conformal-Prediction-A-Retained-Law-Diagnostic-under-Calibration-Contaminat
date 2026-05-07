from __future__ import annotations
from math import sqrt
from typing import Iterable
import numpy as np
def finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]
def mean_std_se(values: Iterable[float]) -> tuple[float, float, float]:
    array = finite_values(values)
    if array.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), 0.0, 0.0)
    std = float(array.std(ddof=0))
    return (float(array.mean()), std, float(std / sqrt(array.size)))
