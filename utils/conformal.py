from __future__ import annotations
from dataclasses import dataclass
from math import ceil
import numpy as np
from experiments.utils.models import Predictor
Array = np.ndarray
@dataclass(frozen=True)
class CalibrationResult:
    tau: float
    keep_mask: Array
    n_keep: int
def calibrate_from_scores(scores: Array, alpha: float, keep_mask: Array | None = None) -> CalibrationResult:
    scores = np.asarray(scores, dtype=float)
    if keep_mask is None:
        keep_mask = np.ones(scores.shape[0], dtype=bool)
    else:
        keep_mask = np.asarray(keep_mask, dtype=bool)
    retained = scores[keep_mask]
    n_keep = int(retained.size)
    if n_keep == 0:
        return CalibrationResult(tau=float("inf"), keep_mask=keep_mask, n_keep=0)
    rank = ceil((n_keep + 1) * (1.0 - alpha))
    if rank == n_keep + 1:
        return CalibrationResult(tau=float("inf"), keep_mask=keep_mask, n_keep=n_keep)
    tau = float(np.partition(retained, rank - 1)[rank - 1])
    return CalibrationResult(tau=tau, keep_mask=keep_mask, n_keep=n_keep)
def coverage_and_width(
    predictor: Predictor,
    x: Array,
    y: Array,
    tau: float,
) -> tuple[float, float]:
    lower, upper = predictor.interval(x, tau)
    y = np.asarray(y, dtype=float)
    coverage = float(np.mean((y >= lower) & (y <= upper)))
    width = float(np.mean(upper - lower))
    return (coverage, width)
