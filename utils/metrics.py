from __future__ import annotations
from math import isfinite
import numpy as np
Array = np.ndarray
def left_quantile(values: Array, level: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("inf")
    rank = max(int(np.ceil(level * values.size)) - 1, 0)
    ordered = np.sort(values)
    return float(ordered[min(rank, ordered.size - 1)])
def kolmogorov_distance(left: Array, right: Array) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if left.size == 0 or right.size == 0:
        return 1.0
    support = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, support, side="right") / left.size
    right_cdf = np.searchsorted(right, support, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))
def estimate_local_density(values: Array, level: float, window: float = 0.05) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if values.size < 20:
        return float("nan")
    low = max(level - window, 1e-3)
    high = min(level + window, 1.0 - 1e-3)
    q_low = float(np.quantile(values, low))
    q_high = float(np.quantile(values, high))
    span = q_high - q_low
    if span <= 1e-8:
        return float("inf")
    return float((high - low) / span)
def covariance_bias_summary(nonconformity: Array, keep_mask: Array) -> tuple[float, float]:
    nonconformity = np.asarray(nonconformity, dtype=float)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if nonconformity.size == 0:
        return (float("nan"), float("nan"))
    n = int(nonconformity.size)
    n_keep = int(np.sum(keep_mask))
    if n_keep <= 0:
        return (float("nan"), float("nan"))
    order = np.argsort(nonconformity, kind="mergesort")
    values = nonconformity[order]
    keep_sorted = keep_mask[order].astype(float)
    cum_total = np.arange(1, n + 1, dtype=float)
    cum_keep = np.cumsum(keep_sorted)
    unique_end = np.r_[values[1:] != values[:-1], True]
    f_full = cum_total[unique_end] / float(n)
    f_keep = cum_keep[unique_end] / float(n_keep)
    diff = f_keep - f_full
    max_direct = float(np.max(np.abs(diff))) if diff.size else 0.0
    return (max_direct, max_direct)
def clean_dirty_retention(clean_scores: Array, dirty_scores: Array, threshold: float, epsilon: float) -> tuple[float, float, float, float]:
    p_c = float(np.mean(np.asarray(clean_scores, dtype=float) <= threshold))
    p_d = float(np.mean(np.asarray(dirty_scores, dtype=float) <= threshold))
    mu_keep = (1.0 - epsilon) * p_c + epsilon * p_d
    tilde_eps = float(epsilon * p_d / mu_keep) if mu_keep > 0.0 else 1.0
    return (p_c, p_d, mu_keep, tilde_eps)
def safe_gap(observed: float, bound: float) -> float:
    if not isfinite(observed) or not isfinite(bound):
        return float("nan")
    return float(observed - bound)

def cdf_excess(source: Array, target: Array, *, empty_value: float = 1.0) -> float:
    """Return sup_x {F_source(x) - F_target(x)} for empirical laws.

    This is the one-sided CDF discrepancy needed for clean-target conformal
    transfer.  When the source empirical law is empty, ``empty_value`` is used;
    callers should usually set it to 0.0 only when the corresponding mixture
    weight is exactly zero.
    """
    source = np.sort(np.asarray(source, dtype=float))
    target = np.sort(np.asarray(target, dtype=float))
    if target.size == 0:
        return float("nan")
    if source.size == 0:
        return float(empty_value)
    support = np.sort(np.concatenate([source, target]))
    source_cdf = np.searchsorted(source, support, side="right") / float(source.size)
    target_cdf = np.searchsorted(target, support, side="right") / float(target.size)
    return float(max(0.0, np.max(source_cdf - target_cdf)))

def weighted_cdf_excess(
    sources: list[Array] | tuple[Array, ...],
    weights: list[float] | tuple[float, ...],
    target: Array,
    *,
    empty_value: float = 1.0,
) -> float:
    """Return sup_x {sum_j w_j F_j(x) - F_target(x)}.

    Components with zero weight are skipped.  If a positive-weight component is
    empty, the returned value is conservative and at least ``empty_value``.
    """
    target = np.sort(np.asarray(target, dtype=float))
    if target.size == 0:
        return float("nan")
    arrays: list[Array] = []
    active_weights: list[float] = []
    saw_empty_positive_weight = False
    for source, weight in zip(sources, weights):
        weight = float(weight)
        if weight <= 0.0 or not np.isfinite(weight):
            continue
        array = np.sort(np.asarray(source, dtype=float))
        if array.size == 0:
            saw_empty_positive_weight = True
            continue
        arrays.append(array)
        active_weights.append(weight)
    if not arrays:
        return float(empty_value if saw_empty_positive_weight else 0.0)
    support = np.sort(np.concatenate([target, *arrays]))
    mixture_cdf = np.zeros_like(support, dtype=float)
    for array, weight in zip(arrays, active_weights):
        mixture_cdf += weight * (np.searchsorted(array, support, side="right") / float(array.size))
    target_cdf = np.searchsorted(target, support, side="right") / float(target.size)
    excess = float(max(0.0, np.max(mixture_cdf - target_cdf)))
    if saw_empty_positive_weight:
        excess = max(excess, float(empty_value))
    return excess

def dkw_radius(n: int, delta: float) -> float:
    """Two-sided DKW radius sqrt(log(2/delta)/(2n))."""
    n = int(n)
    delta = float(delta)
    if n <= 0 or delta <= 0.0:
        return float("inf")
    return float(np.sqrt(np.log(2.0 / min(delta, 1.0)) / (2.0 * n)))

def hoeffding_radius(n: int, delta: float) -> float:
    """Hoeffding radius for a Bernoulli mean."""
    n = int(n)
    delta = float(delta)
    if n <= 0 or delta <= 0.0:
        return float("inf")
    return float(np.sqrt(np.log(2.0 / min(delta, 1.0)) / (2.0 * n)))

def clipped_probability(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(min(1.0, max(0.0, value)))
