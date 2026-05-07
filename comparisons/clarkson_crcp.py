
from __future__ import annotations

"""
Direct comparison utilities for Clarkson et al. (2024), "Split Conformal
Prediction under Data Contamination".

This module is intentionally self-contained.  It re-implements the small pieces
needed for synthetic classification comparisons from the public refactored
repository rather than importing that repository at runtime.  The implemented
methods are:

  * APS split conformal prediction on noisy calibration labels.
  * CRCP / Contamination Robust Conformal Prediction, using the plug-in
    ECDF correction and finite-sample C term from robust_cp.py.
  * NACP-style binary-search APS correction, following nacp.py.
  * Optional covariate-trimmed APS baselines that are analogous to the
    trimming mechanism in the present paper.  These are deliberately included
    as diagnostics: under label-only noise, covariate trimming should not
    reliably separate corrupted labels.

The formulas match the public code structure:
  - robust_cp.py: ScoreECDF, g_hat, compute_C, compute_adjusted_i
  - aps.py: ProbabilityAccumulator / APS scores and prediction sets
  - nacp.py: NACP_aps binary search

The code uses clean test labels and noisy train/calibration labels, matching
the synthetic setup in the Clarkson et al. repository.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.special import softmax


Array = np.ndarray


# ---------------------------------------------------------------------------
# Dataset and label-noise utilities
# ---------------------------------------------------------------------------

def generate_logistic(rng: np.random.Generator, n: int, p: int, K: int) -> tuple[Array, Array]:
    """Gaussian logistic model used in Clarkson et al.'s synthetic experiment."""
    beta = rng.normal(0.0, 1.0, size=(p, K))
    X = rng.normal(0.0, 1.0, size=(n, p))
    probs = softmax(X @ beta, axis=1)
    y = np.array([rng.choice(K, p=row) for row in probs], dtype=int)
    return X, y


def generate_hypercube(
    rng: np.random.Generator,
    n: int,
    p: int,
    K: int,
    sep: float = 1.0,
    prop_informative: float = 0.75,
) -> tuple[Array, Array]:
    """Hypercube-style classification task from the Clarkson et al. repository."""
    X, y = make_classification(
        n_samples=n,
        n_classes=K,
        n_features=p,
        n_informative=max(1, int(p * prop_informative)),
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=sep,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def make_random_flip(eps: float, K: int) -> Array:
    """Noise matrix P used by the public code: (1-eps)I + eps/K 11^T."""
    return (1.0 - eps) * np.eye(K) + (eps / K) * np.ones((K, K))


def sample_noisy_labels(
    rng: np.random.Generator,
    y: Array,
    P: Array,
    p_clean: Array | None = None,
    p_noisy: Array | None = None,
) -> Array:
    """Sample noisy labels following label_noise.py from the public repository.

    The repository uses probabilities
        P[j, :] * p_tilde / p[j]
    conditional on clean class j.  In the synthetic experiments p and p_tilde
    are taken uniform, so this reduces to P[j, :].
    """
    y = np.asarray(y, dtype=int)
    K = P.shape[0]
    if p_clean is None:
        p_clean = np.ones(K) / K
    if p_noisy is None:
        p_noisy = np.ones(K) / K
    y_tilde = np.empty_like(y)
    for j in range(K):
        idx = np.where(y == j)[0]
        if idx.size == 0:
            continue
        probs = P[j, :] * p_noisy / max(float(p_clean[j]), 1e-12)
        probs = probs / probs.sum()
        y_tilde[idx] = rng.choice(K, size=idx.size, p=probs)
    return y_tilde


def get_classifier(name: str, seed: int):
    if name == "log_reg":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=seed)
    if name == "gbt":
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    raise ValueError(f"Unknown classifier {name!r}")


# ---------------------------------------------------------------------------
# APS conformal scores and prediction sets
# ---------------------------------------------------------------------------

def _sorted_prob_data(probs: Array) -> tuple[Array, Array, Array]:
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, axis=1)
    prob_sort = np.take_along_axis(probs, order, axis=1)
    z = np.round(np.cumsum(prob_sort, axis=1), 9)
    ranks = np.empty_like(order)
    for i in range(order.shape[0]):
        ranks[i, order[i]] = np.arange(order.shape[1])
    return order, prob_sort, z, ranks


def aps_randomized_scores(probs: Array, labels: Array, u: Array | None = None) -> Array:
    """Randomized APS nonconformity score.

    This returns cum_prob(label) - prob(label) * U.  It is equivalent to
    1 - ProbabilityAccumulator.calibrate_scores(labels, U) in the public code.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if u is None:
        u = np.ones(labels.shape[0])
    order, prob_sort, z, ranks = _sorted_prob_data(probs)
    label_ranks = ranks[np.arange(labels.shape[0]), labels]
    cum = z[np.arange(labels.shape[0]), label_ranks]
    prob = prob_sort[np.arange(labels.shape[0]), label_ranks]
    return np.minimum(np.maximum(cum - prob * np.asarray(u, dtype=float), 0.0), 1.0)


def aps_sets_from_qhat(
    probs: Array,
    qhat: float,
    u: Array | None = None,
    allow_empty: bool = True,
) -> Array:
    """Return a boolean prediction-set matrix for randomized APS threshold qhat."""
    probs = np.asarray(probs, dtype=float)
    n, K = probs.shape
    if u is None:
        u = np.zeros(n)
    u = np.asarray(u, dtype=float)
    order, prob_sort, z, _ = _sorted_prob_data(probs)
    qhat = float(np.clip(qhat, 0.0, 1.0))
    L = np.argmax(z >= qhat, axis=1).astype(int)
    # If qhat is greater than all cumulative masses due to rounding, include all.
    no_hit = np.all(z < qhat, axis=1)
    L[no_hit] = K - 1
    z_excess = z[np.arange(n), L] - qhat
    denom = np.maximum(prob_sort[np.arange(n), L], 1e-12)
    p_remove = np.clip(z_excess / denom, 0.0, 1.0)
    remove = u <= p_remove
    L = L.copy()
    if allow_empty:
        L[remove] -= 1
    else:
        L[remove] = np.maximum(0, L[remove] - 1)
    sets = np.zeros((n, K), dtype=bool)
    for i in range(n):
        if L[i] >= 0:
            sets[i, order[i, : L[i] + 1]] = True
    return sets


def evaluate_sets(sets: Array, labels: Array) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    covered = sets[np.arange(labels.size), labels]
    sizes = sets.sum(axis=1)
    return {
        "coverage": float(np.mean(covered)),
        "set_size": float(np.mean(sizes)),
        "singleton_rate": float(np.mean(sizes == 1)),
        "empty_rate": float(np.mean(sizes == 0)),
    }


def repo_quantile_index(n: int, alpha: float) -> int:
    """Indexing used by the public run_experiment.py.

    Their code sets i = ceil((n+1)(1-alpha)) and then indexes sorted_scores[i].
    We clip to n-1 to avoid out-of-range failures.
    """
    return int(np.clip(math.ceil((n + 1) * (1.0 - alpha)), 0, n - 1))


def standard_quantile_index(n: int, alpha: float) -> int:
    return int(np.clip(math.ceil((n + 1) * (1.0 - alpha)) - 1, 0, n - 1))


def qhat_from_scores(scores: Array, alpha: float, indexing: str = "repo") -> tuple[float, int]:
    sorted_scores = np.sort(np.asarray(scores, dtype=float))
    idx = repo_quantile_index(sorted_scores.size, alpha) if indexing == "repo" else standard_quantile_index(sorted_scores.size, alpha)
    return float(sorted_scores[idx]), int(idx)


# ---------------------------------------------------------------------------
# Clarkson CRCP implementation
# ---------------------------------------------------------------------------

@dataclass
class ScoreECDF:
    y_noisy: Array
    K: int
    probs: Array
    u: Array

    def __post_init__(self) -> None:
        self.sorted_scores: list[list[Array | None]] = [[None] * self.K for _ in range(self.K)]
        for j in range(self.K):
            idx = np.where(self.y_noisy == j)[0]
            if idx.size == 0:
                continue
            for i in range(self.K):
                dummy = np.full(self.y_noisy.shape[0], i, dtype=int)
                scores_i = aps_randomized_scores(self.probs, dummy, self.u)
                self.sorted_scores[i][j] = np.sort(scores_i[idx])

    def compute(self, q: Array, i: int, j: int) -> Array:
        arr = self.sorted_scores[i][j]
        q = np.asarray(q, dtype=float)
        if arr is None or len(arr) == 0:
            return np.zeros_like(q, dtype=float)
        return np.searchsorted(arr, q, side="right") / float(len(arr))


def g_hat(q: Array, fhat: ScoreECDF, P_inv: Array, p: Array, p_tilde: Array) -> Array:
    q = np.asarray(q, dtype=float)
    K = P_inv.shape[0]
    ret = np.zeros_like(q, dtype=float)
    for i in range(K):
        for j in range(K):
            ret += p[i] * P_inv[j, i] * fhat.compute(q, i, j)
        ret -= p_tilde[i] * fhat.compute(q, i, i)
    return ret


def compute_b(n: int, p_tilde: Array) -> Array:
    return np.sqrt(np.pi / (n * np.maximum(p_tilde, 1e-12))) + (1.0 - p_tilde) ** n


def compute_C(n: int, P_inv: Array, p: Array, p_tilde: Array) -> float:
    K = P_inv.shape[0]
    b = compute_b(n, p_tilde)
    ret = 0.0
    for i in range(K):
        ret += abs(P_inv[i, i] * p[i] - p_tilde[i]) * b[i]
        for j in range(K):
            if j != i:
                ret += abs(P_inv[j, i] * p[i]) * b[j]
    return float(ret)


def compute_adjusted_index(
    alpha: float,
    sorted_scores: Array,
    fhat: ScoreECDF,
    P_inv: Array,
    p: Array,
    p_tilde: Array,
    indexing: str = "repo",
) -> tuple[int, float, bool]:
    n = sorted_scores.shape[0]
    C = compute_C(n, P_inv, p, p_tilde)
    ghats = g_hat(sorted_scores, fhat, P_inv, p, p_tilde)
    ivals = (1.0 - alpha) - ghats + C - (np.arange(1, n + 1) / (n + 1))
    candidates = np.where(ivals <= 0)[0]
    fallback = candidates.size == 0
    if fallback:
        idx = repo_quantile_index(n, alpha) if indexing == "repo" else standard_quantile_index(n, alpha)
    else:
        idx = int(candidates.min())
    return int(np.clip(idx, 0, n - 1)), float(C), bool(fallback)


def crcp_qhat(
    cal_probs: Array,
    noisy_cal_labels: Array,
    alpha: float,
    eps: float,
    cal_u: Array,
    indexing: str = "repo",
) -> dict[str, Any]:
    K = cal_probs.shape[1]
    P = make_random_flip(eps, K)
    p = np.ones(K) / K
    p_tilde = np.ones(K) / K
    fhat = ScoreECDF(noisy_cal_labels, K, cal_probs, cal_u)
    scores = []
    for i in range(K):
        arr = fhat.sorted_scores[i][i]
        if arr is not None:
            scores.append(arr)
    if not scores:
        raise RuntimeError("No diagonal score ECDFs were available.")
    sorted_scores = np.sort(np.concatenate(scores))
    idx, C, fallback = compute_adjusted_index(
        alpha, sorted_scores, fhat, np.linalg.inv(P), p, p_tilde, indexing=indexing
    )
    return {
        "qhat": float(sorted_scores[idx]),
        "index": int(idx),
        "C": float(C),
        "fallback": float(fallback),
    }


# ---------------------------------------------------------------------------
# NACP implementation from the public code
# ---------------------------------------------------------------------------

def nacp_qhat(
    cal_probs: Array,
    noisy_cal_labels: Array,
    alpha: float,
    eps: float,
    tolerance: float = 5e-5,
    q_low: float = 0.5,
    q_high: float = 1.0,
) -> dict[str, Any]:
    K = cal_probs.shape[1]
    left = float(q_low)
    right = float(q_high)
    best_q = right
    min_diff = float("inf")
    # Use deterministic APS sets in the calibration-side binary search, matching
    # the public NACP_aps implementation.
    while right - left > tolerance:
        qhat = 0.5 * (left + right)
        sets = aps_sets_from_qhat(cal_probs, qhat, u=np.ones(cal_probs.shape[0]), allow_empty=True)
        A = evaluate_sets(sets, noisy_cal_labels)["coverage"]
        D = float(np.mean(sets.sum(axis=1)) / K)
        B = (A - eps * D) / max(1.0 - eps, 1e-12)
        diff = B - (1.0 - alpha)
        if diff > 0:
            right = qhat
            if diff < min_diff:
                min_diff = diff
                best_q = qhat
        else:
            left = qhat
    return {
        "qhat": float(best_q),
        "min_diff": float(min_diff),
        "fallback": float(not math.isfinite(min_diff)),
    }


# ---------------------------------------------------------------------------
# Covariate-trimmed APS diagnostics
# ---------------------------------------------------------------------------

def mahalanobis_scores(X: Array, center: Array, covariance: Array) -> Array:
    X = np.asarray(X, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]])
    covariance = 0.5 * (covariance + covariance.T) + 1e-6 * np.eye(covariance.shape[0])
    inv = np.linalg.pinv(covariance)
    diff = X - center
    return np.sqrt(np.maximum(np.sum((diff @ inv) * diff, axis=1), 0.0))


def calibrate_aps(
    cal_probs: Array,
    cal_labels: Array,
    alpha: float,
    rng: np.random.Generator,
    keep_mask: Array | None = None,
    indexing: str = "repo",
) -> dict[str, Any]:
    n = cal_probs.shape[0]
    if keep_mask is None:
        keep_mask = np.ones(n, dtype=bool)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.sum() == 0:
        return {"qhat": 1.0, "index": -1, "n_keep": 0}
    u = rng.uniform(0.0, 1.0, size=n)
    scores = aps_randomized_scores(cal_probs, cal_labels, u)
    qhat, idx = qhat_from_scores(scores[keep_mask], alpha=alpha, indexing=indexing)
    return {"qhat": float(qhat), "index": int(idx), "n_keep": int(keep_mask.sum())}


def evaluate_qhat(
    method_id: str,
    qhat: float,
    test_probs: Array,
    clean_test_labels: Array,
    rng: np.random.Generator,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    u = rng.uniform(0.0, 1.0, size=test_probs.shape[0])
    sets = aps_sets_from_qhat(test_probs, qhat, u=u, allow_empty=True)
    out = evaluate_sets(sets, clean_test_labels)
    out.update({"method_id": method_id, "qhat": float(qhat)})
    if extra:
        out.update(extra)
    return out


def valid_shortest(summary: pd.DataFrame, coverage_target: float) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "model", "eps"]
    for key, sub in summary.groupby(group_cols, dropna=False):
        non_oracle = sub[~sub["method_id"].str.contains("oracle", case=False, na=False)].copy()
        valid = non_oracle[non_oracle["coverage_mean"] >= coverage_target]
        if valid.empty:
            chosen = non_oracle.sort_values(["coverage_mean", "set_size_mean"], ascending=[False, True]).head(1).copy()
            chosen["selection_rule"] = f"fallback_max_coverage_no_method_reached_{coverage_target:.3f}"
        else:
            chosen = valid.sort_values(["set_size_mean", "coverage_mean"], ascending=[True, False]).head(1).copy()
            chosen["selection_rule"] = f"valid_shortest_nonoracle_coverage>={coverage_target:.3f}"
        rows.append(chosen)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_results(raw: pd.DataFrame, alpha: float) -> dict[str, pd.DataFrame]:
    metrics = ["coverage", "set_size", "singleton_rate", "empty_rate", "qhat", "n_keep", "trim_quantile"]
    group_cols = ["dataset", "model", "eps", "method_id", "trim_quantile"]
    agg_frames = []
    for cols, sub in raw.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, cols))
        row["n_reps"] = int(sub["replicate"].nunique())
        for m in metrics:
            if m in sub.columns:
                values = pd.to_numeric(sub[m], errors="coerce")
                row[f"{m}_mean"] = float(values.mean())
                row[f"{m}_sd"] = float(values.std(ddof=1)) if values.count() > 1 else 0.0
                row[f"{m}_se"] = float(row[f"{m}_sd"] / math.sqrt(max(values.count(), 1)))
                row[f"{m}_ci_low"] = float(row[f"{m}_mean"] - 1.96 * row[f"{m}_se"])
                row[f"{m}_ci_high"] = float(row[f"{m}_mean"] + 1.96 * row[f"{m}_se"])
        # Retain useful diagnostic means if available.
        for m in ["crcp_C", "crcp_fallback", "nacp_min_diff", "nacp_fallback", "retained_fraction", "dirty_label_fraction_retained"]:
            if m in sub.columns:
                values = pd.to_numeric(sub[m], errors="coerce")
                row[f"{m}_mean"] = float(values.mean())
        agg_frames.append(row)
    summary = pd.DataFrame(agg_frames)
    summary = summary.sort_values(["dataset", "model", "eps", "method_id", "trim_quantile"], na_position="last")
    highlights = valid_shortest(summary, coverage_target=1.0 - alpha - 0.005)
    return {"summary": summary, "valid_shortest": highlights}


# ---------------------------------------------------------------------------
# Main classification experiment runner
# ---------------------------------------------------------------------------

def run_single_classification_replicate(
    *,
    seed: int,
    dataset: str,
    model: str,
    n_train: int,
    n_cal: int,
    n_test: int,
    p: int,
    K: int,
    eps: float,
    alpha: float,
    trim_quantiles: Iterable[float],
    indexing: str = "repo",
    include_oracle: bool = True,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = n_train + n_cal + n_test
    if dataset == "logistic":
        X, y_clean = generate_logistic(rng, n=n, p=p, K=K)
    elif dataset == "hypercube":
        X, y_clean = generate_hypercube(rng, n=n, p=p, K=K)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    cal_idx = idx[n_train:n_train+n_cal]
    test_idx = idx[n_train+n_cal:]
    P = make_random_flip(eps, K)
    # Train and calibration labels are noisy; test labels remain clean.
    y_noisy_all = y_clean.copy()
    y_noisy_all[np.r_[train_idx, cal_idx]] = sample_noisy_labels(
        rng, y_clean[np.r_[train_idx, cal_idx]], P
    )

    X_train, y_train_noisy = X[train_idx], y_noisy_all[train_idx]
    X_cal, y_cal_noisy = X[cal_idx], y_noisy_all[cal_idx]
    y_cal_clean = y_clean[cal_idx]
    X_test, y_test_clean = X[test_idx], y_clean[test_idx]

    clf = get_classifier(model, seed=seed)
    clf.fit(X_train, y_train_noisy)
    cal_probs = clf.predict_proba(X_cal)
    test_probs = clf.predict_proba(X_test)
    # Some sklearn classifiers may omit classes in tiny smoke settings.
    if cal_probs.shape[1] != K:
        full_cal = np.zeros((cal_probs.shape[0], K))
        full_test = np.zeros((test_probs.shape[0], K))
        for col, cls in enumerate(clf.classes_):
            full_cal[:, int(cls)] = cal_probs[:, col]
            full_test[:, int(cls)] = test_probs[:, col]
        # Keep probabilities nonzero enough for APS ordering.
        full_cal = np.maximum(full_cal, 1e-12); full_cal /= full_cal.sum(axis=1, keepdims=True)
        full_test = np.maximum(full_test, 1e-12); full_test /= full_test.sum(axis=1, keepdims=True)
        cal_probs, test_probs = full_cal, full_test

    rows: list[dict[str, Any]] = []
    base_extra = {
        "seed": seed,
        "dataset": dataset,
        "model": model,
        "eps": eps,
        "alpha": alpha,
        "n_train": n_train,
        "n_cal": n_cal,
        "n_test": n_test,
        "p": p,
        "K": K,
        "replicate": int(seed),
        "label_noise_rate_realized_cal": float(np.mean(y_cal_noisy != y_cal_clean)),
    }

    # Standard APS split conformal.
    cp_cal = calibrate_aps(cal_probs, y_cal_noisy, alpha, rng, indexing=indexing)
    row = evaluate_qhat("aps_cp_noisy_calibration", cp_cal["qhat"], test_probs, y_test_clean, rng, cp_cal)
    row.update(base_extra)
    row["trim_quantile"] = float("nan")
    rows.append(row)

    # Clarkson CRCP.
    cal_u = rng.uniform(0.0, 1.0, size=n_cal)
    cr = crcp_qhat(cal_probs, y_cal_noisy, alpha=alpha, eps=eps, cal_u=cal_u, indexing=indexing)
    row = evaluate_qhat("clarkson_crcp", cr["qhat"], test_probs, y_test_clean, rng, {"n_keep": n_cal})
    row.update(base_extra)
    row.update({"crcp_C": cr["C"], "crcp_fallback": cr["fallback"], "crcp_index": cr["index"]})
    row["trim_quantile"] = float("nan")
    rows.append(row)

    # NACP comparator, implemented from nacp.py.
    na = nacp_qhat(cal_probs, y_cal_noisy, alpha=alpha, eps=eps)
    row = evaluate_qhat("nacp_aps", na["qhat"], test_probs, y_test_clean, rng, {"n_keep": n_cal})
    row.update(base_extra)
    row.update({"nacp_min_diff": na["min_diff"], "nacp_fallback": na["fallback"]})
    row["trim_quantile"] = float("nan")
    rows.append(row)

    # Covariate-trimmed APS diagnostic, analogous to retained calibration trimming.
    center = X_train.mean(axis=0)
    cov = np.cov(X_train, rowvar=False)
    train_mdist = mahalanobis_scores(X_train, center, cov)
    cal_mdist = mahalanobis_scores(X_cal, center, cov)
    for q in trim_quantiles:
        threshold = float(np.quantile(train_mdist, q))
        keep = cal_mdist <= threshold
        cal = calibrate_aps(cal_probs, y_cal_noisy, alpha, rng, keep_mask=keep, indexing=indexing)
        extra = dict(cal)
        extra.update(
            {
                "trim_threshold": threshold,
                "retained_fraction": float(np.mean(keep)),
                "dirty_label_fraction_retained": float(np.mean((y_cal_noisy != y_cal_clean)[keep])) if np.any(keep) else float("nan"),
            }
        )
        row = evaluate_qhat("covariate_trimmed_aps_mahalanobis", cal["qhat"], test_probs, y_test_clean, rng, extra)
        row.update(base_extra)
        row["trim_quantile"] = float(q)
        rows.append(row)

    # Oracle clean-label retention diagnostic.
    if include_oracle:
        keep = y_cal_noisy == y_cal_clean
        cal = calibrate_aps(cal_probs, y_cal_noisy, alpha, rng, keep_mask=keep, indexing=indexing)
        extra = dict(cal)
        extra.update(
            {
                "retained_fraction": float(np.mean(keep)),
                "dirty_label_fraction_retained": 0.0,
            }
        )
        row = evaluate_qhat("oracle_clean_label_aps", cal["qhat"], test_probs, y_test_clean, rng, extra)
        row.update(base_extra)
        row["trim_quantile"] = float("nan")
        rows.append(row)

    return rows


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else [x]


def stable_int(*items: Any, modulus: int = 10_000) -> int:
    text = "|".join(str(x) for x in items)
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % modulus
    return int(value)


def run_classification_group(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seed_base = int(config.get("seed_base", 123))
    n_reps = int(config.get("n_reps", 25))
    alpha = float(config.get("alpha", 0.1))
    indexing = str(config.get("indexing", "repo"))
    trim_quantiles = [float(x) for x in config.get("trim_quantiles", [0.7, 0.8, 0.9, 0.95])]

    datasets = _as_list(config.get("dataset", "logistic"))
    models = _as_list(config.get("model", "log_reg"))
    eps_values = [float(x) for x in _as_list(config.get("eps", 0.2))]
    n_train = int(config.get("n_train", 10000))
    n_cal = int(config.get("n_cal", 10000))
    n_test = int(config.get("n_test", 10000))
    p = int(config.get("p", 10))
    K = int(config.get("K", 5))
    include_oracle = bool(config.get("include_oracle", True))

    for dataset in datasets:
        for model in models:
            for eps in eps_values:
                for rep in range(n_reps):
                    seed = seed_base + 100000 * rep + 1000 * int(round(eps * 1000)) + stable_int(dataset, model, modulus=997)
                    start = time.perf_counter()
                    rep_rows = run_single_classification_replicate(
                        seed=seed,
                        dataset=dataset,
                        model=model,
                        n_train=n_train,
                        n_cal=n_cal,
                        n_test=n_test,
                        p=p,
                        K=K,
                        eps=eps,
                        alpha=alpha,
                        trim_quantiles=trim_quantiles,
                        indexing=indexing,
                        include_oracle=include_oracle,
                    )
                    for row in rep_rows:
                        row["runtime_seconds_replicate"] = time.perf_counter() - start
                        row["replicate"] = rep
                    rows.extend(rep_rows)

    raw = pd.DataFrame(rows)
    raw_path = output_dir / "clarkson_direct_classification_raw.csv"
    raw.to_csv(raw_path, index=False)
    tables = aggregate_results(raw, alpha=alpha)
    summary_path = output_dir / "clarkson_direct_classification_summary.csv"
    valid_path = output_dir / "clarkson_direct_classification_valid_shortest.csv"
    tables["summary"].to_csv(summary_path, index=False)
    tables["valid_shortest"].to_csv(valid_path, index=False)
    metadata = {
        "config": config,
        "num_raw_rows": int(len(raw)),
        "num_summary_rows": int(len(tables["summary"])),
        "paper_comparison": "Clarkson et al. (2024) CRCP synthetic classification setup",
    }
    meta_path = output_dir / "clarkson_direct_classification_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"raw": raw_path, "summary": summary_path, "valid_shortest": valid_path, "metadata": meta_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run direct Clarkson et al. CRCP classification comparison.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file.", default=Path("configs/group_z_clarkson_direct.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/clarkson_direct"))
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = run_classification_group(config, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
