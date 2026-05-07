
from __future__ import annotations

"""
Task-compatible regression comparison with Clarkson et al. (2024).

Clarkson et al.'s public repository contains a simple regression experiment
(cp.py) in which split conformal regression is calibrated on contaminated
residuals and evaluated on clean test points.  It does not implement CRCP for
regression; CRCP is a classification-label-noise adjustment.  This module
therefore compares:

  * clarkson_standard_split: ordinary split conformal regression calibrated on
    the full contaminated calibration split, matching cp.py's regression object.
  * trimmed_mahalanobis_split: covariate-anomaly trimming before residual
    calibration, representing the present paper's trimming mechanism.
  * trimmed_residual_split: residual-score trimming diagnostic.
  * oracle_clean_split: diagnostic benchmark retaining only truly clean
    calibration points.

The experiment is not a claim that Clarkson et al. proposed a regression
competitor beyond ordinary contaminated split conformal.  It is a direct
task-compatible side-by-side on the regression contamination setup.
"""

from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression


Array = np.ndarray


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else [x]


def stable_int(*items: Any, modulus: int = 10_000) -> int:
    text = "|".join(str(x) for x in items)
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % modulus
    return int(value)


def conformal_tau(scores: Array, alpha: float, indexing: str = "standard") -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    if scores.size == 0:
        return float("inf")
    if indexing == "repo":
        idx = int(np.clip(math.ceil((scores.size + 1) * (1.0 - alpha)), 0, scores.size - 1))
    else:
        idx = int(np.clip(math.ceil((scores.size + 1) * (1.0 - alpha)) - 1, 0, scores.size - 1))
    return float(scores[idx])


def coverage_width(model: LinearRegression, X: Array, y: Array, tau: float) -> tuple[float, float]:
    pred = model.predict(X)
    covered = np.abs(y - pred) <= tau
    return float(np.mean(covered)), float(2.0 * tau)


def mahalanobis_scores(X: Array, center: Array, covariance: Array) -> Array:
    covariance = 0.5 * (covariance + covariance.T) + 1e-6 * np.eye(covariance.shape[0])
    inv = np.linalg.pinv(covariance)
    diff = X - center
    return np.sqrt(np.maximum(np.sum((diff @ inv) * diff, axis=1), 0.0))


def generate_clean_regression(
    rng: np.random.Generator,
    n: int,
    p: int,
    beta: Array,
    sigma: float,
) -> tuple[Array, Array]:
    X = rng.normal(0.0, 1.0, size=(n, p))
    y = X @ beta + rng.normal(0.0, sigma, size=n)
    return X, y


def generate_dirty_regression(
    rng: np.random.Generator,
    n: int,
    p: int,
    beta: Array,
    sigma: float,
    family: str,
    shift: float,
    residual_shift: float,
    residual_scale: float,
) -> tuple[Array, Array]:
    if family == "response_shift":
        X = rng.normal(0.0, 1.0, size=(n, p))
        eps = rng.normal(residual_shift, residual_scale * sigma, size=n)
    elif family == "covariate_shift":
        X = rng.normal(shift, 1.0, size=(n, p))
        eps = rng.normal(0.0, sigma, size=n)
    elif family == "high_leverage_response":
        X = rng.normal(shift, 1.0, size=(n, p))
        eps = rng.normal(residual_shift, residual_scale * sigma, size=n)
    elif family == "small_residual":
        X = rng.normal(shift, 1.0, size=(n, p))
        eps = rng.normal(0.0, 0.15 * sigma, size=n)
    else:
        raise ValueError(f"Unknown dirty_family {family!r}")
    y = X @ beta + eps
    return X, y


def sample_calibration_mixture(
    rng: np.random.Generator,
    n: int,
    p: int,
    beta: Array,
    sigma: float,
    epsilon: float,
    dirty_family: str,
    shift: float,
    residual_shift: float,
    residual_scale: float,
) -> tuple[Array, Array, Array]:
    dirty = rng.uniform(size=n) < epsilon
    X = np.empty((n, p), dtype=float)
    y = np.empty(n, dtype=float)
    n_dirty = int(dirty.sum())
    n_clean = n - n_dirty
    if n_clean:
        Xc, yc = generate_clean_regression(rng, n_clean, p, beta, sigma)
        X[~dirty] = Xc
        y[~dirty] = yc
    if n_dirty:
        Xd, yd = generate_dirty_regression(
            rng, n_dirty, p, beta, sigma, dirty_family, shift, residual_shift, residual_scale
        )
        X[dirty] = Xd
        y[dirty] = yd
    return X, y, dirty


def run_single_regression_replicate(
    *,
    seed: int,
    n_train: int,
    n_cal: int,
    n_test: int,
    p: int,
    epsilon: float,
    alpha: float,
    dirty_family: str,
    shift: float,
    residual_shift: float,
    residual_scale: float,
    sigma: float,
    trim_quantiles: Iterable[float],
    indexing: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 1.0, size=p)
    X_train, y_train = generate_clean_regression(rng, n_train, p, beta, sigma)
    model = LinearRegression().fit(X_train, y_train)
    X_cal, y_cal, dirty = sample_calibration_mixture(
        rng, n_cal, p, beta, sigma, epsilon, dirty_family, shift, residual_shift, residual_scale
    )
    X_test_clean, y_test_clean = generate_clean_regression(rng, n_test, p, beta, sigma)
    X_test_mix, y_test_mix, _ = sample_calibration_mixture(
        rng, n_test, p, beta, sigma, epsilon, dirty_family, shift, residual_shift, residual_scale
    )
    residuals = np.abs(y_cal - model.predict(X_cal))
    train_residuals = np.abs(y_train - model.predict(X_train))
    center = X_train.mean(axis=0)
    covariance = np.cov(X_train, rowvar=False)
    train_mdist = mahalanobis_scores(X_train, center, covariance)
    cal_mdist = mahalanobis_scores(X_cal, center, covariance)
    rows = []
    base = {
        "seed": seed,
        "replicate": seed,
        "epsilon": epsilon,
        "alpha": alpha,
        "n_train": n_train,
        "n_cal": n_cal,
        "n_test": n_test,
        "p": p,
        "dirty_family": dirty_family,
        "shift": shift,
        "residual_shift": residual_shift,
        "residual_scale": residual_scale,
        "realized_dirty_fraction": float(np.mean(dirty)),
    }

    def add_method(method_id: str, keep: Array, trim_quantile: float | None, trim_threshold: float | None) -> None:
        tau = conformal_tau(residuals[keep], alpha, indexing=indexing)
        clean_cov, width = coverage_width(model, X_test_clean, y_test_clean, tau)
        mix_cov, _ = coverage_width(model, X_test_mix, y_test_mix, tau)
        n_keep = int(np.sum(keep))
        dirty_retained = float(np.mean(dirty[keep])) if n_keep else float("nan")
        rows.append({
            **base,
            "method_id": method_id,
            "trim_quantile": float(trim_quantile) if trim_quantile is not None else float("nan"),
            "trim_threshold": float(trim_threshold) if trim_threshold is not None else float("nan"),
            "n_keep": n_keep,
            "retained_fraction": float(np.mean(keep)),
            "dirty_fraction_retained": dirty_retained,
            "tau": tau,
            "clean_coverage": clean_cov,
            "mixture_coverage": mix_cov,
            "width": width,
        })

    add_method("clarkson_standard_split", np.ones(n_cal, dtype=bool), None, None)

    # Clarkson repo's regression object does not provide a remedy.  The following
    # are direct trimmed-comparison methods for the present paper.
    for q in trim_quantiles:
        thr = float(np.quantile(train_mdist, q))
        add_method("trimmed_mahalanobis_split", cal_mdist <= thr, q, thr)
    for q in trim_quantiles:
        thr = float(np.quantile(train_residuals, q))
        add_method("trimmed_residual_split", residuals <= thr, q, thr)

    add_method("oracle_clean_split", ~dirty, None, 0.0)
    return rows


def aggregate(raw: pd.DataFrame, alpha: float) -> dict[str, pd.DataFrame]:
    group_cols = ["dirty_family", "epsilon", "method_id", "trim_quantile"]
    metrics = ["clean_coverage", "mixture_coverage", "width", "retained_fraction", "dirty_fraction_retained", "tau", "n_keep"]
    rows = []
    for key, sub in raw.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_reps"] = int(sub["replicate"].nunique())
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_sd"] = float(vals.std(ddof=1)) if vals.count() > 1 else 0.0
            row[f"{m}_se"] = float(row[f"{m}_sd"] / math.sqrt(max(vals.count(), 1)))
            row[f"{m}_ci_low"] = float(row[f"{m}_mean"] - 1.96 * row[f"{m}_se"])
            row[f"{m}_ci_high"] = float(row[f"{m}_mean"] + 1.96 * row[f"{m}_se"])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["dirty_family", "epsilon", "method_id", "trim_quantile"], na_position="last")
    highlights = []
    target = 1.0 - alpha - 0.005
    for key, sub in summary.groupby(["dirty_family", "epsilon"], dropna=False):
        non_oracle = sub[~sub["method_id"].str.contains("oracle", case=False, na=False)].copy()
        valid = non_oracle[non_oracle["clean_coverage_mean"] >= target]
        if valid.empty:
            chosen = non_oracle.sort_values(["clean_coverage_mean", "width_mean"], ascending=[False, True]).head(1).copy()
            chosen["selection_rule"] = f"fallback_max_coverage_no_method_reached_{target:.3f}"
        else:
            chosen = valid.sort_values(["width_mean", "clean_coverage_mean"], ascending=[True, False]).head(1).copy()
            chosen["selection_rule"] = f"valid_shortest_nonoracle_coverage>={target:.3f}"
        highlights.append(chosen)
    return {
        "summary": summary,
        "valid_shortest": pd.concat(highlights, ignore_index=True) if highlights else pd.DataFrame(),
    }


def run_regression_group(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_base = int(config.get("seed_base", 321))
    n_reps = int(config.get("n_reps", 30))
    n_train = int(config.get("n_train", 500))
    n_cal = int(config.get("n_cal", 500))
    n_test = int(config.get("n_test", 2500))
    p = int(config.get("p", 5))
    alpha = float(config.get("alpha", 0.1))
    sigma = float(config.get("sigma", 1.0))
    eps_values = [float(x) for x in _as_list(config.get("epsilon", 0.2))]
    dirty_families = _as_list(config.get("dirty_family", "high_leverage_response"))
    shift = float(config.get("shift", 3.0))
    residual_shift = float(config.get("residual_shift", 3.0))
    residual_scale = float(config.get("residual_scale", 1.0))
    trim_quantiles = [float(x) for x in config.get("trim_quantiles", [0.7, 0.8, 0.9, 0.95])]
    indexing = str(config.get("indexing", "standard"))
    rows = []
    for dirty_family in dirty_families:
        for eps in eps_values:
            for rep in range(n_reps):
                seed = seed_base + rep * 100000 + int(round(eps * 1000)) * 100 + stable_int(dirty_family, modulus=997)
                rows.extend(run_single_regression_replicate(
                    seed=seed,
                    n_train=n_train,
                    n_cal=n_cal,
                    n_test=n_test,
                    p=p,
                    epsilon=eps,
                    alpha=alpha,
                    dirty_family=dirty_family,
                    shift=shift,
                    residual_shift=residual_shift,
                    residual_scale=residual_scale,
                    sigma=sigma,
                    trim_quantiles=trim_quantiles,
                    indexing=indexing,
                ))
    raw = pd.DataFrame(rows)
    raw_path = output_dir / "clarkson_regression_raw.csv"
    raw.to_csv(raw_path, index=False)
    tables = aggregate(raw, alpha=alpha)
    summary_path = output_dir / "clarkson_regression_summary.csv"
    valid_path = output_dir / "clarkson_regression_valid_shortest.csv"
    tables["summary"].to_csv(summary_path, index=False)
    tables["valid_shortest"].to_csv(valid_path, index=False)
    meta_path = output_dir / "clarkson_regression_metadata.json"
    meta_path.write_text(json.dumps({"config": config, "num_raw_rows": len(raw)}, indent=2), encoding="utf-8")
    return {"raw": raw_path, "summary": summary_path, "valid_shortest": valid_path, "metadata": meta_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run task-compatible regression comparison with Clarkson et al.'s regression baseline.")
    parser.add_argument("--config", type=Path, required=True, default=Path("configs/group_z_clarkson_regression.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/clarkson_regression"))
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = run_regression_group(config, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
