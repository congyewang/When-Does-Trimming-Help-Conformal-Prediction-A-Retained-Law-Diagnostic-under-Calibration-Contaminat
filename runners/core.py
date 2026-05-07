from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
from experiments.baselines.api import (
    BaselineContext,
    execute_baseline,
    is_baseline_method,
)
from experiments.baselines.methods import matched_random_keep_mask
from experiments.utils.bounds import (
    additive_mixture_lower_bound,
    bias_covariance_bound,
    bias_tv_bound,
    bridge_delta_s,
    clean_tail_upper,
    dirty_second_moment_lower,
    dirty_tail_lower,
    empirical_retained_contamination_upper,
    localized_clean_retention_lower,
    localized_dirty_retention_upper,
    localized_dirty_second_moment_lower,
    localized_dirty_tail_lower,
    finite_sample_remainder,
    proposition_lower_bound,
    quantile_shift_bound,
    retained_count_lower,
    retained_count_upper,
    theorem_lower_bound,
)
from experiments.utils.config import ExperimentSpec
from experiments.utils.conformal import calibrate_from_scores, coverage_and_width
from experiments.utils.data import (
    clean_sampler_for_spec,
    dirty_sampler_for_spec,
    sample_mixture,
)
from experiments.utils.io import (
    ensure_dir,
    write_json,
    write_rows_csv,
    write_rows_parquet,
)
from experiments.utils.metrics import (
    cdf_excess,
    clean_dirty_retention,
    clipped_probability,
    covariance_bias_summary,
    dkw_radius,
    estimate_local_density,
    hoeffding_radius,
    kolmogorov_distance,
    left_quantile,
    safe_gap,
    weighted_cdf_excess,
)
from experiments.utils.models import build_predictor
from experiments.utils.seeding import derive_seed, make_rng
from experiments.utils.stein import (
    build_evaluation_grid,
    fit_gaussian_mixture_stein_model,
    fit_gaussian_stein_model,
    fit_gmm_mahalanobis_model,
    fit_kde_stein_model,
    fit_two_component_gmm_stein_model,
    stein_feature_envelope,
)
Array = np.ndarray
TRUE_CLEAN_MEAN = 0.0
TRUE_CLEAN_VARIANCE = 1.0
SUPPORTED_STANDARD_GAUSSIAN_SCORE_DGP_FAMILIES = {
    "tradeoff",
    "high_leverage",
    "response_contamination",
    "heavy_tail",
    "gaussian_homoscedastic",
    "heavy_tail_clean",
    "bimodal_clean",
    "low_density_near_quantile",
}
SUPPORTED_BIMODAL_ORACLE_SCORE_DGP_FAMILIES = {
    "bimodal_covariate",
}
@dataclass(frozen=True)
class ThresholdSpec:
    label: str
    quantile: float
    value: float
    source: str
def _threshold_specs(
    spec: ExperimentSpec,
    train_stein_scores: Array,
    reference_stein_scores: Array,
    population_metrics: dict[str, float] | None,
) -> list[ThresholdSpec]:
    if spec.threshold_policy == "fixed_grid":
        return [
            ThresholdSpec(
                label=f"q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(train_stein_scores, quantile)),
                source="train_stein_quantile",
            )
            for quantile in spec.threshold_quantiles
        ]
    if spec.threshold_policy == "clean_reference_only":
        quantile = float(spec.extras.get("reference_quantile", 0.95))
        return [
            ThresholdSpec(
                label=f"clean_ref_q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(reference_stein_scores, quantile)),
                source="clean_reference_quantile",
            )
        ]
    if spec.threshold_policy == "ksd_geometric":
        if population_metrics is None:
            raise ValueError(
                "Population metrics are required for the ksd_geometric threshold policy."
            )
        sigma_c_sq = population_metrics["sigma_c_sq"]
        underline_m_d = population_metrics["underline_m_d"]
        if underline_m_d <= sigma_c_sq or sigma_c_sq <= 0.0:
            fallback = float(np.quantile(train_stein_scores, 0.99))
            return [
                ThresholdSpec(
                    label="ksd_fallback_q=0.990",
                    quantile=0.99,
                    value=fallback,
                    source="ksd_fallback",
                )
            ]
        threshold = float((sigma_c_sq * underline_m_d) ** 0.25)
        return [
            ThresholdSpec(
                label="ksd_geometric",
                quantile=float("nan"),
                value=threshold,
                source="ksd_geometric",
            )
        ]
    if spec.threshold_policy == "ksd_geometric_localized":
        if population_metrics is None:
            raise ValueError(
                "Population metrics are required for the ksd_geometric_localized threshold policy."
            )
        sigma_c_sq = population_metrics["sigma_c_sq"]
        underline_m_d0 = population_metrics["underline_m_d0"]
        q_q = population_metrics["q_Q"]
        if q_q <= 0.0 or underline_m_d0 <= q_q * sigma_c_sq or sigma_c_sq <= 0.0:
            fallback = float(np.quantile(train_stein_scores, 0.99))
            return [
                ThresholdSpec(
                    label="ksd_local_fallback_q=0.990",
                    quantile=0.99,
                    value=fallback,
                    source="ksd_local_fallback",
                )
            ]
        threshold = float(((sigma_c_sq * underline_m_d0) / q_q) ** 0.25)
        return [
            ThresholdSpec(
                label="ksd_geometric_localized",
                quantile=float("nan"),
                value=threshold,
                source="ksd_geometric_localized",
            )
        ]
    if spec.threshold_policy == "oracle":
        return [
            ThresholdSpec(
                label=f"oracle_candidate_q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(train_stein_scores, quantile)),
                source="oracle_grid_candidate",
            )
            for quantile in spec.threshold_quantiles
        ]
    raise ValueError(f"Unknown threshold policy: {spec.threshold_policy}")
def _population_metrics(
    spec: ExperimentSpec,
    stein_model,
    clean_sampler,
    clean_structure_x: Array,
    dirty_structure_x: Array,
    mix_structure_x: Array,
    population_seed: int,
) -> dict[str, Any]:
    pop_rng = np.random.default_rng(population_seed)
    clean_population = clean_sampler(pop_rng, spec.population_sample_size)
    eval_grid = build_evaluation_grid(
        clean_structure_x, dirty_structure_x, spec.evaluation_grid_size
    )
    empirical_eval = stein_model.empirical_score(eval_grid)
    population_eval = stein_model.population_score(eval_grid, clean_population.x)
    clean_population_score = stein_model.population_score(
        clean_structure_x, clean_population.x
    )
    dirty_population_score = stein_model.population_score(
        dirty_structure_x, clean_population.x
    )
    mixture_population_score = stein_model.population_score(
        mix_structure_x, clean_population.x
    )
    clean_empirical_score = stein_model.empirical_score(clean_structure_x)
    support_points = np.concatenate(
        [clean_population.x, clean_structure_x[: min(clean_structure_x.size, 128)]]
    )
    b_xi, l_xi = stein_feature_envelope(stein_model, eval_grid, support_points)
    delta_s = bridge_delta_s(
        b_xi=b_xi,
        l_xi=l_xi,
        rho=spec.bridge_rho,
        delta=spec.bridge_delta,
        ell=spec.clean_reference_size,
        dimension=1,
        covering_number=spec.evaluation_grid_size,
    )
    # This "true" score diagnostic uses the standard-Gaussian clean x law. The
    true_score_clean = _oracle_clean_score_for_spec(spec, clean_structure_x)
    estimated_score_clean = stein_model.score_function(clean_structure_x)
    e_s = float(np.max(np.abs(estimated_score_clean - true_score_clean)))
    b_p = float(np.max(np.abs(true_score_clean)))
    b_s = float(np.max(np.abs(estimated_score_clean)))
    l_stein = stein_model.kappa0 * b_s + stein_model.kappa1
    sigma_c_hp_sq = (
        2.0 * (stein_model.kappa0**2) * (e_s**2)
        + 2.0
        * ((stein_model.kappa0 * b_p + stein_model.kappa1) ** 2)
        / spec.clean_reference_size
        + (l_stein**2)
        * np.sqrt((8.0 * np.log(1.0 / spec.bridge_delta)) / spec.clean_reference_size)
    )
    m_c_pop = float(np.mean(clean_population_score**2))
    d_sq = float(np.mean(mixture_population_score**2))
    underline_m_d = dirty_second_moment_lower(
        d_sq=d_sq,
        sigma_c_sq=m_c_pop,
        epsilon=spec.epsilon,
    )
    score_bound = float(
        np.max(
            np.concatenate(
                [
                    empirical_eval,
                    population_eval,
                    np.abs(clean_empirical_score),
                    np.abs(clean_population_score),
                    np.abs(dirty_population_score),
                ]
            )
        )
    )
    working_radius = _extra_float(spec, "working_radius", 3.0)
    fit_mean_array = np.asarray(stein_model.fit_mean, dtype=float).reshape(-1)
    fit_variance_array = np.asarray(stein_model.fit_variance, dtype=float).reshape(-1)
    working_center = _extra_float(spec, "working_center", float(fit_mean_array[0]))
    working_scale = _extra_float(spec, "working_scale", float(np.sqrt(max(float(fit_variance_array[0]), 1e-12))))
    working_half_width = working_radius * max(working_scale, 1e-12)
    def in_working_set(x: Array) -> Array:
        x_matrix = _feature_matrix(x)
        if x_matrix.shape[1] == 1:
            return np.abs(x_matrix[:, 0] - working_center) <= working_half_width
        center = np.mean(x_matrix, axis=0)
        return np.sqrt(np.sum((x_matrix - center) ** 2, axis=1)) <= working_half_width
    clean_u0_mask = in_working_set(clean_structure_x)
    dirty_u0_mask = in_working_set(dirty_structure_x)
    mix_u0_mask = in_working_set(mix_structure_x)
    eval_u0_mask = in_working_set(eval_grid)
    zeta_p = float(1.0 - np.mean(clean_u0_mask))
    zeta_q = float(1.0 - np.mean(dirty_u0_mask))
    q_q = float(1.0 - zeta_q)
    d0_sq = float(np.mean((mixture_population_score**2) * mix_u0_mask))
    m_c0 = float(np.mean((clean_population_score**2) * clean_u0_mask))
    m_d0_observed = float(np.mean((dirty_population_score**2) * dirty_u0_mask))
    underline_m_d0 = localized_dirty_second_moment_lower(
        d0_sq=d0_sq,
        sigma_c_sq=m_c_pop,
        epsilon=spec.epsilon,
    )
    local_score_arrays = []
    if np.any(eval_u0_mask):
        local_score_arrays.append(np.abs(population_eval[eval_u0_mask]))
    if np.any(clean_u0_mask):
        local_score_arrays.append(np.abs(clean_population_score[clean_u0_mask]))
    if np.any(dirty_u0_mask):
        local_score_arrays.append(np.abs(dirty_population_score[dirty_u0_mask]))
    if np.any(mix_u0_mask):
        local_score_arrays.append(np.abs(mixture_population_score[mix_u0_mask]))
    local_score_bound = (
        float(np.max(np.concatenate(local_score_arrays)))
        if local_score_arrays
        else float("nan")
    )
    combined_x = np.concatenate([clean_structure_x, dirty_structure_x])
    combined_score = np.concatenate([clean_population_score, dirty_population_score])
    fit_mean_array = np.asarray(stein_model.fit_mean, dtype=float).reshape(-1)
    fit_variance_array = np.asarray(stein_model.fit_variance, dtype=float).reshape(-1)
    mahalanobis_score = _mahalanobis_scores(combined_x, fit_mean_array, np.sqrt(np.maximum(fit_variance_array, 1e-12)))
    score_rank_spearman_mahalanobis = _rank_correlation(
        combined_score, mahalanobis_score
    )
    return {
        "evaluation_grid_min": float(eval_grid.min()),
        "evaluation_grid_max": float(eval_grid.max()),
        "sup_score_error": float(np.max(np.abs(empirical_eval - population_eval))),
        "score_model_sup_error": float(e_s),
        "score_model_mse": float(
            np.mean((estimated_score_clean - true_score_clean) ** 2)
        ),
        "score_fit_mean": float(stein_model.fit_mean),
        "score_fit_variance": float(stein_model.fit_variance),
        "score_bandwidth": float(stein_model.bandwidth),
        "b_xi": b_xi,
        "l_xi": l_xi,
        "delta_s": delta_s,
        "clean_population_score": clean_population_score,
        "dirty_population_score": dirty_population_score,
        "mixture_population_score": mixture_population_score,
        "clean_empirical_second_moment": float(np.mean(clean_empirical_score**2)),
        "clean_population_second_moment": m_c_pop,
        "clean_second_moment_hp_bound": float(sigma_c_hp_sq),
        "d_sq": d_sq,
        "underline_m_d": float(underline_m_d),
        "score_bound": score_bound,
        "empirical_eval_max": float(empirical_eval.max()),
        "population_eval_max": float(population_eval.max()),
        "sigma_c_sq": m_c_pop,
        "working_radius": working_radius,
        "working_center": working_center,
        "working_scale": working_scale,
        "zeta_P": zeta_p,
        "zeta_Q": zeta_q,
        "q_Q": q_q,
        "D0_sq": d0_sq,
        "m_c0": m_c0,
        "m_d0_observed": m_d0_observed,
        "underline_m_d0": float(underline_m_d0),
        "local_score_bound": local_score_bound,
        "score_rank_spearman_mahalanobis": score_rank_spearman_mahalanobis,
    }
def _row_base(
    spec: ExperimentSpec, replicate: int, method_id: str, threshold: ThresholdSpec
) -> dict[str, Any]:
    row = {
        "group_id": spec.group_id,
        "config_id": spec.config_id,
        "experiment_title": spec.title,
        "description": spec.description,
        "replicate": replicate,
        "method_id": method_id,
        "threshold_policy": spec.threshold_policy,
        "threshold_label": threshold.label,
        "threshold_quantile": threshold.quantile,
        "threshold_value": threshold.value,
        "threshold_source": threshold.source,
        "threshold_is_fallback": float("fallback" in threshold.label or "fallback" in threshold.source),
        "threshold_is_oracle_selected": float(threshold.source == "oracle_selected"),
        "threshold_is_adaptive_policy": float(
            threshold.source in {
                "ksd_geometric",
                "ksd_geometric_localized",
                "ksd_fallback",
                "ksd_local_fallback",
            }
        ),
        "alpha": spec.alpha,
        "epsilon": spec.epsilon,
        "train_size": spec.train_size,
        "calibration_size": spec.calibration_size,
        "clean_reference_size": spec.clean_reference_size,
        "clean_test_size": spec.clean_test_size,
        "mixture_test_size": spec.mixture_test_size,
        "structure_sample_size": spec.structure_sample_size,
        "threshold_policy_name": spec.threshold_policy,
        "mc_reps": spec.mc_reps,
        "random_seed": spec.random_seed,
        "dgp_family": spec.dgp_family,
        "contamination_family": spec.contamination_family,
        "predictor_type": spec.predictor_type,
        "representation_family": spec.representation_family,
        "eta_fraction": spec.eta_fraction,
        "bridge_delta": spec.bridge_delta,
        "bridge_rho": spec.bridge_rho,
        "population_sample_size": spec.population_sample_size,
        "evaluation_grid_size": spec.evaluation_grid_size,
        "compute_population_metrics": spec.compute_population_metrics,
        "include_mixture_eval": spec.include_mixture_eval,
        "notes": spec.notes,
    }
    for key, value in spec.extras.items():
        row[key] = value
    return row
def _extra_float(spec: ExperimentSpec, key: str, default: float) -> float:
    value = spec.extras.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
def _rank_correlation(left: Array, right: Array) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 3:
        return float("nan")
    left_rank = np.argsort(np.argsort(left)).astype(float)
    right_rank = np.argsort(np.argsort(right)).astype(float)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denom = float(np.sqrt(np.sum(left_rank**2) * np.sum(right_rank**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(left_rank * right_rank) / denom)
def _feature_matrix(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x.reshape(x.shape[0], -1)
def _score_norm_scores(score_model: Any, x: Array) -> Array:
    score = np.asarray(score_model.score_function(x), dtype=float)
    if score.ndim == 1:
        return np.abs(score)
    return np.sqrt(np.sum(score**2, axis=1))
def _mahalanobis_scores(x: Array, center: Array | float, scale: Array | float) -> Array:
    x_matrix = _feature_matrix(x)
    center_array = np.asarray(center, dtype=float)
    center_vector = np.full(x_matrix.shape[1], float(center_array), dtype=float) if center_array.ndim == 0 else center_array.reshape(-1)[: x_matrix.shape[1]]
    diff = x_matrix - center_vector
    scale_array = np.asarray(scale, dtype=float)
    if scale_array.ndim == 0:
        scalar_scale = max(float(scale_array), 1e-12)
        return np.sqrt(np.sum((diff / scalar_scale) ** 2, axis=1))
    if scale_array.ndim == 1:
        scale_vector = np.maximum(scale_array.reshape(-1)[: x_matrix.shape[1]], 1e-12)
        return np.sqrt(np.sum((diff / scale_vector) ** 2, axis=1))
    covariance = 0.5 * (scale_array + scale_array.T) + 1e-6 * np.eye(scale_array.shape[0])
    inverse = np.linalg.pinv(covariance)
    return np.sqrt(np.maximum(np.sum((diff @ inverse) * diff, axis=1), 0.0))
def _mahalanobis_threshold_specs(
    spec: ExperimentSpec,
    train_mahalanobis_scores: Array,
    reference_mahalanobis_scores: Array,
) -> list[ThresholdSpec]:
    if spec.threshold_policy == "fixed_grid":
        return [
            ThresholdSpec(
                label=f"mahal_q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(train_mahalanobis_scores, quantile)),
                source="train_mahalanobis_quantile",
            )
            for quantile in spec.threshold_quantiles
        ]
    if spec.threshold_policy == "clean_reference_only":
        quantile = float(spec.extras.get("reference_quantile", 0.95))
        return [
            ThresholdSpec(
                label=f"mahal_clean_ref_q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(reference_mahalanobis_scores, quantile)),
                source="clean_reference_mahalanobis_quantile",
            )
        ]
    if spec.threshold_policy == "oracle":
        return [
            ThresholdSpec(
                label=f"mahal_oracle_candidate_q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(train_mahalanobis_scores, quantile)),
                source="mahalanobis_oracle_grid_candidate",
            )
            for quantile in spec.threshold_quantiles
        ]
    raise ValueError(
        "mahalanobis_trim currently supports threshold_policy in "
        "{'fixed_grid', 'clean_reference_only', 'oracle'}; "
        f"received {spec.threshold_policy!r}."
    )
def _generic_score_threshold_specs(
    spec: ExperimentSpec,
    train_scores: Array,
    reference_scores: Array,
    *,
    label_prefix: str,
    source_prefix: str,
) -> list[ThresholdSpec]:
    if spec.threshold_policy == "fixed_grid":
        return [
            ThresholdSpec(
                label=f"{label_prefix}_q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(train_scores, quantile)),
                source=f"train_{source_prefix}_quantile",
            )
            for quantile in spec.threshold_quantiles
        ]
    if spec.threshold_policy == "clean_reference_only":
        quantile = float(spec.extras.get("reference_quantile", 0.95))
        return [
            ThresholdSpec(
                label=f"{label_prefix}_clean_ref_q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(reference_scores, quantile)),
                source=f"clean_reference_{source_prefix}_quantile",
            )
        ]
    if spec.threshold_policy == "oracle":
        return [
            ThresholdSpec(
                label=f"{label_prefix}_oracle_candidate_q={quantile:.3f}",
                quantile=quantile,
                value=float(np.quantile(train_scores, quantile)),
                source=f"{source_prefix}_oracle_grid_candidate",
            )
            for quantile in spec.threshold_quantiles
        ]
    raise ValueError(
        f"{label_prefix} thresholds currently support threshold_policy in "
        "{'fixed_grid', 'clean_reference_only', 'oracle'}; "
        f"received {spec.threshold_policy!r}."
    )
def _oracle_clean_score_for_spec(spec: ExperimentSpec, x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if spec.dgp_family in SUPPORTED_STANDARD_GAUSSIAN_SCORE_DGP_FAMILIES:
        return -(x - TRUE_CLEAN_MEAN) / TRUE_CLEAN_VARIANCE
    if spec.dgp_family == "bimodal_covariate":
        weights = np.asarray([0.5, 0.5], dtype=float)
        means = np.asarray([-2.0, 2.0], dtype=float)
        variances = np.asarray([0.49, 0.49], dtype=float)
        log_terms = []
        component_scores = []
        for weight, mean, variance in zip(weights, means, variances):
            log_density = (
                np.log(max(float(weight), 1e-12))
                - 0.5 * np.log(float(variance))
                - 0.5 * ((x - mean) ** 2) / float(variance)
            )
            log_terms.append(log_density)
            component_scores.append(-(x - mean) / float(variance))
        log_terms_array = np.vstack(log_terms)
        max_log = np.max(log_terms_array, axis=0)
        responsibilities = np.exp(log_terms_array - max_log)
        responsibilities /= np.sum(responsibilities, axis=0, keepdims=True)
        component_scores_array = np.vstack(component_scores)
        return np.sum(responsibilities * component_scores_array, axis=0)
    return -(x - TRUE_CLEAN_MEAN) / TRUE_CLEAN_VARIANCE
def _validate_oracle_score_support(spec: ExperimentSpec) -> None:
    if (
        spec.representation_family != "x_only"
        or spec.dgp_family not in SUPPORTED_STANDARD_GAUSSIAN_SCORE_DGP_FAMILIES
    ):
        supported = ", ".join(sorted(SUPPORTED_STANDARD_GAUSSIAN_SCORE_DGP_FAMILIES))
        raise ValueError(
            "score_model_mode='oracle' is only supported when the clean representation law follows "
            "the standard-Gaussian x-only assumption used by the current Stein diagnostics. "
            f"Received dgp_family={spec.dgp_family!r}, representation_family={spec.representation_family!r}. "
            f"Supported dgp_family values: {supported}."
        )
def _stein_model_for_spec(spec: ExperimentSpec, x_fit: Array, x_reference: Array):
    score_model_mode = str(spec.extras.get("score_model_mode", "well_specified"))
    bandwidth_scale = _extra_float(spec, "bandwidth_scale", 1.0)
    if score_model_mode == "oracle":
        _validate_oracle_score_support(spec)
        return fit_gaussian_stein_model(
            x_fit,
            x_reference,
            bandwidth_scale=bandwidth_scale,
            use_oracle_score=True,
            oracle_mean=TRUE_CLEAN_MEAN,
            oracle_variance=TRUE_CLEAN_VARIANCE,
        )
    if score_model_mode == "oracle_bimodal_mixture":
        if spec.dgp_family not in SUPPORTED_BIMODAL_ORACLE_SCORE_DGP_FAMILIES:
            raise ValueError(
                "score_model_mode='oracle_bimodal_mixture' requires "
                "dgp_family='bimodal_covariate'."
            )
        return fit_gaussian_mixture_stein_model(
            x_reference,
            bandwidth_scale=bandwidth_scale,
            weights=(0.5, 0.5),
            means=(-2.0, 2.0),
            variances=(0.49, 0.49),
        )
    if score_model_mode == "fitted_bimodal_mixture":
        return fit_two_component_gmm_stein_model(
            x_fit,
            x_reference,
            bandwidth_scale=bandwidth_scale,
            max_iter=int(spec.extras.get("gmm_max_iter", 100)),
            min_variance=float(spec.extras.get("gmm_min_variance", 1e-4)),
        )
    if score_model_mode in {"kde_score", "kde_stein", "nonparametric_kde"}:
        return fit_kde_stein_model(
            x_fit,
            x_reference,
            bandwidth_scale=bandwidth_scale,
            score_bandwidth_scale=spec.extras.get("score_bandwidth_scale", None),
        )
    if score_model_mode == "well_specified":
        return fit_gaussian_stein_model(
            x_fit,
            x_reference,
            bandwidth_scale=bandwidth_scale,
        )
    if score_model_mode == "mis_specified":
        return fit_gaussian_stein_model(
            x_fit,
            x_reference,
            bandwidth_scale=bandwidth_scale,
            fit_mean_offset=_extra_float(spec, "fit_mean_offset", 0.0),
            fit_variance_scale=_extra_float(spec, "fit_variance_scale", 1.0),
        )
    raise ValueError(f"Unknown score_model_mode: {score_model_mode}")
def _append_one_sided_transfer_metrics(
    row: dict[str, Any],
    spec: ExperimentSpec,
    threshold_value: float,
    clean_structure_scores: Array,
    dirty_structure_scores: Array,
    clean_structure_nonconformity: Array,
    dirty_structure_nonconformity: Array,
    p_c: float,
    p_d: float,
    mu_keep: float,
    tilde_epsilon: float,
) -> None:
    clean_structure_scores = np.asarray(clean_structure_scores, dtype=float)
    dirty_structure_scores = np.asarray(dirty_structure_scores, dtype=float)
    clean_nonconformity = np.asarray(clean_structure_nonconformity, dtype=float)
    dirty_nonconformity = np.asarray(dirty_structure_nonconformity, dtype=float)
    clean_keep_structure = clean_structure_scores <= threshold_value
    dirty_keep_structure = dirty_structure_scores <= threshold_value
    retained_clean_nonconformity = clean_nonconformity[clean_keep_structure]
    retained_dirty_nonconformity = dirty_nonconformity[dirty_keep_structure]
    delta_trim_plus = cdf_excess(
        retained_clean_nonconformity,
        clean_nonconformity,
        empty_value=1.0,
    )
    d_q_plus_empty_value = 0.0 if p_d <= 0.0 or tilde_epsilon <= 0.0 else 1.0
    d_q_plus = cdf_excess(
        retained_dirty_nonconformity,
        clean_nonconformity,
        empty_value=d_q_plus_empty_value,
    )
    retained_to_clean_plus_direct = weighted_cdf_excess(
        [retained_clean_nonconformity, retained_dirty_nonconformity],
        [1.0 - tilde_epsilon, tilde_epsilon],
        clean_nonconformity,
        empty_value=1.0,
    )
    one_sided_transfer_loss = (1.0 - tilde_epsilon) * delta_trim_plus + tilde_epsilon * d_q_plus
    d_q_symmetric = (
        0.0
        if p_d <= 0.0 or retained_dirty_nonconformity.size == 0
        else kolmogorov_distance(clean_nonconformity, retained_dirty_nonconformity)
    )
    symmetric_transfer_loss_empirical_dq = (1.0 - tilde_epsilon) * row.get("trimming_bias", float("nan")) + tilde_epsilon * d_q_symmetric
    remainder = finite_sample_remainder(mu_keep, spec.calibration_size, spec.eta_fraction)
    lpop_plus = 1.0 - spec.alpha - one_sided_transfer_loss
    lpop_plus_direct = 1.0 - spec.alpha - retained_to_clean_plus_direct
    lpop_symmetric_empirical_dq = 1.0 - spec.alpha - symmetric_transfer_loss_empirical_dq
    theorem_lower_bound_plus = (
        lpop_plus - remainder if isfinite(remainder) else float("-inf")
    )
    theorem_lower_bound_plus_direct = (
        lpop_plus_direct - remainder if isfinite(remainder) else float("-inf")
    )
    theorem_lower_bound_symmetric_empirical_dq = (
        lpop_symmetric_empirical_dq - remainder if isfinite(remainder) else float("-inf")
    )
    audit_delta = _extra_float(spec, "audit_delta", 0.05)
    audit_piece_delta = max(audit_delta / 8.0, 1e-12)
    clean_full_n = int(clean_nonconformity.size)
    clean_retained_n = int(retained_clean_nonconformity.size)
    dirty_retained_n = int(retained_dirty_nonconformity.size)
    pc_lcb = clipped_probability(p_c - hoeffding_radius(clean_structure_scores.size, audit_piece_delta))
    pd_ucb = clipped_probability(p_d + hoeffding_radius(dirty_structure_scores.size, audit_piece_delta))
    audit_mu_keep_lower = (1.0 - spec.epsilon) * pc_lcb
    audit_tilde_epsilon_upper = (
        spec.epsilon * pd_ucb
        / ((1.0 - spec.epsilon) * pc_lcb + spec.epsilon * pd_ucb)
        if ((1.0 - spec.epsilon) * pc_lcb + spec.epsilon * pd_ucb) > 0.0
        else float("nan")
    )
    clean_full_rad = dkw_radius(clean_full_n, audit_piece_delta)
    clean_retained_rad = dkw_radius(clean_retained_n, audit_piece_delta)
    dirty_retained_rad = dkw_radius(dirty_retained_n, audit_piece_delta)
    delta_trim_plus_ucb = clipped_probability(delta_trim_plus + clean_full_rad + clean_retained_rad)
    d_q_plus_ucb = clipped_probability(d_q_plus + clean_full_rad + dirty_retained_rad)
    audit_one_sided_loss_upper = (
        (1.0 - audit_tilde_epsilon_upper) * delta_trim_plus_ucb
        + audit_tilde_epsilon_upper * d_q_plus_ucb
        if isfinite(audit_tilde_epsilon_upper)
        else float("nan")
    )
    audit_lpop_plus_lower = (
        1.0 - spec.alpha - audit_one_sided_loss_upper
        if isfinite(audit_one_sided_loss_upper)
        else float("nan")
    )
    audit_remainder = finite_sample_remainder(
        audit_mu_keep_lower,
        spec.calibration_size,
        spec.eta_fraction,
    )
    audit_theorem_lower_bound_plus = (
        audit_lpop_plus_lower - audit_remainder
        if isfinite(audit_lpop_plus_lower) and isfinite(audit_remainder)
        else float("-inf")
    )
    row.update(
        {
            "n_clean_structure_keep": int(clean_retained_n),
            "n_dirty_structure_keep": int(dirty_retained_n),
            "delta_trim_plus": delta_trim_plus,
            "d_q_plus": d_q_plus,
            "retained_to_clean_plus_direct": retained_to_clean_plus_direct,
            "one_sided_transfer_loss": one_sided_transfer_loss,
            "one_sided_decomposition_slack": one_sided_transfer_loss - retained_to_clean_plus_direct,
            "lpop_plus": lpop_plus,
            "lpop_plus_direct": lpop_plus_direct,
            "theorem_lower_bound_plus": theorem_lower_bound_plus,
            "theorem_lower_bound_plus_direct": theorem_lower_bound_plus_direct,
            "d_q_symmetric": d_q_symmetric,
            "symmetric_transfer_loss_empirical_dq": symmetric_transfer_loss_empirical_dq,
            "lpop_symmetric_empirical_dq": lpop_symmetric_empirical_dq,
            "theorem_lower_bound_symmetric_empirical_dq": theorem_lower_bound_symmetric_empirical_dq,
            "one_sided_dirty_component": tilde_epsilon * d_q_plus,
            "one_sided_clean_component": (1.0 - tilde_epsilon) * delta_trim_plus,
            "d_q_plus_defined": float(dirty_retained_n > 0 or p_d <= 0.0),
            "delta_trim_plus_defined": float(clean_retained_n > 0),
            "audit_delta": audit_delta,
            "audit_p_c_lower": pc_lcb,
            "audit_p_d_upper": pd_ucb,
            "audit_mu_keep_lower": audit_mu_keep_lower,
            "audit_tilde_epsilon_upper": audit_tilde_epsilon_upper,
            "audit_delta_trim_plus_upper": delta_trim_plus_ucb,
            "audit_d_q_plus_upper": d_q_plus_ucb,
            "audit_one_sided_loss_upper": audit_one_sided_loss_upper,
            "audit_lpop_plus_lower": audit_lpop_plus_lower,
            "audit_theorem_lower_bound_plus": audit_theorem_lower_bound_plus,
            "audit_remainder": audit_remainder,
            "audit_clean_full_dkw_radius": clean_full_rad,
            "audit_clean_retained_dkw_radius": clean_retained_rad,
            "audit_dirty_retained_dkw_radius": dirty_retained_rad,
        }
    )

def _append_threshold_metrics(
    row: dict[str, Any],
    spec: ExperimentSpec,
    threshold_value: float,
    keep_mask: Array,
    calibration_dirty: Array,
    clean_structure_scores: Array,
    dirty_structure_scores: Array,
    clean_structure_nonconformity: Array,
    dirty_structure_nonconformity: Array,
    population_metrics: dict[str, Any] | None,
) -> None:
    p_c, p_d, mu_keep, tilde_epsilon = clean_dirty_retention(
        clean_structure_scores,
        dirty_structure_scores,
        threshold_value,
        spec.epsilon,
    )
    retained_clean_nonconformity = clean_structure_nonconformity[
        clean_structure_scores <= threshold_value
    ]
    bias = kolmogorov_distance(
        clean_structure_nonconformity, retained_clean_nonconformity
    )
    direct_bias, covariance_bias = covariance_bias_summary(
        clean_structure_nonconformity, clean_structure_scores <= threshold_value
    )
    level = 1.0 - spec.alpha
    clean_quantile = left_quantile(clean_structure_nonconformity, level)
    retained_quantile = left_quantile(retained_clean_nonconformity, level)
    quantile_shift = abs(retained_quantile - clean_quantile)
    lambda_min_est = estimate_local_density(clean_structure_nonconformity, level)
    theorem_bound = theorem_lower_bound(
        spec.alpha,
        bias,
        tilde_epsilon,
        mu_keep,
        spec.calibration_size,
        spec.eta_fraction,
    )
    proposition_bound = proposition_lower_bound(
        spec.alpha,
        spec.epsilon,
        bias,
        tilde_epsilon,
        mu_keep,
        spec.calibration_size,
        spec.eta_fraction,
    )
    additive_bound = additive_mixture_lower_bound(
        spec.alpha,
        spec.epsilon,
        bias,
        tilde_epsilon,
        mu_keep,
        spec.calibration_size,
        spec.eta_fraction,
    )
    eta = spec.eta_fraction * mu_keep
    pi_c = (1.0 - spec.epsilon) * p_c
    pi_d = spec.epsilon * p_d
    n_clean_keep = int(np.sum((~calibration_dirty) & keep_mask))
    n_dirty_keep = int(np.sum(calibration_dirty & keep_mask))
    empirical_retained_contamination_value = (
        float(n_dirty_keep / max(int(keep_mask.sum()), 1)) if keep_mask.any() else 0.0
    )
    quantile_shift_bound_bias_value = quantile_shift_bound(bias, lambda_min_est)
    quantile_shift_bound_tv_value = quantile_shift_bound(1.0 - p_c, lambda_min_est)
    empirical_retained_contamination_upper_value = (
        empirical_retained_contamination_upper(pi_d, mu_keep, eta)
    )
    retained_contamination_upper_violation_value = (
        float(
            empirical_retained_contamination_value
            > empirical_retained_contamination_upper_value
        )
        if isfinite(empirical_retained_contamination_upper_value)
        else float("nan")
    )
    row.update(
        {
            "n_clean_keep": n_clean_keep,
            "n_dirty_keep": n_dirty_keep,
            "n_keep": int(keep_mask.sum()),
            "retained_fraction": float(np.mean(keep_mask)),
            "empirical_retained_contamination": empirical_retained_contamination_value,
            "p_c": p_c,
            "p_d": p_d,
            "p_c_minus_p_d": p_c - p_d,
            "mu_keep": mu_keep,
            "tilde_epsilon": tilde_epsilon,
            "tilde_epsilon_reduction": spec.epsilon - tilde_epsilon,
            "trimming_bias": bias,
            "trimming_bias_covariance": covariance_bias,
            "trimming_bias_direct": direct_bias,
            "net_tradeoff": (spec.epsilon - tilde_epsilon) - bias,
            "bias_tv_bound": bias_tv_bound(p_c),
            "bias_cov_bound": bias_covariance_bound(p_c),
            "clean_quantile": clean_quantile,
            "retained_clean_quantile": retained_quantile,
            "quantile_shift": quantile_shift,
            "lambda_min_est": lambda_min_est,
            "inverse_lambda_min_est": (
                1.0 / lambda_min_est
                if isfinite(lambda_min_est) and lambda_min_est > 0.0
                else float("nan")
            ),
            "quantile_shift_bound_bias": quantile_shift_bound_bias_value,
            "quantile_shift_bound_tv": quantile_shift_bound_tv_value,
            "quantile_shift_bound_bias_looseness": quantile_shift_bound_bias_value
            - quantile_shift,
            "quantile_shift_bound_tv_looseness": quantile_shift_bound_tv_value
            - quantile_shift,
            "quantile_shift_bound_bias_ratio": (
                quantile_shift_bound_bias_value / max(quantile_shift, 1e-8)
                if isfinite(quantile_shift_bound_bias_value)
                else float("nan")
            ),
            "theorem_lower_bound": theorem_bound,
            "proposition_lower_bound": proposition_bound,
            "additive_mixture_lower_bound": additive_bound,
            "pi_c": pi_c,
            "pi_d": pi_d,
            "eta": eta,
            "n_clean_keep_lower_bound": retained_count_lower(
                pi_c, spec.calibration_size, eta
            ),
            "n_dirty_keep_upper_bound": retained_count_upper(
                pi_d, spec.calibration_size, eta
            ),
            "n_keep_lower_bound": retained_count_lower(
                mu_keep, spec.calibration_size, eta
            ),
            "empirical_retained_contamination_upper": empirical_retained_contamination_upper_value,
            "retained_contamination_upper_defined": float(
                isfinite(empirical_retained_contamination_upper_value)
            ),
            "clean_count_violation": float(
                n_clean_keep < retained_count_lower(pi_c, spec.calibration_size, eta)
            ),
            "dirty_count_violation": float(
                n_dirty_keep > retained_count_upper(pi_d, spec.calibration_size, eta)
            ),
            "keep_count_violation": float(
                int(keep_mask.sum())
                < retained_count_lower(mu_keep, spec.calibration_size, eta)
            ),
            "retained_contamination_upper_violation": retained_contamination_upper_violation_value,
        }
    )
    _append_one_sided_transfer_metrics(
        row=row,
        spec=spec,
        threshold_value=threshold_value,
        clean_structure_scores=clean_structure_scores,
        dirty_structure_scores=dirty_structure_scores,
        clean_structure_nonconformity=clean_structure_nonconformity,
        dirty_structure_nonconformity=dirty_structure_nonconformity,
        p_c=p_c,
        p_d=p_d,
        mu_keep=mu_keep,
        tilde_epsilon=tilde_epsilon,
    )
    if population_metrics is None:
        row.update(
            {
                "sup_score_error": float("nan"),
                "score_model_sup_error": float("nan"),
                "score_model_mse": float("nan"),
                "score_fit_mean": float("nan"),
                "score_fit_variance": float("nan"),
                "score_bandwidth": float("nan"),
                "bridge_delta_s": float("nan"),
                "bridge_event_holds": float("nan"),
                "bridge_p_c_lower": float("nan"),
                "bridge_p_c_upper": float("nan"),
                "bridge_p_d_lower": float("nan"),
                "bridge_p_d_upper": float("nan"),
                "bridge_tilde_epsilon_upper": float("nan"),
                "bridge_mu_keep_lower": float("nan"),
                "clean_second_moment_empirical": float("nan"),
                "clean_second_moment_hp_bound": float("nan"),
                "clean_second_moment_violation": float("nan"),
                "d_sq": float("nan"),
                "dirty_second_moment_lower": float("nan"),
                "clean_tail_upper": float("nan"),
                "dirty_tail_lower": float("nan"),
                "observed_clean_tail": float("nan"),
                "observed_dirty_tail": float("nan"),
                "explicit_t_minus": float("nan"),
                "explicit_t_plus": float("nan"),
                "fully_explicit_bias": float("nan"),
                "explicit_fail_t_minus_nonpositive": float("nan"),
                "explicit_fail_udr_nan": float("nan"),
                "explicit_fail_mu_lower_nan": float("nan"),
                "explicit_fail_bar_eps_nan": float("nan"),
                "explicit_fail_fully_bias_nan": float("nan"),
                "semi_explicit_finite": float("nan"),
                "fully_explicit_finite": float("nan"),
                "semi_explicit_informative": float("nan"),
                "fully_explicit_informative": float("nan"),
                "explicit_bound_noninformative": float("nan"),
                "semi_explicit_bound": float("nan"),
                "fully_explicit_bound": float("nan"),
                "semi_explicit_gap": float("nan"),
                "fully_explicit_gap": float("nan"),
                "working_radius": float("nan"),
                "working_center": float("nan"),
                "working_scale": float("nan"),
                "zeta_P": float("nan"),
                "zeta_Q": float("nan"),
                "q_Q": float("nan"),
                "D0_sq": float("nan"),
                "m_c0": float("nan"),
                "m_d0_observed": float("nan"),
                "underline_m_d0": float("nan"),
                "local_score_bound": float("nan"),
                "localized_clean_lower": float("nan"),
                "localized_dirty_upper": float("nan"),
                "localized_bar_epsilon": float("nan"),
                "localized_mu_lower": float("nan"),
                "localized_dirty_tail_lower": float("nan"),
                "localized_feasible": float("nan"),
                "explicit_pass_t_minus_positive": float("nan"),
                "explicit_pass_t_plus_lt_local_bound": float("nan"),
                "explicit_pass_dirty_tail_moment": float("nan"),
                "explicit_pass_clean_lower_positive": float("nan"),
                "explicit_pass_dirty_upper_finite": float("nan"),
                "explicit_pass_bar_epsilon_finite": float("nan"),
                "explicit_pass_mu_lower_finite": float("nan"),
                "explicit_pass_fully_bias_finite": float("nan"),
                "explicit_feasibility_all_pass": float("nan"),
                "explicit_fail_t_plus_ge_l0": float("nan"),
                "explicit_fail_dirty_tail_condition": float("nan"),
                "explicit_fail_clean_lower_nonpositive": float("nan"),
                "score_rank_spearman_mahalanobis": float("nan"),
            }
        )
        return
    delta_s = float(population_metrics["delta_s"])
    clean_pop_score = population_metrics["clean_population_score"]
    dirty_pop_score = population_metrics["dirty_population_score"]
    sigma_c_sq = float(population_metrics["sigma_c_sq"])
    underline_m_d = float(population_metrics["underline_m_d"])
    score_bound = float(population_metrics["score_bound"])
    zeta_p = float(population_metrics["zeta_P"])
    zeta_q = float(population_metrics["zeta_Q"])
    q_q = float(population_metrics["q_Q"])
    underline_m_d0 = float(population_metrics["underline_m_d0"])
    local_score_bound = float(population_metrics["local_score_bound"])
    t_minus = threshold_value - delta_s
    t_plus = threshold_value + delta_s
    p_c_minus = float(np.mean(clean_pop_score <= t_minus)) if t_minus > 0.0 else 0.0
    p_c_plus = float(np.mean(clean_pop_score <= t_plus))
    p_d_minus = float(np.mean(dirty_pop_score <= t_minus)) if t_minus > 0.0 else 0.0
    p_d_plus = float(np.mean(dirty_pop_score <= t_plus))
    bridge_tilde_eps = (
        spec.epsilon
        * p_d_plus
        / ((1.0 - spec.epsilon) * p_c_minus + spec.epsilon * p_d_plus)
        if ((1.0 - spec.epsilon) * p_c_minus + spec.epsilon * p_d_plus) > 0.0
        else float("nan")
    )
    bridge_mu_lower = (1.0 - spec.epsilon) * p_c_minus + spec.epsilon * p_d_minus
    clean_tail = clean_tail_upper(sigma_c_sq, max(threshold_value, 1e-6))
    dirty_tail = dirty_tail_lower(
        max(underline_m_d, 0.0),
        max(threshold_value, 1e-6),
        max(score_bound, threshold_value + 1e-6),
    )
    observed_clean_tail = float(np.mean(clean_pop_score > threshold_value))
    observed_dirty_tail = float(np.mean(dirty_pop_score > threshold_value))
    l_c_r = localized_clean_retention_lower(
        sigma_c_sq=sigma_c_sq,
        t_minus=t_minus,
        zeta_p=zeta_p,
    )
    u_d_r = localized_dirty_retention_upper(
        m_d0_lower=underline_m_d0,
        t_plus=t_plus,
        l0=local_score_bound,
        zeta_q=zeta_q,
    )
    localized_dirty_tail = localized_dirty_tail_lower(
        m_d0_lower=underline_m_d0,
        t_plus=t_plus,
        l0=local_score_bound,
        zeta_q=zeta_q,
    )
    localized_feasible = float(
        np.isfinite(l_c_r)
        and np.isfinite(u_d_r)
        and l_c_r > 0.0
        and q_q > 0.0
        and t_plus > 0.0
        and np.isfinite(local_score_bound)
        and t_plus < local_score_bound
        and q_q * (t_plus**2) < underline_m_d0
    )
    explicit_pass_t_minus_positive = float(t_minus > 0.0)
    explicit_pass_t_plus_lt_local_bound = float(
        np.isfinite(local_score_bound) and t_plus < local_score_bound
    )
    explicit_pass_dirty_tail_moment = float(
        np.isfinite(q_q)
        and np.isfinite(underline_m_d0)
        and q_q > 0.0
        and t_plus > 0.0
        and q_q * (t_plus**2) < underline_m_d0
    )
    explicit_pass_clean_lower_positive = float(np.isfinite(l_c_r) and l_c_r > 0.0)
    explicit_pass_dirty_upper_finite = float(np.isfinite(u_d_r))
    bar_eps = (
        spec.epsilon * u_d_r / ((1.0 - spec.epsilon) * l_c_r + spec.epsilon * u_d_r)
        if np.isfinite(l_c_r)
        and np.isfinite(u_d_r)
        and l_c_r > 0.0
        and u_d_r >= 0.0
        and ((1.0 - spec.epsilon) * l_c_r + spec.epsilon * u_d_r) > 0.0
        else float("nan")
    )
    explicit_pass_bar_epsilon_finite = float(np.isfinite(bar_eps))
    mu_lower = (
        (1.0 - spec.epsilon) * l_c_r
        if np.isfinite(l_c_r) and l_c_r > 0.0
        else float("nan")
    )
    explicit_pass_mu_lower_finite = float(np.isfinite(mu_lower))
    semi_explicit_bound = (
        theorem_lower_bound(
            spec.alpha,
            bias,
            bar_eps,
            mu_lower,
            spec.calibration_size,
            spec.eta_fraction,
        )
        if np.isfinite(bar_eps) and np.isfinite(mu_lower)
        else float("nan")
    )
    fully_explicit_bias = 1.0 - l_c_r if np.isfinite(l_c_r) else float("nan")
    explicit_pass_fully_bias_finite = float(np.isfinite(fully_explicit_bias))
    explicit_feasibility_all_pass = float(
        bool(explicit_pass_t_minus_positive)
        and bool(explicit_pass_t_plus_lt_local_bound)
        and bool(explicit_pass_dirty_tail_moment)
        and bool(explicit_pass_clean_lower_positive)
        and bool(explicit_pass_dirty_upper_finite)
        and bool(explicit_pass_bar_epsilon_finite)
        and bool(explicit_pass_mu_lower_finite)
        and bool(explicit_pass_fully_bias_finite)
    )
    fully_explicit_bound = (
        theorem_lower_bound(
            spec.alpha,
            fully_explicit_bias,
            bar_eps,
            mu_lower,
            spec.calibration_size,
            spec.eta_fraction,
        )
        if np.isfinite(bar_eps)
        and np.isfinite(mu_lower)
        and np.isfinite(fully_explicit_bias)
        else float("nan")
    )
    semi_explicit_finite = float(np.isfinite(semi_explicit_bound))
    fully_explicit_finite = float(np.isfinite(fully_explicit_bound))
    semi_explicit_informative = float(
        np.isfinite(semi_explicit_bound) and semi_explicit_bound >= 0.0
    )
    fully_explicit_informative = float(
        np.isfinite(fully_explicit_bound) and fully_explicit_bound >= 0.0
    )
    row.update(
        {
            "sup_score_error": float(population_metrics["sup_score_error"]),
            "score_model_sup_error": float(population_metrics["score_model_sup_error"]),
            "score_model_mse": float(population_metrics["score_model_mse"]),
            "score_fit_mean": float(population_metrics["score_fit_mean"]),
            "score_fit_variance": float(population_metrics["score_fit_variance"]),
            "score_bandwidth": float(population_metrics["score_bandwidth"]),
            "bridge_delta_s": delta_s,
            "bridge_event_holds": float(
                population_metrics["sup_score_error"] <= delta_s
            ),
            "bridge_p_c_lower": p_c_minus,
            "bridge_p_c_upper": p_c_plus,
            "bridge_p_d_lower": p_d_minus,
            "bridge_p_d_upper": p_d_plus,
            "bridge_tilde_epsilon_upper": bridge_tilde_eps,
            "bridge_mu_keep_lower": bridge_mu_lower,
            "clean_second_moment_empirical": float(
                population_metrics["clean_empirical_second_moment"]
            ),
            "clean_second_moment_hp_bound": float(
                population_metrics["clean_second_moment_hp_bound"]
            ),
            "clean_second_moment_violation": float(
                population_metrics["clean_empirical_second_moment"]
                > population_metrics["clean_second_moment_hp_bound"]
            ),
            "d_sq": float(population_metrics["d_sq"]),
            "dirty_second_moment_lower": underline_m_d,
            "working_radius": float(population_metrics["working_radius"]),
            "working_center": float(population_metrics["working_center"]),
            "working_scale": float(population_metrics["working_scale"]),
            "zeta_P": zeta_p,
            "zeta_Q": zeta_q,
            "q_Q": q_q,
            "D0_sq": float(population_metrics["D0_sq"]),
            "m_c0": float(population_metrics["m_c0"]),
            "m_d0_observed": float(population_metrics["m_d0_observed"]),
            "underline_m_d0": underline_m_d0,
            "local_score_bound": local_score_bound,
            "score_rank_spearman_mahalanobis": float(population_metrics["score_rank_spearman_mahalanobis"]),
            "clean_tail_upper": clean_tail,
            "dirty_tail_lower": dirty_tail,
            "localized_dirty_tail_lower": localized_dirty_tail,
            "observed_clean_tail": observed_clean_tail,
            "observed_dirty_tail": observed_dirty_tail,
            "l_c_r": l_c_r,
            "u_d_r": u_d_r,
            "localized_clean_lower": l_c_r,
            "localized_dirty_upper": u_d_r,
            "localized_bar_epsilon": bar_eps,
            "localized_mu_lower": mu_lower,
            "localized_feasible": localized_feasible,
            "explicit_t_minus": t_minus,
            "explicit_t_plus": t_plus,
            "fully_explicit_bias": fully_explicit_bias,
            "explicit_pass_t_minus_positive": explicit_pass_t_minus_positive,
            "explicit_pass_t_plus_lt_local_bound": explicit_pass_t_plus_lt_local_bound,
            "explicit_pass_dirty_tail_moment": explicit_pass_dirty_tail_moment,
            "explicit_pass_clean_lower_positive": explicit_pass_clean_lower_positive,
            "explicit_pass_dirty_upper_finite": explicit_pass_dirty_upper_finite,
            "explicit_pass_bar_epsilon_finite": explicit_pass_bar_epsilon_finite,
            "explicit_pass_mu_lower_finite": explicit_pass_mu_lower_finite,
            "explicit_pass_fully_bias_finite": explicit_pass_fully_bias_finite,
            "explicit_feasibility_all_pass": explicit_feasibility_all_pass,
            "explicit_fail_t_minus_nonpositive": float(t_minus <= 0.0),
            "explicit_fail_t_plus_ge_l0": float(not (np.isfinite(local_score_bound) and t_plus < local_score_bound)),
            "explicit_fail_dirty_tail_condition": float(not bool(explicit_pass_dirty_tail_moment)),
            "explicit_fail_clean_lower_nonpositive": float(not bool(explicit_pass_clean_lower_positive)),
            "explicit_fail_udr_nan": float(not np.isfinite(u_d_r)),
            "explicit_fail_mu_lower_nan": float(not np.isfinite(mu_lower)),
            "explicit_fail_bar_eps_nan": float(not np.isfinite(bar_eps)),
            "explicit_fail_fully_bias_nan": float(not np.isfinite(fully_explicit_bias)),
            "semi_explicit_finite": semi_explicit_finite,
            "fully_explicit_finite": fully_explicit_finite,
            "semi_explicit_informative": semi_explicit_informative,
            "fully_explicit_informative": fully_explicit_informative,
            "explicit_bound_noninformative": float(
                not (
                    bool(semi_explicit_informative) or bool(fully_explicit_informative)
                )
            ),
            "semi_explicit_bound": semi_explicit_bound,
            "fully_explicit_bound": fully_explicit_bound,
            "semi_explicit_gap": float("nan"),
            "fully_explicit_gap": float("nan"),
        }
    )
def run_experiment_spec(
    spec: ExperimentSpec, output_root: Path
) -> list[dict[str, Any]]:
    ensure_dir(output_root)
    clean_sampler = clean_sampler_for_spec(spec)
    dirty_sampler = dirty_sampler_for_spec(spec)
    rows: list[dict[str, Any]] = []
    for replicate in range(spec.mc_reps):
        fit_rng = make_rng(spec.random_seed, replicate, stage=1)
        reference_rng = make_rng(spec.random_seed, replicate, stage=2)
        calibration_rng = make_rng(spec.random_seed, replicate, stage=3)
        clean_test_rng = make_rng(spec.random_seed, replicate, stage=4)
        mixture_test_rng = make_rng(spec.random_seed, replicate, stage=5)
        structure_rng = make_rng(spec.random_seed, replicate, stage=6)
        random_trim_rng = make_rng(spec.random_seed, replicate, stage=8)
        fit_sample = clean_sampler(fit_rng, spec.train_size)
        reference_sample = clean_sampler(reference_rng, spec.clean_reference_size)
        calibration_sample = sample_mixture(
            calibration_rng,
            size=spec.calibration_size,
            epsilon=spec.epsilon,
            clean_sampler=clean_sampler,
            dirty_sampler=dirty_sampler,
        )
        clean_test = clean_sampler(clean_test_rng, spec.clean_test_size)
        mixture_test = sample_mixture(
            mixture_test_rng,
            size=spec.mixture_test_size,
            epsilon=spec.epsilon,
            clean_sampler=clean_sampler,
            dirty_sampler=dirty_sampler,
        )
        clean_structure = clean_sampler(structure_rng, spec.structure_sample_size)
        dirty_structure = dirty_sampler(structure_rng, spec.structure_sample_size)
        mix_structure = sample_mixture(
            structure_rng,
            size=spec.structure_sample_size,
            epsilon=spec.epsilon,
            clean_sampler=clean_sampler,
            dirty_sampler=dirty_sampler,
        )
        linear_predictor = build_predictor(
            fit_sample.x, fit_sample.y, "linear", spec.alpha
        )
        cqr_predictor = build_predictor(
            fit_sample.x, fit_sample.y, "cqr_linear", spec.alpha
        )
        stein_model = _stein_model_for_spec(spec, fit_sample.x, reference_sample.x)
        train_stein_scores = stein_model.empirical_score(fit_sample.x)
        reference_stein_scores = stein_model.empirical_score(reference_sample.x)
        calibration_stein_scores = stein_model.empirical_score(calibration_sample.x)
        clean_structure_stein_scores = stein_model.empirical_score(clean_structure.x)
        dirty_structure_stein_scores = stein_model.empirical_score(dirty_structure.x)
        fit_feature_matrix = _feature_matrix(fit_sample.x)
        mahalanobis_center = np.mean(fit_feature_matrix, axis=0)
        mahalanobis_covariance = np.cov(fit_feature_matrix, rowvar=False)
        if np.ndim(mahalanobis_covariance) == 0:
            mahalanobis_covariance = np.asarray([[float(mahalanobis_covariance)]])
        mahalanobis_covariance = np.asarray(mahalanobis_covariance, dtype=float).reshape(
            fit_feature_matrix.shape[1], fit_feature_matrix.shape[1]
        )
        train_mahalanobis_scores = _mahalanobis_scores(fit_sample.x, mahalanobis_center, mahalanobis_covariance)
        reference_mahalanobis_scores = _mahalanobis_scores(reference_sample.x, mahalanobis_center, mahalanobis_covariance)
        calibration_mahalanobis_scores = _mahalanobis_scores(calibration_sample.x, mahalanobis_center, mahalanobis_covariance)
        clean_structure_mahalanobis_scores = _mahalanobis_scores(clean_structure.x, mahalanobis_center, mahalanobis_covariance)
        dirty_structure_mahalanobis_scores = _mahalanobis_scores(dirty_structure.x, mahalanobis_center, mahalanobis_covariance)
        train_score_norm_scores = _score_norm_scores(stein_model, fit_sample.x)
        reference_score_norm_scores = _score_norm_scores(stein_model, reference_sample.x)
        calibration_score_norm_scores = _score_norm_scores(stein_model, calibration_sample.x)
        clean_structure_score_norm_scores = _score_norm_scores(stein_model, clean_structure.x)
        dirty_structure_score_norm_scores = _score_norm_scores(stein_model, dirty_structure.x)
        gmm_mahalanobis_model = fit_gmm_mahalanobis_model(
            fit_sample.x,
            n_components=int(spec.extras.get("gmm_components", 2)),
            max_iter=int(spec.extras.get("gmm_max_iter", 100)),
            min_covar=float(spec.extras.get("gmm_min_covar", 1e-4)),
        )
        train_gmm_mahalanobis_scores = gmm_mahalanobis_model.score(fit_sample.x)
        reference_gmm_mahalanobis_scores = gmm_mahalanobis_model.score(reference_sample.x)
        calibration_gmm_mahalanobis_scores = gmm_mahalanobis_model.score(calibration_sample.x)
        clean_structure_gmm_mahalanobis_scores = gmm_mahalanobis_model.score(clean_structure.x)
        dirty_structure_gmm_mahalanobis_scores = gmm_mahalanobis_model.score(dirty_structure.x)
        calibration_linear_scores = linear_predictor.nonconformity(
            calibration_sample.x, calibration_sample.y
        )
        clean_structure_linear_scores = linear_predictor.nonconformity(
            clean_structure.x, clean_structure.y
        )
        train_linear_scores = linear_predictor.nonconformity(fit_sample.x, fit_sample.y)
        dirty_structure_linear_scores = linear_predictor.nonconformity(
            dirty_structure.x, dirty_structure.y
        )
        calibration_cqr_scores = cqr_predictor.nonconformity(
            calibration_sample.x, calibration_sample.y
        )
        clean_structure_cqr_scores = cqr_predictor.nonconformity(
            clean_structure.x, clean_structure.y
        )
        dirty_structure_cqr_scores = cqr_predictor.nonconformity(
            dirty_structure.x, dirty_structure.y
        )
        baseline_context = BaselineContext(
            spec=spec,
            calibration_sample=calibration_sample,
            clean_structure=clean_structure,
            dirty_structure=dirty_structure,
            linear_predictor=linear_predictor,
            cqr_predictor=cqr_predictor,
            calibration_linear_scores=calibration_linear_scores,
            calibration_cqr_scores=calibration_cqr_scores,
            clean_structure_linear_scores=clean_structure_linear_scores,
            dirty_structure_linear_scores=dirty_structure_linear_scores,
            clean_structure_cqr_scores=clean_structure_cqr_scores,
            dirty_structure_cqr_scores=dirty_structure_cqr_scores,
        )
        population_metrics = None
        if spec.compute_population_metrics:
            population_metrics = _population_metrics(
                spec=spec,
                stein_model=stein_model,
                clean_sampler=clean_sampler,
                clean_structure_x=clean_structure.x,
                dirty_structure_x=dirty_structure.x,
                mix_structure_x=mix_structure.x,
                population_seed=derive_seed(spec.random_seed, replicate, 9),
            )
        threshold_specs = _threshold_specs(
            spec, train_stein_scores, reference_stein_scores, population_metrics
        )
        selected_threshold_specs = threshold_specs
        if spec.threshold_policy == "oracle":
            oracle_candidates: list[tuple[float, ThresholdSpec, Any]] = []
            for threshold in threshold_specs:
                keep_mask = calibration_stein_scores <= threshold.value
                calibration = calibrate_from_scores(
                    calibration_linear_scores, alpha=spec.alpha, keep_mask=keep_mask
                )
                clean_cov, _ = coverage_and_width(
                    linear_predictor, clean_test.x, clean_test.y, calibration.tau
                )
                oracle_candidates.append((clean_cov, threshold, calibration))
            best_cov, best_threshold, best_calibration = max(
                oracle_candidates, key=lambda item: item[0]
            )
            _ = best_cov
            selected_threshold_specs = [
                ThresholdSpec(
                    "oracle_selected",
                    best_threshold.quantile,
                    best_threshold.value,
                    "oracle_selected",
                )
            ]
            oracle_calibration_lookup = {best_threshold.value: best_calibration}
        else:
            oracle_calibration_lookup = {}
        stein_keep_counts: dict[str, int] = {}
        for method_id in spec.methods:
            if method_id == "stein_trimmed":
                thresholds = selected_threshold_specs
            elif method_id == "residual_trim":
                thresholds = [
                    ThresholdSpec(
                        label=f"residual_q={quantile:.3f}",
                        quantile=quantile,
                        value=float(np.quantile(train_linear_scores, quantile)),
                        source="train_residual_quantile",
                    )
                    for quantile in spec.threshold_quantiles
                ]
            elif method_id == "mahalanobis_trim":
                thresholds = _mahalanobis_threshold_specs(
                    spec,
                    train_mahalanobis_scores=train_mahalanobis_scores,
                    reference_mahalanobis_scores=reference_mahalanobis_scores,
                )
            elif method_id == "gmm_mahalanobis_trim":
                thresholds = _generic_score_threshold_specs(
                    spec,
                    train_gmm_mahalanobis_scores,
                    reference_gmm_mahalanobis_scores,
                    label_prefix="gmm_mahal",
                    source_prefix="gmm_mahalanobis",
                )
            elif method_id == "score_norm_trim":
                thresholds = _generic_score_threshold_specs(
                    spec,
                    train_score_norm_scores,
                    reference_score_norm_scores,
                    label_prefix="score_norm",
                    source_prefix="score_norm",
                )
            elif method_id == "random_trim":
                thresholds = [
                    ThresholdSpec(
                        label=f"random_match_{threshold.label}",
                        quantile=threshold.quantile,
                        value=threshold.value,
                        source="matched_random_trim",
                    )
                    for threshold in selected_threshold_specs
                ]
            elif is_baseline_method(method_id):
                thresholds = [
                    ThresholdSpec(
                        label=method_id,
                        quantile=float("nan"),
                        value=float("nan"),
                        source="baseline_adapter",
                    )
                ]
            else:
                raise ValueError(f"Unknown method_id: {method_id}")
            for threshold in thresholds:
                if method_id == "stein_trimmed":
                    row = _row_base(spec, replicate, method_id, threshold)
                    start = perf_counter()
                    keep_mask = calibration_stein_scores <= threshold.value
                    calibration = oracle_calibration_lookup.get(threshold.value)
                    if calibration is None:
                        calibration = calibrate_from_scores(
                            calibration_linear_scores,
                            alpha=spec.alpha,
                            keep_mask=keep_mask,
                        )
                    predictor = linear_predictor
                    clean_structure_scores = clean_structure_stein_scores
                    dirty_structure_scores = dirty_structure_stein_scores
                    clean_structure_nonconformity = clean_structure_linear_scores
                    dirty_structure_nonconformity = dirty_structure_linear_scores
                    stein_keep_counts[threshold.label] = calibration.n_keep
                    threshold_metric_value = threshold.value
                elif method_id == "mahalanobis_trim":
                    row = _row_base(spec, replicate, method_id, threshold)
                    start = perf_counter()
                    keep_mask = calibration_mahalanobis_scores <= threshold.value
                    calibration = calibrate_from_scores(
                        calibration_linear_scores,
                        alpha=spec.alpha,
                        keep_mask=keep_mask,
                    )
                    predictor = linear_predictor
                    clean_structure_scores = clean_structure_mahalanobis_scores
                    dirty_structure_scores = dirty_structure_mahalanobis_scores
                    clean_structure_nonconformity = clean_structure_linear_scores
                    dirty_structure_nonconformity = dirty_structure_linear_scores
                    threshold_metric_value = threshold.value
                elif method_id == "gmm_mahalanobis_trim":
                    row = _row_base(spec, replicate, method_id, threshold)
                    start = perf_counter()
                    keep_mask = calibration_gmm_mahalanobis_scores <= threshold.value
                    calibration = calibrate_from_scores(calibration_linear_scores, alpha=spec.alpha, keep_mask=keep_mask)
                    predictor = linear_predictor
                    clean_structure_scores = clean_structure_gmm_mahalanobis_scores
                    dirty_structure_scores = dirty_structure_gmm_mahalanobis_scores
                    clean_structure_nonconformity = clean_structure_linear_scores
                    dirty_structure_nonconformity = dirty_structure_linear_scores
                    threshold_metric_value = threshold.value
                elif method_id == "score_norm_trim":
                    row = _row_base(spec, replicate, method_id, threshold)
                    start = perf_counter()
                    keep_mask = calibration_score_norm_scores <= threshold.value
                    calibration = calibrate_from_scores(calibration_linear_scores, alpha=spec.alpha, keep_mask=keep_mask)
                    predictor = linear_predictor
                    clean_structure_scores = clean_structure_score_norm_scores
                    dirty_structure_scores = dirty_structure_score_norm_scores
                    clean_structure_nonconformity = clean_structure_linear_scores
                    dirty_structure_nonconformity = dirty_structure_linear_scores
                    threshold_metric_value = threshold.value
                elif method_id == "random_trim":
                    row = _row_base(spec, replicate, method_id, threshold)
                    start = perf_counter()
                    if not stein_keep_counts:
                        raise RuntimeError(
                            "random_trim requires stein_trimmed to be evaluated first in each experiment."
                        )
                    match_label = threshold.label.replace("random_match_", "", 1)
                    matched_n_keep = stein_keep_counts[match_label]
                    keep_mask = matched_random_keep_mask(
                        random_trim_rng, spec.calibration_size, matched_n_keep
                    )
                    calibration = calibrate_from_scores(
                        calibration_linear_scores, alpha=spec.alpha, keep_mask=keep_mask
                    )
                    predictor = linear_predictor
                    clean_structure_scores = np.full_like(
                        clean_structure_stein_scores, fill_value=0.0
                    )
                    dirty_structure_scores = np.full_like(
                        dirty_structure_stein_scores, fill_value=0.0
                    )
                    clean_structure_nonconformity = clean_structure_linear_scores
                    dirty_structure_nonconformity = dirty_structure_linear_scores
                    threshold_metric_value = float(
                        matched_n_keep / spec.calibration_size
                    )
                elif method_id == "residual_trim":
                    row = _row_base(spec, replicate, method_id, threshold)
                    start = perf_counter()
                    keep_mask = calibration_linear_scores <= threshold.value
                    calibration = calibrate_from_scores(
                        calibration_linear_scores, alpha=spec.alpha, keep_mask=keep_mask
                    )
                    predictor = linear_predictor
                    clean_structure_scores = clean_structure_linear_scores
                    dirty_structure_scores = dirty_structure_linear_scores
                    clean_structure_nonconformity = clean_structure_linear_scores
                    dirty_structure_nonconformity = dirty_structure_linear_scores
                    threshold_metric_value = threshold.value
                elif is_baseline_method(method_id):
                    baseline_run = execute_baseline(method_id, baseline_context)
                    threshold = ThresholdSpec(
                        label=baseline_run.threshold_label,
                        quantile=baseline_run.threshold_quantile,
                        value=baseline_run.threshold_value,
                        source=baseline_run.threshold_source,
                    )
                    row = _row_base(spec, replicate, method_id, threshold)
                    row.update(baseline_run.extra_row_fields)
                    start = perf_counter()
                    keep_mask = baseline_run.keep_mask
                    calibration = baseline_run.calibration
                    predictor = baseline_run.predictor
                    clean_structure_scores = baseline_run.clean_structure_scores
                    dirty_structure_scores = baseline_run.dirty_structure_scores
                    clean_structure_nonconformity = (
                        baseline_run.clean_structure_nonconformity
                    )
                    dirty_structure_nonconformity = baseline_run.predictor.nonconformity(
                        dirty_structure.x, dirty_structure.y
                    )
                    threshold_metric_value = baseline_run.threshold_value
                if method_id in {
                    "stein_trimmed",
                    "residual_trim",
                    "mahalanobis_trim",
                    "gmm_mahalanobis_trim",
                    "score_norm_trim",
                } or is_baseline_method(method_id):
                    _append_threshold_metrics(
                        row=row,
                        spec=spec,
                        threshold_value=threshold_metric_value,
                        keep_mask=keep_mask,
                        calibration_dirty=calibration_sample.dirty,
                        clean_structure_scores=clean_structure_scores,
                        dirty_structure_scores=dirty_structure_scores,
                        clean_structure_nonconformity=clean_structure_nonconformity,
                        dirty_structure_nonconformity=dirty_structure_nonconformity,
                        population_metrics=(
                            population_metrics if method_id == "stein_trimmed" else None
                        ),
                    )
                elif method_id == "random_trim":
                    retained_fraction = float(np.mean(keep_mask))
                    row.update(
                        {
                            "n_clean_keep": int(
                                np.sum((~calibration_sample.dirty) & keep_mask)
                            ),
                            "n_dirty_keep": int(
                                np.sum(calibration_sample.dirty & keep_mask)
                            ),
                            "n_keep": int(keep_mask.sum()),
                            "retained_fraction": retained_fraction,
                            "empirical_retained_contamination": float(
                                np.mean(calibration_sample.dirty[keep_mask])
                                if keep_mask.any()
                                else 0.0
                            ),
                            "p_c": retained_fraction,
                            "p_d": retained_fraction,
                            "p_c_minus_p_d": 0.0,
                            "mu_keep": retained_fraction,
                            "tilde_epsilon": spec.epsilon,
                            "tilde_epsilon_reduction": 0.0,
                            "trimming_bias": 0.0,
                            "trimming_bias_covariance": 0.0,
                            "trimming_bias_direct": 0.0,
                            "net_tradeoff": 0.0,
                            "bias_tv_bound": 1.0 - retained_fraction,
                            "bias_cov_bound": bias_covariance_bound(
                                max(retained_fraction, 1e-6)
                            ),
                            "clean_quantile": left_quantile(
                                clean_structure_linear_scores, 1.0 - spec.alpha
                            ),
                            "retained_clean_quantile": left_quantile(
                                clean_structure_linear_scores, 1.0 - spec.alpha
                            ),
                            "quantile_shift": 0.0,
                            "lambda_min_est": estimate_local_density(
                                clean_structure_linear_scores, 1.0 - spec.alpha
                            ),
                            "quantile_shift_bound_bias": 0.0,
                            "quantile_shift_bound_tv": quantile_shift_bound(
                                1.0 - retained_fraction,
                                estimate_local_density(
                                    clean_structure_linear_scores, 1.0 - spec.alpha
                                ),
                            ),
                            "theorem_lower_bound": theorem_lower_bound(
                                spec.alpha,
                                0.0,
                                spec.epsilon,
                                retained_fraction,
                                spec.calibration_size,
                                spec.eta_fraction,
                            ),
                            "proposition_lower_bound": proposition_lower_bound(
                                spec.alpha,
                                spec.epsilon,
                                0.0,
                                spec.epsilon,
                                retained_fraction,
                                spec.calibration_size,
                                spec.eta_fraction,
                            ),
                            "additive_mixture_lower_bound": additive_mixture_lower_bound(
                                spec.alpha,
                                spec.epsilon,
                                0.0,
                                spec.epsilon,
                                retained_fraction,
                                spec.calibration_size,
                                spec.eta_fraction,
                            ),
                        }
                    )
                    _append_one_sided_transfer_metrics(
                        row=row,
                        spec=spec,
                        threshold_value=threshold_metric_value,
                        clean_structure_scores=clean_structure_scores,
                        dirty_structure_scores=dirty_structure_scores,
                        clean_structure_nonconformity=clean_structure_nonconformity,
                        dirty_structure_nonconformity=dirty_structure_nonconformity,
                        p_c=retained_fraction,
                        p_d=retained_fraction,
                        mu_keep=retained_fraction,
                        tilde_epsilon=spec.epsilon,
                    )
                clean_coverage, width = coverage_and_width(
                    predictor, clean_test.x, clean_test.y, calibration.tau
                )
                mixture_coverage, _ = coverage_and_width(
                    predictor, mixture_test.x, mixture_test.y, calibration.tau
                )
                row.update(
                    {
                        "tau": calibration.tau,
                        "clean_test_coverage": clean_coverage,
                        "contaminated_mixture_coverage": (
                            mixture_coverage
                            if spec.include_mixture_eval
                            else float("nan")
                        ),
                        "average_interval_width": width,
                    }
                )
                row["bound_gap_theorem"] = safe_gap(
                    clean_coverage, row["theorem_lower_bound"]
                )
                row["bound_gap_proposition"] = safe_gap(
                    mixture_coverage, row["proposition_lower_bound"]
                )
                row["bound_gap_additive"] = safe_gap(
                    mixture_coverage, row["additive_mixture_lower_bound"]
                )
                row["semi_explicit_gap"] = safe_gap(
                    clean_coverage, row.get("semi_explicit_bound", float("nan"))
                )
                row["fully_explicit_gap"] = safe_gap(
                    clean_coverage, row.get("fully_explicit_bound", float("nan"))
                )
                row["runtime_seconds"] = perf_counter() - start
                rows.append(row)
    metadata = {
        "spec": spec.as_metadata(),
        "num_rows": len(rows),
    }
    group_raw_dir = output_root / "raw" / spec.group_id
    ensure_dir(group_raw_dir)
    write_json(group_raw_dir / f"{spec.config_id}_metadata.json", metadata)
    return rows
def persist_raw_rows(
    group_id: str, rows: list[dict[str, Any]], output_root: Path
) -> None:
    raw_dir = output_root / "raw" / group_id
    ensure_dir(raw_dir)
    write_rows_csv(raw_dir / "replicate_results.csv", rows)
    write_rows_parquet(raw_dir / "replicate_results.parquet", rows)