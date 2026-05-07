from __future__ import annotations
from collections import defaultdict
from math import exp
import math
from typing import Any
NUMERIC_FIELDS = [
    "clean_test_coverage_mean",
    "average_interval_width_mean",
    "tilde_epsilon_mean",
    "trimming_bias_mean",
    "quantile_shift_mean",
    "theorem_lower_bound_mean",
    "theorem_lower_bound_plus_mean",
    "lpop_plus_mean",
    "delta_trim_plus_mean",
    "d_q_plus_mean",
    "one_sided_transfer_loss_mean",
    "audit_theorem_lower_bound_plus_mean",
    "bound_gap_theorem_mean",
    "retained_fraction_mean",
    "p_c_mean",
    "p_d_mean",
    "p_c_minus_p_d_mean",
    "net_tradeoff_mean",
]
DENSITY_REGIME_LABELS = {
    "gaussian_homoscedastic": "gaussian_homoscedastic",
    "heavy_tail_clean": "heavy_tailed",
    "bimodal_clean": "bimodal",
    "low_density_near_quantile": "low_density_near_quantile",
}
SCORE_MODEL_MODE_ORDER = {
    "oracle": 0,
    "well_specified": 1,
    "mis_specified": 2,
}
METADATA_FIELDS = [
    "group_id",
    "config_id",
    "experiment_title",
    "description",
    "alpha",
    "epsilon",
    "train_size",
    "calibration_size",
    "clean_reference_size",
    "clean_test_size",
    "mixture_test_size",
    "structure_sample_size",
    "eta_fraction",
    "bridge_delta",
    "bridge_rho",
    "population_sample_size",
    "evaluation_grid_size",
    "compute_population_metrics",
    "include_mixture_eval",
    "mc_reps",
    "random_seed",
    "dgp_family",
    "contamination_family",
    "predictor_type",
    "representation_family",
    "notes",
    "dirty_loc",
    "dirty_scale",
    "leverage_loc",
    "separation_strength",
    "phase_region",
    "score_model_mode",
    "bandwidth_scale",
    "fit_mean_offset",
    "fit_variance_scale",
    "working_radius",
    "working_center",
    "working_scale",
    "replicate_count",
]
def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed
def _safe_diff(left: Any, right: Any) -> float:
    left_value = _as_float(left)
    right_value = _as_float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return float("nan")
    return left_value - right_value
def _separation_value(row: dict[str, Any]) -> float:
    separation = _as_float(row.get("separation_strength"))
    if math.isfinite(separation):
        return separation
    return _as_float(row.get("dirty_loc"))
def _best_stein_key(row: dict[str, Any]) -> tuple[float, float, float]:
    threshold_quantile = _as_float(row.get("threshold_quantile"))
    if not math.isfinite(threshold_quantile):
        threshold_quantile = 1.0
    return (
        _as_float(row.get("clean_test_coverage_mean")),
        _as_float(row.get("theorem_lower_bound_mean")),
        -threshold_quantile,
    )
def _best_stein_tradeoff_key(row: dict[str, Any]) -> tuple[float, float, float]:
    threshold_quantile = _as_float(row.get("threshold_quantile"))
    if not math.isfinite(threshold_quantile):
        threshold_quantile = 1.0
    return (
        _as_float(row.get("net_tradeoff_mean")),
        _as_float(row.get("clean_test_coverage_mean")),
        -threshold_quantile,
    )
def _group_rows(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        grouped[str(row["config_id"])][str(row["method_id"])].append(row)
    return grouped
def _first_method_row(method_rows: dict[str, list[dict[str, Any]]], method_ids: tuple[str, ...]) -> dict[str, Any] | None:
    for method_id in method_ids:
        rows = method_rows.get(method_id, [])
        if rows:
            return rows[0]
    return None
def _select_stein_row(summary_rows: list[dict[str, Any]], selection_rule: str) -> dict[str, Any]:
    if selection_rule == "coverage":
        return max(summary_rows, key=_best_stein_key)
    if selection_rule == "tradeoff":
        return max(summary_rows, key=_best_stein_tradeoff_key)
    raise ValueError(f"Unknown selection_rule: {selection_rule}")
def _joined_stein_rows(summary_rows: list[dict[str, Any]], *, selection_rule: str = "coverage") -> list[dict[str, Any]]:
    joined_rows: list[dict[str, Any]] = []
    grouped = _group_rows(summary_rows)
    for config_id, method_rows in grouped.items():
        stein_rows = method_rows.get("stein_trimmed", [])
        if not stein_rows:
            continue
        stein_row = _select_stein_row(stein_rows, selection_rule)
        ordinary_row = _first_method_row(method_rows, ("ordinary_split_conformal", "split_conformal"))
        oracle_row = _first_method_row(method_rows, ("clean_oracle_trim", "clean_oracle"))
        residual_row = _first_method_row(method_rows, ("residual_trim",))
        random_row = _first_method_row(method_rows, ("random_trim",))
        joined: dict[str, Any] = {field: stein_row.get(field) for field in METADATA_FIELDS if field in stein_row}
        joined.update(
            {
                "config_id": config_id,
                "selected_method_id": "stein_trimmed",
                "selected_threshold_label": stein_row.get("threshold_label"),
                "selected_threshold_quantile": stein_row.get("threshold_quantile"),
                "selected_threshold_source": stein_row.get("threshold_source"),
                "selected_threshold_value_mean": _as_float(stein_row.get("threshold_value_mean")),
                "selection_rule": selection_rule,
                "selection_metric_mean": _as_float(
                    stein_row.get("clean_test_coverage_mean")
                    if selection_rule == "coverage"
                    else stein_row.get("net_tradeoff_mean")
                ),
                "coverage_gain_vs_ordinary_mean": _safe_diff(
                    stein_row.get("clean_test_coverage_mean"),
                    ordinary_row.get("clean_test_coverage_mean") if ordinary_row else None,
                ),
                "oracle_gap_mean": _safe_diff(
                    oracle_row.get("clean_test_coverage_mean") if oracle_row else None,
                    stein_row.get("clean_test_coverage_mean"),
                ),
                "width_gap_to_oracle_mean": _safe_diff(
                    stein_row.get("average_interval_width_mean"),
                    oracle_row.get("average_interval_width_mean") if oracle_row else None,
                ),
            }
        )
        for prefix, row in (
            ("stein", stein_row),
            ("ordinary", ordinary_row),
            ("clean_oracle", oracle_row),
            ("residual", residual_row),
            ("random_trim", random_row),
        ):
            for field in NUMERIC_FIELDS:
                joined[f"{prefix}_{field}"] = _as_float(row.get(field)) if row is not None else float("nan")
        joined_rows.append(joined)
    joined_rows.sort(
        key=lambda row: (
            _as_float(row.get("calibration_size")),
            _as_float(row.get("epsilon")),
            _separation_value(row),
        )
    )
    return joined_rows
def build_group_n_best_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _joined_stein_rows(summary_rows, selection_rule="coverage")
def build_group_n_best_tradeoff_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _joined_stein_rows(summary_rows, selection_rule="tradeoff")
def build_group_o_phase_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _joined_stein_rows(summary_rows, selection_rule="coverage")
def build_group_p_density_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if str(row.get("method_id")) != "stein_trimmed":
            continue
        lambda_min = _as_float(row.get("lambda_min_est_mean"))
        quantile_shift = _as_float(row.get("quantile_shift_mean"))
        bias_bound = _as_float(row.get("quantile_shift_bound_bias_mean"))
        tv_bound = _as_float(row.get("quantile_shift_bound_tv_mean"))
        enriched = dict(row)
        enriched.update(
            {
                "density_regime": DENSITY_REGIME_LABELS.get(str(row.get("dgp_family")), str(row.get("dgp_family"))),
                "bound_looseness_bias_mean": _safe_diff(bias_bound, quantile_shift),
                "bound_looseness_tv_mean": _safe_diff(tv_bound, quantile_shift),
                "inverse_lambda_min_est_mean": 1.0 / lambda_min if math.isfinite(lambda_min) and lambda_min > 0.0 else float("nan"),
                "bound_efficiency_ratio_mean": (
                    bias_bound / max(quantile_shift, 1e-8) if math.isfinite(bias_bound) else float("nan")
                ),
            }
        )
        rows.append(enriched)
    rows.sort(
        key=lambda row: (
            str(row.get("density_regime")),
            _as_float(row.get("epsilon")),
            _as_float(row.get("threshold_quantile")),
        )
    )
    return rows
def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator):
        return float("nan")
    if denominator <= 0.0:
        return float("inf")
    return numerator / denominator
def _finite_average(*values: float) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values) / len(finite_values))
def build_group_q_violation_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if str(row.get("method_id")) != "stein_trimmed":
            continue
        calibration_size = _as_float(row.get("calibration_size"))
        eta_mean = _as_float(row.get("eta_mean"))
        nominal_proxy = exp(-2.0 * calibration_size * (eta_mean**2)) if math.isfinite(calibration_size) and math.isfinite(eta_mean) else float("nan")
        clean_rate = _as_float(row.get("clean_count_violation_mean"))
        dirty_rate = _as_float(row.get("dirty_count_violation_mean"))
        keep_rate = _as_float(row.get("keep_count_violation_mean"))
        upper_rate = _as_float(row.get("retained_contamination_upper_violation_mean"))
        upper_defined_fraction = _as_float(row.get("retained_contamination_upper_defined_mean"))
        enriched = dict(row)
        enriched.update(
            {
                "clean_violation_rate": clean_rate,
                "dirty_violation_rate": dirty_rate,
                "keep_violation_rate": keep_rate,
                "contamination_upper_exceedance_rate": upper_rate,
                "contamination_upper_defined_fraction": upper_defined_fraction,
                "contamination_upper_undefined_fraction": (
                    1.0 - upper_defined_fraction if math.isfinite(upper_defined_fraction) else float("nan")
                ),
                "nominal_bound_proxy": nominal_proxy,
                "clean_conservativeness_ratio": _safe_ratio(nominal_proxy, clean_rate),
                "dirty_conservativeness_ratio": _safe_ratio(nominal_proxy, dirty_rate),
                "keep_conservativeness_ratio": _safe_ratio(nominal_proxy, keep_rate),
                "contamination_upper_conservativeness_ratio": _safe_ratio(nominal_proxy, upper_rate),
            }
        )
        rows.append(enriched)
    rows.sort(
        key=lambda row: (
            str(row.get("contamination_family")),
            _as_float(row.get("epsilon")),
            _as_float(row.get("threshold_quantile")),
            _as_float(row.get("calibration_size")),
        )
    )
    return rows
def build_group_r_ladder_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if str(row.get("method_id")) != "stein_trimmed":
            continue
        sup_score_error = _as_float(row.get("sup_score_error_mean"))
        bridge_delta = _as_float(row.get("bridge_delta_s_mean"))
        bridge_upper = _as_float(row.get("bridge_tilde_epsilon_upper_mean"))
        tilde_epsilon = _as_float(row.get("tilde_epsilon_mean"))
        enriched = dict(row)
        enriched.update(
            {
                "bridge_margin_mean": _safe_diff(bridge_delta, sup_score_error),
                "bridge_upper_slack_mean": _safe_diff(bridge_upper, tilde_epsilon),
                "score_error_ratio_mean": _safe_ratio(sup_score_error, bridge_delta),
                "score_model_error_ratio_mean": _safe_ratio(
                    _as_float(row.get("score_model_sup_error_mean")),
                    bridge_delta,
                ),
            }
        )
        rows.append(enriched)
    rows.sort(
        key=lambda row: (
            SCORE_MODEL_MODE_ORDER.get(str(row.get("score_model_mode")), 99),
            _as_float(row.get("epsilon")),
            _as_float(row.get("clean_reference_size")),
        )
    )
    return rows
def build_group_s_stability_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if str(row.get("method_id")) != "stein_trimmed":
            continue
        semi_finite = _as_float(row.get("semi_explicit_finite_mean"))
        fully_finite = _as_float(row.get("fully_explicit_finite_mean"))
        semi_informative = _as_float(row.get("semi_explicit_informative_mean"))
        fully_informative = _as_float(row.get("fully_explicit_informative_mean"))
        enriched = dict(row)
        enriched.update(
            {
                "finite_explicit_fraction": _finite_average(semi_finite, fully_finite),
                "informative_explicit_fraction": _finite_average(semi_informative, fully_informative),
                "explicit_fail_t_minus_nonpositive_rate": _as_float(row.get("explicit_fail_t_minus_nonpositive_mean")),
                "explicit_fail_udr_nan_rate": _as_float(row.get("explicit_fail_udr_nan_mean")),
                "explicit_fail_mu_lower_nan_rate": _as_float(row.get("explicit_fail_mu_lower_nan_mean")),
                "explicit_fail_bar_eps_nan_rate": _as_float(row.get("explicit_fail_bar_eps_nan_mean")),
                "explicit_fail_fully_bias_nan_rate": _as_float(row.get("explicit_fail_fully_bias_nan_mean")),
                "explicit_bound_noninformative_rate": _as_float(row.get("explicit_bound_noninformative_mean")),
                "semi_minus_theorem_mean": _safe_diff(
                    row.get("semi_explicit_bound_mean"),
                    row.get("theorem_lower_bound_mean"),
                ),
                "fully_minus_semi_mean": _safe_diff(
                    row.get("fully_explicit_bound_mean"),
                    row.get("semi_explicit_bound_mean"),
                ),
            }
        )
        rows.append(enriched)
    rows.sort(
        key=lambda row: (
            _as_float(row.get("clean_reference_size")),
            _as_float(row.get("population_sample_size")),
            _as_float(row.get("evaluation_grid_size")),
            _as_float(row.get("bandwidth_scale")),
            _as_float(row.get("threshold_quantile")),
        )
    )
    return rows
DIAGNOSTIC_METHOD_ORDER = [
    "stein_trimmed",
    "score_norm_trim",
    "mahalanobis_trim",
    "gmm_mahalanobis_trim",
    "ordinary_split_conformal",
    "clarkson_contaminated_split",
    "clean_oracle_trim",
    "random_trim",
    "residual_trim",
    "cqr_conformal",
]
PAIRWISE_METRICS = {
    "clean_test_coverage_mean": "clean_coverage",
    "clean_test_coverage_se": "clean_coverage_se",
    "contaminated_mixture_coverage_mean": "mixture_coverage",
    "average_interval_width_mean": "width",
    "average_interval_width_se": "width_se",
    "retained_fraction_mean": "retained_fraction",
    "p_c_mean": "p_c",
    "p_d_mean": "p_d",
    "p_c_minus_p_d_mean": "pc_minus_pd",
    "tilde_epsilon_mean": "tilde_epsilon",
    "trimming_bias_mean": "trimming_bias",
    "theorem_lower_bound_mean": "theorem_lower_bound",
    "theorem_lower_bound_plus_mean": "theorem_lower_bound_plus",
    "lpop_plus_mean": "lpop_plus",
    "delta_trim_plus_mean": "delta_trim_plus",
    "d_q_plus_mean": "d_q_plus",
    "one_sided_transfer_loss_mean": "one_sided_transfer_loss",
    "audit_theorem_lower_bound_plus_mean": "audit_theorem_lower_bound_plus",
    "bound_gap_theorem_mean": "bound_gap",
}
NONORACLE_EXCLUDED = {"clean_oracle_trim", "clean_oracle"}
def _method_order(method_id: str) -> int:
    try:
        return DIAGNOSTIC_METHOD_ORDER.index(method_id)
    except ValueError:
        return 100
def _quantile_key(row: dict[str, Any]) -> str | None:
    q = _as_float(row.get("threshold_quantile"))
    if math.isfinite(q):
        return f"{q:.3f}"
    label = str(row.get("threshold_label", ""))
    import re
    match = re.search(r"q=([0-9.]+)", label)
    if match:
        try:
            return f"{float(match.group(1)):.3f}"
        except ValueError:
            return None
    return None
def _nominal(row: dict[str, Any]) -> float:
    alpha = _as_float(row.get("alpha"))
    return 1.0 - alpha if math.isfinite(alpha) else 0.9
def _ci_low(row: dict[str, Any], metric: str) -> float:
    mean = _as_float(row.get(f"{metric}_mean"))
    se = _as_float(row.get(f"{metric}_se"))
    return mean - 1.96 * se if math.isfinite(mean) and math.isfinite(se) else float("nan")
def _ci_high(row: dict[str, Any], metric: str) -> float:
    mean = _as_float(row.get(f"{metric}_mean"))
    se = _as_float(row.get(f"{metric}_se"))
    return mean + 1.96 * se if math.isfinite(mean) and math.isfinite(se) else float("nan")
def _copy_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in METADATA_FIELDS if field in row}
def _valid_shortest_select(
    rows: list[dict[str, Any]],
    *,
    coverage_tolerance: float = 0.005,
    exclude_oracle: bool = True,
) -> tuple[dict[str, Any], str]:
    candidates = [row for row in rows if not (exclude_oracle and str(row.get("method_id")) in NONORACLE_EXCLUDED)]
    if not candidates:
        candidates = rows
    target = _nominal(candidates[0]) - coverage_tolerance
    valid = [
        row
        for row in candidates
        if _as_float(row.get("clean_test_coverage_mean")) >= target
        and math.isfinite(_as_float(row.get("average_interval_width_mean")))
    ]
    if valid:
        return (
            min(
                valid,
                key=lambda row: (
                    _as_float(row.get("average_interval_width_mean")),
                    _as_float(row.get("tilde_epsilon_mean"), 1.0),
                    -_as_float(row.get("p_c_minus_p_d_mean"), -1.0),
                ),
            ),
            f"near_valid_shortest_nonoracle_clean_coverage>={target:.3f}",
        )
    return (
        max(
            candidates,
            key=lambda row: (
                _as_float(row.get("clean_test_coverage_mean"), -float("inf")),
                -_as_float(row.get("average_interval_width_mean"), float("inf")),
            ),
        ),
        f"fallback_max_clean_coverage_nonoracle_no_method_reached_near_valid_{target:.3f}",
    )
def build_method_level_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        clean_mean = _as_float(row.get("clean_test_coverage_mean"))
        width_mean = _as_float(row.get("average_interval_width_mean"))
        enriched = _copy_metadata(row)
        enriched.update(
            {
                "method_id": row.get("method_id"),
                "threshold_label": row.get("threshold_label"),
                "threshold_quantile": row.get("threshold_quantile"),
                "threshold_source": row.get("threshold_source"),
                "nominal_clean_coverage": _nominal(row),
                "clean_coverage_margin_mean": clean_mean - _nominal(row) if math.isfinite(clean_mean) else float("nan"),
                "clean_test_coverage_mean": row.get("clean_test_coverage_mean"),
                "clean_test_coverage_se": row.get("clean_test_coverage_se"),
                "clean_test_coverage_ci95_low": clean_mean - 1.96 * _as_float(row.get("clean_test_coverage_se")) if math.isfinite(clean_mean) and math.isfinite(_as_float(row.get("clean_test_coverage_se"))) else float("nan"),
                "clean_test_coverage_ci95_high": clean_mean + 1.96 * _as_float(row.get("clean_test_coverage_se")) if math.isfinite(clean_mean) and math.isfinite(_as_float(row.get("clean_test_coverage_se"))) else float("nan"),
                "contaminated_mixture_coverage_mean": row.get("contaminated_mixture_coverage_mean"),
                "contaminated_mixture_coverage_se": row.get("contaminated_mixture_coverage_se"),
                "average_interval_width_mean": row.get("average_interval_width_mean"),
                "average_interval_width_se": row.get("average_interval_width_se"),
                "average_interval_width_ci95_low": width_mean - 1.96 * _as_float(row.get("average_interval_width_se")) if math.isfinite(width_mean) and math.isfinite(_as_float(row.get("average_interval_width_se"))) else float("nan"),
                "average_interval_width_ci95_high": width_mean + 1.96 * _as_float(row.get("average_interval_width_se")) if math.isfinite(width_mean) and math.isfinite(_as_float(row.get("average_interval_width_se"))) else float("nan"),
                "retained_fraction_mean": row.get("retained_fraction_mean"),
                "retained_fraction_se": row.get("retained_fraction_se"),
                "p_c_mean": row.get("p_c_mean"),
                "p_d_mean": row.get("p_d_mean"),
                "p_c_minus_p_d_mean": row.get("p_c_minus_p_d_mean"),
                "tilde_epsilon_mean": row.get("tilde_epsilon_mean"),
                "trimming_bias_mean": row.get("trimming_bias_mean"),
                "net_tradeoff_mean": row.get("net_tradeoff_mean"),
                "theorem_lower_bound_mean": row.get("theorem_lower_bound_mean"),
                "theorem_lower_bound_plus_mean": row.get("theorem_lower_bound_plus_mean"),
                "lpop_plus_mean": row.get("lpop_plus_mean"),
                "lpop_plus_direct_mean": row.get("lpop_plus_direct_mean"),
                "delta_trim_plus_mean": row.get("delta_trim_plus_mean"),
                "d_q_plus_mean": row.get("d_q_plus_mean"),
                "d_q_symmetric_mean": row.get("d_q_symmetric_mean"),
                "one_sided_transfer_loss_mean": row.get("one_sided_transfer_loss_mean"),
                "one_sided_clean_component_mean": row.get("one_sided_clean_component_mean"),
                "one_sided_dirty_component_mean": row.get("one_sided_dirty_component_mean"),
                "audit_theorem_lower_bound_plus_mean": row.get("audit_theorem_lower_bound_plus_mean"),
                "audit_lpop_plus_lower_mean": row.get("audit_lpop_plus_lower_mean"),
                "audit_delta_trim_plus_upper_mean": row.get("audit_delta_trim_plus_upper_mean"),
                "audit_d_q_plus_upper_mean": row.get("audit_d_q_plus_upper_mean"),
                "threshold_is_fallback_mean": row.get("threshold_is_fallback_mean"),
                "bound_gap_theorem_mean": row.get("bound_gap_theorem_mean"),
                "replicate_count": row.get("replicate_count"),
            }
        )
        rows.append(enriched)
    rows.sort(key=lambda row: (str(row.get("config_id")), _method_order(str(row.get("method_id"))), _as_float(row.get("threshold_quantile"))))
    return rows
def build_valid_shortest_rows(summary_rows: list[dict[str, Any]], *, coverage_tolerance: float = 0.005) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped = _group_rows(summary_rows)
    for config_id, method_rows in sorted(grouped.items()):
        chunk = [row for rows in method_rows.values() for row in rows]
        selected, rule = _valid_shortest_select(chunk, coverage_tolerance=coverage_tolerance, exclude_oracle=True)
        clean_mean = _as_float(selected.get("clean_test_coverage_mean"))
        out_row = _copy_metadata(selected)
        out_row.update(
            {
                "config_id": config_id,
                "selected_method_id": selected.get("method_id"),
                "selected_threshold_label": selected.get("threshold_label"),
                "selected_threshold_quantile": selected.get("threshold_quantile"),
                "selection_rule": rule,
                "nominal_clean_coverage": _nominal(selected),
                "clean_coverage_margin_mean": clean_mean - _nominal(selected) if math.isfinite(clean_mean) else float("nan"),
            }
        )
        for field in (
            "clean_test_coverage_mean",
            "clean_test_coverage_se",
            "contaminated_mixture_coverage_mean",
            "average_interval_width_mean",
            "average_interval_width_se",
            "retained_fraction_mean",
            "p_c_mean",
            "p_d_mean",
            "p_c_minus_p_d_mean",
            "tilde_epsilon_mean",
            "trimming_bias_mean",
            "theorem_lower_bound_mean",
            "theorem_lower_bound_plus_mean",
            "lpop_plus_mean",
            "delta_trim_plus_mean",
            "d_q_plus_mean",
            "one_sided_transfer_loss_mean",
            "audit_theorem_lower_bound_plus_mean",
            "threshold_is_fallback_mean",
            "bound_gap_theorem_mean",
        ):
            out_row[field] = selected.get(field)
        out.append(out_row)
    return out
def _attach_method_metrics(output: dict[str, Any], prefix: str, row: dict[str, Any] | None) -> None:
    for source, suffix in PAIRWISE_METRICS.items():
        output[f"{prefix}_{suffix}"] = _as_float(row.get(source)) if row is not None else float("nan")
def build_same_quantile_rows(
    summary_rows: list[dict[str, Any]],
    *,
    methods: tuple[str, ...] = ("stein_trimmed", "score_norm_trim", "mahalanobis_trim", "gmm_mahalanobis_trim"),
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    configs: set[str] = set()
    quantiles: set[str] = set()
    for row in summary_rows:
        method_id = str(row.get("method_id"))
        if method_id not in methods:
            continue
        q = _quantile_key(row)
        if q is None:
            continue
        config_id = str(row.get("config_id"))
        indexed[(config_id, method_id, q)] = row
        configs.add(config_id)
        quantiles.add(q)
    out: list[dict[str, Any]] = []
    for config_id in sorted(configs):
        for q in sorted(quantiles, key=lambda x: float(x)):
            if not any((config_id, method, q) in indexed for method in methods):
                continue
            base_row = next((indexed.get((config_id, method, q)) for method in methods if indexed.get((config_id, method, q)) is not None), None)
            if base_row is None:
                continue
            row_out = _copy_metadata(base_row)
            row_out.update({"config_id": config_id, "quantile": q})
            for method in methods:
                prefix = method.replace("_trimmed", "").replace("_trim", "").replace("mahalanobis", "mahal")
                method_row = indexed.get((config_id, method, q))
                row_out[f"{prefix}_threshold_label"] = method_row.get("threshold_label") if method_row else ""
                _attach_method_metrics(row_out, prefix, method_row)
            stein = indexed.get((config_id, "stein_trimmed", q))
            mahal = indexed.get((config_id, "mahalanobis_trim", q))
            score_norm = indexed.get((config_id, "score_norm_trim", q))
            gmm = indexed.get((config_id, "gmm_mahalanobis_trim", q))
            row_out.update(
                {
                    "delta_clean_coverage_stein_minus_mahal": _safe_diff(stein.get("clean_test_coverage_mean") if stein else None, mahal.get("clean_test_coverage_mean") if mahal else None),
                    "delta_width_stein_minus_mahal": _safe_diff(stein.get("average_interval_width_mean") if stein else None, mahal.get("average_interval_width_mean") if mahal else None),
                    "delta_p_d_mahal_minus_stein": _safe_diff(mahal.get("p_d_mean") if mahal else None, stein.get("p_d_mean") if stein else None),
                    "delta_tilde_epsilon_mahal_minus_stein": _safe_diff(mahal.get("tilde_epsilon_mean") if mahal else None, stein.get("tilde_epsilon_mean") if stein else None),
                    "delta_pc_minus_pd_stein_minus_mahal": _safe_diff(stein.get("p_c_minus_p_d_mean") if stein else None, mahal.get("p_c_minus_p_d_mean") if mahal else None),
                    "delta_clean_coverage_stein_minus_score_norm": _safe_diff(stein.get("clean_test_coverage_mean") if stein else None, score_norm.get("clean_test_coverage_mean") if score_norm else None),
                    "delta_p_d_score_norm_minus_stein": _safe_diff(score_norm.get("p_d_mean") if score_norm else None, stein.get("p_d_mean") if stein else None),
                    "delta_p_d_gmm_minus_stein": _safe_diff(gmm.get("p_d_mean") if gmm else None, stein.get("p_d_mean") if stein else None),
                }
            )
            out.append(row_out)
    return out
def _closest_by_retention(target_row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    target_retention = _as_float(target_row.get("retained_fraction_mean"))
    finite_candidates = [row for row in candidates if math.isfinite(_as_float(row.get("retained_fraction_mean")))]
    if not math.isfinite(target_retention) or not finite_candidates:
        return None
    return min(
        finite_candidates,
        key=lambda row: (
            abs(_as_float(row.get("retained_fraction_mean")) - target_retention),
            abs(_as_float(row.get("threshold_quantile"), 0.0) - _as_float(target_row.get("threshold_quantile"), 0.0)),
        ),
    )
def build_same_retention_rows(
    summary_rows: list[dict[str, Any]],
    *,
    target_method: str = "stein_trimmed",
    comparator_methods: tuple[str, ...] = ("score_norm_trim", "mahalanobis_trim", "gmm_mahalanobis_trim"),
) -> list[dict[str, Any]]:
    grouped = _group_rows(summary_rows)
    out: list[dict[str, Any]] = []
    for config_id, method_rows in sorted(grouped.items()):
        target_rows = method_rows.get(target_method, [])
        for target_row in sorted(target_rows, key=lambda row: _as_float(row.get("threshold_quantile"))):
            for comparator in comparator_methods:
                match = _closest_by_retention(target_row, method_rows.get(comparator, []))
                if match is None:
                    continue
                row_out = _copy_metadata(target_row)
                row_out.update(
                    {
                        "config_id": config_id,
                        "target_method_id": target_method,
                        "target_threshold_label": target_row.get("threshold_label"),
                        "target_threshold_quantile": target_row.get("threshold_quantile"),
                        "comparator_method_id": comparator,
                        "comparator_threshold_label": match.get("threshold_label"),
                        "comparator_threshold_quantile": match.get("threshold_quantile"),
                        "retention_gap_abs": abs(_as_float(target_row.get("retained_fraction_mean")) - _as_float(match.get("retained_fraction_mean"))),
                    }
                )
                _attach_method_metrics(row_out, "target", target_row)
                _attach_method_metrics(row_out, "comparator", match)
                row_out.update(
                    {
                        "delta_clean_coverage_target_minus_comparator": _safe_diff(target_row.get("clean_test_coverage_mean"), match.get("clean_test_coverage_mean")),
                        "delta_width_target_minus_comparator": _safe_diff(target_row.get("average_interval_width_mean"), match.get("average_interval_width_mean")),
                        "delta_p_d_comparator_minus_target": _safe_diff(match.get("p_d_mean"), target_row.get("p_d_mean")),
                        "delta_tilde_epsilon_comparator_minus_target": _safe_diff(match.get("tilde_epsilon_mean"), target_row.get("tilde_epsilon_mean")),
                        "delta_pc_minus_pd_target_minus_comparator": _safe_diff(target_row.get("p_c_minus_p_d_mean"), match.get("p_c_minus_p_d_mean")),
                    }
                )
                out.append(row_out)
    return out
def build_clarkson_side_by_side_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_rows(summary_rows)
    out: list[dict[str, Any]] = []
    methods = (
        "ordinary_split_conformal",
        "clarkson_contaminated_split",
        "stein_trimmed",
        "clean_oracle_trim",
        "cqr_conformal",
        "residual_trim",
    )
    for config_id, method_rows in sorted(grouped.items()):
        base = _first_method_row(method_rows, methods)
        if base is None:
            continue
        row_out = _copy_metadata(base)
        row_out.update({"config_id": config_id})
        for method in methods:
            rows = method_rows.get(method, [])
            if method == "stein_trimmed" and rows:
                selected, _ = _valid_shortest_select(rows, exclude_oracle=False)
            else:
                selected = rows[0] if rows else None
            prefix = method.replace("ordinary_split_conformal", "ordinary").replace("clarkson_contaminated_split", "clarkson").replace("clean_oracle_trim", "clean_oracle").replace("stein_trimmed", "stein").replace("_trim", "").replace("_conformal", "")
            row_out[f"{prefix}_threshold_label"] = selected.get("threshold_label") if selected else ""
            _attach_method_metrics(row_out, prefix, selected)
        out.append(row_out)
    return out
def build_explicit_feasibility_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if str(row.get("method_id")) != "stein_trimmed":
            continue
        enriched = _copy_metadata(row)
        enriched.update(
            {
                "method_id": row.get("method_id"),
                "threshold_label": row.get("threshold_label"),
                "threshold_quantile": row.get("threshold_quantile"),
                "clean_test_coverage_mean": row.get("clean_test_coverage_mean"),
                "average_interval_width_mean": row.get("average_interval_width_mean"),
                "theorem_lower_bound_mean": row.get("theorem_lower_bound_mean"),
                "semi_explicit_bound_mean": row.get("semi_explicit_bound_mean"),
                "fully_explicit_bound_mean": row.get("fully_explicit_bound_mean"),
                "semi_explicit_finite_rate": _as_float(row.get("semi_explicit_finite_mean")),
                "fully_explicit_finite_rate": _as_float(row.get("fully_explicit_finite_mean")),
                "semi_explicit_informative_rate": _as_float(row.get("semi_explicit_informative_mean")),
                "fully_explicit_informative_rate": _as_float(row.get("fully_explicit_informative_mean")),
                "explicit_pass_t_minus_positive_rate": _as_float(row.get("explicit_pass_t_minus_positive_mean")),
                "explicit_pass_t_plus_lt_local_bound_rate": _as_float(row.get("explicit_pass_t_plus_lt_local_bound_mean")),
                "explicit_pass_dirty_tail_moment_rate": _as_float(row.get("explicit_pass_dirty_tail_moment_mean")),
                "explicit_pass_clean_lower_positive_rate": _as_float(row.get("explicit_pass_clean_lower_positive_mean")),
                "explicit_pass_dirty_upper_finite_rate": _as_float(row.get("explicit_pass_dirty_upper_finite_mean")),
                "explicit_pass_bar_epsilon_finite_rate": _as_float(row.get("explicit_pass_bar_epsilon_finite_mean")),
                "explicit_feasibility_all_pass_rate": _as_float(row.get("explicit_feasibility_all_pass_mean")),
                "localized_feasible_rate": _as_float(row.get("localized_feasible_mean")),
                "explicit_fail_t_minus_nonpositive_rate": _as_float(row.get("explicit_fail_t_minus_nonpositive_mean")),
                "explicit_fail_t_plus_ge_l0_rate": _as_float(row.get("explicit_fail_t_plus_ge_l0_mean")),
                "explicit_fail_dirty_tail_condition_rate": _as_float(row.get("explicit_fail_dirty_tail_condition_mean")),
                "explicit_fail_clean_lower_nonpositive_rate": _as_float(row.get("explicit_fail_clean_lower_nonpositive_mean")),
                "explicit_fail_udr_nan_rate": _as_float(row.get("explicit_fail_udr_nan_mean")),
                "explicit_fail_bar_eps_nan_rate": _as_float(row.get("explicit_fail_bar_eps_nan_mean")),
                "explicit_bound_noninformative_rate": _as_float(row.get("explicit_bound_noninformative_mean")),
                "local_score_bound_mean": row.get("local_score_bound_mean"),
                "explicit_t_plus_mean": row.get("explicit_t_plus_mean"),
                "underline_m_d0_mean": row.get("underline_m_d0_mean"),
                "q_Q_mean": row.get("q_Q_mean"),
                "localized_clean_lower_mean": row.get("localized_clean_lower_mean"),
                "localized_dirty_upper_mean": row.get("localized_dirty_upper_mean"),
                "localized_bar_epsilon_mean": row.get("localized_bar_epsilon_mean"),
            }
        )
        rows.append(enriched)
    rows.sort(key=lambda row: (str(row.get("config_id")), _as_float(row.get("threshold_quantile")), _as_float(row.get("clean_reference_size")), _as_float(row.get("bandwidth_scale"))))
    return rows
def build_group_derived_artifacts(group_id: str, summary_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    outputs: dict[str, list[dict[str, Any]]] = {}
    if group_id in {"group_n_regime_map", "smoke_regime_map"}:
        outputs.update(
            {
                "best": build_group_n_best_rows(summary_rows),
                "best_tradeoff": build_group_n_best_tradeoff_rows(summary_rows),
            }
        )
    if group_id in {"group_o_separation_phase_diagram", "smoke_phase_diagram"}:
        outputs["phase"] = build_group_o_phase_rows(summary_rows)
    if group_id in {"group_p_quantile_density_stress", "smoke_quantile_density"}:
        outputs["density"] = build_group_p_density_rows(summary_rows)
    if group_id in {"group_q_concentration_calibration", "smoke_concentration_calibration"}:
        outputs["violation_rates"] = build_group_q_violation_rows(summary_rows)
    if group_id in {"group_r_bridge_ladder", "smoke_bridge_ladder"}:
        outputs["ladder"] = build_group_r_ladder_rows(summary_rows)
    if group_id in {"group_s_explicit_bound_stability", "smoke_explicit_stability", "group_j_explicit_bounds", "group_v_localized_certificate"}:
        outputs["stability"] = build_group_s_stability_rows(summary_rows)
        outputs["explicit_feasibility"] = build_explicit_feasibility_rows(summary_rows)
    if group_id in {"group_v_localized_certificate"}:
        outputs["method_level"] = build_method_level_rows(summary_rows)
    if group_id in {"group_x_ksd_mahalanobis_advantage", "smoke_ksd_mahalanobis_advantage", "group_y_stein_generalization", "smoke_stein_generalization", "group_t_nongaussian_covariate"}:
        outputs["method_level"] = build_method_level_rows(summary_rows)
        outputs["valid_shortest"] = build_valid_shortest_rows(summary_rows)
        outputs["same_quantile"] = build_same_quantile_rows(summary_rows)
        outputs["same_retention"] = build_same_retention_rows(summary_rows)
    if group_id in {"group_l_baselines", "smoke_baselines"}:
        outputs["method_level"] = build_method_level_rows(summary_rows)
        outputs["valid_shortest"] = build_valid_shortest_rows(summary_rows)
        outputs["clarkson_side_by_side"] = build_clarkson_side_by_side_rows(summary_rows)
    return outputs