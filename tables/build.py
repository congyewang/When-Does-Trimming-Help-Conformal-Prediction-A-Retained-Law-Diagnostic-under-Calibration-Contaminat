from __future__ import annotations
import math
from pathlib import Path
from typing import Any
from experiments.utils.io import write_rows_csv
ORACLE_METHODS = {"clean_oracle_trim", "clean_oracle"}
def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed
def _coverage_target(row: dict[str, object]) -> float:
    alpha = _as_float(row.get("alpha"), 0.1)
    return 1.0 - alpha
def _method_is_oracle(row: dict[str, object]) -> bool:
    return str(row.get("method_id", "")) in ORACLE_METHODS
def _valid_shortest_row(
    rows: list[dict[str, object]],
    *,
    coverage_tolerance: float = 0.005,
    exclude_oracle: bool = True,
) -> tuple[dict[str, object], str]:
    candidate_rows = [row for row in rows if not (exclude_oracle and _method_is_oracle(row))]
    if not candidate_rows:
        candidate_rows = rows
    target = _coverage_target(candidate_rows[0]) - coverage_tolerance
    valid = [
        row
        for row in candidate_rows
        if _as_float(row.get("clean_test_coverage_mean")) >= target
        and math.isfinite(_as_float(row.get("average_interval_width_mean")))
    ]
    if valid:
        return (
            min(
                valid,
                key=lambda row: (
                    _as_float(row.get("average_interval_width_mean")),
                    -_as_float(row.get("clean_test_coverage_mean")),
                    _as_float(row.get("tilde_epsilon_mean"), 1.0),
                ),
            ),
            f"near_valid_shortest_nonoracle_clean_coverage>={target:.3f}",
        )
    return (
        max(
            candidate_rows,
            key=lambda row: (
                _as_float(row.get("clean_test_coverage_mean"), -float("inf")),
                -_as_float(row.get("average_interval_width_mean"), float("inf")),
            ),
        ),
        f"fallback_max_clean_coverage_nonoracle_no_method_reached_near_valid_{target:.3f}",
    )
def _coverage_best_row(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(
        rows,
        key=lambda row: (
            _as_float(row.get("clean_test_coverage_mean"), -float("inf")),
            -_as_float(row.get("average_interval_width_mean"), float("inf")),
        ),
    )
def _highlight_payload(config_id: str, row: dict[str, object], selection_rule: str) -> dict[str, object]:
    target = _coverage_target(row)
    clean_cov = _as_float(row.get("clean_test_coverage_mean"))
    width = _as_float(row.get("average_interval_width_mean"))
    return {
        "config_id": config_id,
        "best_method_id": row.get("method_id"),
        "best_threshold_label": row.get("threshold_label"),
        "best_threshold_quantile": row.get("threshold_quantile"),
        "selection_rule": selection_rule,
        "nominal_clean_coverage": target,
        "clean_coverage_margin_mean": clean_cov - target if math.isfinite(clean_cov) else float("nan"),
        "clean_test_coverage_mean": row.get("clean_test_coverage_mean"),
        "clean_test_coverage_se": row.get("clean_test_coverage_se"),
        "clean_test_coverage_ci95_low": clean_cov - 1.96 * _as_float(row.get("clean_test_coverage_se")) if math.isfinite(clean_cov) and math.isfinite(_as_float(row.get("clean_test_coverage_se"))) else float("nan"),
        "clean_test_coverage_ci95_high": clean_cov + 1.96 * _as_float(row.get("clean_test_coverage_se")) if math.isfinite(clean_cov) and math.isfinite(_as_float(row.get("clean_test_coverage_se"))) else float("nan"),
        "contaminated_mixture_coverage_mean": row.get("contaminated_mixture_coverage_mean"),
        "average_interval_width_mean": row.get("average_interval_width_mean"),
        "average_interval_width_se": row.get("average_interval_width_se"),
        "average_interval_width_ci95_low": width - 1.96 * _as_float(row.get("average_interval_width_se")) if math.isfinite(width) and math.isfinite(_as_float(row.get("average_interval_width_se"))) else float("nan"),
        "average_interval_width_ci95_high": width + 1.96 * _as_float(row.get("average_interval_width_se")) if math.isfinite(width) and math.isfinite(_as_float(row.get("average_interval_width_se"))) else float("nan"),
        "retained_fraction_mean": row.get("retained_fraction_mean"),
        "p_c_mean": row.get("p_c_mean"),
        "p_d_mean": row.get("p_d_mean"),
        "p_c_minus_p_d_mean": row.get("p_c_minus_p_d_mean"),
        "tilde_epsilon_mean": row.get("tilde_epsilon_mean"),
        "trimming_bias_mean": row.get("trimming_bias_mean"),
        "theorem_lower_bound_mean": row.get("theorem_lower_bound_mean"),
        "theorem_lower_bound_plus_mean": row.get("theorem_lower_bound_plus_mean"),
        "lpop_plus_mean": row.get("lpop_plus_mean"),
        "delta_trim_plus_mean": row.get("delta_trim_plus_mean"),
        "d_q_plus_mean": row.get("d_q_plus_mean"),
        "one_sided_transfer_loss_mean": row.get("one_sided_transfer_loss_mean"),
        "audit_theorem_lower_bound_plus_mean": row.get("audit_theorem_lower_bound_plus_mean"),
        "threshold_is_fallback_mean": row.get("threshold_is_fallback_mean"),
        "bound_gap_theorem_mean": row.get("bound_gap_theorem_mean"),
        "replicate_count": row.get("replicate_count"),
    }
def build_group_highlights(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    highlights: list[dict[str, object]] = []
    by_config: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_config.setdefault(str(row["config_id"]), []).append(row)
    for config_id, rows in sorted(by_config.items()):
        best, rule = _valid_shortest_row(rows, coverage_tolerance=0.005, exclude_oracle=True)
        highlights.append(_highlight_payload(config_id, best, rule))
    write_rows_csv(output_path, highlights)
def build_group_coverage_best_highlights(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    highlights: list[dict[str, object]] = []
    by_config: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_config.setdefault(str(row["config_id"]), []).append(row)
    for config_id, rows in sorted(by_config.items()):
        best = _coverage_best_row(rows)
        highlights.append(_highlight_payload(config_id, best, "coverage_best_including_oracle"))
    write_rows_csv(output_path, highlights)