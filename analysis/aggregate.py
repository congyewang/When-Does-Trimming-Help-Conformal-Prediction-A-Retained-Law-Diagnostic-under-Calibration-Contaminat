from __future__ import annotations
from collections import defaultdict
import math
from typing import Any
from experiments.utils.stats import mean_std_se
GROUP_FIELDS = [
    "group_id",
    "config_id",
    "experiment_title",
    "description",
    "method_id",
    "baseline_adapter_id",
    "baseline_tier",
    "baseline_appendix_only",
    "threshold_policy",
    "threshold_label",
    "threshold_quantile",
    "threshold_source",
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
]
EXCLUDE_FROM_AGGREGATION = set(GROUP_FIELDS + ["replicate"])
ADAPTIVE_THRESHOLD_POLICIES = {"ksd_geometric", "ksd_geometric_localized"}

def _group_field_value(row: dict[str, Any], field: str) -> Any:
    """Return the grouping value for a raw row.

    Adaptive KSD policies may realize either a geometric threshold or a fallback
    threshold in different Monte Carlo replicates.  For policy-level summaries
    these should be aggregated together; fallback frequency is reported through
    numeric fields such as ``threshold_is_fallback_mean``.
    """
    threshold_policy = str(row.get("threshold_policy", ""))
    if threshold_policy in ADAPTIVE_THRESHOLD_POLICIES:
        if field == "threshold_label":
            return f"policy:{threshold_policy}"
        if field == "threshold_source":
            return "adaptive_policy_realized"
        if field == "threshold_quantile":
            return None
    return row.get(field)
def _normalize_group_value(value: Any) -> Any:
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_normalize_group_value(_group_field_value(row, field)) for field in GROUP_FIELDS)
        grouped[key].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, chunk in grouped.items():
        summary = {field: value for field, value in zip(GROUP_FIELDS, key)}
        metric_keys = [metric for metric in chunk[0].keys() if metric not in EXCLUDE_FROM_AGGREGATION]
        for metric in metric_keys:
            raw_values = [row.get(metric) for row in chunk]
            numeric_values: list[float] = []
            for value in raw_values:
                if value is None:
                    continue
                try:
                    numeric_values.append(float(value))
                except (TypeError, ValueError):
                    numeric_values = []
                    break
            if not numeric_values:
                continue
            mean, std, se = mean_std_se(numeric_values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_se"] = se
        summary["replicate_count"] = len(chunk)
        summary_rows.append(summary)
    summary_rows.sort(key=lambda row: (row["config_id"], row["method_id"], str(row["threshold_label"])))
    return summary_rows