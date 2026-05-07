from __future__ import annotations
import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any
NUMERIC = [
    "clean_test_coverage_mean",
    "contaminated_mixture_coverage_mean",
    "average_interval_width_mean",
    "retained_fraction_mean",
    "p_c_mean",
    "p_d_mean",
    "p_c_minus_p_d_mean",
    "tilde_epsilon_mean",
    "trimming_bias_mean",
    "bound_gap_theorem_mean",
]
OUT_FIELDS = [
    "config_id",
    "quantile",
    "stein_threshold",
    "mahal_threshold",
    "stein_clean_coverage",
    "mahal_clean_coverage",
    "stein_width",
    "mahal_width",
    "stein_p_c",
    "mahal_p_c",
    "stein_p_d",
    "mahal_p_d",
    "stein_pc_minus_pd",
    "mahal_pc_minus_pd",
    "stein_tilde_epsilon",
    "mahal_tilde_epsilon",
    "stein_trimming_bias",
    "mahal_trimming_bias",
    "delta_clean_coverage_stein_minus_mahal",
    "delta_width_stein_minus_mahal",
    "delta_p_d_mahal_minus_stein",
    "delta_tilde_epsilon_mahal_minus_stein",
    "delta_pc_minus_pd_stein_minus_mahal",
]
BEST_FIELDS = [
    "config_id",
    "selected_method_id",
    "selected_threshold_label",
    "selection_rule",
    "clean_test_coverage_mean",
    "average_interval_width_mean",
    "contaminated_mixture_coverage_mean",
    "retained_fraction_mean",
    "p_c_mean",
    "p_d_mean",
    "p_c_minus_p_d_mean",
    "tilde_epsilon_mean",
    "trimming_bias_mean",
    "bound_gap_theorem_mean",
]
def as_float(x: Any) -> float:
    try:
        if x is None or x == "":
            return math.nan
        return float(x)
    except (TypeError, ValueError):
        return math.nan
def quantile_from_label(label: str) -> str | None:
    m = re.search(r"(?:mahal_)?q=([0-9.]+)", label)
    if not m:
        return None
    return f"{float(m.group(1)):.3f}"
def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))
def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
def compare_same_quantiles(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        method = row.get("method_id", "")
        if method not in {"stein_trimmed", "mahalanobis_trim"}:
            continue
        q = quantile_from_label(row.get("threshold_label", ""))
        if q is None:
            continue
        indexed[(row.get("config_id", ""), method, q)] = row
    out: list[dict[str, Any]] = []
    configs = sorted({key[0] for key in indexed})
    qs = sorted({key[2] for key in indexed})
    for config_id in configs:
        for q in qs:
            s = indexed.get((config_id, "stein_trimmed", q))
            m = indexed.get((config_id, "mahalanobis_trim", q))
            if s is None or m is None:
                continue
            row: dict[str, Any] = {
                "config_id": config_id,
                "quantile": q,
                "stein_threshold": s.get("threshold_label", ""),
                "mahal_threshold": m.get("threshold_label", ""),
            }
            mapping = {
                "clean_test_coverage_mean": "clean_coverage",
                "average_interval_width_mean": "width",
                "p_c_mean": "p_c",
                "p_d_mean": "p_d",
                "p_c_minus_p_d_mean": "pc_minus_pd",
                "tilde_epsilon_mean": "tilde_epsilon",
                "trimming_bias_mean": "trimming_bias",
            }
            for src, dst in mapping.items():
                row[f"stein_{dst}"] = as_float(s.get(src))
                row[f"mahal_{dst}"] = as_float(m.get(src))
            row["delta_clean_coverage_stein_minus_mahal"] = row["stein_clean_coverage"] - row["mahal_clean_coverage"]
            row["delta_width_stein_minus_mahal"] = row["stein_width"] - row["mahal_width"]
            row["delta_p_d_mahal_minus_stein"] = row["mahal_p_d"] - row["stein_p_d"]
            row["delta_tilde_epsilon_mahal_minus_stein"] = row["mahal_tilde_epsilon"] - row["stein_tilde_epsilon"]
            row["delta_pc_minus_pd_stein_minus_mahal"] = row["stein_pc_minus_pd"] - row["mahal_pc_minus_pd"]
            out.append(row)
    return out
def valid_shortest_best(rows: list[dict[str, str]], alpha: float, tolerance: float) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_config.setdefault(row.get("config_id", ""), []).append(row)
    out: list[dict[str, Any]] = []
    target = 1.0 - alpha
    for config_id, chunk in sorted(by_config.items()):
        valid = [r for r in chunk if as_float(r.get("clean_test_coverage_mean")) >= target - tolerance]
        if valid:
            chosen = min(valid, key=lambda r: as_float(r.get("average_interval_width_mean")))
            rule = f"min_width_subject_to_clean_coverage>={target - tolerance:.3f}"
        else:
            chosen = max(chunk, key=lambda r: as_float(r.get("clean_test_coverage_mean")))
            rule = f"max_clean_coverage_no_method_reached_{target - tolerance:.3f}"
        out_row: dict[str, Any] = {
            "config_id": config_id,
            "selected_method_id": chosen.get("method_id", ""),
            "selected_threshold_label": chosen.get("threshold_label", ""),
            "selection_rule": rule,
        }
        for field in BEST_FIELDS[4:]:
            out_row[field] = chosen.get(field, "")
        out.append(out_row)
    return out
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--coverage-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    rows = load_rows(args.summary_csv)
    comparison = compare_same_quantiles(rows)
    best = valid_shortest_best(rows, args.alpha, args.coverage_tolerance)
    compare_path = args.output_dir / "group_x_ksd_vs_mahalanobis_same_quantile.csv"
    best_path = args.output_dir / "group_x_valid_shortest_highlights.csv"
    write_csv(compare_path, comparison, OUT_FIELDS)
    write_csv(best_path, best, BEST_FIELDS)
    print(f"wrote {compare_path}")
    print(f"wrote {best_path}")
    if comparison:
        interesting = [r for r in comparison if r["config_id"] != "bimodal_center_negative_control"]
        if interesting:
            avg_pd_gap = sum(r["delta_p_d_mahal_minus_stein"] for r in interesting) / len(interesting)
            avg_tilde_gap = sum(r["delta_tilde_epsilon_mahal_minus_stein"] for r in interesting) / len(interesting)
            print(f"mean Mahalanobis-minus-Stein p_d gap on slope configs: {avg_pd_gap:.4f}")
            print(f"mean Mahalanobis-minus-Stein tilde_epsilon gap on slope configs: {avg_tilde_gap:.4f}")
if __name__ == "__main__":
    main()
