from __future__ import annotations
import os
os.environ.setdefault("EXPERIMENTS_SKIP_FIGURES", "1")
import csv
from pathlib import Path
from experiments.runners.groups import run_group_config
from experiments.utils.config import load_group_config
ROOT = Path(__file__).resolve().parents[1]
def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
def test_bimodal_slope_ksd_separates_better_than_mahalanobis(tmp_path: Path) -> None:
    config = load_group_config(ROOT / "configs" / "smoke_ksd_mahalanobis_advantage.yaml")
    run_group_config(config, tmp_path)
    rows = _read_summary(
        tmp_path / "aggregated" / "smoke_ksd_mahalanobis_advantage_summary.csv"
    )
    stein_rows = [row for row in rows if row["method_id"] == "stein_trimmed"]
    mahal_rows = [row for row in rows if row["method_id"] == "mahalanobis_trim"]
    assert stein_rows
    assert mahal_rows
    best_stein = max(stein_rows, key=lambda row: float(row["p_c_minus_p_d_mean"]))
    best_mahal = max(mahal_rows, key=lambda row: float(row["p_c_minus_p_d_mean"]))
    assert float(best_stein["p_c_minus_p_d_mean"]) > 0.5
    assert float(best_stein["tilde_epsilon_mean"]) < 0.05
    assert float(best_mahal["p_c_minus_p_d_mean"]) < 0.0
    assert float(best_mahal["p_d_mean"]) > 0.95
def test_kde_stein_generalization_smoke_outputs_new_comparators(tmp_path: Path) -> None:
    config = load_group_config(ROOT / "configs" / "smoke_stein_generalization.yaml")
    run_group_config(config, tmp_path)
    rows = _read_summary(
        tmp_path / "aggregated" / "smoke_stein_generalization_summary.csv"
    )
    methods = {row["method_id"] for row in rows}
    for required in {
        "stein_trimmed",
        "score_norm_trim",
        "mahalanobis_trim",
        "gmm_mahalanobis_trim",
        "ordinary_split_conformal",
        "clean_oracle_trim",
    }:
        assert required in methods
    stein_rows = [row for row in rows if row["method_id"] == "stein_trimmed"]
    mahal_rows = [row for row in rows if row["method_id"] == "mahalanobis_trim"]
    assert stein_rows and mahal_rows
    best_stein = max(stein_rows, key=lambda row: float(row["p_c_minus_p_d_mean"]))
    best_mahal = max(mahal_rows, key=lambda row: float(row["p_c_minus_p_d_mean"]))
    assert float(best_stein["p_c_minus_p_d_mean"]) > float(best_mahal["p_c_minus_p_d_mean"])
    assert float(best_stein["tilde_epsilon_mean"]) < 0.05
