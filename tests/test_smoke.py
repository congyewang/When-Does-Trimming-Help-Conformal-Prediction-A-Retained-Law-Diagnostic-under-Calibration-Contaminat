from __future__ import annotations
import csv
from dataclasses import replace
import math
import tempfile
import unittest
from pathlib import Path
from experiments.analysis.index import build_experiment_index
from experiments.figures.render import render_group_figure
from experiments.runners.core import _validate_oracle_score_support
from experiments.runners.groups import run_group_config
from experiments.utils.config import load_group_config
from experiments.utils.io import read_rows_csv
ROOT = Path(__file__).resolve().parents[1]
def _finite_numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(column, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values
def _distinct_rounded(values: list[float], digits: int = 8) -> set[float]:
    return {round(value, digits) for value in values if math.isfinite(value)}
class SmokeTests(unittest.TestCase):
    def test_smoke_configs_create_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            configs = [
                load_group_config(ROOT / "configs" / "smoke_tradeoff.yaml"),
                load_group_config(ROOT / "configs" / "smoke_bridge.yaml"),
                load_group_config(ROOT / "configs" / "smoke_baselines.yaml"),
            ]
            for config in configs:
                run_group_config(config, output_root)
            expected_summary_paths = [
                output_root / "aggregated" / "smoke_tradeoff_summary.csv",
                output_root / "aggregated" / "smoke_bridge_summary.csv",
                output_root / "aggregated" / "smoke_baselines_summary.csv",
            ]
            for summary_path in expected_summary_paths:
                self.assertTrue(summary_path.exists(), summary_path)
                with summary_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    columns = reader.fieldnames or []
                for required in [
                    "clean_test_coverage_mean",
                    "average_interval_width_mean",
                    "retained_fraction_mean",
                    "bound_gap_theorem_mean",
                ]:
                    self.assertIn(required, columns)
            for figure_name in ["smoke_tradeoff", "smoke_bridge", "smoke_baselines"]:
                self.assertTrue((output_root / "figures" / f"{figure_name}.png").exists())
                self.assertTrue((output_root / "figures" / f"{figure_name}.pdf").exists())
            index_text = build_experiment_index(configs, output_root)
            self.assertIn("smoke_tradeoff", index_text)
    def test_phase1_groups_create_outputs_and_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            configs = [
                load_group_config(ROOT / "configs" / "smoke_regime_map.yaml"),
                load_group_config(ROOT / "configs" / "smoke_phase_diagram.yaml"),
            ]
            for config in configs:
                run_group_config(config, output_root)
            regime_summary = output_root / "aggregated" / "smoke_regime_map_summary.csv"
            phase_summary = output_root / "aggregated" / "smoke_phase_diagram_summary.csv"
            self.assertTrue(regime_summary.exists(), regime_summary)
            self.assertTrue(phase_summary.exists(), phase_summary)
            with regime_summary.open(newline="", encoding="utf-8") as handle:
                regime_columns = csv.DictReader(handle).fieldnames or []
            with phase_summary.open(newline="", encoding="utf-8") as handle:
                phase_columns = csv.DictReader(handle).fieldnames or []
            for required in ["p_c_minus_p_d_mean", "net_tradeoff_mean", "separation_strength"]:
                self.assertIn(required, regime_columns)
            for required in ["p_c_minus_p_d_mean", "separation_strength", "phase_region"]:
                self.assertIn(required, phase_columns)
            regime_best = output_root / "aggregated" / "smoke_regime_map_best.csv"
            regime_best_tradeoff = output_root / "aggregated" / "smoke_regime_map_best_tradeoff.csv"
            phase_table = output_root / "aggregated" / "smoke_phase_diagram_phase.csv"
            self.assertTrue(regime_best.exists(), regime_best)
            self.assertTrue(regime_best_tradeoff.exists(), regime_best_tradeoff)
            self.assertTrue(phase_table.exists(), phase_table)
            with regime_best.open(newline="", encoding="utf-8") as handle:
                regime_best_columns = csv.DictReader(handle).fieldnames or []
            with regime_best_tradeoff.open(newline="", encoding="utf-8") as handle:
                regime_best_tradeoff_columns = csv.DictReader(handle).fieldnames or []
            with phase_table.open(newline="", encoding="utf-8") as handle:
                phase_table_columns = csv.DictReader(handle).fieldnames or []
            for required in ["coverage_gain_vs_ordinary_mean", "oracle_gap_mean", "width_gap_to_oracle_mean", "selection_rule"]:
                self.assertIn(required, regime_best_columns)
                self.assertIn(required, regime_best_tradeoff_columns)
                self.assertIn(required, phase_table_columns)
            self.assertIn("selection_metric_mean", regime_best_tradeoff_columns)
            rerender_root = output_root / "rerendered"
            for config in configs:
                cached_rows = read_rows_csv(output_root / "aggregated" / f"{config.group_id}_summary.csv")
                render_group_figure(config, cached_rows, rerender_root / config.group_id)
                self.assertTrue((rerender_root / f"{config.group_id}.png").exists())
                self.assertTrue((rerender_root / f"{config.group_id}.pdf").exists())
            index_text = build_experiment_index(configs, output_root)
            self.assertIn("best_tradeoff", index_text)
            self.assertIn("Derived Tables Used For Figures", index_text)
    def test_phase2_groups_create_outputs_and_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            configs = [
                load_group_config(ROOT / "configs" / "smoke_quantile_density.yaml"),
                load_group_config(ROOT / "configs" / "smoke_concentration_calibration.yaml"),
            ]
            for config in configs:
                run_group_config(config, output_root)
            quantile_summary = output_root / "aggregated" / "smoke_quantile_density_summary.csv"
            concentration_summary = output_root / "aggregated" / "smoke_concentration_calibration_summary.csv"
            self.assertTrue(quantile_summary.exists(), quantile_summary)
            self.assertTrue(concentration_summary.exists(), concentration_summary)
            with quantile_summary.open(newline="", encoding="utf-8") as handle:
                quantile_columns = csv.DictReader(handle).fieldnames or []
            with concentration_summary.open(newline="", encoding="utf-8") as handle:
                concentration_columns = csv.DictReader(handle).fieldnames or []
            for required in ["lambda_min_est_mean", "quantile_shift_bound_bias_mean", "quantile_shift_bound_bias_looseness_mean"]:
                self.assertIn(required, quantile_columns)
            for required in ["clean_count_violation_mean", "dirty_count_violation_mean", "retained_contamination_upper_violation_mean"]:
                self.assertIn(required, concentration_columns)
            quantile_density = output_root / "aggregated" / "smoke_quantile_density_density.csv"
            concentration_violation_rates = output_root / "aggregated" / "smoke_concentration_calibration_violation_rates.csv"
            self.assertTrue(quantile_density.exists(), quantile_density)
            self.assertTrue(concentration_violation_rates.exists(), concentration_violation_rates)
            with quantile_density.open(newline="", encoding="utf-8") as handle:
                quantile_density_columns = csv.DictReader(handle).fieldnames or []
            with concentration_violation_rates.open(newline="", encoding="utf-8") as handle:
                concentration_violation_columns = csv.DictReader(handle).fieldnames or []
            for required in ["density_regime", "bound_looseness_bias_mean", "inverse_lambda_min_est_mean", "bound_efficiency_ratio_mean"]:
                self.assertIn(required, quantile_density_columns)
            for required in [
                "clean_violation_rate",
                "dirty_violation_rate",
                "keep_violation_rate",
                "contamination_upper_exceedance_rate",
                "contamination_upper_defined_fraction",
                "nominal_bound_proxy",
            ]:
                self.assertIn(required, concentration_violation_columns)
            quantile_density_rows = read_rows_csv(quantile_density)
            density_regimes = {str(row.get("density_regime")) for row in quantile_density_rows if str(row.get("density_regime"))}
            self.assertGreater(len(density_regimes), 1)
            lambda_values = _finite_numeric_values(quantile_density_rows, "lambda_min_est_mean")
            self.assertTrue(lambda_values)
            self.assertGreater(len(_distinct_rounded(lambda_values)), 1)
            concentration_violation_rows = read_rows_csv(concentration_violation_rates)
            proxy_values = _finite_numeric_values(concentration_violation_rows, "nominal_bound_proxy")
            self.assertTrue(proxy_values)
            self.assertGreater(len(_distinct_rounded(proxy_values)), 1)
            defined_fraction_values = _finite_numeric_values(
                concentration_violation_rows,
                "contamination_upper_defined_fraction",
            )
            self.assertTrue(defined_fraction_values)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in defined_fraction_values))
            rate_columns = [
                "clean_violation_rate",
                "dirty_violation_rate",
                "keep_violation_rate",
                "contamination_upper_exceedance_rate",
            ]
            finite_rate_columns = {
                column: _finite_numeric_values(concentration_violation_rows, column) for column in rate_columns
            }
            self.assertTrue(any(values for values in finite_rate_columns.values()))
            self.assertTrue(
                len(
                    {
                        tuple(
                            round(value, 8) if math.isfinite(value) else float("nan")
                            for value in (
                                float(row.get("clean_violation_rate", "nan")),
                                float(row.get("dirty_violation_rate", "nan")),
                                float(row.get("keep_violation_rate", "nan")),
                                float(row.get("contamination_upper_exceedance_rate", "nan")),
                                float(row.get("nominal_bound_proxy", "nan")),
                            )
                        )
                        for row in concentration_violation_rows
                    }
                )
                > 1
            )
            rerender_root = output_root / "rerendered_phase2"
            for config in configs:
                cached_rows = read_rows_csv(output_root / "aggregated" / f"{config.group_id}_summary.csv")
                render_group_figure(config, cached_rows, rerender_root / config.group_id)
                self.assertTrue((rerender_root / f"{config.group_id}.png").exists())
                self.assertTrue((rerender_root / f"{config.group_id}.pdf").exists())
            index_text = build_experiment_index(configs, output_root)
            self.assertIn("smoke_quantile_density_density.csv", index_text)
            self.assertIn("smoke_concentration_calibration_violation_rates.csv", index_text)
    def test_phase3_groups_create_outputs_and_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            configs = [
                load_group_config(ROOT / "configs" / "smoke_bridge_ladder.yaml"),
                load_group_config(ROOT / "configs" / "smoke_explicit_stability.yaml"),
            ]
            for config in configs:
                run_group_config(config, output_root)
            bridge_summary = output_root / "aggregated" / "smoke_bridge_ladder_summary.csv"
            explicit_summary = output_root / "aggregated" / "smoke_explicit_stability_summary.csv"
            self.assertTrue(bridge_summary.exists(), bridge_summary)
            self.assertTrue(explicit_summary.exists(), explicit_summary)
            with bridge_summary.open(newline="", encoding="utf-8") as handle:
                bridge_columns = csv.DictReader(handle).fieldnames or []
            with explicit_summary.open(newline="", encoding="utf-8") as handle:
                explicit_columns = csv.DictReader(handle).fieldnames or []
            for required in [
                "sup_score_error_mean",
                "score_model_sup_error_mean",
                "bridge_delta_s_mean",
                "bridge_event_holds_mean",
                "bridge_tilde_epsilon_upper_mean",
            ]:
                self.assertIn(required, bridge_columns)
            for required in [
                "semi_explicit_bound_mean",
                "fully_explicit_bound_mean",
                "explicit_fail_t_minus_nonpositive_mean",
                "explicit_fail_udr_nan_mean",
                "explicit_bound_noninformative_mean",
            ]:
                self.assertIn(required, explicit_columns)
            bridge_ladder = output_root / "aggregated" / "smoke_bridge_ladder_ladder.csv"
            explicit_stability = output_root / "aggregated" / "smoke_explicit_stability_stability.csv"
            self.assertTrue(bridge_ladder.exists(), bridge_ladder)
            self.assertTrue(explicit_stability.exists(), explicit_stability)
            with bridge_ladder.open(newline="", encoding="utf-8") as handle:
                bridge_ladder_columns = csv.DictReader(handle).fieldnames or []
            with explicit_stability.open(newline="", encoding="utf-8") as handle:
                explicit_stability_columns = csv.DictReader(handle).fieldnames or []
            for required in [
                "score_model_mode",
                "bridge_margin_mean",
                "bridge_upper_slack_mean",
                "score_error_ratio_mean",
                "score_model_sup_error_mean",
            ]:
                self.assertIn(required, bridge_ladder_columns)
            for required in [
                "finite_explicit_fraction",
                "informative_explicit_fraction",
                "explicit_fail_t_minus_nonpositive_rate",
                "explicit_fail_udr_nan_rate",
                "explicit_bound_noninformative_rate",
                "fully_minus_semi_mean",
            ]:
                self.assertIn(required, explicit_stability_columns)
            bridge_ladder_rows = read_rows_csv(bridge_ladder)
            bridge_modes = {str(row.get("score_model_mode")) for row in bridge_ladder_rows if str(row.get("score_model_mode"))}
            self.assertGreater(len(bridge_modes), 1)
            bridge_model_errors = _finite_numeric_values(bridge_ladder_rows, "score_model_sup_error_mean")
            self.assertTrue(bridge_model_errors)
            self.assertGreater(len(_distinct_rounded(bridge_model_errors)), 1)
            bridge_upper_slack = _finite_numeric_values(bridge_ladder_rows, "bridge_upper_slack_mean")
            self.assertTrue(bridge_upper_slack)
            explicit_stability_rows = read_rows_csv(explicit_stability)
            threshold_quantiles = _finite_numeric_values(explicit_stability_rows, "threshold_quantile")
            self.assertGreater(len(_distinct_rounded(threshold_quantiles)), 1)
            finite_fractions = _finite_numeric_values(explicit_stability_rows, "finite_explicit_fraction")
            informative_fractions = _finite_numeric_values(explicit_stability_rows, "informative_explicit_fraction")
            self.assertTrue(finite_fractions)
            self.assertTrue(informative_fractions)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in finite_fractions))
            self.assertTrue(all(0.0 <= value <= 1.0 for value in informative_fractions))
            failure_columns = [
                "explicit_fail_t_minus_nonpositive_rate",
                "explicit_fail_udr_nan_rate",
                "explicit_fail_mu_lower_nan_rate",
                "explicit_fail_bar_eps_nan_rate",
                "explicit_fail_fully_bias_nan_rate",
                "explicit_bound_noninformative_rate",
            ]
            finite_failure_columns = {
                column: _finite_numeric_values(explicit_stability_rows, column) for column in failure_columns
            }
            self.assertTrue(any(values for values in finite_failure_columns.values()))
            rerender_root = output_root / "rerendered_phase3"
            for config in configs:
                cached_rows = read_rows_csv(output_root / "aggregated" / f"{config.group_id}_summary.csv")
                render_group_figure(config, cached_rows, rerender_root / config.group_id)
                self.assertTrue((rerender_root / f"{config.group_id}.png").exists())
                self.assertTrue((rerender_root / f"{config.group_id}.pdf").exists())
            index_text = build_experiment_index(configs, output_root)
            self.assertIn("smoke_bridge_ladder_ladder.csv", index_text)
            self.assertIn("smoke_explicit_stability_stability.csv", index_text)
    def test_phase3_oracle_guard_rejects_unsupported_clean_law(self) -> None:
        oracle_spec = load_group_config(ROOT / "configs" / "smoke_bridge_ladder.yaml").experiments[0]
        unsupported = replace(
            oracle_spec,
            dgp_family="unsupported_non_gaussian",
        )
        with self.assertRaisesRegex(ValueError, "score_model_mode='oracle'"):
            _validate_oracle_score_support(unsupported)
if __name__ == "__main__":
    unittest.main()
