from __future__ import annotations
from pathlib import Path
from experiments.utils.config import GroupConfig
def _derived_table_paths(config: GroupConfig) -> list[str]:
    if config.group_id in {"group_n_regime_map", "smoke_regime_map"}:
        return [
            f"results/aggregated/{config.group_id}_best.csv",
            f"results/aggregated/{config.group_id}_best_tradeoff.csv",
        ]
    if config.group_id in {"group_o_separation_phase_diagram", "smoke_phase_diagram"}:
        return [f"results/aggregated/{config.group_id}_phase.csv"]
    if config.group_id in {"group_p_quantile_density_stress", "smoke_quantile_density"}:
        return [f"results/aggregated/{config.group_id}_density.csv"]
    if config.group_id in {"group_q_concentration_calibration", "smoke_concentration_calibration"}:
        return [f"results/aggregated/{config.group_id}_violation_rates.csv"]
    if config.group_id in {"group_r_bridge_ladder", "smoke_bridge_ladder"}:
        return [f"results/aggregated/{config.group_id}_ladder.csv"]
    if config.group_id in {"group_s_explicit_bound_stability", "smoke_explicit_stability"}:
        return [f"results/aggregated/{config.group_id}_stability.csv"]
    if config.group_id == "group_v_localized_certificate":
        return [
            f"results/aggregated/{config.group_id}_stability.csv",
            f"results/aggregated/{config.group_id}_explicit_feasibility.csv",
            f"results/aggregated/{config.group_id}_method_level.csv",
        ]
    return []
def build_experiment_index(group_configs: list[GroupConfig], output_root: Path) -> str:
    lines = [
        "# Experiment Index",
        "",
        "| Theory result | Group | Config file | Raw Summary | Derived Tables Used For Figures | Figure |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for config in group_configs:
        summary_path = f"results/aggregated/{config.group_id}_summary.csv"
        derived_paths = "; ".join(f"`{path}`" for path in _derived_table_paths(config)) or "-"
        figure_path = f"results/figures/{config.group_id}.png"
        for theory_ref in config.theory_refs:
            lines.append(
                f"| `{theory_ref}` | `{config.group_id}` | `configs/{config.path.name}` | `{summary_path}` | {derived_paths} | `{figure_path}` |"
            )
    lines.extend(
        [
            "",
            f"Generated from `{output_root}`.",
        ]
    )
    return "\n".join(lines)