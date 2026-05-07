from __future__ import annotations
import os
from pathlib import Path
from experiments.analysis.aggregate import aggregate_rows
from experiments.analysis.derived import build_group_derived_artifacts
from experiments.tables.build import build_group_coverage_best_highlights, build_group_highlights
from experiments.utils.config import GroupConfig
from experiments.utils.io import write_json, write_rows_csv
from experiments.runners.core import persist_raw_rows, run_experiment_spec

def run_group_config(group_config: GroupConfig, output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in group_config.experiments:
        rows.extend(run_experiment_spec(spec, output_root=output_root))
    persist_raw_rows(group_config.group_id, rows, output_root)
    summary_rows = aggregate_rows(rows)
    aggregated_dir = output_root / "aggregated"
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(aggregated_dir / f"{group_config.group_id}_summary.csv", summary_rows)
    derived_outputs = build_group_derived_artifacts(group_config.group_id, summary_rows)
    for suffix, derived_rows in derived_outputs.items():
        if derived_rows:
            write_rows_csv(aggregated_dir / f"{group_config.group_id}_{suffix}.csv", derived_rows)
    build_group_highlights(summary_rows, tables_dir / f"{group_config.group_id}_highlights.csv")
    build_group_coverage_best_highlights(summary_rows, tables_dir / f"{group_config.group_id}_coverage_best_highlights.csv")
    if os.environ.get("EXPERIMENTS_SKIP_FIGURES", "0") not in {"1", "true", "TRUE", "yes", "YES"}:
        from experiments.figures.render import render_group_figure
        render_group_figure(group_config, summary_rows, figures_dir / group_config.group_id)
    write_json(
        aggregated_dir / f"{group_config.group_id}_metadata.json",
        {
            "group_id": group_config.group_id,
            "title": group_config.title,
            "description": group_config.description,
            "figure_kind": group_config.figure_kind,
            "theory_refs": list(group_config.theory_refs),
            "num_replicate_rows": len(rows),
            "num_summary_rows": len(summary_rows),
            "derived_tables": [f"{group_config.group_id}_{suffix}.csv" for suffix in derived_outputs if derived_outputs[suffix]],
            "figures_skipped": os.environ.get("EXPERIMENTS_SKIP_FIGURES", "0") in {"1", "true", "TRUE", "yes", "YES"},
        },
    )
    return summary_rows
