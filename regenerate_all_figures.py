from __future__ import annotations
import csv
import sys
from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.analysis.index import build_experiment_index
from experiments.figures.render import render_group_figure
from experiments.utils.config import load_all_group_configs
def _read_summary_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows
def main() -> int:
    root = Path(__file__).resolve().parent
    output_root = root / "results"
    group_configs = load_all_group_configs(root / "configs")
    for group_config in group_configs:
        summary_path = output_root / "aggregated" / f"{group_config.group_id}_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing cached summary for {group_config.group_id}: {summary_path}")
        summary_rows = _read_summary_rows(summary_path)
        render_group_figure(group_config, summary_rows, output_root / "figures" / group_config.group_id)
    index_text = build_experiment_index(group_configs, output_root=output_root)
    (root / "EXPERIMENT_INDEX.md").write_text(index_text, encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
