from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.analysis.derived import build_group_derived_artifacts
from experiments.tables.build import build_group_coverage_best_highlights, build_group_highlights
from experiments.utils.io import write_rows_csv


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild highlight and diagnostic tables from an aggregated summary CSV. "
            "This is useful for existing runs: no experiment rerun is required."
        )
    )
    parser.add_argument("group_id", help="Group id, for example group_y_stein_generalization")
    parser.add_argument("summary_csv", type=Path, help="Path to <group>_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for output CSVs. Defaults to the summary CSV directory.")
    args = parser.parse_args()

    rows = _load_csv(args.summary_csv)
    output_dir = args.output_dir or args.summary_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    derived = build_group_derived_artifacts(args.group_id, rows)
    for suffix, derived_rows in derived.items():
        if derived_rows:
            write_rows_csv(output_dir / f"{args.group_id}_{suffix}.csv", derived_rows)

    build_group_highlights(rows, output_dir / f"{args.group_id}_highlights.csv")
    build_group_coverage_best_highlights(rows, output_dir / f"{args.group_id}_coverage_best_highlights.csv")
    print(f"wrote diagnostics for {args.group_id} to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
