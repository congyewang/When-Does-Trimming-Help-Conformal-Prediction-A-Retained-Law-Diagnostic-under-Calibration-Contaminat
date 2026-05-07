from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from typing import Any
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"No rows provided for CSV output: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
def write_rows_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if os.environ.get("EXPERIMENTS_WRITE_PARQUET", "0") != "1":
        return
    ensure_dir(path.parent)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return
    if not rows:
        raise ValueError(f"No rows provided for Parquet output: {path}")
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    columns = {key: [row.get(key) for row in rows] for key in keys}
    try:
        table = pa.Table.from_pydict(columns)
        pq.write_table(table, path)
    except Exception:
        return
def read_rows_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
