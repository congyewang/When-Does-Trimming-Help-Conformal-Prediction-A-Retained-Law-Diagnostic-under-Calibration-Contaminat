from __future__ import annotations
import argparse
import sys
from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.runners.groups import run_group_config
from experiments.utils.config import load_group_config
def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the baseline comparison suite from a saved config.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(root / "configs" / "baselines" / "comparison_suite.yaml"),
        help="Path to a baseline group config. Defaults to the full baseline comparison suite.",
    )
    parser.add_argument(
        "--output-root",
        default=str(root / "results"),
        help="Root directory where raw results, summaries, and figures are written.",
    )
    return parser.parse_args()
def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    group_config = load_group_config(config_path)
    run_group_config(group_config, output_root=output_root)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
