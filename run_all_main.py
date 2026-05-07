from __future__ import annotations
import sys
from pathlib import Path
from tqdm import tqdm
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.analysis.index import build_experiment_index
from experiments.runners.groups import run_group_config
from experiments.utils.config import load_all_group_configs
def main() -> int:
    root = Path(__file__).resolve().parent
    config_dir = root / "configs"
    output_root = root / "results"
    group_configs = load_all_group_configs(config_dir)
    for group_config in tqdm(group_configs, desc="Running group configs"):
        run_group_config(group_config, output_root=output_root)
    index_text = build_experiment_index(group_configs, output_root=output_root)
    (root / "EXPERIMENT_INDEX.md").write_text(index_text, encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
