from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.analysis.index import build_experiment_index
from experiments.runners.groups import run_group_config
from experiments.utils.config import load_group_config
SUPPLEMENTAL_GROUPS = (
    "group_a_main_theorem.yaml",
    "group_b_mixture.yaml",
    "group_c_ideal_interpolation.yaml",
    "group_d_no_separation.yaml",
    "group_j_explicit_bounds.yaml",
    "group_s_explicit_bound_stability.yaml",
    "group_v_localized_certificate.yaml",
    "group_w_large_m_tightness.yaml",
    "group_t_nongaussian_covariate.yaml",
    "group_x_ksd_mahalanobis_advantage.yaml",
    "group_y_stein_generalization.yaml",
    "group_u_label_only_failure.yaml",
)
def _resolve_group_paths(config_dir: Path, names: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for name in names:
        path = config_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing group config: {path}")
        paths.append(path)
    return paths
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the supplemental experiments needed for the revised paper tables."
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=list(SUPPLEMENTAL_GROUPS),
        help="Config filenames under experiments/configs. Defaults to the recommended supplemental groups.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to experiments/results.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure rendering and only write raw, aggregated, derived, and highlight tables.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_dir = root / "configs"
    output_root = args.output_root or (root / "results")
    if args.skip_figures:
        os.environ["EXPERIMENTS_SKIP_FIGURES"] = "1"
    group_configs = [load_group_config(path) for path in _resolve_group_paths(config_dir, args.groups)]
    for group_config in tqdm(group_configs, desc="Running supplemental groups"):
        run_group_config(group_config, output_root=output_root)
    index_text = build_experiment_index(group_configs, output_root=output_root)
    (root / "SUPPLEMENTAL_EXPERIMENT_INDEX.md").write_text(index_text, encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
