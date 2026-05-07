from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
REQUIRED_SPEC_FIELDS = {
    "alpha",
    "epsilon",
    "train_size",
    "calibration_size",
    "clean_reference_size",
    "threshold_policy",
    "mc_reps",
    "random_seed",
    "dgp_family",
    "contamination_family",
    "predictor_type",
}
STANDALONE_GROUP_CONFIG_NAMES = {
    "group_z_clarkson_direct.yaml",
    "group_z_clarkson_regression.yaml",
}
@dataclass(frozen=True)
class ExperimentSpec:
    group_id: str
    config_id: str
    title: str
    description: str
    alpha: float
    epsilon: float
    train_size: int
    calibration_size: int
    clean_reference_size: int
    threshold_policy: str
    mc_reps: int
    random_seed: int
    dgp_family: str
    contamination_family: str
    predictor_type: str
    clean_test_size: int
    mixture_test_size: int
    structure_sample_size: int
    threshold_quantiles: tuple[float, ...]
    methods: tuple[str, ...]
    representation_family: str = "x_only"
    eta_fraction: float = 0.2
    bridge_delta: float = 0.1
    bridge_rho: float = 0.1
    population_sample_size: int = 4096
    evaluation_grid_size: int = 256
    compute_population_metrics: bool = False
    include_mixture_eval: bool = False
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    @classmethod
    def from_dict(cls, group_id: str, payload: dict[str, Any]) -> "ExperimentSpec":
        data = dict(payload)
        missing = sorted(REQUIRED_SPEC_FIELDS.difference(data))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Missing required spec fields: {missing_text}")
        known_fields = {
            "group_id",
            "config_id",
            "title",
            "description",
            "alpha",
            "epsilon",
            "train_size",
            "calibration_size",
            "clean_reference_size",
            "threshold_policy",
            "mc_reps",
            "random_seed",
            "dgp_family",
            "contamination_family",
            "predictor_type",
            "clean_test_size",
            "mixture_test_size",
            "structure_sample_size",
            "threshold_quantiles",
            "methods",
            "representation_family",
            "eta_fraction",
            "bridge_delta",
            "bridge_rho",
            "population_sample_size",
            "evaluation_grid_size",
            "compute_population_metrics",
            "include_mixture_eval",
            "notes",
        }
        extras = {key: value for key, value in data.items() if key not in known_fields}
        return cls(
            group_id=group_id,
            config_id=str(data["config_id"]),
            title=str(data.get("title", data["config_id"])),
            description=str(data.get("description", "")),
            alpha=float(data["alpha"]),
            epsilon=float(data["epsilon"]),
            train_size=int(data["train_size"]),
            calibration_size=int(data["calibration_size"]),
            clean_reference_size=int(data["clean_reference_size"]),
            threshold_policy=str(data["threshold_policy"]),
            mc_reps=int(data["mc_reps"]),
            random_seed=int(data["random_seed"]),
            dgp_family=str(data["dgp_family"]),
            contamination_family=str(data["contamination_family"]),
            predictor_type=str(data["predictor_type"]),
            clean_test_size=int(data.get("clean_test_size", 4000)),
            mixture_test_size=int(data.get("mixture_test_size", 4000)),
            structure_sample_size=int(data.get("structure_sample_size", 3000)),
            threshold_quantiles=tuple(float(value) for value in data.get("threshold_quantiles", [])),
            methods=tuple(str(value) for value in data.get("methods", ["stein_trimmed"])),
            representation_family=str(data.get("representation_family", "x_only")),
            eta_fraction=float(data.get("eta_fraction", 0.2)),
            bridge_delta=float(data.get("bridge_delta", 0.1)),
            bridge_rho=float(data.get("bridge_rho", 0.1)),
            population_sample_size=int(data.get("population_sample_size", 4096)),
            evaluation_grid_size=int(data.get("evaluation_grid_size", 256)),
            compute_population_metrics=bool(data.get("compute_population_metrics", False)),
            include_mixture_eval=bool(data.get("include_mixture_eval", False)),
            notes=str(data.get("notes", "")),
            extras=extras,
        )
    def as_metadata(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "config_id": self.config_id,
            "title": self.title,
            "description": self.description,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "train_size": self.train_size,
            "calibration_size": self.calibration_size,
            "clean_reference_size": self.clean_reference_size,
            "threshold_policy": self.threshold_policy,
            "mc_reps": self.mc_reps,
            "random_seed": self.random_seed,
            "dgp_family": self.dgp_family,
            "contamination_family": self.contamination_family,
            "predictor_type": self.predictor_type,
            "clean_test_size": self.clean_test_size,
            "mixture_test_size": self.mixture_test_size,
            "structure_sample_size": self.structure_sample_size,
            "threshold_quantiles": list(self.threshold_quantiles),
            "methods": list(self.methods),
            "representation_family": self.representation_family,
            "eta_fraction": self.eta_fraction,
            "bridge_delta": self.bridge_delta,
            "bridge_rho": self.bridge_rho,
            "population_sample_size": self.population_sample_size,
            "evaluation_grid_size": self.evaluation_grid_size,
            "compute_population_metrics": self.compute_population_metrics,
            "include_mixture_eval": self.include_mixture_eval,
            "notes": self.notes,
            "extras": self.extras,
        }
@dataclass(frozen=True)
class GroupConfig:
    path: Path
    group_id: str
    title: str
    description: str
    figure_kind: str
    theory_refs: tuple[str, ...]
    experiments: tuple[ExperimentSpec, ...]
def _merge_defaults(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged
def load_group_config(path: Path) -> GroupConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Group config {path} must contain a mapping.")
    if payload.get("standalone", False) or path.name in STANDALONE_GROUP_CONFIG_NAMES:
        raise ValueError(f"Standalone config {path} cannot be loaded as a group config.")
    if "group_id" not in payload:
        raise ValueError(f"Group config {path} is missing required top-level field: group_id")
    defaults = dict(payload.get("defaults", {}))
    experiments = []
    for item in payload.get("experiments", []):
        merged = _merge_defaults(defaults, item)
        experiments.append(ExperimentSpec.from_dict(str(payload["group_id"]), merged))
    return GroupConfig(
        path=path,
        group_id=str(payload["group_id"]),
        title=str(payload.get("title", payload["group_id"])),
        description=str(payload.get("description", "")),
        figure_kind=str(payload.get("figure_kind", "threshold_sweep")),
        theory_refs=tuple(str(ref) for ref in payload.get("theory_refs", [])),
        experiments=tuple(experiments),
    )
def load_all_group_configs(config_dir: Path) -> list[GroupConfig]:
    group_configs: list[GroupConfig] = []
    for path in sorted(config_dir.glob("group_*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if path.name in STANDALONE_GROUP_CONFIG_NAMES or (
            isinstance(payload, dict) and payload.get("standalone", False)
        ):
            continue
        group_configs.append(load_group_config(path))
    return group_configs
