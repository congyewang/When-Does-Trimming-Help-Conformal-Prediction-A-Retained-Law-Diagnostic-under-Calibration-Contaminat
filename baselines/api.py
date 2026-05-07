from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from experiments.utils.config import ExperimentSpec
from experiments.utils.conformal import CalibrationResult, calibrate_from_scores
from experiments.utils.data import Dataset
from experiments.utils.models import Predictor
Array = np.ndarray
@dataclass(frozen=True)
class BaselineContext:
    spec: ExperimentSpec
    calibration_sample: Dataset
    clean_structure: Dataset
    dirty_structure: Dataset
    linear_predictor: Predictor
    cqr_predictor: Predictor
    calibration_linear_scores: Array
    calibration_cqr_scores: Array
    clean_structure_linear_scores: Array
    dirty_structure_linear_scores: Array
    clean_structure_cqr_scores: Array
    dirty_structure_cqr_scores: Array
@dataclass(frozen=True)
class BaselineRun:
    calibration: CalibrationResult
    keep_mask: Array
    predictor: Predictor
    clean_structure_scores: Array
    dirty_structure_scores: Array
    clean_structure_nonconformity: Array
    threshold_label: str
    threshold_quantile: float
    threshold_value: float
    threshold_source: str
    extra_row_fields: dict[str, Any] = field(default_factory=dict)
BaselineRunner = Callable[[BaselineContext], BaselineRun]
@dataclass(frozen=True)
class BaselineAdapter:
    adapter_id: str
    problem: str
    comparable_object: str
    citation: str
    tier: str
    runner: BaselineRunner
    appendix_only: bool = False
    allowed_contamination_families: tuple[str, ...] = ()
    def execute(self, context: BaselineContext) -> BaselineRun:
        if self.allowed_contamination_families and context.spec.contamination_family not in self.allowed_contamination_families:
            allowed = ", ".join(self.allowed_contamination_families)
            raise ValueError(
                f"Baseline `{self.adapter_id}` is only compatible with contamination families "
                f"[{allowed}], but received `{context.spec.contamination_family}`."
            )
        run = self.runner(context)
        metadata = dict(run.extra_row_fields)
        metadata.update(
            {
                "baseline_adapter_id": self.adapter_id,
                "baseline_problem": self.problem,
                "baseline_comparable_object": self.comparable_object,
                "baseline_citation": self.citation,
                "baseline_tier": self.tier,
                "baseline_appendix_only": float(self.appendix_only),
            }
        )
        return BaselineRun(
            calibration=run.calibration,
            keep_mask=run.keep_mask,
            predictor=run.predictor,
            clean_structure_scores=run.clean_structure_scores,
            dirty_structure_scores=run.dirty_structure_scores,
            clean_structure_nonconformity=run.clean_structure_nonconformity,
            threshold_label=run.threshold_label,
            threshold_quantile=run.threshold_quantile,
            threshold_value=run.threshold_value,
            threshold_source=run.threshold_source,
            extra_row_fields=metadata,
        )
def _all_keep_mask(size: int) -> Array:
    return np.ones(size, dtype=bool)
def _all_keep_scores(template: Array) -> Array:
    return np.zeros_like(template, dtype=float)
def _ordinary_split(context: BaselineContext, method_label: str) -> BaselineRun:
    keep_mask = _all_keep_mask(context.spec.calibration_size)
    return BaselineRun(
        calibration=calibrate_from_scores(context.calibration_linear_scores, alpha=context.spec.alpha, keep_mask=keep_mask),
        keep_mask=keep_mask,
        predictor=context.linear_predictor,
        clean_structure_scores=_all_keep_scores(context.clean_structure_linear_scores),
        dirty_structure_scores=_all_keep_scores(context.dirty_structure_linear_scores),
        clean_structure_nonconformity=context.clean_structure_linear_scores,
        threshold_label=method_label,
        threshold_quantile=float("nan"),
        threshold_value=0.0,
        threshold_source="baseline_all_keep",
    )
def _cqr_split(context: BaselineContext, method_label: str) -> BaselineRun:
    keep_mask = _all_keep_mask(context.spec.calibration_size)
    return BaselineRun(
        calibration=calibrate_from_scores(context.calibration_cqr_scores, alpha=context.spec.alpha, keep_mask=keep_mask),
        keep_mask=keep_mask,
        predictor=context.cqr_predictor,
        clean_structure_scores=_all_keep_scores(context.clean_structure_cqr_scores),
        dirty_structure_scores=_all_keep_scores(context.dirty_structure_cqr_scores),
        clean_structure_nonconformity=context.clean_structure_cqr_scores,
        threshold_label=method_label,
        threshold_quantile=float("nan"),
        threshold_value=0.0,
        threshold_source="baseline_all_keep",
    )
def _clean_oracle_trim(context: BaselineContext) -> BaselineRun:
    keep_mask = ~context.calibration_sample.dirty
    return BaselineRun(
        calibration=calibrate_from_scores(context.calibration_linear_scores, alpha=context.spec.alpha, keep_mask=keep_mask),
        keep_mask=keep_mask,
        predictor=context.linear_predictor,
        clean_structure_scores=np.zeros_like(context.clean_structure_linear_scores, dtype=float),
        dirty_structure_scores=np.ones_like(context.dirty_structure_linear_scores, dtype=float),
        clean_structure_nonconformity=context.clean_structure_linear_scores,
        threshold_label="clean_oracle_trim",
        threshold_quantile=float("nan"),
        threshold_value=0.5,
        threshold_source="oracle_dirty_indicator",
    )
BASELINE_ADAPTERS: dict[str, BaselineAdapter] = {
    "ordinary_split_conformal": BaselineAdapter(
        adapter_id="ordinary_split_conformal",
        problem="Split conformal regression under the observed contaminated calibration split.",
        comparable_object="Vanilla split conformal using every contaminated calibration point.",
        citation="Lei et al. (2018)",
        tier="main",
        runner=lambda context: _ordinary_split(context, method_label="ordinary_split_conformal"),
    ),
    "cqr_conformal": BaselineAdapter(
        adapter_id="cqr_conformal",
        problem="Conformalized quantile regression on the observed contaminated calibration split.",
        comparable_object="CQR calibrated on the same contaminated calibration sample used by the trimmed method.",
        citation="Romano et al. (2019)",
        tier="main",
        runner=lambda context: _cqr_split(context, method_label="cqr_conformal"),
    ),
    "clean_oracle_trim": BaselineAdapter(
        adapter_id="clean_oracle_trim",
        problem="Diagnostic upper benchmark with access to the latent contamination indicator.",
        comparable_object="Retain exactly the truly clean points from the same contaminated calibration sample.",
        citation="Repository diagnostic oracle",
        tier="main",
        runner=_clean_oracle_trim,
    ),
    "clarkson_contaminated_split": BaselineAdapter(
        adapter_id="clarkson_contaminated_split",
        problem="Split conformal under contaminated calibration data.",
        comparable_object="The paper's task-compatible regression comparator: vanilla contaminated-calibration split conformal.",
        citation="Clarkson et al. (2024)",
        tier="main",
        runner=lambda context: _ordinary_split(context, method_label="clarkson_contaminated_split"),
    ),
    "einbinder_noisy_residual": BaselineAdapter(
        adapter_id="einbinder_noisy_residual",
        problem="Noisy-label conformal regression with residual scores.",
        comparable_object="Ordinary conformal calibration applied directly to noisy responses.",
        citation="Einbinder et al. (2024)",
        tier="appendix",
        appendix_only=True,
        allowed_contamination_families=("response_contamination",),
        runner=lambda context: _ordinary_split(context, method_label="einbinder_noisy_residual"),
    ),
    "einbinder_noisy_cqr": BaselineAdapter(
        adapter_id="einbinder_noisy_cqr",
        problem="Noisy-label conformal regression with CQR scores.",
        comparable_object="CQR-style noisy-label conformal calibration on the observed noisy-response calibration split.",
        citation="Einbinder et al. (2024)",
        tier="appendix",
        appendix_only=True,
        allowed_contamination_families=("response_contamination",),
        runner=lambda context: _cqr_split(context, method_label="einbinder_noisy_cqr"),
    ),
}
BASELINE_ALIASES = {
    "split_conformal": "ordinary_split_conformal",
    "cqr": "cqr_conformal",
    "clean_oracle": "clean_oracle_trim",
}
def canonical_baseline_id(method_id: str) -> str | None:
    adapter_id = BASELINE_ALIASES.get(method_id, method_id)
    if adapter_id in BASELINE_ADAPTERS:
        return adapter_id
    return None
def is_baseline_method(method_id: str) -> bool:
    return canonical_baseline_id(method_id) is not None
def get_baseline_adapter(method_id: str) -> BaselineAdapter:
    adapter_id = canonical_baseline_id(method_id)
    if adapter_id is None:
        raise KeyError(f"Unknown baseline method: {method_id}")
    return BASELINE_ADAPTERS[adapter_id]
def execute_baseline(method_id: str, context: BaselineContext) -> BaselineRun:
    return get_baseline_adapter(method_id).execute(context)
