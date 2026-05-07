from __future__ import annotations
import numpy as np
from experiments.analysis.aggregate import aggregate_rows
from experiments.utils.metrics import cdf_excess, weighted_cdf_excess

def test_cdf_excess_detects_left_shift() -> None:
    clean = np.array([0.0, 1.0, 2.0, 3.0])
    retained = np.array([0.0, 0.5, 1.0, 1.5])
    assert cdf_excess(retained, clean) > 0.0
    assert cdf_excess(clean, retained) < cdf_excess(retained, clean)

def test_weighted_cdf_excess_is_no_larger_than_component_bound() -> None:
    clean = np.array([0.0, 1.0, 2.0, 3.0])
    retained_clean = np.array([0.0, 1.0, 2.0, 3.0])
    retained_dirty = np.array([0.0, 0.0, 0.0, 0.0])
    eps = 0.25
    direct = weighted_cdf_excess([retained_clean, retained_dirty], [1.0 - eps, eps], clean)
    decomp = (1.0 - eps) * cdf_excess(retained_clean, clean) + eps * cdf_excess(retained_dirty, clean)
    assert direct <= decomp + 1e-12

def test_adaptive_thresholds_are_aggregated_at_policy_level() -> None:
    base = {
        "group_id": "group_v_localized_certificate",
        "config_id": "cfg",
        "experiment_title": "title",
        "description": "desc",
        "method_id": "stein_trimmed",
        "baseline_adapter_id": None,
        "baseline_tier": None,
        "baseline_appendix_only": None,
        "threshold_policy": "ksd_geometric_localized",
        "threshold_quantile": float("nan"),
        "alpha": 0.1,
        "epsilon": 0.2,
        "train_size": 10,
        "calibration_size": 10,
        "clean_reference_size": 10,
        "clean_test_size": 10,
        "mixture_test_size": 10,
        "structure_sample_size": 10,
        "eta_fraction": 0.2,
        "bridge_delta": 0.1,
        "bridge_rho": 0.1,
        "population_sample_size": 100,
        "evaluation_grid_size": 10,
        "compute_population_metrics": True,
        "include_mixture_eval": True,
        "mc_reps": 2,
        "random_seed": 1,
        "dgp_family": "gaussian_homoscedastic",
        "contamination_family": "partial_separation",
        "predictor_type": "linear",
        "representation_family": "x_only",
        "notes": "",
        "dirty_loc": 4.0,
        "dirty_scale": 0.8,
        "leverage_loc": None,
        "separation_strength": None,
        "phase_region": None,
        "score_model_mode": "well_specified",
        "bandwidth_scale": None,
        "fit_mean_offset": None,
        "fit_variance_scale": None,
        "working_radius": 3.0,
    }
    rows = []
    for rep, label, source, fallback in [
        (0, "ksd_geometric_localized", "ksd_geometric_localized", 0.0),
        (1, "ksd_local_fallback_q=0.990", "ksd_local_fallback", 1.0),
    ]:
        row = dict(base)
        row.update(
            {
                "replicate": rep,
                "threshold_label": label,
                "threshold_source": source,
                "clean_test_coverage": 0.8 + 0.1 * rep,
                "threshold_is_fallback": fallback,
            }
        )
        rows.append(row)
    summary = aggregate_rows(rows)
    assert len(summary) == 1
    assert summary[0]["replicate_count"] == 2
    assert summary[0]["threshold_label"] == "policy:ksd_geometric_localized"
    assert summary[0]["threshold_is_fallback_mean"] == 0.5
