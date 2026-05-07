from __future__ import annotations
import os
from pathlib import Path
from typing import Any
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
from experiments.analysis.derived import (
    build_group_n_best_rows,
    build_group_o_phase_rows,
    build_group_p_density_rows,
    build_group_q_violation_rows,
    build_group_r_ladder_rows,
    build_group_s_stability_rows,
)
from experiments.utils.config import GroupConfig
def _save(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
def _filter(
    summary_rows: list[dict[str, Any]], **kwargs: object
) -> list[dict[str, Any]]:
    rows = summary_rows
    for key, value in kwargs.items():
        rows = [row for row in rows if row.get(key) == value]
    return rows
def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
def _separation_value(row: dict[str, Any]) -> float:
    separation = _coerce_float(row.get("separation_strength"))
    if np.isfinite(separation):
        return separation
    return _coerce_float(row.get("dirty_loc"))
def _plot_threshold_sweep(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    rows = [
        row
        for row in summary_rows
        if np.isfinite(_coerce_float(row.get("threshold_quantile", np.nan)))
    ]
    rows.sort(key=lambda row: _coerce_float(row["threshold_quantile"]))
    x = np.asarray(
        [_coerce_float(row["threshold_quantile"]) for row in rows], dtype=float
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(
        x,
        [row["clean_test_coverage_mean"] for row in rows],
        marker="o",
        label="Clean coverage",
    )
    axes[0].plot(
        x,
        [row["theorem_lower_bound_mean"] for row in rows],
        marker="s",
        label="Theorem bound",
    )
    axes[0].set_title("Coverage")
    axes[1].plot(
        x,
        [row["tilde_epsilon_mean"] for row in rows],
        marker="o",
        label="Effective contamination",
    )
    axes[1].plot(
        x,
        [row["trimming_bias_mean"] for row in rows],
        marker="s",
        label="Trimming bias",
    )
    axes[1].set_title("Decomposition")
    axes[2].plot(
        x,
        [row["retained_fraction_mean"] for row in rows],
        marker="o",
        label="Retained fraction",
    )
    axes[2].plot(
        x,
        [row["average_interval_width_mean"] for row in rows],
        marker="s",
        label="Width",
    )
    axes[2].set_title("Retention and width")
    axes[3].plot(
        x,
        [row["bound_gap_theorem_mean"] for row in rows],
        marker="o",
        label="Coverage - theorem bound",
    )
    axes[3].plot(
        x,
        [row["quantile_shift_mean"] for row in rows],
        marker="s",
        label="Quantile shift",
    )
    axes[3].set_title("Slack and quantile shift")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("Threshold quantile")
        axis.legend()
    _save(fig, output_stem)
def _plot_mixture(summary_rows: list[dict[str, Any]], output_stem: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if np.isfinite(_coerce_float(row.get("threshold_quantile", np.nan)))
    ]
    rows.sort(key=lambda row: _coerce_float(row["threshold_quantile"]))
    x = np.asarray(
        [_coerce_float(row["threshold_quantile"]) for row in rows], dtype=float
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        x,
        [row["clean_test_coverage_mean"] for row in rows],
        marker="o",
        label="Clean coverage",
    )
    axes[0].plot(
        x,
        [row["contaminated_mixture_coverage_mean"] for row in rows],
        marker="s",
        label="Mixture coverage",
    )
    axes[0].plot(
        x,
        [row["proposition_lower_bound_mean"] for row in rows],
        marker="d",
        label="Proposition bound",
    )
    axes[0].plot(
        x,
        [row["additive_mixture_lower_bound_mean"] for row in rows],
        marker="^",
        label="Additive bound",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[0].set_xlabel("Threshold quantile")
    axes[0].set_title("Mixture coverage")
    axes[1].plot(
        x,
        [row["tilde_epsilon_mean"] for row in rows],
        marker="o",
        label="Effective contamination",
    )
    axes[1].plot(
        x,
        [row["retained_fraction_mean"] for row in rows],
        marker="s",
        label="Retained fraction",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[1].set_xlabel("Threshold quantile")
    axes[1].set_title("Retention")
    _save(fig, output_stem)
def _plot_grouped_line(
    summary_rows: list[dict[str, Any]], output_stem: Path, x_field: str, title: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    method_ids = sorted({str(row["method_id"]) for row in summary_rows})
    for method_id in method_ids:
        rows = [row for row in summary_rows if row["method_id"] == method_id]
        rows.sort(key=lambda row: _coerce_float(row.get(x_field, np.nan)))
        x = [_coerce_float(row.get(x_field, np.nan)) for row in rows]
        axes[0].plot(
            x,
            [row["clean_test_coverage_mean"] for row in rows],
            marker="o",
            label=method_id,
        )
        axes[1].plot(
            x,
            [row["average_interval_width_mean"] for row in rows],
            marker="o",
            label=method_id,
        )
    axes[0].set_title(f"{title}: coverage")
    axes[1].set_title(f"{title}: width")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
        axis.set_xlabel(x_field)
    _save(fig, output_stem)
def _plot_ksd_tail(summary_rows: list[dict[str, Any]], output_stem: Path) -> None:
    rows = [row for row in summary_rows if str(row.get("method_id")) == "stein_trimmed"]
    rows.sort(key=lambda row: _separation_value(row))
    x = np.asarray([_separation_value(row) for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(
        x,
        [_coerce_float(row.get("observed_clean_tail_mean")) for row in rows],
        marker="o",
        label="Observed clean tail",
    )
    axes[0].plot(
        x,
        [_coerce_float(row.get("clean_tail_upper_mean")) for row in rows],
        marker="s",
        label="Markov upper",
    )
    axes[0].set_title("Clean tail")
    axes[1].plot(
        x,
        [_coerce_float(row.get("observed_dirty_tail_mean")) for row in rows],
        marker="o",
        label="Observed dirty tail",
    )
    axes[1].plot(
        x,
        [
            _coerce_float(
                row.get(
                    "localized_dirty_tail_lower_mean", row.get("dirty_tail_lower_mean")
                )
            )
            for row in rows
        ],
        marker="s",
        label="Localized dirty-tail lower",
    )
    axes[1].set_title("Dirty tail")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("dirty_loc / separation")
        axis.legend(fontsize=8)
    _save(fig, output_stem)
def _plot_retained_concentration(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    rows = [row for row in summary_rows if str(row.get("method_id")) == "stein_trimmed"]
    rows.sort(key=lambda row: _coerce_float(row.get("calibration_size")))
    x = np.asarray(
        [_coerce_float(row.get("calibration_size")) for row in rows], dtype=float
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()
    axes[0].plot(
        x,
        [
            _coerce_float(row.get("n_clean_keep_mean"))
            / max(_coerce_float(row.get("calibration_size")), 1.0)
            for row in rows
        ],
        marker="o",
        label="Observed clean keep / m",
    )
    axes[0].plot(
        x,
        [
            _coerce_float(row.get("n_clean_keep_lower_bound_mean"))
            / max(_coerce_float(row.get("calibration_size")), 1.0)
            for row in rows
        ],
        marker="s",
        label="Hoeffding lower / m",
    )
    axes[0].set_title("Clean retained count")
    axes[1].plot(
        x,
        [
            _coerce_float(row.get("n_dirty_keep_mean"))
            / max(_coerce_float(row.get("calibration_size")), 1.0)
            for row in rows
        ],
        marker="o",
        label="Observed dirty keep / m",
    )
    axes[1].plot(
        x,
        [
            _coerce_float(row.get("n_dirty_keep_upper_bound_mean"))
            / max(_coerce_float(row.get("calibration_size")), 1.0)
            for row in rows
        ],
        marker="s",
        label="Hoeffding upper / m",
    )
    axes[1].set_title("Dirty retained count")
    axes[2].plot(
        x,
        [
            _coerce_float(row.get("n_keep_mean"))
            / max(_coerce_float(row.get("calibration_size")), 1.0)
            for row in rows
        ],
        marker="o",
        label="Observed keep / m",
    )
    axes[2].plot(
        x,
        [
            _coerce_float(row.get("n_keep_lower_bound_mean"))
            / max(_coerce_float(row.get("calibration_size")), 1.0)
            for row in rows
        ],
        marker="s",
        label="Hoeffding lower / m",
    )
    axes[2].set_title("Total retained count")
    axes[3].plot(
        x,
        [
            _coerce_float(row.get("empirical_retained_contamination_mean"))
            for row in rows
        ],
        marker="o",
        label="Observed retained contamination",
    )
    axes[3].plot(
        x,
        [
            _coerce_float(row.get("empirical_retained_contamination_upper_mean"))
            for row in rows
        ],
        marker="s",
        label="Upper bound",
    )
    axes[3].set_title("Retained contamination ratio")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("calibration_size")
        axis.legend(fontsize=8)
    _save(fig, output_stem)
def _plot_reference_concentration(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    rows = [row for row in summary_rows if str(row.get("method_id")) == "stein_trimmed"]
    rows.sort(key=lambda row: _coerce_float(row.get("clean_reference_size")))
    x = np.asarray(
        [_coerce_float(row.get("clean_reference_size")) for row in rows], dtype=float
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(
        x,
        [_coerce_float(row.get("clean_second_moment_empirical_mean")) for row in rows],
        marker="o",
        label="Empirical second moment",
    )
    axes[0].plot(
        x,
        [_coerce_float(row.get("clean_second_moment_hp_bound_mean")) for row in rows],
        marker="s",
        label="HP bound",
    )
    axes[0].set_title("Clean second moment")
    axes[1].plot(
        x,
        [_coerce_float(row.get("sup_score_error_mean")) for row in rows],
        marker="o",
        label="Sup score error",
    )
    axes[1].plot(
        x,
        [_coerce_float(row.get("bridge_delta_s_mean")) for row in rows],
        marker="s",
        label="Bridge tolerance",
    )
    axes[1].set_title("Score bridge")
    axes[2].plot(
        x,
        [_coerce_float(row.get("clean_second_moment_violation_mean")) for row in rows],
        marker="o",
        label="Violation rate",
    )
    axes[2].set_title("Second-moment violation")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("clean_reference_size")
        axis.legend(fontsize=8)
    _save(fig, output_stem)
def _plot_bridge(summary_rows: list[dict[str, Any]], output_stem: Path) -> None:
    rows = list(summary_rows)
    rows.sort(key=lambda row: float(row["clean_reference_size"]))
    x = [float(row["clean_reference_size"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(
        x,
        [row["sup_score_error_mean"] for row in rows],
        marker="o",
        label="Sup score error",
    )
    axes[0].plot(
        x, [row["bridge_delta_s_mean"] for row in rows], marker="s", label="Delta_S"
    )
    axes[1].plot(
        x,
        [row["bridge_event_holds_mean"] for row in rows],
        marker="o",
        label="Bridge event rate",
    )
    axes[1].plot(
        x,
        [
            1.0 - float(row["bridge_delta"]) if "bridge_delta" in row else np.nan
            for row in rows
        ],
        marker="s",
        label="Nominal",
    )
    axes[2].plot(
        x,
        [row["bridge_tilde_epsilon_upper_mean"] for row in rows],
        marker="o",
        label="Bridge eps upper",
    )
    axes[2].plot(
        x, [row["tilde_epsilon_mean"] for row in rows], marker="s", label="Observed eps"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
        axis.set_xlabel("clean_reference_size")
    _save(fig, output_stem)
def _plot_explicit_bounds(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    rows = [
        row
        for row in summary_rows
        if np.isfinite(_coerce_float(row.get("threshold_quantile", np.nan)))
    ]
    rows.sort(key=lambda row: _coerce_float(row["threshold_quantile"]))
    x = np.asarray(
        [_coerce_float(row["threshold_quantile"]) for row in rows], dtype=float
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        x,
        [row["clean_test_coverage_mean"] for row in rows],
        marker="o",
        label="Coverage",
    )
    ax.plot(
        x,
        [row["theorem_lower_bound_mean"] for row in rows],
        marker="s",
        label="Generic theorem",
    )
    ax.plot(
        x,
        [row["semi_explicit_bound_mean"] for row in rows],
        marker="^",
        label="Semi explicit",
    )
    ax.plot(
        x,
        [row["fully_explicit_bound_mean"] for row in rows],
        marker="d",
        label="Fully explicit",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_xlabel("Threshold quantile")
    ax.set_title("Explicit Stein bounds")
    _save(fig, output_stem)
def _plot_bar(
    summary_rows: list[dict[str, Any]], output_stem: Path, metric: str, title: str
) -> None:
    labels = [f"{row['config_id']}|{row['method_id']}" for row in summary_rows]
    values = [float(row[f"{metric}_mean"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 4))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_stem)
def _plot_multi_metric_bars(
    summary_rows: list[dict[str, Any]], output_stem: Path, title: str
) -> None:
    rows = list(summary_rows)
    rows.sort(key=lambda row: (str(row.get("config_id")), str(row.get("method_id"))))
    labels = [f"{row.get('config_id')}|{row.get('method_id')}" for row in rows]
    x = np.arange(len(rows))
    metrics = [
        ("clean_test_coverage", "Clean coverage"),
        ("average_interval_width", "Interval width"),
        ("tilde_epsilon", "Effective contamination"),
        ("retained_fraction", "Retained fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(max(12, 0.55 * len(labels)), 8))
    axes = axes.ravel()
    for axis, (metric, metric_title) in zip(axes, metrics):
        means = [_coerce_float(row.get(f"{metric}_mean")) for row in rows]
        ses = [_coerce_float(row.get(f"{metric}_se")) for row in rows]
        axis.bar(x, means, yerr=ses, capsize=2)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axis.set_title(metric_title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    _save(fig, output_stem)
def _finite_bounds(
    rows: list[dict[str, Any]], metric: str, diverging: bool = False
) -> tuple[float, float]:
    finite_values = [_coerce_float(row.get(metric)) for row in rows]
    finite_values = [value for value in finite_values if np.isfinite(value)]
    if not finite_values:
        return (-1.0, 1.0) if diverging else (0.0, 1.0)
    vmin = min(finite_values)
    vmax = max(finite_values)
    if diverging:
        bound = max(abs(vmin), abs(vmax), 1e-6)
        return -bound, bound
    if np.isclose(vmin, vmax):
        pad = 0.05 * max(abs(vmin), 1.0)
        return vmin - pad, vmax + pad
    return vmin, vmax
def _plot_heatmap_regime(summary_rows: list[dict[str, Any]], output_stem: Path) -> None:
    best_rows = build_group_n_best_rows(summary_rows)
    if not best_rows:
        _plot_threshold_sweep(summary_rows, output_stem)
        return
    calibration_sizes = sorted(
        value
        for value in {_coerce_float(row.get("calibration_size")) for row in best_rows}
        if np.isfinite(value)
    )
    epsilons = sorted(
        value
        for value in {_coerce_float(row.get("epsilon")) for row in best_rows}
        if np.isfinite(value)
    )
    separations = sorted(
        {
            _separation_value(row)
            for row in best_rows
            if np.isfinite(_separation_value(row))
        }
    )
    epsilon_index = {value: idx for idx, value in enumerate(epsilons)}
    separation_index = {value: idx for idx, value in enumerate(separations)}
    metrics = [
        ("coverage_gain_vs_ordinary_mean", "Coverage Gain vs split", True, "coolwarm"),
        ("stein_net_tradeoff_mean", "Net trade-off", True, "coolwarm"),
        ("stein_theorem_lower_bound_mean", "Theorem lower bound", False, "viridis"),
        ("oracle_gap_mean", "Oracle gap", False, "magma_r"),
    ]
    fig, axes = plt.subplots(
        len(calibration_sizes),
        len(metrics),
        figsize=(3.2 * len(metrics), 2.2 * len(calibration_sizes)),
        squeeze=False,
    )
    for col_idx, (metric, title, diverging, cmap_name) in enumerate(metrics):
        vmin, vmax = _finite_bounds(best_rows, metric, diverging=diverging)
        image = None
        for row_idx, calibration_size in enumerate(calibration_sizes):
            axis = axes[row_idx, col_idx]
            grid = np.full((len(epsilons), len(separations)), np.nan, dtype=float)
            for row in best_rows:
                row_cal = _coerce_float(row.get("calibration_size"))
                if not np.isfinite(row_cal) or not np.isclose(
                    row_cal, calibration_size
                ):
                    continue
                epsilon = _coerce_float(row.get("epsilon"))
                separation = _separation_value(row)
                metric_value = _coerce_float(row.get(metric))
                if not np.isfinite(epsilon) or not np.isfinite(separation):
                    continue
                epsilon_idx = epsilon_index.get(epsilon)
                separation_idx = separation_index.get(separation)
                if epsilon_idx is None or separation_idx is None:
                    continue
                grid[epsilon_idx, separation_idx] = metric_value
            image = axis.imshow(
                grid,
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_name,
                interpolation="nearest",
            )
            axis.set_xticks(range(len(separations)))
            axis.set_xticklabels(
                [f"{value:g}" for value in separations], rotation=45, ha="right"
            )
            axis.set_yticks(range(len(epsilons)))
            axis.set_yticklabels([f"{value:.2f}" for value in epsilons])
            if row_idx == 0:
                axis.set_title(title)
            if row_idx == len(calibration_sizes) - 1:
                axis.set_xlabel("separation_strength")
            if col_idx == 0:
                axis.set_ylabel(f"epsilon\nm={int(calibration_size)}")
        if image is not None:
            fig.colorbar(image, ax=axes[:, col_idx], shrink=0.72)
    _save(fig, output_stem)
def _phase_regions(rows: list[dict[str, Any]]) -> tuple[float, float, float] | None:
    # These spans come only from config-defined `phase_region` annotations attached to
    # each separation setting. They are not estimated from the empirical curves.
    separations = sorted(
        {_separation_value(row) for row in rows if np.isfinite(_separation_value(row))}
    )
    if not separations:
        return None
    no_sep = [
        _separation_value(row)
        for row in rows
        if row.get("phase_region") == "no_separation"
    ]
    near_ideal = [
        _separation_value(row)
        for row in rows
        if row.get("phase_region") == "near_ideal"
    ]
    no_sep = [value for value in no_sep if np.isfinite(value)]
    near_ideal = [value for value in near_ideal if np.isfinite(value)]
    if not no_sep or not near_ideal:
        return None
    return (min(separations), max(no_sep), min(near_ideal))
def _shade_phase_regions(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    bounds = _phase_regions(rows)
    if bounds is None:
        return
    x_min, no_sep_max, near_ideal_min = bounds
    x_max = max(
        _separation_value(row) for row in rows if np.isfinite(_separation_value(row))
    )
    axis.axvspan(x_min, no_sep_max, color="#dff0d8", alpha=0.25, zorder=0)
    axis.axvspan(no_sep_max, near_ideal_min, color="#fcf8e3", alpha=0.30, zorder=0)
    axis.axvspan(near_ideal_min, x_max, color="#d9edf7", alpha=0.25, zorder=0)
    for label, left, right in (
        ("config: no separation", x_min, no_sep_max),
        ("config: transition", no_sep_max, near_ideal_min),
        ("config: near ideal", near_ideal_min, x_max),
    ):
        axis.text(
            (left + right) / 2.0,
            0.98,
            label,
            ha="center",
            va="top",
            transform=axis.get_xaxis_transform(),
            fontsize=8,
        )
def _plot_phase_diagram(summary_rows: list[dict[str, Any]], output_stem: Path) -> None:
    phase_rows = build_group_o_phase_rows(summary_rows)
    if not phase_rows:
        _plot_threshold_sweep(summary_rows, output_stem)
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    metric_specs = [
        ("stein_p_c_minus_p_d_mean", "p_c - p_d"),
        ("stein_tilde_epsilon_mean", "Effective contamination"),
        ("oracle_gap_mean", "Oracle gap"),
        ("width_gap_to_oracle_mean", "Width gap to oracle"),
    ]
    epsilon_values = sorted(
        value
        for value in {_coerce_float(row.get("epsilon")) for row in phase_rows}
        if np.isfinite(value)
    )
    calibration_sizes = sorted(
        value
        for value in {_coerce_float(row.get("calibration_size")) for row in phase_rows}
        if np.isfinite(value)
    )
    epsilon_colors = {
        epsilon: plt.get_cmap("tab10")(index)
        for index, epsilon in enumerate(epsilon_values)
    }
    line_styles = {
        calibration_size: style
        for calibration_size, style in zip(calibration_sizes, ["-", "--", "-.", ":"])
    }
    for axis, (metric, title) in zip(axes, metric_specs):
        _shade_phase_regions(axis, phase_rows)
        for epsilon in epsilon_values:
            for calibration_size in calibration_sizes:
                rows = [
                    row
                    for row in phase_rows
                    if np.isclose(_coerce_float(row.get("epsilon")), epsilon)
                    and np.isclose(
                        _coerce_float(row.get("calibration_size")), calibration_size
                    )
                ]
                rows.sort(key=_separation_value)
                if not rows:
                    continue
                x = [_separation_value(row) for row in rows]
                y = [_coerce_float(row.get(metric)) for row in rows]
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.8,
                    color=epsilon_colors[epsilon],
                    linestyle=line_styles.get(calibration_size, "-"),
                    label=f"eps={epsilon:.2f}, m={int(calibration_size)}",
                )
        if metric != "stein_tilde_epsilon_mean":
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.grid(alpha=0.25)
        axis.set_title(title)
        axis.set_xlabel("separation_strength")
    fig.suptitle(
        "Shaded bands are config-defined phase annotations", y=1.02, fontsize=11
    )
    axes[-1].legend(fontsize=8, ncol=2)
    _save(fig, output_stem)
def _plot_quantile_density(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    density_rows = build_group_p_density_rows(summary_rows)
    if not density_rows:
        _plot_threshold_sweep(summary_rows, output_stem)
        return
    regime_markers = {
        "gaussian_homoscedastic": "o",
        "heavy_tailed": "s",
        "bimodal": "^",
        "low_density_near_quantile": "D",
    }
    regime_colors = {
        "gaussian_homoscedastic": "#3b6fb6",
        "heavy_tailed": "#d07c2c",
        "bimodal": "#2c9a5f",
        "low_density_near_quantile": "#b6436e",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    lambda_values = np.asarray(
        [_coerce_float(row.get("lambda_min_est_mean")) for row in density_rows],
        dtype=float,
    )
    scatter = axes[0].scatter(
        [_coerce_float(row.get("trimming_bias_mean")) for row in density_rows],
        [_coerce_float(row.get("quantile_shift_mean")) for row in density_rows],
        c=lambda_values,
        cmap="viridis",
        s=60,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    for regime, marker in regime_markers.items():
        regime_rows = [
            row for row in density_rows if row.get("density_regime") == regime
        ]
        if not regime_rows:
            continue
        axes[0].scatter(
            [_coerce_float(row.get("trimming_bias_mean")) for row in regime_rows],
            [_coerce_float(row.get("quantile_shift_mean")) for row in regime_rows],
            c=[_coerce_float(row.get("lambda_min_est_mean")) for row in regime_rows],
            cmap="viridis",
            s=60,
            alpha=0.85,
            marker=marker,
            edgecolors="black",
            linewidths=0.3,
            label=regime,
        )
    axes[0].set_title("Shift vs trimming bias")
    axes[0].set_xlabel("trimming_bias")
    axes[0].set_ylabel("quantile_shift")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.colorbar(scatter, ax=axes[0], shrink=0.8, label="lambda_min_est")
    regime_order = [row.get("density_regime") for row in density_rows]
    regime_order = list(dict.fromkeys(regime_order))
    epsilon_styles = {0.10: "-", 0.20: "--"}
    for regime in regime_order:
        for epsilon in sorted(
            {
                _coerce_float(row.get("epsilon"))
                for row in density_rows
                if row.get("density_regime") == regime
            }
        ):
            rows = [
                row
                for row in density_rows
                if row.get("density_regime") == regime
                and np.isclose(_coerce_float(row.get("epsilon")), epsilon)
            ]
            rows.sort(key=lambda row: _coerce_float(row.get("threshold_quantile")))
            if not rows:
                continue
            axes[1].plot(
                [_coerce_float(row.get("threshold_quantile")) for row in rows],
                [_coerce_float(row.get("bound_looseness_bias_mean")) for row in rows],
                marker=regime_markers.get(str(regime), "o"),
                color=regime_colors.get(str(regime), "#333333"),
                linestyle=epsilon_styles.get(round(float(epsilon), 2), "-"),
                label=f"{regime}, eps={epsilon:.2f}",
            )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Bias-bound looseness")
    axes[1].set_xlabel("threshold_quantile")
    axes[1].set_ylabel("bound - observed shift")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    for regime in regime_order:
        rows = [row for row in density_rows if row.get("density_regime") == regime]
        axes[2].scatter(
            [_coerce_float(row.get("inverse_lambda_min_est_mean")) for row in rows],
            [_coerce_float(row.get("bound_efficiency_ratio_mean")) for row in rows],
            color=regime_colors.get(str(regime), "#333333"),
            marker=regime_markers.get(str(regime), "o"),
            s=60,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
            label=regime,
        )
    axes[2].set_title("Bound ratio vs inverse density")
    axes[2].set_xlabel("1 / lambda_min_est")
    axes[2].set_ylabel("bound / observed shift")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    _save(fig, output_stem)
def _aggregate_concentration_for_plot(
    violation_rows: list[dict[str, Any]],
    metric: str,
) -> dict[tuple[str, float], float]:
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in violation_rows:
        family = str(row.get("contamination_family"))
        calibration_size = _coerce_float(row.get("calibration_size"))
        value = _coerce_float(row.get(metric))
        if not np.isfinite(calibration_size) or not np.isfinite(value):
            continue
        grouped.setdefault((family, calibration_size), []).append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}
def _plot_concentration_calibration(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    violation_rows = build_group_q_violation_rows(summary_rows)
    if not violation_rows:
        _plot_threshold_sweep(summary_rows, output_stem)
        return
    families = sorted({str(row.get("contamination_family")) for row in violation_rows})
    calibration_sizes = sorted(
        {
            _coerce_float(row.get("calibration_size"))
            for row in violation_rows
            if np.isfinite(_coerce_float(row.get("calibration_size")))
        }
    )
    family_colors = {
        family: plt.get_cmap("tab10")(index) for index, family in enumerate(families)
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = [
        ("clean_violation_rate", "clean"),
        ("dirty_violation_rate", "dirty"),
        ("keep_violation_rate", "keep"),
    ]
    for family in families:
        for metric, label in metric_specs:
            points = _aggregate_concentration_for_plot(violation_rows, metric)
            y = [points.get((family, size), np.nan) for size in calibration_sizes]
            axes[0].plot(
                calibration_sizes,
                y,
                marker="o",
                linestyle={"clean": "-", "dirty": "--", "keep": ":"}[label],
                color=family_colors[family],
                label=f"{family} | {label}",
            )
    axes[0].set_title("Count-bound violation rates")
    axes[0].set_xlabel("calibration_size")
    axes[0].set_ylabel("violation rate")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    exceed_points = _aggregate_concentration_for_plot(
        violation_rows, "contamination_upper_exceedance_rate"
    )
    nominal_points = _aggregate_concentration_for_plot(
        violation_rows, "nominal_bound_proxy"
    )
    for family in families:
        axes[1].plot(
            calibration_sizes,
            [exceed_points.get((family, size), np.nan) for size in calibration_sizes],
            marker="o",
            color=family_colors[family],
            label=f"{family} exceedance",
        )
        axes[1].plot(
            calibration_sizes,
            [nominal_points.get((family, size), np.nan) for size in calibration_sizes],
            marker="s",
            linestyle="--",
            color=family_colors[family],
            alpha=0.8,
            label=f"{family} nominal proxy",
        )
    axes[1].set_title("Retained contamination upper bound")
    axes[1].set_xlabel("calibration_size")
    axes[1].set_ylabel("rate / proxy")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    bar_metrics = [
        ("clean_violation_rate", "clean"),
        ("dirty_violation_rate", "dirty"),
        ("keep_violation_rate", "keep"),
        ("contamination_upper_exceedance_rate", "upper"),
    ]
    bar_width = 0.18
    x = np.arange(len(families))
    for idx, (metric, label) in enumerate(bar_metrics):
        points = _aggregate_concentration_for_plot(violation_rows, metric)
        values = []
        for family in families:
            family_values = [
                value
                for (key_family, _), value in points.items()
                if key_family == family
            ]
            values.append(float(np.mean(family_values)) if family_values else np.nan)
        axes[2].bar(x + (idx - 1.5) * bar_width, values, width=bar_width, label=label)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(families, rotation=20, ha="right")
    axes[2].set_title("Average violation diagnostics")
    axes[2].set_ylabel("mean rate")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    _save(fig, output_stem)
def _plot_bridge_ladder(summary_rows: list[dict[str, Any]], output_stem: Path) -> None:
    ladder_rows = build_group_r_ladder_rows(summary_rows)
    if not ladder_rows:
        _plot_bridge(summary_rows, output_stem)
        return
    mode_order = ["oracle", "well_specified", "mis_specified"]
    mode_colors = {
        "oracle": "#2c9a5f",
        "well_specified": "#3b6fb6",
        "mis_specified": "#b6436e",
    }
    epsilon_styles = {
        epsilon: style
        for epsilon, style in zip(
            sorted(
                {
                    _coerce_float(row.get("epsilon"))
                    for row in ladder_rows
                    if np.isfinite(_coerce_float(row.get("epsilon")))
                }
            ),
            ["-", "--", "-.", ":"],
        )
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for mode in mode_order:
        mode_rows = [
            row for row in ladder_rows if str(row.get("score_model_mode")) == mode
        ]
        if not mode_rows:
            continue
        for epsilon in sorted(
            {
                _coerce_float(row.get("epsilon"))
                for row in mode_rows
                if np.isfinite(_coerce_float(row.get("epsilon")))
            }
        ):
            rows = [
                row
                for row in mode_rows
                if np.isclose(_coerce_float(row.get("epsilon")), epsilon)
            ]
            rows.sort(key=lambda row: _coerce_float(row.get("clean_reference_size")))
            if not rows:
                continue
            x = [_coerce_float(row.get("clean_reference_size")) for row in rows]
            label = f"{mode}, eps={epsilon:.2f}"
            style = epsilon_styles.get(epsilon, "-")
            color = mode_colors.get(mode, "#333333")
            axes[0].plot(
                x,
                [_coerce_float(row.get("sup_score_error_mean")) for row in rows],
                marker="o",
                linestyle=style,
                color=color,
                label=label,
            )
            axes[1].plot(
                x,
                [_coerce_float(row.get("bridge_event_holds_mean")) for row in rows],
                marker="o",
                linestyle=style,
                color=color,
                label=label,
            )
            axes[2].plot(
                x,
                [_coerce_float(row.get("bridge_upper_slack_mean")) for row in rows],
                marker="o",
                linestyle=style,
                color=color,
                label=label,
            )
        axes[3].scatter(
            [_coerce_float(row.get("score_model_sup_error_mean")) for row in mode_rows],
            [_coerce_float(row.get("clean_test_coverage_mean")) for row in mode_rows],
            color=mode_colors.get(mode, "#333333"),
            s=65,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
            label=mode,
        )
    axes[0].set_title("Empirical bridge error")
    axes[0].set_xlabel("clean_reference_size")
    axes[0].set_ylabel("sup_score_error")
    axes[1].set_title("Bridge event rate")
    axes[1].set_xlabel("clean_reference_size")
    axes[1].set_ylabel("bridge_event_holds")
    axes[2].set_title("Bridge upper slack")
    axes[2].set_xlabel("clean_reference_size")
    axes[2].set_ylabel("bridge_tilde_eps_upper - tilde_eps")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[3].set_title("Coverage vs score-model error")
    axes[3].set_xlabel("score_model_sup_error")
    axes[3].set_ylabel("clean_test_coverage")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[3].legend(fontsize=8)
    _save(fig, output_stem)
def _aggregate_rows_by_reference(
    stability_rows: list[dict[str, Any]],
    metric: str,
) -> dict[float, float]:
    grouped: dict[float, list[float]] = {}
    for row in stability_rows:
        reference_size = _coerce_float(row.get("clean_reference_size"))
        value = _coerce_float(row.get(metric))
        if not np.isfinite(reference_size) or not np.isfinite(value):
            continue
        grouped.setdefault(reference_size, []).append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}
def _aggregate_rows_by_reference_and_threshold(
    stability_rows: list[dict[str, Any]],
    metric: str,
) -> dict[tuple[float, float], float]:
    grouped: dict[tuple[float, float], list[float]] = {}
    for row in stability_rows:
        reference_size = _coerce_float(row.get("clean_reference_size"))
        threshold_quantile = _coerce_float(row.get("threshold_quantile"))
        value = _coerce_float(row.get(metric))
        if (
            not np.isfinite(reference_size)
            or not np.isfinite(threshold_quantile)
            or not np.isfinite(value)
        ):
            continue
        grouped.setdefault((reference_size, threshold_quantile), []).append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}
def _plot_explicit_stability(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    stability_rows = build_group_s_stability_rows(summary_rows)
    if not stability_rows:
        _plot_explicit_bounds(summary_rows, output_stem)
        return
    reference_sizes = sorted(
        {
            _coerce_float(row.get("clean_reference_size"))
            for row in stability_rows
            if np.isfinite(_coerce_float(row.get("clean_reference_size")))
        }
    )
    threshold_quantiles = sorted(
        {
            _coerce_float(row.get("threshold_quantile"))
            for row in stability_rows
            if np.isfinite(_coerce_float(row.get("threshold_quantile")))
        }
    )
    reference_colors = {
        reference_size: plt.get_cmap("tab10")(index)
        for index, reference_size in enumerate(reference_sizes)
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    semi_finite = _aggregate_rows_by_reference(
        stability_rows, "semi_explicit_finite_mean"
    )
    fully_finite = _aggregate_rows_by_reference(
        stability_rows, "fully_explicit_finite_mean"
    )
    semi_informative = _aggregate_rows_by_reference(
        stability_rows, "semi_explicit_informative_mean"
    )
    fully_informative = _aggregate_rows_by_reference(
        stability_rows, "fully_explicit_informative_mean"
    )
    axes[0].plot(
        reference_sizes,
        [semi_finite.get(size, np.nan) for size in reference_sizes],
        marker="o",
        label="semi finite",
    )
    axes[0].plot(
        reference_sizes,
        [fully_finite.get(size, np.nan) for size in reference_sizes],
        marker="s",
        label="fully finite",
    )
    axes[0].plot(
        reference_sizes,
        [semi_informative.get(size, np.nan) for size in reference_sizes],
        marker="^",
        linestyle="--",
        label="semi informative",
    )
    axes[0].plot(
        reference_sizes,
        [fully_informative.get(size, np.nan) for size in reference_sizes],
        marker="d",
        linestyle="--",
        label="fully informative",
    )
    axes[0].set_title("Explicit stability vs reference size")
    axes[0].set_xlabel("clean_reference_size")
    axes[0].set_ylabel("fraction")
    failure_metrics = [
        ("explicit_fail_t_minus_nonpositive_rate", "t- <= 0"),
        ("explicit_fail_udr_nan_rate", "u_d^R nan"),
        ("explicit_fail_mu_lower_nan_rate", "mu_lower nan"),
        ("explicit_fail_bar_eps_nan_rate", "bar_eps nan"),
        ("explicit_fail_fully_bias_nan_rate", "bias nan"),
        ("explicit_bound_noninformative_rate", "noninformative"),
    ]
    failure_values = []
    for metric, _ in failure_metrics:
        values = [_coerce_float(row.get(metric)) for row in stability_rows]
        finite_values = [value for value in values if np.isfinite(value)]
        failure_values.append(
            float(np.mean(finite_values)) if finite_values else np.nan
        )
    axes[1].bar(
        np.arange(len(failure_metrics)),
        failure_values,
        color=["#b6436e", "#d07c2c", "#3b6fb6", "#7c6fd0", "#4a4a4a", "#2c9a5f"],
    )
    axes[1].set_xticks(np.arange(len(failure_metrics)))
    axes[1].set_xticklabels(
        [label for _, label in failure_metrics], rotation=25, ha="right"
    )
    axes[1].set_title("Average failure-cause rates")
    axes[1].set_ylabel("mean rate")
    semi_gap = _aggregate_rows_by_reference_and_threshold(
        stability_rows, "semi_explicit_gap_mean"
    )
    fully_gap = _aggregate_rows_by_reference_and_threshold(
        stability_rows, "fully_explicit_gap_mean"
    )
    noninformative = _aggregate_rows_by_reference_and_threshold(
        stability_rows, "explicit_bound_noninformative_rate"
    )
    for reference_size in reference_sizes:
        color = reference_colors.get(reference_size, "#333333")
        axes[2].plot(
            threshold_quantiles,
            [
                semi_gap.get((reference_size, quantile), np.nan)
                for quantile in threshold_quantiles
            ],
            marker="o",
            color=color,
            linestyle="-",
            label=f"semi r={int(reference_size)}",
        )
        axes[2].plot(
            threshold_quantiles,
            [
                fully_gap.get((reference_size, quantile), np.nan)
                for quantile in threshold_quantiles
            ],
            marker="s",
            color=color,
            linestyle="--",
            label=f"fully r={int(reference_size)}",
        )
        axes[3].plot(
            threshold_quantiles,
            [
                noninformative.get((reference_size, quantile), np.nan)
                for quantile in threshold_quantiles
            ],
            marker="o",
            color=color,
            linestyle="-",
            label=f"r={int(reference_size)}",
        )
    axes[2].set_title("Explicit bound gaps vs threshold")
    axes[2].set_xlabel("threshold_quantile")
    axes[2].set_ylabel("coverage - explicit bound")
    axes[3].set_title("Noninformative-rate vs threshold")
    axes[3].set_xlabel("threshold_quantile")
    axes[3].set_ylabel("noninformative rate")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=7, ncol=2)
    axes[3].legend(fontsize=8)
    _save(fig, output_stem)
def _plot_localized_certificate(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    rows = [row for row in summary_rows if str(row.get("method_id")) == "stein_trimmed"]
    rows.sort(
        key=lambda row: (
            _coerce_float(row.get("dirty_loc")),
            _coerce_float(row.get("working_radius_mean", row.get("working_radius"))),
        )
    )
    x = np.arange(len(rows))
    labels = [
        f"loc={_coerce_float(row.get('dirty_loc')):.1f},R={_coerce_float(row.get('working_radius_mean', row.get('working_radius'))):.1f}"
        for row in rows
    ]
    fig, axes = plt.subplots(2, 2, figsize=(max(12, 0.65 * len(labels)), 8))
    axes = axes.ravel()
    axes[0].bar(x, [_coerce_float(row.get("localized_feasible_mean")) for row in rows])
    axes[0].set_title("Localized certificate feasibility rate")
    axes[1].bar(
        x, [_coerce_float(row.get("localized_bar_epsilon_mean")) for row in rows]
    )
    axes[1].bar(
        x, [_coerce_float(row.get("tilde_epsilon_mean")) for row in rows], alpha=0.5
    )
    axes[1].set_title("Certified vs observed retained contamination")
    axes[2].bar(
        x,
        [_coerce_float(row.get("semi_explicit_bound_mean")) for row in rows],
        label="Semi explicit",
    )
    axes[2].bar(
        x,
        [_coerce_float(row.get("clean_test_coverage_mean")) for row in rows],
        alpha=0.5,
        label="Observed coverage",
    )
    axes[2].set_title("Localized explicit bound vs coverage")
    axes[2].legend(fontsize=8)
    axes[3].bar(
        x, [_coerce_float(row.get("zeta_P_mean")) for row in rows], label="zeta_P"
    )
    axes[3].bar(
        x,
        [_coerce_float(row.get("zeta_Q_mean")) for row in rows],
        alpha=0.5,
        label="zeta_Q",
    )
    axes[3].set_title("Working-set tail penalties")
    axes[3].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    _save(fig, output_stem)
def _plot_large_m_tightness(
    summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    rows = [row for row in summary_rows if str(row.get("method_id")) == "stein_trimmed"]
    rows.sort(key=lambda row: _coerce_float(row.get("calibration_size")))
    x = np.asarray(
        [_coerce_float(row.get("calibration_size")) for row in rows], dtype=float
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(
        x,
        [_coerce_float(row.get("clean_test_coverage_mean")) for row in rows],
        marker="o",
        label="Observed coverage",
    )
    axes[0].plot(
        x,
        [_coerce_float(row.get("theorem_lower_bound_mean")) for row in rows],
        marker="s",
        label="Theorem lower bound",
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Coverage and lower bound")
    axes[1].plot(
        x,
        [_coerce_float(row.get("bound_gap_theorem_mean")) for row in rows],
        marker="o",
        label="Coverage - theorem bound",
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Bound gap")
    axes[2].plot(
        x,
        [_coerce_float(row.get("tilde_epsilon_mean")) for row in rows],
        marker="o",
        label="tilde epsilon",
    )
    axes[2].plot(
        x,
        [_coerce_float(row.get("trimming_bias_mean")) for row in rows],
        marker="s",
        label="bias",
    )
    axes[2].set_xscale("log")
    axes[2].set_title("Population loss terms")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("calibration_size")
        axis.legend(fontsize=8)
    _save(fig, output_stem)
def render_group_figure(
    group_config: GroupConfig, summary_rows: list[dict[str, Any]], output_stem: Path
) -> None:
    kind = group_config.figure_kind
    if kind == "threshold_sweep":
        _plot_threshold_sweep(summary_rows, output_stem)
    elif kind == "mixture":
        _plot_mixture(summary_rows, output_stem)
    elif kind == "interpolation":
        _plot_grouped_line(
            summary_rows, output_stem, x_field="dirty_loc", title="Interpolation"
        )
    elif kind == "calibration_size":
        _plot_grouped_line(
            summary_rows,
            output_stem,
            x_field="calibration_size",
            title="Calibration size",
        )
    elif kind == "retained_concentration":
        _plot_retained_concentration(summary_rows, output_stem)
    elif kind == "ksd_tail":
        _plot_ksd_tail(summary_rows, output_stem)
    elif kind == "reference_concentration":
        _plot_reference_concentration(summary_rows, output_stem)
    elif kind == "clean_reference_concentration":
        _plot_reference_concentration(summary_rows, output_stem)
    elif kind == "bridge":
        _plot_bridge(summary_rows, output_stem)
    elif kind == "explicit_bounds":
        _plot_explicit_bounds(summary_rows, output_stem)
    elif kind == "baseline_compare":
        _plot_multi_metric_bars(summary_rows, output_stem, title=group_config.title)
    elif kind == "stress_compare":
        _plot_multi_metric_bars(summary_rows, output_stem, title=group_config.title)
    elif kind == "multi_metric_baseline":
        _plot_multi_metric_bars(summary_rows, output_stem, title=group_config.title)
    elif kind == "multi_metric_stress":
        _plot_multi_metric_bars(summary_rows, output_stem, title=group_config.title)
    elif kind == "heatmap_regime":
        _plot_heatmap_regime(summary_rows, output_stem)
    elif kind == "phase_diagram":
        _plot_phase_diagram(summary_rows, output_stem)
    elif kind == "quantile_density":
        _plot_quantile_density(summary_rows, output_stem)
    elif kind == "concentration_calibration":
        _plot_concentration_calibration(summary_rows, output_stem)
    elif kind == "bridge_ladder":
        _plot_bridge_ladder(summary_rows, output_stem)
    elif kind == "explicit_stability":
        _plot_explicit_stability(summary_rows, output_stem)
    elif kind == "localized_certificate":
        _plot_localized_certificate(summary_rows, output_stem)
    elif kind == "large_m_tightness":
        _plot_large_m_tightness(summary_rows, output_stem)
    elif kind == "nongaussian_covariate":
        _plot_multi_metric_bars(summary_rows, output_stem, title=group_config.title)
    elif kind == "label_only_failure":
        _plot_multi_metric_bars(summary_rows, output_stem, title=group_config.title)
    else:
        _plot_threshold_sweep(summary_rows, output_stem)