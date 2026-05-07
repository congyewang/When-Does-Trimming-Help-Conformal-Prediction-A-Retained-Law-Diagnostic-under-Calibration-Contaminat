from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from experiments.utils.config import ExperimentSpec
Array = np.ndarray
@dataclass(frozen=True)
class Dataset:
    x: Array
    y: Array
    dirty: Array
Sampler = Callable[[np.random.Generator, int], Dataset]
def _dataset(x: Array, y: Array, dirty: bool) -> Dataset:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    dirty_mask = np.full(x.shape[0], dirty, dtype=bool)
    return Dataset(x=x, y=y, dirty=dirty_mask)
def _as_matrix(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x.reshape(x.shape[0], -1)
def _linear_mean(x: Array) -> Array:
    features = _as_matrix(x)
    if features.shape[1] == 1:
        return features[:, 0]
    coeff = np.linspace(1.0, -0.6, features.shape[1])
    return features @ coeff
def sample_clean_tradeoff(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=0.0, scale=1.0, size=size)
    sigma = 0.6 * (1.0 + 0.6 * np.abs(x))
    y = x + sigma * rng.normal(size=size)
    return _dataset(x, y, dirty=False)
def sample_clean_homoscedastic(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=0.0, scale=1.0, size=size)
    y = x + 0.6 * rng.normal(size=size)
    return _dataset(x, y, dirty=False)
def sample_clean_bimodal_covariate(rng: np.random.Generator, size: int) -> Dataset:
    component = rng.random(size=size) < 0.5
    x_left = rng.normal(loc=-2.0, scale=0.7, size=size)
    x_right = rng.normal(loc=2.0, scale=0.7, size=size)
    x = np.where(component, x_left, x_right)
    y = x + 0.6 * rng.normal(size=size)
    return _dataset(x, y, dirty=False)
def sample_clean_t_covariate(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.standard_t(df=3.0, size=size) / np.sqrt(3.0)
    y = x + 0.6 * rng.normal(size=size)
    return _dataset(x, y, dirty=False)
def sample_clean_skewed_covariate(rng: np.random.Generator, size: int) -> Dataset:
    g = rng.normal(loc=0.0, scale=1.0, size=size)
    raw = np.exp(0.8 * g)
    mean = np.exp(0.5 * 0.8**2)
    var = (np.exp(0.8**2) - 1.0) * np.exp(0.8**2)
    x = (raw - mean) / np.sqrt(var)
    y = x + 0.6 * rng.normal(size=size)
    return _dataset(x, y, dirty=False)
def sample_clean_skewed_mixture_covariate(rng: np.random.Generator, size: int) -> Dataset:
    component = rng.random(size=size) < 0.62
    g = rng.normal(size=size)
    raw = np.exp(0.75 * g)
    mean = np.exp(0.5 * 0.75**2)
    var = (np.exp(0.75**2) - 1.0) * np.exp(0.75**2)
    skewed = (raw - mean) / np.sqrt(var)
    left_x1 = -1.7 + 0.45 * rng.normal(size=size)
    right_x1 = 0.95 + 0.85 * skewed
    x1 = np.where(component, left_x1, right_x1)
    x2 = np.where(component, -0.55 + 0.35 * rng.normal(size=size), 0.75 + 0.35 * np.tanh(right_x1) + 0.25 * rng.normal(size=size))
    x = np.column_stack([x1, x2])
    sigma = 0.45 + 0.12 * np.abs(x1)
    y = _linear_mean(x) + sigma * rng.normal(size=size)
    return _dataset(x, y, dirty=False)
def sample_clean_banana_covariate(rng: np.random.Generator, size: int) -> Dataset:
    x1 = rng.normal(loc=0.0, scale=1.15, size=size)
    x2 = 0.55 * (x1**2 - 1.0) + 0.28 * rng.normal(size=size)
    x = np.column_stack([x1, x2])
    sigma = 0.42 + 0.10 * np.abs(x1)
    y = _linear_mean(x) + sigma * rng.normal(size=size)
    return _dataset(x, y, dirty=False)
def sample_clean_spiral_covariate(rng: np.random.Generator, size: int) -> Dataset:
    theta = rng.uniform(0.55 * np.pi, 3.75 * np.pi, size=size)
    radius = 0.18 + 0.115 * theta
    x1 = radius * np.cos(theta) + 0.08 * rng.normal(size=size)
    x2 = radius * np.sin(theta) + 0.08 * rng.normal(size=size)
    x = np.column_stack([x1, x2])
    x = (x - np.asarray([0.05, 0.0])) / np.asarray([1.15, 1.15])
    y = _linear_mean(x) + 0.45 * rng.normal(size=size)
    return _dataset(x, y, dirty=False)
def sample_clean_two_moons_covariate(rng: np.random.Generator, size: int) -> Dataset:
    component = rng.random(size=size) < 0.5
    theta = rng.uniform(0.0, np.pi, size=size)
    upper = np.column_stack([np.cos(theta), np.sin(theta)])
    lower = np.column_stack([1.0 - np.cos(theta), -np.sin(theta) + 0.55])
    raw = np.where(component[:, None], upper, lower)
    raw = raw + 0.065 * rng.normal(size=(size, 2))
    x = (raw - np.asarray([0.5, 0.28])) / np.asarray([1.05, 1.05])
    sigma = 0.38 + 0.06 * np.abs(x[:, 0])
    y = _linear_mean(x) + sigma * rng.normal(size=size)
    return _dataset(x, y, dirty=False)

def _signed_magnitudes(rng: np.random.Generator, magnitudes: Array) -> Array:
    signs = rng.choice(np.array([-1.0, 1.0]), size=np.asarray(magnitudes).shape[0])
    return signs * np.asarray(magnitudes, dtype=float)
def sample_clean_heavy_tail(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=0.0, scale=1.0, size=size)
    noise = 0.35 * rng.standard_t(df=3.0, size=size)
    y = x + noise
    return _dataset(x, y, dirty=False)
def sample_clean_bimodal(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=0.0, scale=1.0, size=size)
    component = rng.random(size=size) < 0.55
    small_mode = np.abs(rng.normal(loc=0.28, scale=0.05, size=size))
    large_mode = np.abs(rng.normal(loc=1.05, scale=0.10, size=size))
    magnitudes = np.where(component, small_mode, large_mode)
    y = x + _signed_magnitudes(rng, magnitudes)
    return _dataset(x, y, dirty=False)
def sample_clean_low_density_near_quantile(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=0.0, scale=1.0, size=size)
    component = rng.random(size=size) < 0.90
    low_mode = np.abs(rng.normal(loc=0.24, scale=0.03, size=size))
    high_mode = np.abs(rng.normal(loc=1.55, scale=0.08, size=size))
    magnitudes = np.where(component, low_mode, high_mode)
    y = x + _signed_magnitudes(rng, magnitudes)
    return _dataset(x, y, dirty=False)
def sample_dirty_tradeoff(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=6.0, scale=1.0, size=size)
    y = x + 0.05 * rng.normal(size=size)
    return _dataset(x, y, dirty=True)
def sample_dirty_ideal(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=8.0, scale=0.5, size=size)
    y = x + 0.05 * rng.normal(size=size)
    return _dataset(x, y, dirty=True)
def sample_dirty_no_separation(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=0.0, scale=1.0, size=size)
    y = x.copy()
    return _dataset(x, y, dirty=True)
def sample_dirty_partial_separation(rng: np.random.Generator, size: int, dirty_loc: float, dirty_scale: float) -> Dataset:
    x = rng.normal(loc=dirty_loc, scale=dirty_scale, size=size)
    y = x + 0.05 * rng.normal(size=size)
    return _dataset(x, y, dirty=True)
def sample_dirty_high_leverage(rng: np.random.Generator, size: int, leverage_loc: float = 10.0) -> Dataset:
    x = rng.normal(loc=leverage_loc, scale=0.8, size=size)
    y = 0.5 * x + rng.normal(scale=0.2, size=size)
    return _dataset(x, y, dirty=True)
def sample_dirty_response_contamination(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.normal(loc=0.0, scale=1.0, size=size)
    y = x + rng.normal(loc=0.0, scale=4.0, size=size)
    return _dataset(x, y, dirty=True)
def sample_dirty_heavy_tail(rng: np.random.Generator, size: int) -> Dataset:
    x = rng.standard_t(df=2.5, size=size) * 4.0 + 3.0
    y = x + rng.standard_t(df=2.5, size=size)
    return _dataset(x, y, dirty=True)
def sample_dirty_localized_covariate(rng: np.random.Generator, size: int, loc: tuple[float, ...], scale: float, y_shift: float = 2.0) -> Dataset:
    loc_array = np.asarray(loc, dtype=float)
    x = rng.normal(loc=loc_array, scale=float(scale), size=(size, loc_array.size))
    y = _linear_mean(x) + float(y_shift) + 0.12 * rng.normal(size=size)
    if loc_array.size == 1:
        x = x[:, 0]
    return _dataset(x, y, dirty=True)
def sample_mixture(rng: np.random.Generator, size: int, epsilon: float, clean_sampler: Sampler, dirty_sampler: Sampler) -> Dataset:
    dirty_mask = rng.random(size=size) < epsilon
    n_dirty = int(dirty_mask.sum())
    n_clean = size - n_dirty
    clean = clean_sampler(rng, n_clean)
    dirty = dirty_sampler(rng, n_dirty)
    template_x = clean.x if n_clean > 0 else dirty.x
    feature_shape = template_x.shape[1:]
    x = np.empty((size, *feature_shape), dtype=float)
    y = np.empty(size, dtype=float)
    if n_clean:
        x[~dirty_mask] = clean.x
        y[~dirty_mask] = clean.y
    if n_dirty:
        x[dirty_mask] = dirty.x
        y[dirty_mask] = dirty.y
    return Dataset(x=x, y=y, dirty=dirty_mask)
def clean_sampler_for_spec(spec: ExperimentSpec) -> Sampler:
    if spec.dgp_family in {"tradeoff", "high_leverage", "response_contamination", "heavy_tail"}:
        return sample_clean_tradeoff
    if spec.dgp_family == "gaussian_homoscedastic": return sample_clean_homoscedastic
    if spec.dgp_family in {"ideal_separation", "no_separation"}: return sample_clean_homoscedastic
    if spec.dgp_family == "heavy_tail_clean": return sample_clean_heavy_tail
    if spec.dgp_family == "bimodal_clean": return sample_clean_bimodal
    if spec.dgp_family == "low_density_near_quantile": return sample_clean_low_density_near_quantile
    if spec.dgp_family == "bimodal_covariate": return sample_clean_bimodal_covariate
    if spec.dgp_family == "t_covariate": return sample_clean_t_covariate
    if spec.dgp_family == "skewed_covariate": return sample_clean_skewed_covariate
    if spec.dgp_family == "skewed_mixture_covariate": return sample_clean_skewed_mixture_covariate
    if spec.dgp_family == "banana_covariate": return sample_clean_banana_covariate
    if spec.dgp_family in {"spiral_covariate", "swiss_roll_covariate"}: return sample_clean_spiral_covariate
    if spec.dgp_family == "two_moons_covariate": return sample_clean_two_moons_covariate
    return sample_clean_homoscedastic
def dirty_sampler_for_spec(spec: ExperimentSpec) -> Sampler:
    family = spec.contamination_family
    if family == "separated_covariate": return sample_dirty_tradeoff
    if family == "ideal_covariate": return sample_dirty_ideal
    if family == "no_separation": return sample_dirty_no_separation
    if family == "partial_separation":
        dirty_loc = float(spec.extras.get("dirty_loc", 4.0)); dirty_scale = float(spec.extras.get("dirty_scale", 0.7))
        return lambda rng, size: sample_dirty_partial_separation(rng, size, dirty_loc=dirty_loc, dirty_scale=dirty_scale)
    if family == "high_leverage":
        leverage_loc = float(spec.extras.get("leverage_loc", 10.0))
        return lambda rng, size: sample_dirty_high_leverage(rng, size, leverage_loc=leverage_loc)
    if family == "response_contamination": return sample_dirty_response_contamination
    if family == "heavy_tail": return sample_dirty_heavy_tail
    if family in {"localized_covariate", "banana_bridge", "spiral_center", "swiss_roll_hole", "two_moons_gap"}:
        if family == "banana_bridge": default_loc = (0.0, 1.10); default_scale = 0.16
        elif family in {"spiral_center", "swiss_roll_hole"}: default_loc = (0.0, 0.0); default_scale = 0.14
        elif family == "two_moons_gap": default_loc = (0.0, 0.0); default_scale = 0.12
        else: default_loc = (0.0, 0.0); default_scale = 0.18
        loc = (float(spec.extras.get("dirty_loc_x", default_loc[0])), float(spec.extras.get("dirty_loc_y", default_loc[1])))
        dirty_scale = float(spec.extras.get("dirty_scale", default_scale))
        y_shift = float(spec.extras.get("dirty_y_shift", 2.0))
        return lambda rng, size: sample_dirty_localized_covariate(rng, size, loc=loc, scale=dirty_scale, y_shift=y_shift)
    raise ValueError(f"Unknown contamination family: {family}")
