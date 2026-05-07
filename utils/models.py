from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
Array = np.ndarray
class Predictor(Protocol):
    def nonconformity(self, x: Array, y: Array) -> Array:
        ...
    def interval(self, x: Array, tau: float) -> tuple[Array, Array]:
        ...
def _feature_matrix(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x.reshape(x.shape[0], -1)
def _design_matrix(x: Array) -> Array:
    features = _feature_matrix(x)
    return np.column_stack([np.ones(features.shape[0], dtype=float), features])
@dataclass(frozen=True)
class LinearRegressor:
    coefficients: Array
    @property
    def intercept(self) -> float:
        return float(self.coefficients[0])
    @property
    def slope(self) -> Array:
        slope = np.asarray(self.coefficients[1:], dtype=float)
        return slope[0] if slope.size == 1 else slope
    def predict(self, x: Array) -> Array:
        return _design_matrix(x) @ np.asarray(self.coefficients, dtype=float)
    def nonconformity(self, x: Array, y: Array) -> Array:
        return np.abs(np.asarray(y, dtype=float) - self.predict(x))
    def interval(self, x: Array, tau: float) -> tuple[Array, Array]:
        pred = self.predict(x)
        tau = float(tau)
        return pred - tau, pred + tau
@dataclass(frozen=True)
class QuantileLinearRegressor:
    coefficients: Array
    quantile: float
    def predict(self, x: Array) -> Array:
        return _design_matrix(x) @ np.asarray(self.coefficients, dtype=float)
@dataclass(frozen=True)
class ConformalizedQuantilePredictor:
    lower: QuantileLinearRegressor
    upper: QuantileLinearRegressor
    def nonconformity(self, x: Array, y: Array) -> Array:
        lower = self.lower.predict(x)
        upper = self.upper.predict(x)
        y = np.asarray(y, dtype=float)
        return np.maximum(lower - y, y - upper)
    def interval(self, x: Array, tau: float) -> tuple[Array, Array]:
        lower = self.lower.predict(x)
        upper = self.upper.predict(x)
        tau = float(tau)
        return lower - tau, upper + tau
def fit_linear_regressor(x: Array, y: Array) -> LinearRegressor:
    y = np.asarray(y, dtype=float)
    design = _design_matrix(x)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return LinearRegressor(coefficients=np.asarray(coeffs, dtype=float))
def _standardized_design(x: Array) -> tuple[Array, Array, Array]:
    features = _feature_matrix(x)
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (features - center) / scale
    return np.column_stack([np.ones(standardized.shape[0], dtype=float), standardized]), center, scale
def _unstandardize_coefficients(beta: Array, center: Array, scale: Array) -> Array:
    beta = np.asarray(beta, dtype=float)
    slope = beta[1:] / scale
    intercept = beta[0] - np.sum(beta[1:] * center / scale)
    return np.concatenate([[intercept], slope])
def fit_quantile_regressor(x: Array, y: Array, quantile: float) -> QuantileLinearRegressor:
    """Fit a linear quantile regressor using a NumPy-only smoothed pinball solver.

    The original experiments used scipy.optimize at import time.  The solver here avoids a
    heavyweight dependency in the experiment entry points while preserving the intended CQR
    baseline behavior.  It minimizes a smooth approximation to the check loss with Adam.
    """
    y = np.asarray(y, dtype=float)
    design, center, scale = _standardized_design(x)
    n, p = design.shape
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    # A scale-adaptive smoothing parameter for the pinball kink.
    eps = max(1e-3, 0.02 * float(np.std(y) + 1e-8))
    lr = 0.04 / np.sqrt(max(p, 1))
    ridge = 1e-6
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    beta1 = 0.9
    beta2 = 0.999
    for step in range(1, 1201):
        residual = y - design @ beta
        # Smooth approximation to d rho_q(r) / d r: q - 1{r < 0}.
        clipped = np.clip(-residual / eps, -40.0, 40.0)
        sigmoid_neg = 1.0 / (1.0 + np.exp(-clipped))
        dloss_dr = quantile - sigmoid_neg
        grad = -(design.T @ dloss_dr) / float(n)
        grad[1:] += ridge * beta[1:]
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        mhat = m / (1.0 - beta1**step)
        vhat = v / (1.0 - beta2**step)
        beta -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        if step in {400, 800}:
            lr *= 0.5
    coeffs = _unstandardize_coefficients(beta, center, scale)
    return QuantileLinearRegressor(coefficients=np.asarray(coeffs, dtype=float), quantile=float(quantile))
def build_predictor(x: Array, y: Array, predictor_type: str, alpha: float) -> Predictor:
    if predictor_type == "linear":
        return fit_linear_regressor(x, y)
    if predictor_type == "cqr_linear":
        lower = fit_quantile_regressor(x, y, quantile=alpha / 2.0)
        upper = fit_quantile_regressor(x, y, quantile=1.0 - alpha / 2.0)
        return ConformalizedQuantilePredictor(lower=lower, upper=upper)
    raise ValueError(f"Unknown predictor type: {predictor_type}")
