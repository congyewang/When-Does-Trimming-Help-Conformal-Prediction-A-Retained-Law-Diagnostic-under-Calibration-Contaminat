from __future__ import annotations
from dataclasses import dataclass
from math import exp, log
import numpy as np
Array = np.ndarray
def _as_feature_matrix(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return values.reshape(1, 1)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values.reshape(values.shape[0], -1)
def _maybe_vector(score_matrix: Array) -> Array:
    score_matrix = np.asarray(score_matrix, dtype=float)
    if score_matrix.ndim == 2 and score_matrix.shape[1] == 1:
        return score_matrix[:, 0]
    return score_matrix
def _squared_distances(left: Array, right: Array) -> Array:
    left = _as_feature_matrix(left); right = _as_feature_matrix(right)
    diff = left[:, None, :] - right[None, :, :]
    return np.sum(diff**2, axis=2)
def median_heuristic(values: Array) -> float:
    values = _as_feature_matrix(values)
    if values.shape[0] <= 1: return 1.0
    diffs = np.sqrt(_squared_distances(values, values))
    upper = diffs[np.triu_indices(values.shape[0], k=1)]
    upper = upper[upper > 0.0]
    if upper.size == 0:
        return max(float(np.std(values)), 1e-3)
    return float(max(np.median(upper), 1e-3))
def _regularized_inverse_and_logdet(covariance: Array, min_covar: float) -> tuple[Array, float]:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0: covariance = covariance.reshape(1, 1)
    covariance = 0.5 * (covariance + covariance.T) + float(min_covar) * np.eye(covariance.shape[0])
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        covariance = covariance + 10.0 * float(min_covar) * np.eye(covariance.shape[0])
        sign, logdet = np.linalg.slogdet(covariance)
    return np.linalg.pinv(covariance), float(logdet)
@dataclass(frozen=True)
class GaussianSteinModel:
    fit_mean: Array | float
    fit_variance: Array | float
    bandwidth: float
    reference: Array
    def _mean_vector(self, dimension: int) -> Array:
        mean = np.asarray(self.fit_mean, dtype=float)
        if mean.ndim == 0: return np.full(dimension, float(mean), dtype=float)
        return mean.reshape(-1)[:dimension]
    def _variance_vector(self, dimension: int) -> Array:
        variance = np.asarray(self.fit_variance, dtype=float)
        if variance.ndim == 0: return np.full(dimension, max(float(variance), 1e-8), dtype=float)
        if variance.ndim == 2: variance = np.diag(variance)
        return np.maximum(variance.reshape(-1)[:dimension], 1e-8)
    def _score_matrix(self, x: Array) -> Array:
        x_matrix = _as_feature_matrix(x)
        return -(x_matrix - self._mean_vector(x_matrix.shape[1])) / self._variance_vector(x_matrix.shape[1])
    def score_function(self, x: Array) -> Array:
        return _maybe_vector(self._score_matrix(x))
    def _kernel_matrix(self, u: Array, z: Array) -> Array:
        return np.exp(-0.5 * _squared_distances(u, z) / (self.bandwidth**2))
    def kernel(self, u: Array, z: Array) -> Array:
        return self._kernel_matrix(u, z)
    def grad_first(self, u: Array, z: Array) -> Array:
        u_matrix = _as_feature_matrix(u); z_matrix = _as_feature_matrix(z)
        kernel = self._kernel_matrix(u_matrix, z_matrix)
        grad = -((u_matrix[:, None, :] - z_matrix[None, :, :]) / (self.bandwidth**2)) * kernel[:, :, None]
        return grad[:, :, 0] if grad.shape[2] == 1 else grad
    def feature_pairwise(self, u: Array, z: Array) -> Array:
        u_matrix = _as_feature_matrix(u); z_matrix = _as_feature_matrix(z)
        kernel = self._kernel_matrix(u_matrix, z_matrix)
        scores = self._score_matrix(u_matrix)
        grad = -((u_matrix[:, None, :] - z_matrix[None, :, :]) / (self.bandwidth**2)) * kernel[:, :, None]
        return kernel[:, :, None] * scores[:, None, :] + grad
    def feature(self, u: Array, z: Array) -> Array:
        features = self.feature_pairwise(u, z)
        return features[:, :, 0] if features.shape[2] == 1 else features
    def _witness_from_reference(self, z: Array, reference: Array, batch_size: int = 2048) -> Array:
        z_matrix = _as_feature_matrix(z); ref = _as_feature_matrix(reference)
        output = np.empty((z_matrix.shape[0], z_matrix.shape[1]), dtype=float)
        for start in range(0, z_matrix.shape[0], batch_size):
            stop = min(start + batch_size, z_matrix.shape[0])
            output[start:stop] = np.mean(self.feature_pairwise(ref, z_matrix[start:stop]), axis=0)
        return output
    def empirical_witness(self, z: Array, batch_size: int = 2048) -> Array:
        return _maybe_vector(self._witness_from_reference(z, self.reference, batch_size=batch_size))
    def empirical_score(self, z: Array, batch_size: int = 2048) -> Array:
        witness = self._witness_from_reference(z, self.reference, batch_size=batch_size)
        score = self._score_matrix(z)
        return np.sqrt(np.sum(witness**2, axis=1) + np.sum(score**2, axis=1))
    def population_score(self, z: Array, clean_sample: Array) -> Array:
        witness = self._witness_from_reference(z, clean_sample)
        score = self._score_matrix(z)
        return np.sqrt(np.sum(witness**2, axis=1) + np.sum(score**2, axis=1))
    @property
    def kappa0(self) -> float: return 1.0
    @property
    def kappa1(self) -> float: return float(exp(-0.5) / self.bandwidth)
@dataclass(frozen=True)
class GaussianMixtureSteinModel(GaussianSteinModel):
    weights: tuple[float, float] = (0.5, 0.5)
    means: tuple[float, float] = (-2.0, 2.0)
    variances: tuple[float, float] = (0.49, 0.49)
    def _score_matrix(self, x: Array) -> Array:
        x_matrix = _as_feature_matrix(x)
        if x_matrix.shape[1] != 1: return super()._score_matrix(x_matrix)
        x_vector = x_matrix[:, 0]
        weights = np.asarray(self.weights, dtype=float); means = np.asarray(self.means, dtype=float); variances = np.asarray(self.variances, dtype=float)
        log_terms = []; component_scores = []
        for weight, mean, variance in zip(weights, means, variances):
            variance = max(float(variance), 1e-8)
            log_terms.append(log(max(float(weight), 1e-12)) - 0.5 * log(variance) - 0.5 * ((x_vector - mean) ** 2) / variance)
            component_scores.append(-(x_vector - mean) / variance)
        log_terms_array = np.vstack(log_terms); max_log = np.max(log_terms_array, axis=0)
        resp = np.exp(log_terms_array - max_log); resp /= np.sum(resp, axis=0, keepdims=True)
        return np.sum(resp * np.vstack(component_scores), axis=0).reshape(-1, 1)
@dataclass(frozen=True)
class KDESteinModel(GaussianSteinModel):
    score_reference: Array | None = None
    score_bandwidth: float | None = None
    def _score_reference_matrix(self) -> Array:
        return _as_feature_matrix(self.reference if self.score_reference is None else self.score_reference)
    def _score_bandwidth_value(self) -> float:
        return float(self.bandwidth if self.score_bandwidth is None else self.score_bandwidth)
    def _score_matrix(self, x: Array) -> Array:
        x_matrix = _as_feature_matrix(x); ref = self._score_reference_matrix(); bandwidth = max(self._score_bandwidth_value(), 1e-8)
        output = np.empty_like(x_matrix, dtype=float)
        for start in range(0, x_matrix.shape[0], 2048):
            stop = min(start + 2048, x_matrix.shape[0]); block = x_matrix[start:stop]
            diff = block[:, None, :] - ref[None, :, :]
            kernel = np.exp(-0.5 * np.sum(diff**2, axis=2) / (bandwidth**2))
            denom = np.maximum(np.sum(kernel, axis=1, keepdims=True), 1e-12)
            output[start:stop] = -np.sum(kernel[:, :, None] * diff, axis=1) / (bandwidth**2 * denom)
        return output
@dataclass(frozen=True)
class GMMMahalanobisModel:
    weights: Array
    means: Array
    covariances: Array
    min_covar: float = 1e-4
    def score(self, x: Array) -> Array:
        x_matrix = _as_feature_matrix(x); distances = []
        for mean, covariance in zip(self.means, self.covariances):
            inverse, _ = _regularized_inverse_and_logdet(covariance, self.min_covar)
            diff = x_matrix - mean
            distances.append(np.sum((diff @ inverse) * diff, axis=1))
        return np.sqrt(np.maximum(np.min(np.vstack(distances), axis=0), 0.0))
def fit_gaussian_stein_model(x_fit: Array, x_reference: Array, *, bandwidth_scale: float = 1.0, fit_mean_offset: float = 0.0, fit_variance_scale: float = 1.0, use_oracle_score: bool = False, oracle_mean: float | Array = 0.0, oracle_variance: float | Array = 1.0) -> GaussianSteinModel:
    x_fit_matrix = _as_feature_matrix(x_fit); x_reference_matrix = _as_feature_matrix(x_reference)
    if use_oracle_score:
        mean = np.asarray(oracle_mean, dtype=float); variance = np.asarray(oracle_variance, dtype=float)
    else:
        mean = np.mean(x_fit_matrix, axis=0); variance = np.var(x_fit_matrix, axis=0)
    mean = np.asarray(mean, dtype=float) + float(fit_mean_offset); variance = np.maximum(np.asarray(variance, dtype=float) * float(fit_variance_scale), 1e-6)
    if mean.size == 1: mean = float(mean.reshape(-1)[0])
    if variance.size == 1: variance = float(variance.reshape(-1)[0])
    return GaussianSteinModel(fit_mean=mean, fit_variance=variance, bandwidth=max(median_heuristic(x_reference_matrix) * float(bandwidth_scale), 1e-6), reference=np.asarray(x_reference, dtype=float).copy())
def fit_kde_stein_model(x_fit: Array, x_reference: Array, *, bandwidth_scale: float = 1.0, score_bandwidth_scale: float | None = None) -> KDESteinModel:
    x_fit_matrix = _as_feature_matrix(x_fit); x_reference_matrix = _as_feature_matrix(x_reference)
    witness_bandwidth = max(median_heuristic(x_reference_matrix) * float(bandwidth_scale), 1e-6)
    score_scale = float(bandwidth_scale if score_bandwidth_scale is None else score_bandwidth_scale)
    score_bandwidth = max(median_heuristic(x_fit_matrix) * score_scale, 1e-6)
    mean = np.mean(x_fit_matrix, axis=0); variance = np.maximum(np.var(x_fit_matrix, axis=0), 1e-6)
    mean_out: Array | float = float(mean[0]) if mean.size == 1 else mean
    variance_out: Array | float = float(variance[0]) if variance.size == 1 else variance
    return KDESteinModel(fit_mean=mean_out, fit_variance=variance_out, bandwidth=witness_bandwidth, reference=np.asarray(x_reference, dtype=float).copy(), score_reference=np.asarray(x_fit, dtype=float).copy(), score_bandwidth=score_bandwidth)
def fit_gaussian_mixture_stein_model(x_reference: Array, *, bandwidth_scale: float = 1.0, weights: tuple[float, float] = (0.5, 0.5), means: tuple[float, float] = (-2.0, 2.0), variances: tuple[float, float] = (0.49, 0.49)) -> GaussianMixtureSteinModel:
    x_reference = np.asarray(x_reference, dtype=float)
    weights_array = np.asarray(weights, dtype=float); means_array = np.asarray(means, dtype=float); variances_array = np.asarray(variances, dtype=float)
    mixture_mean = float(np.sum(weights_array * means_array)); mixture_second = float(np.sum(weights_array * (variances_array + means_array**2)))
    return GaussianMixtureSteinModel(fit_mean=mixture_mean, fit_variance=max(mixture_second - mixture_mean**2, 1e-6), bandwidth=max(median_heuristic(x_reference) * float(bandwidth_scale), 1e-6), reference=x_reference.copy(), weights=weights, means=means, variances=variances)
def fit_two_component_gmm_stein_model(x_fit: Array, x_reference: Array, *, bandwidth_scale: float = 1.0, max_iter: int = 100, min_variance: float = 1e-4) -> GaussianMixtureSteinModel | GaussianSteinModel:
    x_fit = np.asarray(x_fit, dtype=float).reshape(-1); x_reference = np.asarray(x_reference, dtype=float).reshape(-1)
    if x_fit.size < 4: return fit_gaussian_stein_model(x_fit, x_reference, bandwidth_scale=bandwidth_scale)
    weights = np.asarray([0.5, 0.5], dtype=float); means = np.asarray([np.quantile(x_fit, 0.25), np.quantile(x_fit, 0.75)], dtype=float)
    overall_var = max(float(np.var(x_fit)), min_variance); variances = np.asarray([overall_var / 2.0, overall_var / 2.0], dtype=float)
    for _ in range(max_iter):
        log_terms = []
        for weight, mean, variance in zip(weights, means, variances):
            variance = max(float(variance), min_variance)
            log_terms.append(log(max(float(weight), 1e-12)) - 0.5 * log(variance) - 0.5 * ((x_fit - mean) ** 2) / variance)
        log_terms_array = np.vstack(log_terms); max_log = np.max(log_terms_array, axis=0)
        resp = np.exp(log_terms_array - max_log); resp /= np.sum(resp, axis=0, keepdims=True)
        nk = np.maximum(resp.sum(axis=1), 1e-12); weights = nk / float(x_fit.size); means = (resp @ x_fit) / nk
        variances = np.sum(resp * ((x_fit[None, :] - means[:, None]) ** 2), axis=1) / nk; variances = np.maximum(variances, min_variance)
    order = np.argsort(means); weights = weights[order]; means = means[order]; variances = variances[order]
    return fit_gaussian_mixture_stein_model(x_reference, bandwidth_scale=bandwidth_scale, weights=(float(weights[0]), float(weights[1])), means=(float(means[0]), float(means[1])), variances=(float(variances[0]), float(variances[1])))
def fit_gmm_mahalanobis_model(x_fit: Array, *, n_components: int = 2, max_iter: int = 100, min_covar: float = 1e-4) -> GMMMahalanobisModel:
    x_matrix = _as_feature_matrix(x_fit); n, dimension = x_matrix.shape; n_components = max(1, min(int(n_components), max(n, 1)))
    centered = x_matrix - np.mean(x_matrix, axis=0)
    if n <= n_components:
        mean = np.mean(x_matrix, axis=0, keepdims=True); covariance = np.cov(x_matrix, rowvar=False)
        if np.ndim(covariance) == 0: covariance = np.asarray([[float(covariance)]])
        return GMMMahalanobisModel(weights=np.ones(1), means=mean, covariances=np.asarray(covariance, dtype=float).reshape(dimension, dimension)[None, :, :], min_covar=min_covar)
    _, _, vt = np.linalg.svd(centered, full_matrices=False); projection = centered @ vt[0]
    targets = np.percentile(projection, np.linspace(10.0, 90.0, n_components))
    means = np.vstack([x_matrix[np.argmin(np.abs(projection - target))] for target in targets])
    weights = np.full(n_components, 1.0 / n_components, dtype=float)
    base_cov = np.cov(x_matrix, rowvar=False)
    if np.ndim(base_cov) == 0: base_cov = np.asarray([[float(base_cov)]])
    covariances = np.repeat((np.asarray(base_cov, dtype=float).reshape(dimension, dimension) + min_covar * np.eye(dimension))[None, :, :], n_components, axis=0)
    for _ in range(max_iter):
        log_terms = []
        for weight, mean, covariance in zip(weights, means, covariances):
            inverse, logdet = _regularized_inverse_and_logdet(covariance, min_covar); diff = x_matrix - mean
            mahal_sq = np.sum((diff @ inverse) * diff, axis=1)
            log_terms.append(log(max(float(weight), 1e-12)) - 0.5 * (logdet + mahal_sq + dimension * log(2.0 * np.pi)))
        log_terms_array = np.vstack(log_terms); max_log = np.max(log_terms_array, axis=0)
        resp = np.exp(log_terms_array - max_log); resp /= np.maximum(np.sum(resp, axis=0, keepdims=True), 1e-12)
        nk = np.maximum(resp.sum(axis=1), 1e-12); weights = nk / float(n); means = (resp @ x_matrix) / nk[:, None]
        new_covariances = []
        for k in range(n_components):
            diff = x_matrix - means[k]
            new_covariances.append((diff.T * resp[k]) @ diff / nk[k] + min_covar * np.eye(dimension))
        covariances = np.asarray(new_covariances, dtype=float)
    return GMMMahalanobisModel(weights=weights, means=means, covariances=covariances, min_covar=min_covar)
def build_evaluation_grid(clean_x: Array, dirty_x: Array, grid_size: int) -> Array:
    clean_x_matrix = _as_feature_matrix(clean_x); dirty_x_matrix = _as_feature_matrix(dirty_x); combined = np.vstack([clean_x_matrix, dirty_x_matrix])
    low = np.min(combined, axis=0); high = np.max(combined, axis=0); span = np.maximum(high - low, 1.0); low = low - 0.5 * span; high = high + 0.5 * span
    if combined.shape[1] == 1: return np.linspace(float(low[0]), float(high[0]), grid_size, dtype=float)
    per_dim = int(np.ceil(grid_size ** (1.0 / combined.shape[1]))); axes = [np.linspace(low[j], high[j], per_dim, dtype=float) for j in range(combined.shape[1])]
    mesh = np.meshgrid(*axes, indexing="ij"); grid = np.column_stack([axis.reshape(-1) for axis in mesh])
    return grid[:grid_size]
def stein_feature_envelope(model: GaussianSteinModel, eval_grid: Array, support_points: Array) -> tuple[float, float]:
    eval_grid_matrix = _as_feature_matrix(eval_grid); support_matrix = _as_feature_matrix(support_points); sample_support = support_matrix[: min(64, support_matrix.shape[0])]
    features = model.feature_pairwise(sample_support, eval_grid_matrix); norms = np.sqrt(np.sum(features**2, axis=2)); b_xi = float(norms.max()) if norms.size else 0.0
    if eval_grid_matrix.shape[0] <= 1: return (b_xi, 0.0)
    if eval_grid_matrix.shape[1] == 1:
        order = np.argsort(eval_grid_matrix[:, 0]); sorted_grid = eval_grid_matrix[order]; diffs = np.diff(sorted_grid[:, 0]); diffs = np.where(np.abs(diffs) < 1e-12, 1e-12, diffs); lipschitz = 0.0
        for u in sample_support:
            values = model.feature_pairwise(u.reshape(1, -1), sorted_grid)[0]; value_norm = np.sqrt(np.sum(values**2, axis=1)); quotient = np.abs(np.diff(value_norm) / diffs)
            if quotient.size: lipschitz = max(lipschitz, float(quotient.max()))
        return (b_xi, lipschitz)
    nearest = np.sqrt(_squared_distances(eval_grid_matrix, eval_grid_matrix)); nearest[nearest == 0.0] = np.nan; spacing = float(np.nanmin(nearest)) if np.isfinite(nearest).any() else 1.0
    return (b_xi, b_xi / max(spacing, 1e-6))
