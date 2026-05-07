from __future__ import annotations
from math import exp, isfinite, log, sqrt
def theorem_mixture_loss(bias: float, tilde_epsilon: float) -> float:
    return (1.0 - tilde_epsilon) * bias + tilde_epsilon
def finite_sample_remainder(mu_keep: float, m_cal: int, eta_fraction: float) -> float:
    if mu_keep <= 0.0:
        return float("inf")
    eta = eta_fraction * mu_keep
    if eta <= 0.0 or eta >= mu_keep:
        return float("inf")
    finite_sample = sqrt(log(max(m_cal, 2)) / (m_cal * max(mu_keep - eta, 1e-12)))
    finite_sample += exp(-2.0 * m_cal * eta**2) + 2.0 * (m_cal**-2)
    return finite_sample
def theorem_lower_bound(
    alpha: float,
    bias: float,
    tilde_epsilon: float,
    mu_keep: float,
    m_cal: int,
    eta_fraction: float,
) -> float:
    if mu_keep <= 0.0:
        return float("-inf")
    remainder = finite_sample_remainder(mu_keep, m_cal, eta_fraction)
    if not isfinite(remainder):
        return float("-inf")
    return 1.0 - alpha - theorem_mixture_loss(bias, tilde_epsilon) - remainder
def proposition_lower_bound(
    alpha: float,
    epsilon: float,
    bias: float,
    tilde_epsilon: float,
    mu_keep: float,
    m_cal: int,
    eta_fraction: float,
) -> float:
    return (1.0 - epsilon) * theorem_lower_bound(
        alpha, bias, tilde_epsilon, mu_keep, m_cal, eta_fraction
    )
def additive_mixture_lower_bound(
    alpha: float,
    epsilon: float,
    bias: float,
    tilde_epsilon: float,
    mu_keep: float,
    m_cal: int,
    eta_fraction: float,
) -> float:
    remainder = finite_sample_remainder(mu_keep, m_cal, eta_fraction)
    if not isfinite(remainder):
        return float("-inf")
    return (
        (1.0 - epsilon) * (1.0 - alpha)
        - theorem_mixture_loss(bias, tilde_epsilon)
        - remainder
    )
def retained_count_upper(pi_value: float, m_cal: int, eta: float) -> float:
    return m_cal * (pi_value + eta)
def retained_count_lower(pi_value: float, m_cal: int, eta: float) -> float:
    return m_cal * max(pi_value - eta, 0.0)
def empirical_retained_contamination_upper(
    pi_d: float, mu_keep: float, eta: float
) -> float:
    if mu_keep <= eta:
        return float("inf")
    return (pi_d + eta) / (mu_keep - eta)
def bias_tv_bound(p_c: float) -> float:
    return 1.0 - p_c
def bias_covariance_bound(p_c: float) -> float:
    if p_c <= 0.0:
        return float("inf")
    return 0.5 * sqrt((1.0 - p_c) / p_c)
def quantile_shift_bound(delta: float, lambda_min: float) -> float:
    if lambda_min <= 0.0:
        return float("inf")
    return delta / lambda_min
def dirty_second_moment_lower(d_sq: float, sigma_c_sq: float, epsilon: float) -> float:
    if epsilon <= 0.0:
        return float("nan")
    return (d_sq - (1.0 - epsilon) * sigma_c_sq) / epsilon
def clean_tail_upper(sigma_c_sq: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 1.0
    return min(1.0, sigma_c_sq / (threshold**2))
def dirty_tail_lower(m_d: float, threshold: float, score_bound: float) -> float:
    if threshold <= 0.0 or score_bound <= threshold or threshold**2 >= m_d:
        return 0.0
    numerator = max(m_d - threshold**2, 0.0)
    denominator = max(score_bound**2 - threshold**2, 1e-12)
    return max(0.0, min(1.0, numerator / denominator))
def localized_dirty_second_moment_lower(
    d0_sq: float, sigma_c_sq: float, epsilon: float
) -> float:
    if epsilon <= 0.0:
        return float("nan")
    return (d0_sq - (1.0 - epsilon) * sigma_c_sq) / epsilon
def localized_clean_retention_lower(
    sigma_c_sq: float, t_minus: float, zeta_p: float
) -> float:
    if t_minus <= 0.0:
        return float("nan")
    return 1.0 - zeta_p - sigma_c_sq / (t_minus**2)
def localized_dirty_retention_upper(
    m_d0_lower: float,
    t_plus: float,
    l0: float,
    zeta_q: float,
) -> float:
    q_q = 1.0 - zeta_q
    if q_q <= 0.0:
        return float("nan")
    if t_plus <= 0.0 or l0 <= t_plus:
        return float("nan")
    numerator = m_d0_lower - q_q * (t_plus**2)
    denominator = l0**2 - t_plus**2
    if numerator <= 0.0 or denominator <= 0.0:
        return float("nan")
    ratio = numerator / denominator
    if ratio < 0.0 or ratio > 1.0:
        return float("nan")
    return 1.0 - ratio
def localized_dirty_tail_lower(
    m_d0_lower: float,
    t_plus: float,
    l0: float,
    zeta_q: float,
) -> float:
    q_q = 1.0 - zeta_q
    if q_q <= 0.0:
        return float("nan")
    if t_plus <= 0.0 or l0 <= t_plus:
        return float("nan")
    numerator = m_d0_lower - q_q * (t_plus**2)
    denominator = l0**2 - t_plus**2
    if numerator <= 0.0 or denominator <= 0.0:
        return float("nan")
    value = numerator / denominator
    return value if 0.0 <= value <= 1.0 else float("nan")
def bridge_delta_s(
    b_xi: float,
    l_xi: float,
    rho: float,
    delta: float,
    ell: int,
    dimension: int,
    covering_number: int,
) -> float:
    if ell <= 0:
        return float("inf")
    first = 2.0 * l_xi * rho
    second = b_xi * sqrt(
        (8.0 * dimension / ell)
        * log(max((2 * dimension * max(covering_number, 1)) / max(delta, 1e-12), 1.0))
    )
    return first + second
