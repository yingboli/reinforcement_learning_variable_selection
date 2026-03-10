"""
Synthetic data generation for variable selection simulations.

Provides Toeplitz-correlated design with controlled SNR for regression (n >= p).
"""

import numpy as np
from scipy.linalg import cholesky
from typing import Tuple


def toeplitz_covariance(p: int, rho: float) -> np.ndarray:
    """
    Toeplitz covariance matrix: Σ_ij = rho^|i-j|.
    For rho=0 returns identity (uncorrelated).
    """
    if rho == 0:
        return np.eye(p)
    i = np.arange(p, dtype=float)
    j = np.arange(p, dtype=float)
    return np.power(rho, np.abs(np.subtract.outer(i, j)))


def generate_toeplitz_regression(
    n: int,
    p_total: int,
    p_true: int,
    snr: float,
    rho: float,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate regression data with Toeplitz-correlated design and target SNR.

    Model: y = X @ beta + epsilon, X ~ N(0, Sigma), Sigma Toeplitz with rho.
    SNR = Var(X@beta) / sigma^2 (signal variance / noise variance).
    True coefficients are 1 for the first p_true predictors, 0 elsewhere.

    Parameters
    ----------
    n : int
        Sample size.
    p_total : int
        Total number of predictors (must be <= n).
    p_true : int
        Number of true (non-zero) coefficients. Use 0 for null model.
    snr : float
        Target signal-to-noise ratio: Var(X@beta) / sigma^2.
    rho : float
        Toeplitz correlation parameter (0 = uncorrelated, 0.5, 0.9).
    random_state : int
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n, p_total)
        Design matrix (not standardized).
    y : np.ndarray, shape (n,)
        Response.
    true_features : np.ndarray
        Indices of true predictors (0 .. p_true-1). Empty if p_true=0.
    beta : np.ndarray, shape (p_total,)
        True coefficient vector.
    sigma : float
        Residual standard deviation used.
    """
    if p_total > n:
        raise ValueError("Require n >= p_total; do not use p > n case.")
    rng = np.random.default_rng(random_state)

    # Covariance and Cholesky
    Sigma = toeplitz_covariance(p_total, rho)
    L = cholesky(Sigma, lower=True)

    # X ~ N(0, Sigma)
    Z = rng.standard_normal((n, p_total))
    X = Z @ L.T

    # beta: first p_true entries = 1, rest 0
    beta = np.zeros(p_total)
    if p_true > 0:
        beta[:p_true] = 1.0
    true_features = np.arange(p_true) if p_true > 0 else np.array([], dtype=int)

    # Residual sigma to achieve target SNR
    # Var(X@beta) = beta' Sigma beta (population)
    signal_var = float(beta @ Sigma @ beta)
    if p_true == 0:
        sigma = 1.0
        y = rng.standard_normal(n) * sigma
    else:
        if signal_var <= 0:
            sigma = 1.0
        else:
            sigma = np.sqrt(signal_var / snr)
        y = X @ beta + rng.standard_normal(n) * sigma

    return X, y, true_features, beta, sigma
