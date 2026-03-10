"""
Shared reward and model-selection utilities for variable selection.

Used by env_base.py (RL reward computation) and evaluate.py (e.g. MCMC baseline, AIC/BIC/BF selection).
All models use OLS for regression and LogisticRegression(penalty='none') for classification.

Public reward API: all reward-computing functions take (X_train, y_train, selected_indices, ...)
and return the scalar used as reward (e.g. -RMSE, AUC, -AIC, -BIC, log BF).
"""

import numpy as np
from typing import Optional

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score


def g_prior(n: int, p: int) -> float:
    """g-prior hyperparameter: g = max(p^2, n) (Liang et al. 2008)."""
    return float(max(p * p, n))


def _log_likelihood_regression(rss: float, n: int) -> float:
    """
    Gaussian log-likelihood for linear regression: -n/2 * (log(2*pi) + log(sigma2_mle) + 1),
    with sigma2_mle = RSS/n. Internal helper.
    """
    sigma2_mle = max(rss / n, 1e-12)
    return -n / 2 * (np.log(2 * np.pi) + np.log(sigma2_mle) + 1)


def _log_likelihood_binary_classification(y: np.ndarray, p: np.ndarray) -> float:
    """
    Bernoulli log-likelihood: sum( y*log(p) + (1-y)*log(1-p) ). Internal helper.
    """
    p_safe = np.clip(p, 1e-12, 1 - 1e-12)
    y_ = np.asarray(y, dtype=np.float64).ravel()
    return float(np.sum(y_ * np.log(p_safe) + (1 - y_) * np.log(1 - p_safe)))


def _aic(log_lik: float, p_gamma: int) -> float:
    """AIC = -log(L) + p_gamma. Lower is better; reward = -AIC. Internal helper."""
    return -log_lik + p_gamma


def _bic(log_lik: float, p_gamma: int, n: int) -> float:
    """BIC = -log(L) + (p_gamma/2)*log(n). Lower is better; reward = -BIC. Internal helper."""
    return -log_lik + (p_gamma / 2) * np.log(n)


def _log_bayes_factor_regression(r2: float, n: int, p_gamma: int, g: float) -> float:
    """
    Log Bayes factor vs null (intercept-only) under g-prior (Liang et al. 2008). Internal helper.
    """
    r2_safe = np.clip(r2, 1e-10, 1 - 1e-10)
    term1 = ((n - p_gamma - 1) / 2) * np.log(1 + g)
    term2 = ((n - 1) / 2) * np.log(1 + g * (1 - r2_safe))
    return float(term1 - term2)


# Back-compat: formula helpers (evaluate.py / env_base may still use these)
log_likelihood_regression = _log_likelihood_regression
log_likelihood_binary_classification = _log_likelihood_binary_classification
aic = _aic
bic = _bic
log_bayes_factor_regression = _log_bayes_factor_regression


# ---------- Public reward API: (X_train, y_train, selected_indices, ...) ----------


def cv_rmse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: np.ndarray,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> float:
    """
    K-fold cross-validated RMSE for linear regression (OLS on selected features).
    Reward = -RMSE (higher is better). Null model (no selection): predict with mean(y) per fold.
    """
    n = X_train.shape[0]
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    if len(selected_indices) == 0:
        mses = []
        for train_idx, val_idx in kf.split(y_train):
            mean_train = np.mean(y_train[train_idx])
            mses.append(np.mean((y_train[val_idx] - mean_train) ** 2))
        return -np.sqrt(max(np.mean(mses), 1e-12))
    X_sel = X_train[:, selected_indices]
    model = LinearRegression()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        model, X_sel, y_train, cv=kf, scoring="neg_mean_squared_error",
    )
    return -np.sqrt(max(-float(np.mean(scores)), 1e-12))


def cv_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: np.ndarray,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> float:
    """
    K-fold cross-validated AUC for binary classification (LogisticRegression, penalty='none', on selected features).
    Null model (no selection): return 0.5.
    """
    if len(selected_indices) == 0:
        return 0.5
    X_sel = X_train[:, selected_indices]
    model = LogisticRegression(penalty="none", random_state=random_state, max_iter=1000)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_sel, y_train, cv=kf, scoring="roc_auc")
    return float(np.mean(scores))


def reward_regression_aic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    """Reward = -AIC for OLS on selected features. Null: intercept-only (mean)."""
    n = X_train.shape[0]
    if len(selected_indices) == 0:
        rss = np.sum((y_train - np.mean(y_train)) ** 2)
        log_lik = _log_likelihood_regression(rss, n)
        return -_aic(log_lik, 0)
    X_sel = X_train[:, selected_indices]
    model = LinearRegression()
    model.fit(X_sel, y_train)
    y_pred = model.predict(X_sel)
    rss = np.sum((y_train - y_pred) ** 2)
    log_lik = _log_likelihood_regression(rss, n)
    return -_aic(log_lik, len(selected_indices))


def reward_regression_bic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    """Reward = -BIC for OLS on selected features. Null: intercept-only."""
    n = X_train.shape[0]
    if len(selected_indices) == 0:
        rss = np.sum((y_train - np.mean(y_train)) ** 2)
        log_lik = _log_likelihood_regression(rss, n)
        return -_bic(log_lik, 0, n)
    X_sel = X_train[:, selected_indices]
    model = LinearRegression()
    model.fit(X_sel, y_train)
    y_pred = model.predict(X_sel)
    rss = np.sum((y_train - y_pred) ** 2)
    log_lik = _log_likelihood_regression(rss, n)
    return -_bic(log_lik, len(selected_indices), n)


def reward_regression_bayes_factor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: np.ndarray,
    g: Optional[float] = None,
) -> float:
    """Log Bayes factor vs null for OLS on selected features. Null returns 0.0. g from g_prior(n, p) if not given."""
    n, p_total = X_train.shape[0], X_train.shape[1]
    if len(selected_indices) == 0:
        return 0.0
    if g is None:
        g = g_prior(n, p_total)
    X_sel = X_train[:, selected_indices]
    model = LinearRegression()
    model.fit(X_sel, y_train)
    y_pred = model.predict(X_sel)
    r2 = r2_score(y_train, y_pred) if n > 1 else 0.0
    return _log_bayes_factor_regression(r2, n, len(selected_indices), g)


def reward_classification_aic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: np.ndarray,
    random_state: Optional[int] = None,
) -> float:
    """Reward = -AIC for binary logistic regression on selected features. y_train: 0/1 or class labels."""
    n = X_train.shape[0]
    y_bin = _binary_labels(y_train)
    if len(selected_indices) == 0:
        p_arr = np.full(n, np.mean(y_bin), dtype=np.float64)
        log_lik = _log_likelihood_binary_classification(y_bin, p_arr)
        return -_aic(log_lik, 0)
    X_sel = X_train[:, selected_indices]
    model = LogisticRegression(penalty="none", random_state=random_state, max_iter=1000)
    model.fit(X_sel, y_train)
    p = model.predict_proba(X_sel)[:, 1]
    y_bin = (y_train == model.classes_[1]).astype(np.float64)
    log_lik = _log_likelihood_binary_classification(y_bin, p)
    return -_aic(log_lik, len(selected_indices))


def reward_classification_bic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: np.ndarray,
    random_state: Optional[int] = None,
) -> float:
    """Reward = -BIC for binary logistic regression on selected features."""
    n = X_train.shape[0]
    y_bin = _binary_labels(y_train)
    if len(selected_indices) == 0:
        p_arr = np.full(n, np.mean(y_bin), dtype=np.float64)
        log_lik = _log_likelihood_binary_classification(y_bin, p_arr)
        return -_bic(log_lik, 0, n)
    X_sel = X_train[:, selected_indices]
    model = LogisticRegression(penalty="none", random_state=random_state, max_iter=1000)
    model.fit(X_sel, y_train)
    p = model.predict_proba(X_sel)[:, 1]
    y_bin = (y_train == model.classes_[1]).astype(np.float64)
    log_lik = _log_likelihood_binary_classification(y_bin, p)
    return -_bic(log_lik, len(selected_indices), n)


def _binary_labels(y: np.ndarray) -> np.ndarray:
    """Map y to 0/1 using second unique class as 1 (matches LogisticRegression.classes_[1])."""
    uniq = np.unique(y)
    return (y == uniq[1]).astype(np.float64)
