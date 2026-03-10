"""
Evaluation utilities for variable selection methods.

This module provides functions for evaluating selected features and comparing
against baseline methods like lasso, forward selection, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import (
    LinearRegression,
    LassoCV,
    Lasso,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_auc_score
from reward_utils import (
    g_prior,
    log_bayes_factor_regression,
    cv_rmse,
    cv_auc,
    reward_regression_bayes_factor,
)

def evaluate_selection(
    X: np.ndarray,
    y: np.ndarray,
    selected_features: np.ndarray,
    task: str = "regression",
    model=None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate selected features on the given data (e.g. test set).
    Uses OLS for regression and LogisticRegression(penalty='none') for classification.
    Null model (no features) is valid: intercept-only; metrics computed correctly.

    Args:
        X, y: Data to fit and evaluate on (e.g. test set)
        selected_features: Indices of selected features
        task: 'regression' or 'classification'
        model: sklearn estimator (default: LinearRegression / LogisticRegression(penalty='none'))
        random_state: Random seed for reproducible fits (classification default model).

    Returns:
        Dict with task, n_features, selected_features; regression: test_mse, test_r2;
        classification: test_f1, test_auc.
    """
    if model is None:
        model = (
            LinearRegression()
            if task == "regression"
            else LogisticRegression(penalty="none", max_iter=1000, random_state=random_state)
        )

    if len(selected_features) == 0:
        # Null model: intercept only
        if task == "regression":
            y_pred_null = np.full_like(y, np.mean(y), dtype=np.float64)
            return {
                "task": task,
                "n_features": 0,
                "test_mse": float(mean_squared_error(y, y_pred_null)),
                "test_r2": 0.0,
                "selected_features": [],
            }
        else:
            # Classification null: predict with p = mean(y)
            y_bin = (y == np.unique(y)[1]).astype(np.int32)
            p_mean = np.mean(y_bin)
            y_pred_null = (np.full(len(y), p_mean) >= 0.5).astype(np.int32)
            return {
                "task": task,
                "n_features": 0,
                "test_f1": float(f1_score(y_bin, y_pred_null, average="weighted", zero_division=0)),
                "test_auc": 0.5,
                "selected_features": [],
            }

    X_sel = X[:, selected_features]
    model.fit(X_sel, y)
    y_pred = model.predict(X_sel)

    if task == "regression":
        return {
            "task": task,
            "n_features": len(selected_features),
            "test_mse": float(mean_squared_error(y, y_pred)),
            "test_r2": float(r2_score(y, y_pred)),
            "selected_features": selected_features.tolist(),
        }
    else:
        proba = model.predict_proba(X_sel)
        auc = float(roc_auc_score(y, proba[:, 1]))
        return {
            "task": task,
            "n_features": len(selected_features),
            "test_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
            "test_auc": auc,
            "selected_features": selected_features.tolist(),
        }


def _mcmc_metropolis_variable_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 1000,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Metropolis algorithm for Bayesian variable selection (g-prior).
    Propose by flipping one coordinate or swapping two; accept with min(1, BF_new/BF_old).
    Starts from a random subset (random start). Uses a cache for visited models' log BF.
    Returns selected feature indices (gamma where gamma_j=1).
    """
    rng = np.random.default_rng(random_state)
    n, p = X_train.shape
    g = g_prior(n, p)
    cache: Dict[Tuple[int, ...], float] = {}

    def get_log_bf(selected_indices: np.ndarray) -> float:
        key = tuple(sorted(selected_indices.tolist()))
        if key not in cache:
            cache[key] = reward_regression_bayes_factor(
                X_train, y_train, np.asarray(key), g=g
            )
        return cache[key]

    # Random start: random subset of size 0 to min(p, 20)
    n_start = int(rng.integers(0, min(p + 1, 21)))
    start_indices = rng.choice(p, size=n_start, replace=False) if n_start > 0 else np.array([], dtype=np.intp)
    gamma = np.zeros(p, dtype=np.int8)
    gamma[start_indices] = 1
    log_bf = get_log_bf(np.where(gamma)[0])
    best_gamma = gamma.copy()
    best_log_bf = log_bf

    for _ in range(n_iter):
        if rng.random() < 0.5:
            # Flip one coordinate
            j = rng.integers(0, p)
            gamma_new = gamma.copy()
            gamma_new[j] = 1 - gamma_new[j]
        else:
            # Swap two coordinates (must differ: one in, one out)
            in_set = np.where(gamma == 1)[0]
            out_set = np.where(gamma == 0)[0]
            if len(in_set) > 0 and len(out_set) > 0:
                i_in = rng.choice(in_set)
                i_out = rng.choice(out_set)
                gamma_new = gamma.copy()
                gamma_new[i_in], gamma_new[i_out] = 0, 1
            else:
                # No valid swap (all 0 or all 1); flip one instead
                j = rng.integers(0, p)
                gamma_new = gamma.copy()
                gamma_new[j] = 1 - gamma_new[j]
        sel_new = np.where(gamma_new)[0]
        log_bf_new = get_log_bf(sel_new)
        if np.log(rng.random()) < log_bf_new - log_bf:
            gamma = gamma_new
            log_bf = log_bf_new
            if log_bf > best_log_bf:
                best_gamma = gamma.copy()
                best_log_bf = log_bf
    return np.where(best_gamma)[0]


def _lasso_by_criterion(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    criterion: str,
    cv: int,
    alphas: Optional[np.ndarray] = None,
    adaptive: bool = False,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Lasso (or Adaptive Lasso if adaptive=True) by criterion: cv, aic, bic, or bayes_factor. Returns selected indices.
    Adaptive: regression uses OLS MLE for weights; classification uses LogisticRegression(penalty='none') MLE.
    Selection uses coefficient in weighted space (delta) so large w do not amplify noise and cause over-selection."""
    from reward_utils import (
        g_prior,
        log_likelihood_regression,
        log_bayes_factor_regression,
        log_likelihood_binary_classification,
        aic,
        bic,
    )

    n, p = X_train.shape
    # Optional adaptive weights (regression: OLS; classification: LogisticRegression MLE)
    X_fit = X_train
    w = np.ones(p)
    if adaptive:
        if task == "regression":
            ols_init = LinearRegression().fit(X_train, y_train)
            w = 1 / np.maximum(np.abs(ols_init.coef_), 1e-5)
        else:
            logreg_mle = LogisticRegression(penalty="none", max_iter=1000, random_state=random_state).fit(X_train, y_train)
            w = 1 / np.maximum(np.abs(logreg_mle.coef_.ravel()), 1e-5)
        X_fit = X_train * w

    # Classification
    if task != "regression":
        use_cv = criterion in ("cv", "bayes_factor")
        if use_cv:
            logreg_cv = LogisticRegressionCV(
                cv=cv, random_state=random_state, max_iter=1000, penalty="l1", solver="saga"
            )
            logreg_cv.fit(X_fit, y_train)
            # For adaptive: threshold in weighted space (delta) to avoid over-selection from large w
            coef_sel = logreg_cv.coef_.ravel() if adaptive else logreg_cv.coef_.ravel()
            return np.where(np.abs(coef_sel) > 1e-6)[0]
        y_bin = (y_train == np.unique(y_train)[1]).astype(np.float64)
        alphas = alphas if alphas is not None else np.logspace(-4, 4, 50)
        best_score, best_support = -np.inf, np.array([], dtype=int)
        for alpha in alphas:
            C = 1.0 / max(alpha, 1e-10)
            logreg_l1 = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=1000, random_state=random_state)
            logreg_l1.fit(X_fit, y_train)
            coef_sel = logreg_l1.coef_.ravel() if adaptive else logreg_l1.coef_.ravel()
            support = np.where(np.abs(coef_sel) > 1e-6)[0]
            if len(support) == 0:
                log_lik = log_likelihood_binary_classification(y_bin, np.full_like(y_bin, np.mean(y_bin)))
                score = -aic(log_lik, 0) if criterion == "aic" else -bic(log_lik, 0, n)
            else:
                logreg_ols = LogisticRegression(penalty="none", max_iter=1000, random_state=random_state)
                logreg_ols.fit(X_train[:, support], y_train)
                p = logreg_ols.predict_proba(X_train[:, support])[:, 1]
                log_lik = log_likelihood_binary_classification(y_bin, p)
                score = -aic(log_lik, len(support)) if criterion == "aic" else -bic(log_lik, len(support), n)
            if score > best_score:
                best_score, best_support = score, support
        return best_support

    # Regression

    # Regression: CV
    if criterion == "cv":
        lasso = LassoCV(cv=cv, random_state=random_state, max_iter=2000)
        lasso.fit(X_fit, y_train)
        coef_sel = lasso.coef_ if adaptive else lasso.coef_
        return np.where(np.abs(coef_sel) > 1e-6)[0]

    # Regression: AIC / BIC / bayes_factor (reward form: -AIC, -BIC, log BF; maximize)
    alphas = alphas if alphas is not None else np.logspace(-4, 4, 50)
    g_val = g_prior(n, p)
    best_score = -np.inf
    best_support = np.array([], dtype=int)
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=2000, random_state=random_state)
        lasso.fit(X_fit, y_train)
        coef_sel = lasso.coef_ if adaptive else lasso.coef_
        support = np.where(np.abs(coef_sel) > 1e-6)[0]
        if len(support) == 0:
            rss_null = np.sum((y_train - np.mean(y_train)) ** 2)
            log_lik_null = log_likelihood_regression(rss_null, n)
            score = 0.0 if criterion == "bayes_factor" else (-aic(log_lik_null, 0) if criterion == "aic" else -bic(log_lik_null, 0, n))
        else:
            ols = LinearRegression()
            ols.fit(X_train[:, support], y_train)
            y_pred = ols.predict(X_train[:, support])
            rss = np.sum((y_train - y_pred) ** 2)
            r2 = r2_score(y_train, y_pred)
            log_lik = log_likelihood_regression(rss, n)
            p_g = len(support)
            score = log_bayes_factor_regression(r2, n, p_g, g_val) if criterion == "bayes_factor" else (-aic(log_lik, p_g) if criterion == "aic" else -bic(log_lik, p_g, n))
        if score > best_score:
            best_score, best_support = score, support
    return best_support


def _forward_backward_selection_by_criterion(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    criterion: str,
    cv: int,
    direction: str = "forward",
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Forward or backward selection by criterion. Stops when criterion would decrease.
    direction: 'forward' (add features from empty) or 'backward' (remove features from full)."""
    from sklearn.model_selection import cross_val_score, KFold
    from reward_utils import (
        g_prior,
        log_likelihood_regression,
        log_bayes_factor_regression,
        reward_classification_aic,
        reward_classification_bic,
        aic,
        bic,
    )

    n, p = X_train.shape
    g_val = g_prior(n, p) if criterion == "bayes_factor" else None
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state) if random_state is not None else cv

    def _score(indices):
        if len(indices) == 0:
            if task == "regression":
                rss = np.sum((y_train - np.mean(y_train)) ** 2)
                log_lik = log_likelihood_regression(rss, n)
                if criterion == "aic":
                    return -aic(log_lik, 0)
                if criterion == "bic":
                    return -bic(log_lik, 0, n)
                return 0.0
            if task == "classification":
                if criterion == "aic":
                    return reward_classification_aic(X_train, y_train, np.array([], dtype=int), random_state=random_state)
                if criterion == "bic":
                    return reward_classification_bic(X_train, y_train, np.array([], dtype=int), random_state=random_state)
                return 0.5
        X_sel = X_train[:, indices]
        if task == "regression":
            ols = LinearRegression().fit(X_sel, y_train)
            y_pred = ols.predict(X_sel)
            rss = np.sum((y_train - y_pred) ** 2)
            r2 = r2_score(y_train, y_pred)
            log_lik = log_likelihood_regression(rss, n)
            p_g = len(indices)
            if criterion == "cv":
                return cross_val_score(ols, X_sel, y_train, cv=cv_splitter, scoring="r2").mean()
            if criterion == "aic":
                return -aic(log_lik, p_g)
            if criterion == "bic":
                return -bic(log_lik, p_g, n)
            return log_bayes_factor_regression(r2, n, p_g, g_val)
        if criterion == "cv":
            lr = LogisticRegression(penalty="none", max_iter=1000, random_state=random_state)
            return cross_val_score(lr, X_sel, y_train, cv=cv_splitter, scoring="roc_auc").mean()
        if criterion == "aic":
            return reward_classification_aic(X_train, y_train, np.asarray(indices), random_state=random_state)
        return reward_classification_bic(X_train, y_train, np.asarray(indices), random_state=random_state)

    if direction == "forward":
        selected = []
        current_score = _score(selected)
        for _ in range(p):
            best_j = -1
            best_score = -np.inf
            for j in range(p):
                if j in selected:
                    continue
                try_set = selected + [j]
                score = _score(try_set)
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j < 0 or best_score <= current_score:
                break
            selected.append(best_j)
            current_score = best_score
        return np.array(selected)

    # backward
    current = list(range(p))
    current_score = _score(current)
    while len(current) > 1:
        best_drop = -1
        best_score_after = -np.inf
        for idx in range(len(current)):
            try_set = [current[i] for i in range(len(current)) if i != idx]
            score = _score(try_set)
            if score > best_score_after:
                best_score_after = score
                best_drop = idx
        if best_drop < 0 or best_score_after < current_score:
            break
        current = [current[i] for i in range(len(current)) if i != best_drop]
        current_score = best_score_after
    return np.array(current)


def compare_with_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_features: np.ndarray,
    task: str = "regression",
    cv: int = 5,
    selection_criterion: str = "cv",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compare RL-selected features with baseline methods.
    selection_criterion: 'cv' (default), 'aic', 'bic', or 'bayes_factor'. Controls how
    Lasso/Adaptive Lasso/Forward/RFE choose their hyperparameters or feature order.
    Forward and RFE stop when the criterion would decrease.
    Regression: RL, Lasso, Adaptive Lasso, Forward, RFE, MCMC (Metropolis), All.
    Classification: RL, LogisticRegressionCV, Adaptive Lasso, Forward, RFE, All.
    """
    results = []

    estimator = (
        LinearRegression()
        if task == "regression"
        else LogisticRegression(penalty="none", max_iter=1000, random_state=random_state)
    )

    def _row(name: str, res: Dict[str, Any]) -> Dict[str, Any]:
        row = {"method": name, "n_features": res["n_features"]}
        if task == "regression":
            row["test_r2"] = res.get("test_r2")
            row["test_mse"] = res.get("test_mse")
        else:
            row["test_f1"] = res.get("test_f1")
            row["test_auc"] = res.get("test_auc")
        return row

    # 1. RL
    rl_results = evaluate_selection(X_test, y_test, selected_features, task=task, random_state=random_state)
    results.append(_row("RL (PPO)", rl_results))

    # 2. Lasso (regression) or LogisticRegressionCV (classification)
    lasso_features = _lasso_by_criterion(X_train, y_train, task, selection_criterion, cv, random_state=random_state)
    lr_results = evaluate_selection(X_test, y_test, lasso_features, task=task, random_state=random_state)
    label = "LassoCV" if selection_criterion == "cv" else f"Lasso ({selection_criterion})"
    
    results.append(_row(label, lr_results))

    # 2b. Adaptive Lasso (weights: OLS MLE for regression, LogisticRegression MLE for classification)
    try:
        adlasso_features = _lasso_by_criterion(
            X_train, y_train, task, selection_criterion, cv, adaptive=True, random_state=random_state
        )
        adlasso_results = evaluate_selection(X_test, y_test, adlasso_features, task=task, random_state=random_state)
        label = "Adaptive Lasso (CV)" if selection_criterion == "cv" else f"Adaptive Lasso ({selection_criterion})"
        results.append(_row(label, adlasso_results))
    except Exception as e:
        print(f"Adaptive Lasso failed: {e}")

    # 3. Forward Selection
    try:
        forward_features = _forward_backward_selection_by_criterion(
            X_train, y_train, task, selection_criterion, cv, direction="forward", random_state=random_state
        )
        fr_results = evaluate_selection(X_test, y_test, forward_features, task=task, random_state=random_state)
        results.append(_row("Forward Selection", fr_results))
    except Exception as e:
        print(f"Forward selection failed: {e}")

    # 4. Backward selection
    try:
        backward_features = _forward_backward_selection_by_criterion(
            X_train, y_train, task, selection_criterion, cv, direction="backward", random_state=random_state
        )
        backward_results = evaluate_selection(X_test, y_test, backward_features, task=task, random_state=random_state)
        results.append(_row("Backward Selection", backward_results))
    except Exception as e:
        print(f"Backward selection failed: {e}")

    # 5. MCMC (regression only)
    if task == "regression":
        try:
            mcmc_features = _mcmc_metropolis_variable_selection(
                X_train, y_train, random_state=random_state
            )
            mcmc_results = evaluate_selection(X_test, y_test, mcmc_features, task=task, random_state=random_state)
            results.append(_row("MCMC (Metropolis)", mcmc_results))
        except Exception as e:
            print(f"MCMC failed: {e}")

    # 6. All features
    all_features = np.arange(X_train.shape[1])
    all_results = evaluate_selection(X_test, y_test, all_features, task=task, random_state=random_state)
    results.append(_row("All Features", all_results))

    return pd.DataFrame(results)


def compute_precision_recall(
    selected_features: np.ndarray,
    true_features: np.ndarray,
    n_total_features: int,
) -> Dict[str, float]:
    """
    Compute precision and recall for feature selection (if ground truth available).
    
    Args:
        selected_features: Indices of selected features
        true_features: Indices of truly important features
        n_total_features: Total number of features
        
    Returns:
        Dictionary with precision, recall, F1, and accuracy
    """
    selected_set = set(selected_features)
    true_set = set(true_features)
    
    # True positives: features that are both selected and true
    tp = len(selected_set & true_set)
    
    # False positives: features that are selected but not true
    fp = len(selected_set - true_set)
    
    # False negatives: features that are true but not selected
    fn = len(true_set - selected_set)
    
    # True negatives: features that are neither selected nor true
    tn = n_total_features - tp - fp - fn
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_total_features
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def plot_selection_history(
    history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot training history (rewards, number of features, etc.).
    
    Args:
        history: List of dictionaries with training metrics
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    if not history:
        print("No history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract metrics
    episodes = [h.get("episode", i) for i, h in enumerate(history)]
    rewards = [h.get("reward", 0) for h in history]
    n_features = [h.get("n_features", 0) for h in history]
    test_r2 = [h.get("test_r2", None) for h in history]
    
    # Plot 1: Reward over time
    axes[0, 0].plot(episodes, rewards, alpha=0.6)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Reward Over Time")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of features over time
    axes[0, 1].plot(episodes, n_features, alpha=0.6, color="orange")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Number of Selected Features")
    axes[0, 1].set_title("Feature Selection Size Over Time")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Reward vs Number of Features (scatter)
    axes[1, 0].scatter(n_features, rewards, alpha=0.5)
    axes[1, 0].set_xlabel("Number of Selected Features")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].set_title("Reward vs Feature Count")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Test R² over time (if available)
    if any(r2 is not None for r2 in test_r2):
        valid_indices = [i for i, r2 in enumerate(test_r2) if r2 is not None]
        valid_episodes = [episodes[i] for i in valid_indices]
        valid_r2 = [test_r2[i] for i in valid_indices]
        axes[1, 1].plot(valid_episodes, valid_r2, alpha=0.6, color="green")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Test R²")
        axes[1, 1].set_title("Test R² Over Time")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No test R² data", ha="center", va="center")
        axes[1, 1].set_title("Test R² Over Time")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
