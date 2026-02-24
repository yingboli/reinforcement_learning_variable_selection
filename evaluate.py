"""
Evaluation utilities for variable selection methods.

This module provides functions for evaluating selected features and comparing
against baseline methods like lasso, forward selection, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LassoCV, Ridge, LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_features: np.ndarray,
    task: str = "regression",
    model=None,
    cv: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate selected features on test set.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        selected_features: Indices of selected features
        task: 'regression' or 'classification'
        model: sklearn estimator (default: Ridge for regression, LogisticRegression for classification)
        cv: Number of CV folds for cross-validation score
        
    Returns:
        Dictionary with evaluation metrics (test_r2/test_mse for regression,
        test_accuracy/test_f1 for classification, plus n_features, selected_features).
    """
    if model is None:
        model = Ridge(alpha=1.0) if task == "regression" else LogisticRegression(max_iter=1000, random_state=42)
    
    if len(selected_features) == 0:
        if task == "regression":
            return {
                "task": task,
                "n_features": 0,
                "test_mse": np.inf,
                "test_r2": -np.inf,
                "test_mae": np.inf,
                "cv_r2_mean": -np.inf,
                "cv_r2_std": 0.0,
                "selected_features": [],
            }
        else:
            return {
                "task": task,
                "n_features": 0,
                "test_accuracy": 0.0,
                "test_f1": 0.0,
                "cv_accuracy_mean": 0.0,
                "cv_accuracy_std": 0.0,
                "selected_features": [],
            }
    
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    if task == "regression":
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring="r2")
        return {
            "task": task,
            "n_features": len(selected_features),
            "test_mse": mean_squared_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "selected_features": selected_features.tolist(),
        }
    else:
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring="accuracy")
        return {
            "task": task,
            "n_features": len(selected_features),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "selected_features": selected_features.tolist(),
        }


def compare_with_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_features: np.ndarray,
    task: str = "regression",
    n_features_target: Optional[int] = None,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Compare RL-selected features with baseline methods.
    For regression: LassoCV, Forward Selection, RFE, All. For classification: LogisticRegressionCV, Forward, RFE, All.
    """
    results = []
    if n_features_target is None:
        n_features_target = len(selected_features)
    
    estimator = Ridge(alpha=1.0) if task == "regression" else LogisticRegression(max_iter=1000, random_state=42)
    
    def _row(name: str, res: Dict[str, Any]) -> Dict[str, Any]:
        row = {"method": name, "n_features": res["n_features"]}
        if task == "regression":
            row["test_r2"] = res.get("test_r2")
            row["test_mse"] = res.get("test_mse")
            row["cv_r2_mean"] = res.get("cv_r2_mean")
            row["cv_r2_std"] = res.get("cv_r2_std")
        else:
            row["test_accuracy"] = res.get("test_accuracy")
            row["test_f1"] = res.get("test_f1")
            row["cv_accuracy_mean"] = res.get("cv_accuracy_mean")
            row["cv_accuracy_std"] = res.get("cv_accuracy_std")
        return row
    
    # 1. RL
    rl_results = evaluate_selection(
        X_train, y_train, X_test, y_test, selected_features, task=task, cv=cv
    )
    results.append(_row("RL (PPO)", rl_results))
    
    # 2. LassoCV (regression) or LogisticRegressionCV (classification)
    if task == "regression":
        lasso = LassoCV(cv=cv, random_state=42, max_iter=2000)
        lasso.fit(X_train, y_train)
        lasso_features = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        if len(lasso_features) > 0:
            lr_results = evaluate_selection(
                X_train, y_train, X_test, y_test, lasso_features, task=task, cv=cv
            )
            results.append(_row("LassoCV", lr_results))
    else:
        logreg_cv = LogisticRegressionCV(cv=cv, random_state=42, max_iter=1000)
        logreg_cv.fit(X_train, y_train)
        # Use non-zero coefs as "selected" (for multinomial we use any non-zero)
        if hasattr(logreg_cv, "coef_") and logreg_cv.coef_.ndim == 2:
            selected = np.unique(np.where(np.abs(logreg_cv.coef_) > 1e-6)[1])
        else:
            selected = np.where(np.abs(logreg_cv.coef_.ravel()) > 1e-6)[0]
        if len(selected) > 0:
            lr_results = evaluate_selection(
                X_train, y_train, X_test, y_test, selected, task=task, cv=cv
            )
            results.append(_row("LogisticRegressionCV", lr_results))
    
    # 3. Forward Selection
    if n_features_target > 0 and n_features_target <= X_train.shape[1]:
        try:
            forward_selector = SequentialFeatureSelector(
                estimator,
                n_features_to_select=n_features_target,
                direction="forward",
                cv=cv,
                n_jobs=-1,
            )
            forward_selector.fit(X_train, y_train)
            forward_features = np.where(forward_selector.get_support())[0]
            if len(forward_features) > 0:
                fr_results = evaluate_selection(
                    X_train, y_train, X_test, y_test, forward_features, task=task, cv=cv
                )
                results.append(_row("Forward Selection", fr_results))
        except Exception as e:
            print(f"Forward selection failed: {e}")
    
    # 4. RFE
    if n_features_target > 0 and n_features_target <= X_train.shape[1]:
        try:
            rfe = RFE(estimator, n_features_to_select=n_features_target)
            rfe.fit(X_train, y_train)
            rfe_features = np.where(rfe.get_support())[0]
            if len(rfe_features) > 0:
                rfe_results = evaluate_selection(
                    X_train, y_train, X_test, y_test, rfe_features, task=task, cv=cv
                )
                results.append(_row("RFE", rfe_results))
        except Exception as e:
            print(f"RFE failed: {e}")
    
    # 5. All features
    all_features = np.arange(X_train.shape[1])
    all_results = evaluate_selection(
        X_train, y_train, X_test, y_test, all_features, task=task, cv=cv
    )
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
