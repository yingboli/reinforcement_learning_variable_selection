"""
Base environment for reinforcement learning-based variable selection.

Shared logic for reward computation, model fitting, and caching.
Supports regression (Ridge) and classification (LogisticRegression).
Subclasses define action/observation spaces and step/reset semantics.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from typing import Optional, Tuple, Dict, Any


class BaseVariableSelectionEnv(gym.Env):
    """
    Base class for variable selection environments.
    
    Handles common setup (data, model, cache) and reward computation.
    Supports task='regression' (Ridge, reward_type r2/mse) or task='classification'
    (LogisticRegression, reward_type accuracy/f1_weighted/roc_auc).
    Subclasses must define action_space, observation_space, reset(), step(),
    and _get_observation().
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    _REGRESSION_REWARD_TYPES = ("r2", "mse", "cv_rmse", "aic", "bic", "bayes_factor")
    _CLASSIFICATION_REWARD_TYPES = ("accuracy", "f1_weighted", "roc_auc")
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "regression",
        sparsity_penalty: float = 0.01,
        reward_type: Optional[str] = None,
        use_cv: bool = True,
        cv_folds: int = 3,
        model_alpha: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the base variable selection environment.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - continuous for regression, int/labels for classification
            task: 'regression' or 'classification'
            sparsity_penalty: Penalty coefficient α for number of selected features
            reward_type: For regression: 'r2', 'mse', 'cv_rmse', 'aic', 'bic', 'bayes_factor'.
                For classification: 'accuracy', 'f1_weighted', 'roc_auc'. Default: 'r2' / 'accuracy'.
            use_cv: Whether to use cross-validation for reward estimation
            cv_folds: Number of CV folds if use_cv=True
            model_alpha: Regularization (Ridge alpha or LogisticRegression C=1/alpha)
            random_state: Random seed for reproducibility
        """
        super().__init__()
        
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples")
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        self.task = task
        self.sparsity_penalty = sparsity_penalty
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.model_alpha = model_alpha
        self.random_state = random_state
        
        if reward_type is None:
            reward_type = "r2" if task == "regression" else "accuracy"
        if task == "regression" and reward_type not in self._REGRESSION_REWARD_TYPES:
            raise ValueError(f"reward_type must be in {self._REGRESSION_REWARD_TYPES} for regression")
        if task == "classification" and reward_type not in self._CLASSIFICATION_REWARD_TYPES:
            raise ValueError(f"reward_type must be in {self._CLASSIFICATION_REWARD_TYPES} for classification")
        self.reward_type = reward_type
        
        if task == "regression":
            self.model = Ridge(alpha=model_alpha, random_state=random_state)
        else:
            self.model = LogisticRegression(
                C=1.0 / max(model_alpha, 1e-6),
                random_state=random_state,
                max_iter=1000,
            )
        self.cache = {}
        self._n = self.n_samples
        self._p_total = self.n_features
        # g-prior for Bayes factor: g = max(p^2, n) (Liang et al.)
        self._g_prior = max(self._p_total ** 2, self._n) if self.task == "regression" else None
    
    def _compute_reward(self, selected_indices: np.ndarray) -> float:
        """
        Compute reward for a given feature selection.
        Regression: reward = r2 - penalty (or -mse - penalty).
        Classification: reward = accuracy/f1/roc_auc - penalty.
        """
        n_selected = len(selected_indices)
        
        if n_selected == 0:
            # No features: regression = fit intercept only; classification = majority class
            X_dummy = np.zeros((self.n_samples, 1))
            if self.task == "regression":
                self.model.fit(X_dummy, self.y)
                y_pred = self.model.predict(X_dummy)
                rss = np.sum((self.y - y_pred) ** 2)
                r2_0 = r2_score(self.y, y_pred) if self.n_samples > 1 else 0.0
                if self.reward_type == "r2":
                    performance = r2_0
                elif self.reward_type == "mse":
                    performance = -mean_squared_error(self.y, y_pred)
                elif self.reward_type == "cv_rmse":
                    if self.use_cv:
                        scores = cross_val_score(
                            self.model, X_dummy, self.y, cv=self.cv_folds,
                            scoring="neg_mean_squared_error",
                        )
                        performance = -np.sqrt(max(-scores.mean(), 1e-12))
                    else:
                        performance = -np.sqrt(rss / max(self.n_samples, 1))
                elif self.reward_type == "aic":
                    # AIC = n*log(RSS/n) + 2*p_γ, p_γ=0
                    performance = -self.n_samples * np.log(max(rss / self.n_samples, 1e-12))
                elif self.reward_type == "bic":
                    performance = -self.n_samples * np.log(max(rss / self.n_samples, 1e-12))
                elif self.reward_type == "bayes_factor":
                    # BF null vs null: 1, log(BF)=0
                    performance = 0.0
                else:
                    performance = r2_0 if self.reward_type == "r2" else -mean_squared_error(self.y, y_pred)
            else:
                # Classification: majority class baseline
                from collections import Counter
                majority = Counter(self.y).most_common(1)[0][0]
                y_pred = np.full_like(self.y, majority)
                if self.reward_type == "accuracy":
                    performance = accuracy_score(self.y, y_pred)
                elif self.reward_type == "f1_weighted":
                    performance = f1_score(self.y, y_pred, average="weighted", zero_division=0)
                else:
                    performance = accuracy_score(self.y, y_pred)
            return performance - self.sparsity_penalty * 0
        
        cache_key = tuple(sorted(selected_indices))
        if cache_key in self.cache:
            performance = self.cache[cache_key]
        else:
            X_selected = self.X[:, selected_indices]
            if self.task == "regression":
                self.model.fit(X_selected, self.y)
                y_pred = self.model.predict(X_selected)
                rss = np.sum((self.y - y_pred) ** 2)
                r2 = r2_score(self.y, y_pred) if self.n_samples > 1 else 0.0
                p_g = n_selected
                if self.reward_type == "r2":
                    performance = r2
                elif self.reward_type == "mse":
                    performance = -mean_squared_error(self.y, y_pred)
                elif self.reward_type == "cv_rmse":
                    if self.use_cv:
                        scores = cross_val_score(
                            self.model, X_selected, self.y, cv=self.cv_folds,
                            scoring="neg_mean_squared_error",
                        )
                        performance = -np.sqrt(max(-scores.mean(), 1e-12))
                    else:
                        performance = -np.sqrt(rss / max(self.n_samples, 1))
                elif self.reward_type == "aic":
                    # AIC = n*log(RSS/n) + 2*p_γ; reward = -AIC
                    aic = self.n_samples * np.log(max(rss / self.n_samples, 1e-12)) + 2 * p_g
                    performance = -aic
                elif self.reward_type == "bic":
                    # BIC = n*log(RSS/n) + log(n)*p_γ; reward = -BIC
                    bic = self.n_samples * np.log(max(rss / self.n_samples, 1e-12)) + np.log(max(self.n_samples, 1)) * p_g
                    performance = -bic
                elif self.reward_type == "bayes_factor":
                    # BF_γ:γ∅ = (1+g)^((n-p_γ-1)/2) * [1+g(1-R²_γ)]^(-(n-1)/2), g=max(p²,n)
                    r2_safe = np.clip(r2, 1e-10, 1 - 1e-10)
                    bf = (1 + self._g_prior) ** ((self.n_samples - p_g - 1) / 2) * (
                        1 + self._g_prior * (1 - r2_safe)
                    ) ** (-(self.n_samples - 1) / 2)
                    performance = np.log(max(bf, 1e-300))
                else:
                    if self.use_cv:
                        scoring = "r2" if self.reward_type == "r2" else "neg_mean_squared_error"
                        scores = cross_val_score(
                            self.model, X_selected, self.y, cv=self.cv_folds, scoring=scoring
                        )
                        performance = scores.mean() if self.reward_type == "r2" else -scores.mean()
                    else:
                        performance = r2 if self.reward_type == "r2" else -mean_squared_error(self.y, y_pred)
            else:
                if self.use_cv:
                    scoring = self.reward_type  # accuracy, f1_weighted, roc_auc
                    scores = cross_val_score(
                        self.model, X_selected, self.y, cv=self.cv_folds, scoring=scoring
                    )
                    performance = scores.mean()
                else:
                    self.model.fit(X_selected, self.y)
                    y_pred = self.model.predict(X_selected)
                    if self.reward_type == "accuracy":
                        performance = accuracy_score(self.y, y_pred)
                    elif self.reward_type == "f1_weighted":
                        performance = f1_score(self.y, y_pred, average="weighted", zero_division=0)
                    elif self.reward_type == "roc_auc" and hasattr(self.model, "predict_proba"):
                        from sklearn.metrics import roc_auc_score
                        try:
                            proba = self.model.predict_proba(X_selected)
                            performance = roc_auc_score(self.y, proba, multi_class="ovr", average="weighted")
                        except Exception:
                            performance = accuracy_score(self.y, y_pred)
                    else:
                        performance = accuracy_score(self.y, y_pred)
            self.cache[cache_key] = performance
        
        if self.task == "regression":
            # cv_rmse, aic, bic, bayes_factor already encode complexity; no extra sparsity term
            if self.reward_type in ("cv_rmse", "aic", "bic", "bayes_factor"):
                reward = performance
            elif self.reward_type == "r2":
                reward = performance - self.sparsity_penalty * n_selected
            else:
                reward = -performance - self.sparsity_penalty * n_selected
        else:
            reward = performance - self.sparsity_penalty * n_selected
        return reward
    
    def clear_cache(self) -> None:
        """Clear the reward cache."""
        self.cache.clear()
