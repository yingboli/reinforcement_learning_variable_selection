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
        
        # g-prior hyperparameter for Bayes factor: g = max(p², n) (Liang et al. 2008)
        self._g_prior = max(self.n_features ** 2, self.n_samples)
    
    def _compute_log_likelihood(self, rss: float, n: int) -> float:
        """
        Compute log-likelihood for linear regression with Gaussian errors.
        
        Under the assumption y ~ N(Xβ, σ²I), the log-likelihood is:
            LL = -n/2 * log(2π) - n/2 * log(σ²) - RSS/(2σ²)
        
        Using MLE estimate σ² = RSS/n:
            LL = -n/2 * [log(2π) + log(RSS/n) + 1]
        """
        sigma2_mle = max(rss / n, 1e-12)
        log_lik = -n / 2 * (np.log(2 * np.pi) + np.log(sigma2_mle) + 1)
        return log_lik
    
    def _compute_aic(self, rss: float, n: int, p_gamma: int) -> float:
        """
        Compute AIC (Akaike Information Criterion).
        
        From the write-up:
            AIC(γ) = -log(L̂_γ) + p_γ
        
        where p_γ is the number of features included.
        Lower AIC is better (we minimize AIC).
        """
        log_lik = self._compute_log_likelihood(rss, n)
        return -log_lik + p_gamma
    
    def _compute_bic(self, rss: float, n: int, p_gamma: int) -> float:
        """
        Compute BIC (Bayesian Information Criterion).
        
        From the write-up:
            BIC(γ) = -log(L̂_γ) + (p_γ / 2) * log(n)
        
        where p_γ is the number of features included.
        Lower BIC is better (we minimize BIC).
        """
        log_lik = self._compute_log_likelihood(rss, n)
        return -log_lik + (p_gamma / 2) * np.log(n)
    
    def _compute_log_bayes_factor(self, r2: float, n: int, p_gamma: int) -> float:
        """
        Compute log Bayes factor under g-prior (Liang et al. 2008).
        
        From the write-up, comparing model γ to null model γ∅:
            BF_{γ:γ∅} = (1 + g)^{(n - p_γ - 1)/2} * [1 + g(1 - R²_γ)]^{-(n-1)/2}
        
        where:
            - g = max(p², n) is the g-prior hyperparameter
            - p_γ is the number of features in model γ
            - R²_γ is the coefficient of determination
            - n is the sample size
        
        We return log(BF) for numerical stability. Higher is better (we maximize BF).
        """
        g = self._g_prior
        
        # Clip R² to avoid numerical issues at boundaries
        r2_safe = np.clip(r2, 1e-10, 1 - 1e-10)
        
        # Compute log(BF) for numerical stability
        # log(BF) = ((n - p_γ - 1) / 2) * log(1 + g) - ((n - 1) / 2) * log(1 + g(1 - R²))
        term1 = ((n - p_gamma - 1) / 2) * np.log(1 + g)
        term2 = ((n - 1) / 2) * np.log(1 + g * (1 - r2_safe))
        
        log_bf = term1 - term2
        
        return log_bf
    
    def _compute_reward(self, selected_indices: np.ndarray) -> float:
        """
        Compute reward for a given feature selection.
        Regression: reward = r2 - penalty (or -mse - penalty).
        Classification: reward = accuracy/f1/roc_auc - penalty.
        """
        n_selected = len(selected_indices)
        n = self.n_samples
        
        if n_selected == 0:
            # No features: regression = fit intercept only; classification = majority class
            X_dummy = np.zeros((self.n_samples, 1))
            if self.task == "regression":
                self.model.fit(X_dummy, self.y)
                y_pred = self.model.predict(X_dummy)
                rss = np.sum((self.y - y_pred) ** 2)
                r2_0 = r2_score(self.y, y_pred) if self.n_samples > 1 else 0.0
                if self.reward_type == "r2":
                    if self.use_cv:
                        scores = cross_val_score(
                            self.model, X_dummy, self.y, cv=self.cv_folds, scoring="r2"
                        )
                        performance = scores.mean()
                    else:
                        performance = r2_0
                elif self.reward_type == "mse":
                    if self.use_cv:
                        scores = cross_val_score(
                            self.model, X_dummy, self.y, cv=self.cv_folds,
                            scoring="neg_mean_squared_error",
                        )
                        performance = scores.mean()  # Already negative
                    else:
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
                    # Null model: p_γ = 0 (no features selected)
                    aic = self._compute_aic(rss, n, p_gamma=0)
                    performance = -aic
                elif self.reward_type == "bic":
                    # Null model: p_γ = 0 (no features selected)
                    bic = self._compute_bic(rss, n, p_gamma=0)
                    performance = -bic
                elif self.reward_type == "bayes_factor":
                    # BF of null vs null = 1, log(BF) = 0
                    performance = 0.0
                else:
                    performance = r2_0
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
                    if self.use_cv:
                        scores = cross_val_score(
                            self.model, X_selected, self.y, cv=self.cv_folds, scoring="r2"
                        )
                        performance = scores.mean()
                    else:
                        performance = r2
                elif self.reward_type == "mse":
                    if self.use_cv:
                        scores = cross_val_score(
                            self.model, X_selected, self.y, cv=self.cv_folds,
                            scoring="neg_mean_squared_error",
                        )
                        performance = scores.mean()  # Already negative
                    else:
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
                    # p_γ = number of selected features
                    aic = self._compute_aic(rss, n, p_gamma=p_g)
                    performance = -aic
                elif self.reward_type == "bic":
                    # p_γ = number of selected features
                    bic = self._compute_bic(rss, n, p_gamma=p_g)
                    performance = -bic
                elif self.reward_type == "bayes_factor":
                    # Bayes factor under g-prior (Liang et al. 2008)
                    # Uses R² and the closed-form marginal likelihood
                    performance = self._compute_log_bayes_factor(r2, n, p_gamma=p_g)
                else:
                    performance = r2
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
            elif self.reward_type in ("r2", "mse"):
                # For r2: higher is better, performance is R² (positive)
                # For mse: lower is better, performance is -MSE (negative)
                # Both cases: reward = performance - penalty
                reward = performance - self.sparsity_penalty * n_selected
            else:
                reward = performance - self.sparsity_penalty * n_selected
        else:
            reward = performance - self.sparsity_penalty * n_selected
        return reward
    
    def clear_cache(self) -> None:
        """Clear the reward cache."""
        self.cache.clear()
