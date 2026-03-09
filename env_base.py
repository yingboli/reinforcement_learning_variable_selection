"""
Base environment for reinforcement learning-based variable selection.

Shared logic for reward computation, model fitting, and caching.
Supports regression (OLS / normal linear regression) and classification (LogisticRegression).
Subclasses define action/observation spaces and step/reset semantics.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from typing import Optional, Tuple, Dict, Any


class BaseVariableSelectionEnv(gym.Env):
    """
    Base class for variable selection environments.
    
    Handles common setup (data, model, cache) and reward computation.
    Supports task='regression' (OLS, reward_type cv_rmse/aic/bic/bayes_factor) or task='classification'
    (LogisticRegression, reward_type cv_auc/aic/bic).
    Subclasses must define action_space, observation_space, reset(), step(),
    and _get_observation().
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    _REGRESSION_REWARD_TYPES = ("cv_rmse", "aic", "bic", "bayes_factor")
    _CLASSIFICATION_REWARD_TYPES = ("cv_auc", "aic", "bic")

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "regression",
        reward_type: Optional[str] = None,
        cv_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the base variable selection environment.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - continuous for regression, int/labels for classification
            task: 'regression' or 'classification'
            reward_type: For regression: 'cv_rmse', 'aic', 'bic', 'bayes_factor'.
                For classification: 'cv_auc', 'aic', 'bic'. Default: 'cv_rmse' / 'cv_auc'.
            cv_folds: Number of CV folds (for cv_rmse and cv_auc).
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
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        if reward_type is None:
            reward_type = "cv_rmse" if task == "regression" else "cv_auc"
        if task == "regression" and reward_type not in self._REGRESSION_REWARD_TYPES:
            raise ValueError(f"reward_type must be in {self._REGRESSION_REWARD_TYPES} for regression")
        if task == "classification" and reward_type not in self._CLASSIFICATION_REWARD_TYPES:
            raise ValueError(f"reward_type must be in {self._CLASSIFICATION_REWARD_TYPES} for classification")
        self.reward_type = reward_type
        
        if task == "regression":
            self.model = LinearRegression()
        else:
            self.model = LogisticRegression(
                penalty="none",
                random_state=random_state,
                max_iter=1000,
            )
        self.cache = {}
        
        # g-prior hyperparameter for Bayes factor: g = max(p², n) (Liang et al. 2008)
        self._g_prior = max(self.n_features ** 2, self.n_samples)
    
    def _compute_log_likelihood(
        self,
        *,
        rss: Optional[float] = None,
        n: Optional[int] = None,
        X_selected: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute log-likelihood for regression or classification.

        Regression: pass rss and n (Gaussian linear model, σ² = RSS/n).
        Classification: pass X_selected. If None or shape (n, 0), null model (p = mean(y), no fit).
        Otherwise self.model must be already fitted on X_selected and self.y.
        """
        if rss is not None and n is not None:
            sigma2_mle = max(rss / n, 1e-12)
            return -n / 2 * (np.log(2 * np.pi) + np.log(sigma2_mle) + 1)
        if X_selected is None or (
            isinstance(X_selected, np.ndarray)
            and (X_selected.ndim < 2 or X_selected.shape[1] == 0)
        ):
            # Binary null: p = mean(y), no logistic fit. log L = sum( y*log(p) + (1-y)*log(1-p) )
            y_bin = (self.y == np.unique(self.y)[1]).astype(np.float64)
            p = np.mean(y_bin)
            return np.sum(
                y_bin * np.log(p + 1e-12) + (1 - y_bin) * np.log(1 - p + 1e-12)
            )
        if X_selected is not None:
            # Binary logistic: log L = sum( y*log(p) + (1-y)*log(1-p) ), p = P(Y=1)
            proba = self.model.predict_proba(X_selected)
            p = proba[:, 1]
            y_bin = (self.y == self.model.classes_[1]).astype(np.float64)
            return np.sum(
                y_bin * np.log(p + 1e-12) + (1 - y_bin) * np.log(1 - p + 1e-12)
            )
        raise ValueError("Provide (rss, n) or X_selected for classification.")
    
    def _compute_aic(self, log_lik: float, p_gamma: int) -> float:
        """
        AIC = -log(L̂) + p_γ. Used for both regression and classification.
        Lower AIC is better (we minimize AIC); reward = -AIC.
        """
        return -log_lik + p_gamma
    
    def _compute_bic(self, log_lik: float, p_gamma: int) -> float:
        """
        BIC = -log(L̂) + (p_γ / 2) * log(n). Used for both regression and classification.
        Lower BIC is better; reward = -BIC.
        """
        return -log_lik + (p_gamma / 2) * np.log(self.n_samples)
    
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
        Regression: reward = objective (cv_rmse, -aic, -bic, or log bayes_factor; no extra sparsity).
        Classification: reward = cv_auc, -aic, or -bic.
        """
        n_selected = len(selected_indices)
        n = self.n_samples
        
        if n_selected == 0:
            # No features: regression = intercept only (mean of y); classification = majority class
            if self.task == "regression":
                y_mean = np.mean(self.y)
                rss = np.sum((self.y - y_mean) ** 2)
                if self.reward_type == "cv_rmse":
                    # K-fold CV RMSE for intercept-only: in each fold, predict with mean(y_train)
                    kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    mses = []
                    for train_idx, val_idx in kf.split(self.y):
                        mean_train = np.mean(self.y[train_idx])
                        mses.append(np.mean((self.y[val_idx] - mean_train) ** 2))
                    reward = -np.sqrt(max(np.mean(mses), 1e-12))
                elif self.reward_type in ("aic", "bic"):
                    log_lik = self._compute_log_likelihood(rss=rss, n=n)
                    reward = (
                        -self._compute_aic(log_lik, p_gamma=n_selected)
                        if self.reward_type == "aic"
                        else -self._compute_bic(log_lik, p_gamma=n_selected)
                    )
                elif self.reward_type == "bayes_factor":
                    reward = 0.0
                return reward
            else:
                # Classification, n_selected==0
                if self.reward_type == "cv_auc":
                    # No features: constant predictor; AUC = 0.5
                    reward = 0.5
                elif self.reward_type in ("aic", "bic"):
                    log_lik = self._compute_log_likelihood(X_selected=None)
                    reward = (
                        -self._compute_aic(log_lik, p_gamma=n_selected)
                        if self.reward_type == "aic"
                        else -self._compute_bic(log_lik, p_gamma=n_selected)
                    )
                return reward
        
        cache_key = tuple(sorted(selected_indices))
        if cache_key in self.cache:
            return self.cache[cache_key]

        X_selected = self.X[:, selected_indices]
        if self.task == "regression":
            self.model.fit(X_selected, self.y)
            y_pred = self.model.predict(X_selected)
            rss = np.sum((self.y - y_pred) ** 2)
            r2 = r2_score(self.y, y_pred) if self.n_samples > 1 else 0.0
            p_gamma = n_selected

            if self.reward_type == "cv_rmse":
                scores = cross_val_score(
                    self.model, X_selected, self.y, cv=self.cv_folds,
                    scoring="neg_mean_squared_error",
                )
                reward = -np.sqrt(max(-scores.mean(), 1e-12))
            elif self.reward_type == "aic":
                log_lik = self._compute_log_likelihood(rss=rss, n=n)
                reward = -self._compute_aic(log_lik, p_gamma)
            elif self.reward_type == "bic":
                log_lik = self._compute_log_likelihood(rss=rss, n=n)
                reward = -self._compute_bic(log_lik, p_gamma)
            elif self.reward_type == "bayes_factor":
                reward = self._compute_log_bayes_factor(r2, n, p_gamma=p_gamma)
        else:
            # Classification: cv_auc, aic, or bic
            if self.reward_type == "cv_auc":
                scoring = "roc_auc_ovr" if len(np.unique(self.y)) > 2 else "roc_auc"
                scores = cross_val_score(
                    self.model, X_selected, self.y, cv=self.cv_folds, scoring=scoring
                )
                reward = scores.mean()
            elif self.reward_type == "aic":
                self.model.fit(X_selected, self.y)
                log_lik = self._compute_log_likelihood(X_selected=X_selected)
                reward = -self._compute_aic(log_lik, n_selected)
            elif self.reward_type == "bic":
                self.model.fit(X_selected, self.y)
                log_lik = self._compute_log_likelihood(X_selected=X_selected)
                reward = -self._compute_bic(log_lik, n_selected)

        self.cache[cache_key] = reward
        return reward

    def clear_cache(self) -> None:
        """Clear the reward cache."""
        self.cache.clear()
