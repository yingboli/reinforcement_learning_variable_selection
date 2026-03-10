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
from typing import Optional, Tuple, Dict, Any

from reward_utils import (
    g_prior,
    cv_rmse,
    cv_auc,
    reward_regression_aic,
    reward_regression_bic,
    reward_regression_bayes_factor,
    reward_classification_aic,
    reward_classification_bic,
)


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
        
        # g-prior for Bayes factor (from reward_utils)
        self._g_prior = g_prior(self.n_samples, self.n_features)

    def _compute_reward(self, selected_indices: np.ndarray) -> float:
        """
        Compute reward for a given feature selection.
        Regression: reward = objective (cv_rmse, -aic, -bic, or log bayes_factor; no extra sparsity).
        Classification: reward = cv_auc, -aic, or -bic.
        """
        cache_key = tuple(sorted(selected_indices))
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.task == "regression":
            if self.reward_type == "cv_rmse":
                reward = cv_rmse(
                    self.X, self.y, selected_indices,
                    cv_folds=self.cv_folds, random_state=self.random_state,
                )
            elif self.reward_type == "aic":
                reward = reward_regression_aic(self.X, self.y, selected_indices)
            elif self.reward_type == "bic":
                reward = reward_regression_bic(self.X, self.y, selected_indices)
            else:
                reward = reward_regression_bayes_factor(
                    self.X, self.y, selected_indices, g=self._g_prior
                )
        else:
            if self.reward_type == "cv_auc":
                reward = cv_auc(
                    self.X, self.y, selected_indices,
                    cv_folds=self.cv_folds, random_state=self.random_state,
                )
            elif self.reward_type == "aic":
                reward = reward_classification_aic(
                    self.X, self.y, selected_indices, random_state=self.random_state
                )
            else:
                reward = reward_classification_bic(
                    self.X, self.y, selected_indices, random_state=self.random_state
                )

        self.cache[cache_key] = reward
        return reward

    def clear_cache(self) -> None:
        """Clear the reward cache."""
        self.cache.clear()
