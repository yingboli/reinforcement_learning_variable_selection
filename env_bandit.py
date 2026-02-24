"""
Bandit environment for variable selection (one-step MDP).

Agent selects all features via a binary mask; episode terminates immediately.
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from env_base import BaseVariableSelectionEnv


class VariableSelectionEnv(BaseVariableSelectionEnv):
    """
    Variable selection as a bandit problem (one-step MDP).
    
    The agent selects features via a binary mask action space and receives
    a reward. The episode terminates immediately after one step.
    Observation is a dummy constant (no meaningful state).
    """
    
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
        super().__init__(
            X=X, y=y,
            task=task,
            sparsity_penalty=sparsity_penalty,
            reward_type=reward_type,
            use_cv=use_cv,
            cv_folds=cv_folds,
            model_alpha=model_alpha,
            random_state=random_state,
        )
        
        self.action_space = spaces.MultiBinary(self.n_features)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_features,),
            dtype=np.float32,
        )
    
    def _get_observation(self) -> np.ndarray:
        """Dummy observation (bandit has no state)."""
        return np.zeros(self.n_features, dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        return self._get_observation(), {}
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        selected_indices = np.where(action.astype(np.int8) == 1)[0]
        reward = self._compute_reward(selected_indices)
        obs = self._get_observation()
        info = {
            "n_selected": len(selected_indices),
            "selected_features": selected_indices.tolist(),
            "reward": reward,
        }
        return obs, reward, True, False, info
    
    def render(self) -> None:
        pass
    
    def get_selected_features_from_action(self, action: np.ndarray) -> np.ndarray:
        """Return selected feature indices from a binary mask action."""
        return np.where(action.astype(np.int8) == 1)[0]
