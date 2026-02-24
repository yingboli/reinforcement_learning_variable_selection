"""
Sequential MDP environment for variable selection.

Agent adds/removes features one at a time; state is current selection (meaningful).
"""

import numpy as np
from gymnasium import spaces
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional, Tuple, Dict, Any

from env_base import BaseVariableSelectionEnv


class SequentialVariableSelectionEnv(BaseVariableSelectionEnv):
    """
    Sequential MDP for variable selection.
    
    The agent toggles or adds/removes one feature per step. State is the
    current binary feature selection. With gamma=0, only immediate rewards matter.
    If max_episode_steps is None, it defaults to min(max(50, 2*n_features), 500)
    so that large n_features get enough steps.
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
        max_episode_steps: Optional[int] = None,
        model_alpha: float = 1.0,
        action_type: str = "toggle",
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
        
        if action_type not in ["toggle", "add_remove"]:
            raise ValueError("action_type must be 'toggle' or 'add_remove'")
        
        # Scale with n_features so large feature sets get enough steps (cap at 500)
        if max_episode_steps is None:
            max_episode_steps = min(max(50, 2 * self.n_features), 500)
        self.max_episode_steps = max_episode_steps
        self.action_type = action_type
        self.state = np.zeros(self.n_features, dtype=np.int8)
        self.current_step = 0
        
        if action_type == "toggle":
            self.action_space = spaces.Discrete(self.n_features)
        else:
            self.action_space = spaces.Discrete(2 * self.n_features)
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_features,),
            dtype=np.float32,
        )
    
    def _get_observation(self) -> np.ndarray:
        """Current feature selection as observation."""
        return self.state.astype(np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.state = np.zeros(self.n_features, dtype=np.int8)
        self.current_step = 0
        
        if options is not None and options.get("random_start", False):
            n_random = self.np_random.integers(1, min(self.n_features // 2, 10))
            random_indices = self.np_random.choice(
                self.n_features, size=n_random, replace=False
            )
            self.state[random_indices] = 1
        
        obs = self._get_observation()
        info = {
            "n_selected": int(np.sum(self.state)),
            "selected_features": np.where(self.state == 1)[0].tolist(),
        }
        return obs, info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.action_type == "toggle":
            self.state[action] = 1 - self.state[action]
        else:
            if action < self.n_features:
                self.state[action] = 1
            else:
                self.state[action - self.n_features] = 0
        
        self.current_step += 1
        selected_indices = np.where(self.state == 1)[0]
        reward = self._compute_reward(selected_indices)
        terminated = self.current_step >= self.max_episode_steps
        obs = self._get_observation()
        info = {
            "n_selected": len(selected_indices),
            "selected_features": selected_indices.tolist(),
            "reward": reward,
            "step": self.current_step,
            "action": action,
        }
        return obs, reward, terminated, False, info
    
    def render(self) -> None:
        selected_indices = np.where(self.state == 1)[0]
        n_selected = len(selected_indices)
        print(f"Step: {self.current_step}/{self.max_episode_steps}")
        print(f"Selected features: {n_selected}/{self.n_features}")
        if n_selected > 0:
            print(f"Feature indices: {selected_indices.tolist()}")
            X_selected = self.X[:, selected_indices]
            self.model.fit(X_selected, self.y)
            y_pred = self.model.predict(X_selected)
            print(f"R²: {r2_score(self.y, y_pred):.4f}, MSE: {mean_squared_error(self.y, y_pred):.4f}")
    
    def get_selected_features(self) -> np.ndarray:
        """Return currently selected feature indices."""
        return np.where(self.state == 1)[0]
