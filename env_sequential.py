"""
Sequential MDP environment for variable selection.

Agent adds/removes features one at a time; state is current selection (meaningful).
Supports action masking (info["action_mask"]) and an optional stop action.
"""

import numpy as np
from gymnasium import spaces
from gymnasium import Env
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional, Tuple, Dict, Any

from env_base import BaseVariableSelectionEnv


class ActionMaskWrapper(Env):
    """
    Wrapper that maps invalid actions to a valid one before calling step.
    Uses the mask from the previous step's info (so you must have reset or stepped at least once).
    Use with add_remove so the agent never executes no-ops; works with standard PPO.
    """
    
    def __init__(self, env: Env, fallback: str = "first"):
        """
        Args:
            env: SequentialVariableSelectionEnv (or any env that puts action_mask in info).
            fallback: "first" = use first valid action when invalid; "random" = sample uniformly from valid.
        """
        self.env = env
        self.fallback = fallback
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._last_mask: Optional[np.ndarray] = None
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_mask = info.get("action_mask")
        return obs, info
    
    def step(self, action):
        mask = self._last_mask
        if mask is not None and not mask[action]:
            valid = np.where(mask)[0]
            if len(valid) > 0:
                action = int(np.random.choice(valid) if self.fallback == "random" else valid[0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_mask = info.get("action_mask")
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class SequentialVariableSelectionEnv(BaseVariableSelectionEnv):
    """
    Sequential MDP for variable selection.
    
    The agent toggles or adds/removes one feature per step. State is the
    current binary feature selection. With gamma=0, only immediate rewards matter.
    If max_episode_steps is None, it defaults to min(max(50, 2*n_features), 500)
    so that large n_features get enough steps.
    
    - action_type "add_remove": actions 0..n_features-1 = add feature j;
      n_features..2*n_features-1 = remove feature j. Invalid actions (add when
      already in, remove when not in) are no-ops and yield a small penalty;
      info["action_mask"] gives valid actions (True = valid) for use with
      maskable policies.
    - action_type "toggle": actions 0..n_features-1 = toggle feature j. If
      include_stop_action=True, action n_features = stop (terminate with current
      selection).
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
        include_stop_action: bool = True,
        invalid_action_penalty: float = 0.01,
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
        self.include_stop_action = include_stop_action
        self.invalid_action_penalty = invalid_action_penalty
        self.state = np.zeros(self.n_features, dtype=np.int8)
        self.current_step = 0
        
        n_actions = self.n_features if action_type == "toggle" else 2 * self.n_features
        if include_stop_action:
            n_actions += 1
        self.action_space = spaces.Discrete(n_actions)
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_features,),
            dtype=np.float32,
        )
    
    def _get_observation(self) -> np.ndarray:
        """Current feature selection as observation."""
        return self.state.astype(np.float32)
    
    def _get_action_mask(self) -> np.ndarray:
        """Boolean array of shape (action_space.n,): True = valid action."""
        n = self.action_space.n
        mask = np.ones(n, dtype=bool)
        if self.action_type == "add_remove":
            # actions 0..n_features-1: add j valid iff state[j]==0
            # actions n_features..2*n_features-1: remove j valid iff state[j]==1
            for j in range(self.n_features):
                mask[j] = self.state[j] == 0
                mask[self.n_features + j] = self.state[j] == 1
        # stop action (if present) is always valid
        return mask
    
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
        
        # Initialize previous reward for delta reward computation
        # This avoids double-counting when using gamma > 0
        selected_indices = np.where(self.state == 1)[0]
        self._prev_reward = self._compute_reward(selected_indices)
        
        obs = self._get_observation()
        info = {
            "n_selected": int(np.sum(self.state)),
            "selected_features": np.where(self.state == 1)[0].tolist(),
            "action_mask": self._get_action_mask(),
        }
        return obs, info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        stop_action_idx = (
            (self.n_features if self.action_type == "toggle" else 2 * self.n_features)
            if self.include_stop_action else None
        )
        
        if self.include_stop_action and action == stop_action_idx:
            # Stop: terminate with current selection
            # Use delta reward (should be 0 since state didn't change)
            selected_indices = np.where(self.state == 1)[0]
            current_reward = self._compute_reward(selected_indices)
            delta_reward = current_reward - self._prev_reward  # Should be ~0
            obs = self._get_observation()
            info = {
                "n_selected": len(selected_indices),
                "selected_features": selected_indices.tolist(),
                "reward": delta_reward,
                "current_objective": current_reward,
                "step": self.current_step,
                "action": action,
                "action_mask": self._get_action_mask(),
                "stopped": True,
            }
            return obs, delta_reward, True, False, info
        
        invalid = False
        if self.action_type == "toggle":
            self.state[action] = 1 - self.state[action]
        else:
            if action < self.n_features:
                if self.state[action] == 0:
                    self.state[action] = 1
                else:
                    invalid = True  # add when already in
            else:
                j = action - self.n_features
                if self.state[j] == 1:
                    self.state[j] = 0
                else:
                    invalid = True  # remove when not in
        
        self.current_step += 1
        selected_indices = np.where(self.state == 1)[0]
        current_reward = self._compute_reward(selected_indices)
        
        # Use DELTA reward to avoid double-counting with gamma > 0
        # delta_reward = improvement from this action
        # Sum of all delta rewards = final reward (no double counting)
        delta_reward = current_reward - self._prev_reward
        self._prev_reward = current_reward
        
        if invalid:
            delta_reward = delta_reward - self.invalid_action_penalty
        
        terminated = self.current_step >= self.max_episode_steps
        obs = self._get_observation()
        info = {
            "n_selected": len(selected_indices),
            "selected_features": selected_indices.tolist(),
            "reward": delta_reward,
            "current_objective": current_reward,  # For debugging
            "step": self.current_step,
            "action": action,
            "action_mask": self._get_action_mask(),
            "stopped": False,
        }
        return obs, delta_reward, terminated, False, info
    
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
