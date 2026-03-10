"""
Sequential MDP environment for variable selection.

Agent adds/removes features one at a time; state is current selection (meaningful).
Supports action masking (info["action_mask"]). Episodes run for max_episode_steps (no stop action).
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
    current binary feature selection. Episodes run for max_episode_steps (no stop action).
    If max_episode_steps is None, it defaults to min(max(50, 2*n_features), 500)
    so that large n_features get enough steps.
    
    - action_type "add_remove": actions 0..n_features-1 = add feature j;
      n_features..2*n_features-1 = remove feature j. Invalid actions (add when
      already in, remove when not in) are no-ops and yield a small penalty;
      info["action_mask"] gives valid actions (True = valid) for use with
      maskable policies.
    - action_type "toggle": actions 0..n_features-1 = toggle feature j.
    
    Optional: random_start_probability (0–1) starts some episodes from a random
    non-empty selection to avoid under-selection. improvement_bonus_coef adds
    a bonus proportional to (reward_after - reward_before) to emphasize
    improvement over the previous step.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "regression",
        reward_type: Optional[str] = None,
        cv_folds: int = 5,
        max_episode_steps: Optional[int] = None,
        action_type: str = "toggle",
        invalid_action_penalty: float = 0.01,
        random_start_probability: float = 0.0,
        improvement_bonus_coef: float = 0.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            X=X, y=y,
            task=task,
            reward_type=reward_type,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        
        if action_type not in ["toggle", "add_remove"]:
            raise ValueError("action_type must be 'toggle' or 'add_remove'")
        
        # Scale with n_features so large feature sets get enough steps (cap at 500)
        if max_episode_steps is None:
            max_episode_steps = min(max(50, 2 * self.n_features), 500)
        self.max_episode_steps = max_episode_steps
        self.action_type = action_type
        self.invalid_action_penalty = invalid_action_penalty
        self.random_start_probability = float(random_start_probability)
        self.improvement_bonus_coef = float(improvement_bonus_coef)
        self.state = np.zeros(self.n_features, dtype=np.int8)
        self.current_step = 0
        self._last_reward: Optional[float] = None
        
        n_actions = self.n_features if action_type == "toggle" else 2 * self.n_features
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
        return mask
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.state = np.zeros(self.n_features, dtype=np.int8)
        self.current_step = 0
        
        # Explicit options["random_start"] overrides; else use random_start_probability for training
        opt = options or {}
        if "random_start" in opt:
            use_random = bool(opt["random_start"])
        else:
            use_random = self.random_start_probability > 0 and self.np_random.random() < self.random_start_probability
        if use_random:
            n_random = self.np_random.integers(1, min(max(1, self.n_features // 2), 10))
            random_indices = self.np_random.choice(
                self.n_features, size=n_random, replace=False
            )
            self.state[random_indices] = 1
        
        selected_indices = np.where(self.state == 1)[0]
        self._last_reward = self._compute_reward(selected_indices)
        
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
        reward_after = self._compute_reward(selected_indices)
        reward = reward_after
        if self.improvement_bonus_coef != 0 and self._last_reward is not None:
            reward += self.improvement_bonus_coef * (reward_after - self._last_reward)
        self._last_reward = reward_after
        if invalid:
            reward = reward - self.invalid_action_penalty
        terminated = self.current_step >= self.max_episode_steps
        obs = self._get_observation()
        info = {
            "n_selected": len(selected_indices),
            "selected_features": selected_indices.tolist(),
            "reward": reward,
            "step": self.current_step,
            "action": action,
            "action_mask": self._get_action_mask(),
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
