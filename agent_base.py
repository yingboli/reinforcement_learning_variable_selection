"""
Base PPO agent for variable selection.

Shared logic for training, saving, loading, and prediction.
Subclasses set up PPO with environment-specific policy_kwargs and gamma,
and implement select_features().
"""

import numpy as np
from typing import Optional, Dict, Any
from stable_baselines3 import PPO


class BaseVariableSelectionPPO:
    """
    Base wrapper for PPO variable selection agents.
    
    Subclasses provide policy_kwargs and gamma in __init__, and implement
    select_features() (bandit: one step; sequential: full episode).
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        self.env = env
        self.agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
        )
    
    def train(self, total_timesteps: int, log_interval: int = 10) -> None:
        """
        Train the PPO agent.
        PPO uses stochastic sampling during training by default.
        """
        self.agent.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )
    
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ):
        """Predict action for given observation. Return type is env-specific (array or int)."""
        action, _ = self.agent.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str) -> None:
        """Save the trained agent."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load a trained agent."""
        self.agent = PPO.load(path, env=self.env)
    
    def get_agent(self) -> PPO:
        """Return the underlying PPO agent."""
        return self.agent
