"""
PPO agent for sequential variable selection (multi-step MDP, gamma=1).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from agent_base import BaseVariableSelectionPPO


class SequentialFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for sequential MDP; processes current feature selection state.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
        net_arch: Optional[list] = None,
    ):
        super().__init__(observation_space, features_dim)
        if net_arch is None:
            net_arch = [128, 64]
        input_dim = observation_space.shape[0]
        layers = []
        prev_dim = input_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, features_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def _default_policy_kwargs_for_n_features(n_features: int) -> Dict[str, Any]:
    """
    Suggest policy_kwargs (feature extractor + pi/vf sizes) based on n_features.
    Use for large n_features (e.g. 500) so the policy has enough capacity.
    """
    if n_features <= 50:
        return {
            "features_extractor_class": SequentialFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 64, "net_arch": [128, 64]},
            "net_arch": dict(pi=[64, 32], vf=[64, 32]),
        }
    elif n_features <= 200:
        return {
            "features_extractor_class": SequentialFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128, "net_arch": [256, 128]},
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        }
    else:
        # e.g. n_features = 500 or more
        return {
            "features_extractor_class": SequentialFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256, "net_arch": [512, 256]},
            "net_arch": dict(pi=[256, 128], vf=[128, 64]),
        }


class SequentialVariableSelectionPPO(BaseVariableSelectionPPO):
    """
    PPO agent for sequential variable selection (multi-step, gamma=1).
    For large n_features (e.g. 500), pass policy_kwargs from
    _default_policy_kwargs_for_n_features(n_features) so pi/vf scale appropriately.
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 1.0,
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
        if policy_kwargs is None:
            # Default for small/medium n_features; for large n use _default_policy_kwargs_for_n_features(env.n_features)
            policy_kwargs = _default_policy_kwargs_for_n_features(getattr(env, "n_features", 50))
        
        super().__init__(
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
    
    def select_features(self, deterministic: bool = True) -> np.ndarray:
        """Select features by running a full episode from empty selection."""
        obs, _ = self.env.reset(options={"random_start": False})
        while True:
            action = self.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return self.env.get_selected_features()
