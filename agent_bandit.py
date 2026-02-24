"""
PPO agent for bandit variable selection (one-step MDP).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from agent_base import BaseVariableSelectionPPO


class MinimalFeatureExtractor(BaseFeaturesExtractor):
    """
    Minimal feature extractor for bandit problems.
    Observation is always a constant (zeros), so we use a single learnable
    vector as the feature representation. No linear layer: weights would
    never get nonzero gradients when input is always zero.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 32,
    ):
        super().__init__(observation_space, features_dim)
        # Learned constant feature vector (observation is never used)
        self.features = nn.Parameter(torch.zeros(features_dim))
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Broadcast learned constant to batch size; ignore observations
        batch_size = observations.shape[0]
        return self.features.unsqueeze(0).expand(batch_size, -1)


def _default_policy_kwargs_for_n_features(n_features: int) -> Dict[str, Any]:
    """
    Suggest policy_kwargs for bandit. Only pi is scaled with n_features.
    Bandit has no meaningful state: feature extractor and vf are not meaningful
    (constant observation, constant baseline). They stay minimal; only the
    policy (pi) that outputs n_features logits is scaled for capacity.
    """
    # Feature extractor and vf fixed minimal (not meaningful in bandit)
    fe_kwargs = {"features_extractor_class": MinimalFeatureExtractor, "features_extractor_kwargs": {"features_dim": 32}}
    if n_features <= 50:
        return {**fe_kwargs, "net_arch": [dict(pi=[64, 32], vf=[16])]}
    elif n_features <= 200:
        return {**fe_kwargs, "net_arch": [dict(pi=[128, 64], vf=[16])]}
    else:
        return {**fe_kwargs, "net_arch": [dict(pi=[256, 128], vf=[16])]}


class VariableSelectionPPO(BaseVariableSelectionPPO):
    """
    PPO agent for bandit variable selection (one-step, binary mask action).
    Feature extractor and vf are minimal (no meaningful state). For large
    n_features, _default_policy_kwargs_for_n_features(n_features) scales only pi.
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,  # No effect: bandit has one step per episode (no future reward)
        gae_lambda: float = 0.95,  # No effect: advantage is just reward - V(s)
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
            policy_kwargs = _default_policy_kwargs_for_n_features(
                getattr(env, "n_features", 50)
            )
        
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
        """Select features (one step; bandit terminates immediately)."""
        obs, _ = self.env.reset()
        action = self.predict(obs, deterministic=deterministic)
        return self.env.get_selected_features_from_action(action)
