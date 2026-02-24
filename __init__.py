"""
Reinforcement Learning for Variable Selection

A PyTorch-based implementation of RL variable selection using PPO.
"""

from .env_bandit import VariableSelectionEnv
from .agent_bandit import VariableSelectionPPO
from .evaluate import (
    evaluate_selection,
    compare_with_baselines,
    compute_precision_recall,
    plot_selection_history,
)

__version__ = "0.1.0"
__all__ = [
    "VariableSelectionEnv",
    "VariableSelectionPPO",
    "evaluate_selection",
    "compare_with_baselines",
    "compute_precision_recall",
    "plot_selection_history",
]
