# Reinforcement Learning for Variable Selection

This repository implements reinforcement learning-based variable selection methods using Proximal Policy Optimization (PPO). Two approaches are provided:

1. **Bandit Approach** (`env.py`, `agent.py`): One-step MDP where the agent selects all features simultaneously via a binary mask
2. **Sequential MDP Approach** (`env_sequential.py`, `agent_sequential.py`): Multi-step MDP where the agent sequentially adds/removes features one at a time, with gamma=0 (only immediate rewards matter)

Both methods optimize a reward function that balances model performance (MSE/R²) with sparsity.

## Overview

The implementation uses:
- **PyTorch** for neural network components (via stable-baselines3)
- **Gymnasium** for the RL environment interface
- **stable-baselines3** for the PPO algorithm
- **scikit-learn** for model fitting and evaluation

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage - Bandit Approach

Train the bandit RL agent on a synthetic dataset:

```bash
python main.py --dataset synthetic --n_features 50 --total_timesteps 10000
```

### Comparison - Bandit vs Sequential MDP

Compare both approaches:

```bash
python main_comparison.py --dataset synthetic --n_features 50 --total_timesteps 10000
```

This will train both approaches and compare their performance.

### Classification

Run variable selection for classification (reward: accuracy by default):

```bash
python main.py --task classification --dataset breast_cancer --total_timesteps 10000
python main.py --task classification --dataset synthetic --n_features 50
```

Classification reward types: `accuracy`, `f1_weighted`, `roc_auc`. Datasets: `synthetic`, `breast_cancer`.

### Command-Line Arguments

- `--task`: `regression` or `classification` (default: regression)
- `--dataset`: Regression: `synthetic`, `diabetes`, `california`. Classification: `synthetic`, `breast_cancer`
- `--n_samples`: Number of samples (for synthetic data, default: 200)
- `--n_features`: Number of features (for synthetic data, default: 50)
- `--sparsity_penalty`: Sparsity penalty coefficient α (default: 0.01)
- `--reward_type`: Regression: `r2`, `mse`, `cv_rmse` (cross-validated RMSE), `aic`, `bic`, `bayes_factor` (g-prior BF vs null). Classification: `accuracy`, `f1_weighted`, `roc_auc`. Default: `r2` / `accuracy`
- `--total_timesteps`: Total training timesteps (default: 10000)
- `--max_episode_steps`: Maximum steps per episode (default: 50)
- `--learning_rate`: Learning rate for PPO (default: 3e-4)
- `--output_dir`: Output directory for results (default: `./results`)
- `--save_model`: Path to save trained model (optional)

### Example: Custom Training

```bash
python main.py \
    --dataset synthetic \
    --n_features 100 \
    --sparsity_penalty 0.02 \
    --reward_type r2 \
    --total_timesteps 50000 \
    --learning_rate 1e-4 \
    --save_model ./models/rl_agent.zip
```

## Code Structure

```
reinforced_variable_selection/
├── env_base.py             # Base env: reward computation, model, cache
├── env_bandit.py           # Bandit environment (subclass of base)
├── env_sequential.py       # Sequential MDP environment (subclass of base)
├── agent_base.py           # Base PPO wrapper: train, save, load, predict
├── agent_bandit.py         # Bandit PPO agent (subclass of base)
├── agent_sequential.py     # Sequential MDP PPO agent (subclass of base)
├── evaluate.py             # Evaluation utilities and baseline comparisons
├── main.py                 # Main training script (bandit approach)
├── main_comparison.py      # Comparison script (both approaches)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Usage as a Python Package

### Environment

Create and use the environment:

```python
from env_bandit import VariableSelectionEnv
import numpy as np
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=200, n_features=50, n_informative=5, random_state=42)

# Create environment
env = VariableSelectionEnv(
    X, y,
    sparsity_penalty=0.01,
    reward_type="r2",
    use_cv=True,
    cv_folds=3
)

# Reset environment
obs, info = env.reset()

# Take a random action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Agent

Train and use the RL agent:

```python
from agent_bandit import VariableSelectionPPO
from env_bandit import VariableSelectionEnv

# Create environment
env = VariableSelectionEnv(X_train, y_train, sparsity_penalty=0.01)

# Create agent
agent = VariableSelectionPPO(env, learning_rate=3e-4)

# Train
agent.train(total_timesteps=10000)

# Select features
selected_features = agent.select_features(deterministic=True)
print(f"Selected features: {selected_features}")
```

### Evaluation

Evaluate selected features and compare with baselines:

```python
from evaluate import evaluate_selection, compare_with_baselines

# Evaluate selection
results = evaluate_selection(
    X_train, y_train, X_test, y_test, selected_features
)
print(f"Test R²: {results['test_r2']:.4f}")
print(f"Test MSE: {results['test_mse']:.4f}")

# Compare with baselines
comparison_df = compare_with_baselines(
    X_train, y_train, X_test, y_test, selected_features
)
print(comparison_df)
```

## How It Works

### Bandit Approach (One-step MDP)

**Environment** (`env.py`):
- **State**: Constant dummy observation (all zeros) - not meaningful for bandit
- **Action**: Binary mask of length `n_features` (MultiBinary action space)
- **Reward**: Model performance (R² or negative MSE) minus sparsity penalty
- **Episode**: Terminates immediately after one step

**Agent** (`agent.py`):
- Minimal feature extractor (single linear layer)
- Minimal value network (learns constant baseline)
- Policy network learns to select features via binary mask
- Uses stochastic sampling during training

**Training Process**:
1. Agent selects a binary mask (all features at once)
2. Environment computes reward
3. Episode terminates immediately
4. Agent updates policy using PPO

### Sequential MDP Approach (Multi-step, gamma=0)

**Environment** (`env_sequential.py`):
- **State**: Current binary feature selection mask (meaningful!)
- **Action**: Feature index to toggle/add/remove (Discrete action space)
- **Reward**: Model performance minus sparsity penalty (computed at each step)
- **Episode**: Runs for `max_episode_steps` steps
- **Gamma**: 0.0 (only immediate rewards matter, no future discounting)

**Agent** (`agent_sequential.py`):
- Feature extractor processes meaningful state (current selection)
- Value network estimates value of current partial selection
- Policy network learns which feature to toggle based on current state
- Uses gamma=0 (no future discounting)

**Training Process**:
1. Agent starts with empty selection
2. At each step, agent selects which feature to toggle/add/remove
3. Environment updates state and computes reward
4. Agent sees meaningful state (current selection)
5. Episode continues until max steps
6. Agent updates policy using PPO with gamma=0

## Hyperparameters

Key hyperparameters to tune:

- **sparsity_penalty** (α): Controls trade-off between model performance and sparsity
  - Higher values encourage fewer features
  - Typical range: 0.001 - 0.1
  
- **reward_type**: `r2` (bounded, more stable) or `mse` (unbounded)
  
- **learning_rate**: PPO learning rate (typically 1e-4 to 1e-3)
  
- **max_episode_steps**: Maximum steps per episode (controls episode length)

## Evaluation

The implementation includes comparison with baseline methods:
- **LassoCV**: Lasso with cross-validation
- **Forward Selection**: Sequential forward feature selection
- **RFE**: Recursive feature elimination
- **All Features**: Baseline using all features

Results are saved to CSV files in the output directory.

## Tips for Best Results

1. **Data Preprocessing**: Standardize features and optionally standardize targets
2. **Sparsity Penalty**: Start with small values (0.001-0.01) and adjust based on desired sparsity
3. **Training Time**: More timesteps generally lead to better results (try 20k-50k+)
4. **Cross-Validation**: The environment uses CV for reward estimation by default (more robust but slower)
5. **Episode Length**: Adjust `max_episode_steps` based on expected number of features

## Comparison: Bandit vs Sequential MDP

**Bandit Approach**:
- ✅ Faster (one step per episode)
- ✅ Simpler action space (binary mask)
- ❌ No meaningful state/observation
- ❌ No intermediate feedback
- ❌ Less exploration flexibility

**Sequential MDP Approach**:
- ✅ Meaningful state space (current selection)
- ✅ Meaningful value function (value of partial selection)
- ✅ Better exploration (incremental decisions)
- ✅ More natural for variable selection (like stepwise methods)
- ❌ Slower (multiple steps per episode)
- ❌ More complex (need to define episode length)

**Recommendation**: Try both and compare! Sequential MDP often performs better for variable selection because it can see intermediate results and make incremental decisions.

## Limitations and Future Work

- Supports **regression** (Ridge) and **classification** (LogisticRegression)
- Regression reward: `r2` or `mse`. Classification reward: `accuracy`, `f1_weighted`, or `roc_auc`
- Bandit: Binary mask action space can be challenging for high-dimensional problems (>100 features)
- Sequential: Episode length needs tuning for different problem sizes

## References

- Le, Y., Bai, Y., and Zhou, F. Reinforced variable selection. Statistical Theory and Related Fields, pp. 1–18, 2025.

## License

This code is provided for research and educational purposes.
