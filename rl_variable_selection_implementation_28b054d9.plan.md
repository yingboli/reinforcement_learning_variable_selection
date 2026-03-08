---
name: RL Variable Selection Implementation
overview: Implement a reinforcement learning-based variable selection method using PPO with a binary mask action space, optimizing MSE/R² minus a sparsity penalty. The implementation will use PyTorch, sklearn, and gymnasium with stable-baselines3 for the PPO algorithm.
todos:
  - id: setup_env
    content: Create Gymnasium environment (env.py) with state/action/reward definitions
    status: completed
  - id: implement_ppo
    content: Implement PPO agent using stable-baselines3 or custom PyTorch implementation
    status: completed
  - id: create_evaluation
    content: Build evaluation utilities (evaluate.py) for testing and comparing methods
    status: completed
  - id: main_pipeline
    content: Create main training script (main.py) with data loading and training loop
    status: completed
  - id: requirements_docs
    content: Create requirements.txt and README.md with usage examples
    status: completed
isProject: false
---

# Implementation Plan: Reinforcement Learning Variable Selection

## Overview

This plan implements a PPO-based variable selection method where an agent learns to select optimal feature subsets by optimizing a reward function that balances model performance (MSE/R²) with sparsity.

## Architecture

The implementation consists of:

1. **Environment** (`env.py`): Gymnasium environment that defines state, action, and reward
2. **PPO Agent** (`agent.py`): PPO implementation using stable-baselines3 (PyTorch backend)
3. **Evaluation Utilities** (`evaluate.py`): Functions for testing selected features
4. **Main Script** (`main.py`): Training and evaluation pipeline
5. **Requirements** (`requirements.txt`): Dependencies

## Key Components

### 1. Environment (`env.py`)

**State Space:**

- Binary vector of length `n_features` representing currently selected features
- Optionally include additional context (e.g., current model performance, number of selected features)

**Action Space:**

- Binary mask of length `n_features` (MultiBinary action space)
- Each action selects/deselects all features simultaneously

**Reward Function:**

```
reward = -MSE - α × (number of selected features)
```

or alternatively:

```
reward = R² - α × (number of selected features)
```

where `α` is a sparsity penalty coefficient.

**Key Methods:**

- `reset()`: Initialize with empty or random feature selection
- `step(action)`: Update feature selection, fit model, compute reward
- `render()`: Display current state and performance metrics

**Implementation Notes:**

- Use sklearn's `Ridge` or `LinearRegression` for fast model fitting
- Cache model fits when possible to avoid redundant computation
- Use cross-validation for more robust reward estimation (with `cross_val_score`)

### 2. PPO Agent (`agent.py`)

**Using stable-baselines3:**

- Leverage `PPO` from `stable_baselines3` (PyTorch backend)
- Customize policy network architecture if needed
- Handle MultiBinary action space appropriately

**Key Parameters:**

- Learning rate, batch size, number of epochs per update
- Policy network architecture (MLP with appropriate input/output dimensions)
- Clipping parameter (ε) for PPO

**Alternative Custom Implementation:**

- If stable-baselines3 doesn't handle MultiBinary well, implement custom PPO in PyTorch
- Use policy gradient with clipped surrogate objective
- Include value function estimation for advantage calculation

### 3. Evaluation Utilities (`evaluate.py`)

**Functions:**

- `evaluate_selection(X, y, selected_features, model)`: Test selected features on test set
- `compare_with_baselines(X_train, y_train, X_test, y_test, selected_features)`: Compare against lasso, forward selection, etc.
- `plot_selection_history(history)`: Visualize training progress

**Metrics:**

- Test set MSE/R²
- Number of selected features
- Precision/recall for true feature recovery (if ground truth available)

### 4. Main Script (`main.py`)

**Training Pipeline:**

1. Load/preprocess data (standardize using sklearn)
2. Split into train/validation/test sets
3. Create environment with training data
4. Initialize PPO agent
5. Train agent with callbacks for logging
6. Evaluate best policy on validation set
7. Test final selection on test set
8. Compare with baseline methods

**Hyperparameters:**

- Sparsity penalty `α`
- PPO hyperparameters (learning rate, batch size, etc.)
- Environment parameters (max episode length, reward scaling)

## File Structure

```
reinforced_variable_selection/
├── env.py              # Gymnasium environment
├── agent.py            # PPO agent wrapper/custom implementation
├── evaluate.py         # Evaluation utilities
├── main.py             # Main training script
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── examples/           # Example usage scripts
    └── example_usage.py
```

## Dependencies

**Core:**

- `torch` (PyTorch) - Neural networks
- `numpy` - Numerical operations
- `pandas` - Data handling
- `scikit-learn` - Model fitting and evaluation

**RL Libraries:**

- `gymnasium` - RL environment interface
- `stable-baselines3` - PPO implementation (optional if custom implementation preferred)

**Utilities:**

- `matplotlib` - Plotting (for evaluation visualizations)
- `tqdm` - Progress bars

## Implementation Considerations

### Binary Mask Action Space Challenge

MultiBinary action spaces with many features can be challenging for PPO. Options:

1. **Use stable-baselines3 with MultiBinary**: Test if it works well
2. **Custom action space**: Implement a custom action wrapper that handles binary masks
3. **Alternative formulation**: Use sequential add/remove actions instead (simpler but slower)

### Reward Shaping

- Normalize rewards to improve training stability
- Consider using R² instead of MSE (bounded, more interpretable)
- Experiment with different sparsity penalty values

### Computational Efficiency

- Use Ridge regression for fast model fitting
- Consider using a subset of data for reward computation during training
- Cache feature subsets and their scores when possible

### Evaluation Strategy

- Use separate validation set for hyperparameter tuning
- Report performance on held-out test set
- Compare against sklearn's `LassoCV`, `RFE`, and forward selection

## Testing Strategy

1. **Synthetic Data**: Test on known ground truth (e.g., sparse linear model)
2. **Real Datasets**: Test on standard regression datasets (Boston Housing, Diabetes, etc.)
3. **Baseline Comparison**: Compare against lasso, forward selection, and random selection

## Next Steps After Prototype

- Hyperparameter tuning (grid search or Bayesian optimization)
- Extension to classification problems
- Support for different model types (not just linear regression)
- Parallelization for faster training

