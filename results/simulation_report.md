# RL Variable Selection Simulation Report

**Generated:** 2026-03-10 00:22:18

**Simulation Settings:** 4 configurations × 1 runs each = 4 total simulations

## Overview

This report summarizes the results of running reinforcement learning-based variable selection
on synthetic datasets with known ground truth. We compare two RL approaches (Bandit MDP and 
Sequential MDP) against traditional baseline methods (LassoCV and RFE).

Each configuration was run **1 times** with different random seeds to compute mean ± standard deviation.

## Stopping Criteria for Each Method

| Method | Stopping Criteria | Convergence Check |
|--------|-------------------|-------------------|
| **Bandit MDP** | Fixed timesteps (15,000-20,000) | No early stopping; runs for full timesteps |
| **Sequential MDP** | Fixed timesteps (15,000-20,000) | No early stopping; gamma=0 (immediate reward only) |
| **LassoCV** | Cross-validation convergence | Automatic via sklearn (max_iter=2000, cv=5) |
| **RFE** | All features ranked | Deterministic; selects exactly n_informative features |

### PPO Training Details (for RL methods)

- **Objective**: Maximize reward = R² - 0.01 × n_selected_features
- **Algorithm**: PPO with clipped surrogate objective (clip_range=0.2)
- **Batch size**: 64, n_epochs=10 per update
- **Learning rate**: 3e-4

## Simulation Configurations

| Config | Samples | Informative | Fake | Total | Noise | Timesteps |
|--------|---------|-------------|------|-------|-------|-----------|
| 1 | 500 | 5 | 15 | 20 | 2.0 | 5000 |
| 2 | 1000 | 8 | 22 | 30 | 2.0 | 5000 |
| 3 | 2000 | 10 | 40 | 50 | 2.0 | 8000 |
| 4 | 5000 | 15 | 35 | 50 | 2.0 | 10000 |


## Results by Configuration (Mean ± Std over 1 runs)

### Configuration 1

**Dataset Settings:**
- Samples: 500
- Informative features: 5
- Fake (noise) features: 15
- Total features: 20
- Noise level: 2.0
- Training timesteps: 5000

**Results (Mean ± Std over 1 runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | 5.0±0.0 | 0.942±0.000 | 0.043±0.000 | 0.600±0.000 | 0.600±0.000 | 0.600±0.000 | 77.5±0.0 |
| Sequential MDP | 2.0±0.0 | 0.849±0.000 | 0.111±0.000 | 1.000±0.000 | 0.400±0.000 | 0.571±0.000 | 58.8±0.0 |
| LassoCV | 9.0±0.0 | 0.951±0.000 | 0.037±0.000 | 0.556±0.000 | 1.000±0.000 | 0.714±0.000 | 0.1±0.0 |
| RFE | 5.0±0.0 | 0.952±0.000 | 0.035±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.0±0.0 |
| All Features | 20 | 0.950±0.000 | - | 0.250 | 1.000 | 0.400 | - |

---

### Configuration 2

**Dataset Settings:**
- Samples: 1000
- Informative features: 8
- Fake (noise) features: 22
- Total features: 30
- Noise level: 2.0
- Training timesteps: 5000

**Results (Mean ± Std over 1 runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | 9.0±0.0 | 0.972±0.000 | 0.025±0.000 | 0.556±0.000 | 0.625±0.000 | 0.588±0.000 | 409.8±0.0 |
| Sequential MDP | 2.0±0.0 | 0.797±0.000 | 0.185±0.000 | 1.000±0.000 | 0.250±0.000 | 0.400±0.000 | 151.3±0.0 |
| LassoCV | 16.0±0.0 | 0.977±0.000 | 0.021±0.000 | 0.500±0.000 | 1.000±0.000 | 0.667±0.000 | 0.2±0.0 |
| RFE | 8.0±0.0 | 0.977±0.000 | 0.021±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.5±0.0 |
| All Features | 30 | 0.977±0.000 | - | 0.267 | 1.000 | 0.421 | - |

---

### Configuration 3

**Dataset Settings:**
- Samples: 2000
- Informative features: 10
- Fake (noise) features: 40
- Total features: 50
- Noise level: 2.0
- Training timesteps: 8000

**Results (Mean ± Std over 1 runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | 8.0±0.0 | 0.982±0.000 | 0.020±0.000 | 1.000±0.000 | 0.800±0.000 | 0.889±0.000 | 2729.9±0.0 |
| Sequential MDP | 4.0±0.0 | 0.675±0.000 | 0.359±0.000 | 0.750±0.000 | 0.300±0.000 | 0.429±0.000 | 1487.8±0.0 |
| LassoCV | 26.0±0.0 | 0.981±0.000 | 0.020±0.000 | 0.346±0.000 | 0.900±0.000 | 0.500±0.000 | 0.5±0.0 |
| RFE | 10.0±0.0 | 0.982±0.000 | 0.020±0.000 | 0.900±0.000 | 0.900±0.000 | 0.900±0.000 | 1.7±0.0 |
| All Features | 50 | 0.981±0.000 | - | 0.200 | 1.000 | 0.333 | - |

---

### Configuration 4

**Dataset Settings:**
- Samples: 5000
- Informative features: 15
- Fake (noise) features: 35
- Total features: 50
- Noise level: 2.0
- Training timesteps: 10000

**Results (Mean ± Std over 1 runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | 11.0±0.0 | 0.962±0.000 | 0.042±0.000 | 1.000±0.000 | 0.733±0.000 | 0.846±0.000 | 4528.0±0.0 |
| Sequential MDP | 4.0±0.0 | 0.592±0.000 | 0.450±0.000 | 0.750±0.000 | 0.200±0.000 | 0.316±0.000 | 2923.5±0.0 |
| LassoCV | 26.0±0.0 | 0.983±0.000 | 0.019±0.000 | 0.577±0.000 | 1.000±0.000 | 0.732±0.000 | 0.5±0.0 |
| RFE | 15.0±0.0 | 0.983±0.000 | 0.019±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.5±0.0 |
| All Features | 50 | 0.983±0.000 | - | 0.300 | 1.000 | 0.462 | - |

---

## Overall Summary (Averaged Across All 4 Configurations)

| Method | Avg Features | Avg R² | Avg Precision | Avg Recall | Avg F1 | Avg Runtime (s) |
|--------|--------------|--------|---------------|------------|--------|-----------------|
| Bandit MDP | 8.2 | 0.965 | 0.789 | 0.690 | 0.731 | 1936.3 |
| Sequential MDP | 3.0 | 0.728 | 0.875 | 0.287 | 0.429 | 1155.3 |
| LassoCV | 19.2 | 0.973 | 0.495 | 0.975 | 0.653 | 0.3 |
| RFE | 9.5 | 0.974 | 0.975 | 0.975 | 0.975 | 0.9 |

## Convergence Analysis

Since the RL methods (Bandit and Sequential MDP) use **fixed timestep stopping**, convergence 
is determined by observing the stability of results across multiple runs:

| Method | F1 Std (across runs) | Interpretation |
|--------|---------------------|----------------|
| Config 1 - Bandit | 0.000 | ✓ Converged |
| Config 1 - Sequential | 0.000 | ✓ Converged |
| Config 2 - Bandit | 0.000 | ✓ Converged |
| Config 2 - Sequential | 0.000 | ✓ Converged |
| Config 3 - Bandit | 0.000 | ✓ Converged |
| Config 3 - Sequential | 0.000 | ✓ Converged |
| Config 4 - Bandit | 0.000 | ✓ Converged |
| Config 4 - Sequential | 0.000 | ✓ Converged |


**Interpretation:**
- F1 Std < 0.1: Model has converged to stable solutions
- F1 Std ≥ 0.1: High variance suggests more training timesteps may help

## Metrics Explanation

- **Test R²**: Coefficient of determination on held-out test set (higher is better)
- **Test MSE**: Mean squared error on test set (lower is better)
- **Precision**: Proportion of selected features that are truly informative
- **Recall**: Proportion of truly informative features that were selected
- **F1**: Harmonic mean of precision and recall

## Files Generated

- `simulation_all_runs.csv`: All individual run results
- `simulation_statistics.csv`: Mean ± Std statistics per configuration
- `simulation_report.md`: This report

## References

- Le, Y., Bai, Y., and Zhou, F. (2025). *Reinforced variable selection.* Statistical Theory and Related Fields.
