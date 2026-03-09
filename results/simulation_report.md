# RL Variable Selection Simulation Report

**Generated:** 2026-03-09 01:56:13

**Simulation Settings:** 4 configurations × 100 runs each = 400 total simulations

## Overview

This report summarizes the results of running reinforcement learning-based variable selection
on synthetic datasets with known ground truth. We compare two RL approaches (Bandit MDP and 
Sequential MDP) against traditional baseline methods (LassoCV and RFE).

Each configuration was run **100 times** with different random seeds to compute mean ± standard deviation.

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
| 1 | 500 | 5 | 15 | 20 | 5.0 | 5000 |
| 2 | 1000 | 8 | 22 | 30 | 8.0 | 5000 |
| 3 | 2000 | 10 | 40 | 50 | 10.0 | 8000 |
| 4 | 5000 | 15 | 35 | 50 | 12.0 | 10000 |


## Results by Configuration (Mean ± Std over 100 runs)

### Configuration 1

**Dataset Settings:**
- Samples: 500
- Informative features: 5
- Fake (noise) features: 15
- Total features: 20
- Noise level: 5.0
- Training timesteps: 5000

**Results (Mean ± Std over 100 runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | 7.7±2.8 | 0.744±0.127 | 0.246±0.120 | 0.596±0.254 | 0.796±0.198 | 0.662±0.217 | 3.3±0.2 |
| Sequential MDP | 2.9±1.2 | 0.668±0.140 | 0.321±0.135 | 0.879±0.160 | 0.494±0.161 | 0.611±0.125 | 3.1±0.3 |
| LassoCV | 9.8±3.3 | 0.761±0.126 | 0.229±0.114 | 0.521±0.148 | 0.938±0.112 | 0.654±0.126 | 0.0±0.0 |
| RFE | 5.0±0.0 | 0.766±0.124 | 0.224±0.112 | 0.888±0.121 | 0.888±0.121 | 0.888±0.121 | 0.0±0.0 |
| All Features | 20 | 0.758±0.127 | - | 0.250 | 1.000 | 0.400 | - |

---

### Configuration 2

**Dataset Settings:**
- Samples: 1000
- Informative features: 8
- Fake (noise) features: 22
- Total features: 30
- Noise level: 8.0
- Training timesteps: 5000

**Results (Mean ± Std over 100 runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | 7.8±2.6 | 0.707±0.111 | 0.289±0.106 | 0.845±0.208 | 0.771±0.137 | 0.794±0.155 | 8.5±4.3 |
| Sequential MDP | 3.0±1.2 | 0.488±0.136 | 0.508±0.139 | 0.817±0.187 | 0.299±0.101 | 0.425±0.109 | 5.8±0.7 |
| LassoCV | 15.8±4.3 | 0.713±0.114 | 0.283±0.108 | 0.509±0.130 | 0.941±0.094 | 0.648±0.102 | 0.0±0.0 |
| RFE | 8.0±0.0 | 0.718±0.113 | 0.278±0.108 | 0.897±0.111 | 0.897±0.111 | 0.897±0.111 | 0.0±0.0 |
| All Features | 30 | 0.711±0.115 | - | 0.267 | 1.000 | 0.421 | - |

---

### Configuration 3

**Dataset Settings:**
- Samples: 2000
- Informative features: 10
- Fake (noise) features: 40
- Total features: 50
- Noise level: 10.0
- Training timesteps: 8000

**Results (Mean ± Std over 100 runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | 7.5±1.2 | 0.684±0.102 | 0.320±0.109 | 0.992±0.036 | 0.745±0.121 | 0.846±0.085 | 12.1±4.3 |
| Sequential MDP | 3.1±1.3 | 0.371±0.103 | 0.636±0.115 | 0.725±0.222 | 0.220±0.099 | 0.327±0.122 | 9.8±1.3 |
| LassoCV | 22.6±5.2 | 0.687±0.102 | 0.317±0.109 | 0.447±0.101 | 0.964±0.054 | 0.604±0.091 | 0.0±0.0 |
| RFE | 10.0±0.0 | 0.693±0.100 | 0.311±0.107 | 0.924±0.071 | 0.924±0.071 | 0.924±0.071 | 0.0±0.0 |
| All Features | 50 | 0.685±0.102 | - | 0.200 | 1.000 | 0.333 | - |

---

### Configuration 4

**Dataset Settings:**
- Samples: 5000
- Informative features: 15
- Fake (noise) features: 35
- Total features: 50
- Noise level: 12.0
- Training timesteps: 10000

**Results (Mean ± Std over 100 runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | 10.0±1.6 | 0.690±0.079 | 0.309±0.080 | 1.000±0.000 | 0.665±0.109 | 0.793±0.080 | 18.3±4.9 |
| Sequential MDP | 3.4±1.4 | -inf±nan | inf±nan | 0.855±0.186 | 0.188±0.071 | 0.301±0.095 | 14.4±2.4 |
| LassoCV | 30.1±5.0 | 0.703±0.081 | 0.295±0.082 | 0.496±0.074 | 0.973±0.046 | 0.653±0.064 | 0.0±0.0 |
| RFE | 15.0±0.0 | 0.705±0.081 | 0.293±0.081 | 0.929±0.065 | 0.929±0.065 | 0.929±0.065 | 0.0±0.0 |
| All Features | 50 | 0.703±0.081 | - | 0.300 | 1.000 | 0.462 | - |

---

## Overall Summary (Averaged Across All 4 Configurations)

| Method | Avg Features | Avg R² | Avg Precision | Avg Recall | Avg F1 | Avg Runtime (s) |
|--------|--------------|--------|---------------|------------|--------|-----------------|
| Bandit MDP | 8.3 | 0.706 | 0.858 | 0.744 | 0.774 | 10.6 |
| Sequential MDP | 3.1 | -inf | 0.819 | 0.300 | 0.416 | 8.3 |
| LassoCV | 19.6 | 0.716 | 0.493 | 0.954 | 0.640 | 0.0 |
| RFE | 9.5 | 0.721 | 0.910 | 0.910 | 0.910 | 0.0 |

## Convergence Analysis

Since the RL methods (Bandit and Sequential MDP) use **fixed timestep stopping**, convergence 
is determined by observing the stability of results across multiple runs:

| Method | F1 Std (across runs) | Interpretation |
|--------|---------------------|----------------|
| Config 1 - Bandit | 0.217 | ⚠ High variance |
| Config 1 - Sequential | 0.125 | ⚠ High variance |
| Config 2 - Bandit | 0.155 | ⚠ High variance |
| Config 2 - Sequential | 0.109 | ⚠ High variance |
| Config 3 - Bandit | 0.085 | ✓ Converged |
| Config 3 - Sequential | 0.122 | ⚠ High variance |
| Config 4 - Bandit | 0.080 | ✓ Converged |
| Config 4 - Sequential | 0.095 | ✓ Converged |


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
