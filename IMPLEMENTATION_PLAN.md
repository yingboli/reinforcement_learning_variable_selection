# Implementation Plan (from research_plan.pdf)

This document maps the research plan to the codebase and tracks what is done vs. planned.

## 1. Objective functions (Section 2.2)

| Objective | Status | Where |
|----------|--------|--------|
| **Cross-validated RMSE** | Done | `env_base.py`: `reward_type="cv_rmse"`; reward = -CV_RMSE |
| **AIC** | Done | `env_base.py`: `reward_type="aic"`; AIC = n*log(RSS/n) + 2*p_γ, reward = -AIC |
| **BIC** | Done | `env_base.py`: `reward_type="bic"`; BIC = n*log(RSS/n) + log(n)*p_γ, reward = -BIC |
| **Bayes factor (g-prior)** | Done | `env_base.py`: `reward_type="bayes_factor"`; BF vs null, g = max(p²,n), reward = log(BF) |

Use e.g. `--reward_type cv_rmse`, `aic`, `bic`, or `bayes_factor` in regression.

## 2. Baseline search methods (Section 2.3)

| Method | Status | Where |
|--------|--------|--------|
| Forward / backward selection | Done | `evaluate.py`: SequentialFeatureSelector (forward); RFE (backward-style) |
| Lasso | Done | LassoCV in `evaluate.py` |
| **Adaptive Lasso** | Done | `evaluate.py`: weights w_j = 1/\|β̂_j\| from Ridge, then LassoCV on scaled X |
| **MCMC (Metropolis)** | Done | `evaluate.py`: `_mcmc_metropolis_variable_selection`; flip or swap, accept by BF ratio |

## 3. RL methods (Section 3)

| Component | Status | Where |
|-----------|--------|--------|
| Bandit (one-step, policy = NN) | Done | `env_bandit.py`, `agent_bandit.py` |
| MDP (multi-step, toggle) | Done | `env_sequential.py`, `agent_sequential.py` |
| PPO with clipped objective | Done | stable-baselines3 PPO in `agent_base.py` |


## 4. Simulation studies (Section 4.1)

| Item | Status | Notes |
|------|--------|--------|
| Simulation design | Planned | PDF has no details; suggest: known true γ*, vary n/p/sparsity, report selection F1 and test RMSE |
| Selection accuracy (e.g. F1) | Done | `evaluate.py`: `compute_precision_recall(selected, true_features, n_total)` |

To run simulations with known truth: generate data from a sparse linear model, record true features, then compare selected vs true via precision/recall/F1.

## 5. Real data – TabRED (Section 5)

| Item | Status | Notes |
|------|--------|--------|
| TabRED benchmark | Planned | [tabred](https://github.com/yandex-research/tabred): add data loader or instructions to download and pass X, y into existing pipeline |
| Compare prediction accuracy (e.g. test RMSE) | Done | Current evaluation reports test R²/MSE and CV R² |

## 6. Suggested next steps

1. **Simulation script**: Add `scripts/run_simulation.py` that generates data with known γ*, runs bandit/MDP and baselines, reports selection F1 and test RMSE.
2. **TabRED loader**: Add `data/tabred.py` or document how to load TabRED datasets and call existing `main.py` / `main_comparison.py` with the same interface (train/test splits, scaling).
3. **Reward_type in CLI**: Ensure `--reward_type` help lists all options (cv_rmse, aic, bic, bayes_factor for regression).
4. **Experiments**: Run full comparison (bandit vs MDP, multiple reward types, baselines) on synthetic and TabRED as in the plan.
