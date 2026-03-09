#!/usr/bin/env python3
"""
Comprehensive simulation script for RL-based variable selection.

Runs simulations on synthetic datasets with known ground truth,
comparing Bandit MDP, Sequential MDP, and baseline methods.
Outputs detailed results to CSV and a markdown summary.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LassoCV
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, r2_score

from env_bandit import VariableSelectionEnv
from env_sequential import SequentialVariableSelectionEnv
from agent_bandit import VariableSelectionPPO
from agent_sequential import SequentialVariableSelectionPPO
from evaluate import evaluate_selection, compute_precision_recall


def create_synthetic_data_with_known_formula(
    n_samples: int = 200,
    n_informative: int = 5,
    n_fake: int = 45,
    noise: float = 10.0,
    random_state: int = 42,
):
    """
    Create synthetic regression data with known ground truth.
    
    The data is generated using STANDARDIZED X, so the formula
    y = Σ coef_i * X_i + noise is accurate for standardized features.
    
    Returns:
        X_scaled: Standardized feature matrix
        y: Target vector (computed from standardized X)
        true_features: Indices of truly informative features
        formula_description: String describing the true formula
        coef_true: True coefficients for informative features
        scaler_X: The StandardScaler used (for reference)
    """
    np.random.seed(random_state)
    
    n_features = n_informative + n_fake
    
    # Step 1: Generate raw features
    X_informative_raw = np.random.randn(n_samples, n_informative)
    X_fake_raw = np.random.randn(n_samples, n_fake)
    X_raw = np.hstack([X_informative_raw, X_fake_raw])
    
    # Step 2: Shuffle columns (so we don't know which are informative)
    perm = np.random.permutation(n_features)
    X_raw = X_raw[:, perm]
    
    # Step 3: Standardize X FIRST
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    
    # Step 4: Identify where true features ended up after shuffle
    true_features = np.where(perm < n_informative)[0]
    
    # Step 5: Generate coefficients for the true features
    coef_true = np.random.randn(n_informative) * 5
    coef_true = np.round(coef_true, 2)
    
    # Step 6: Compute y using STANDARDIZED X
    # Map coefficients to their positions after shuffling
    coef_full = np.zeros(n_features)
    for i, feat_idx in enumerate(sorted(true_features)):
        # Find which original informative feature this corresponds to
        orig_informative_idx = perm[feat_idx]
        coef_full[feat_idx] = coef_true[orig_informative_idx]
    
    y = X_scaled @ coef_full + np.random.randn(n_samples) * noise
    
    # Step 7: Build formula description
    formula_parts = []
    for feat_idx in sorted(true_features):
        formula_parts.append(f"{coef_full[feat_idx]:.2f}*X{feat_idx}")
    
    formula_description = "y = " + " + ".join(formula_parts) + f" + N(0, {noise})"
    
    return X_scaled, y, true_features, formula_description, coef_true, scaler_X


def check_convergence(rewards_history: list, window: int = 10, threshold: float = 0.01) -> dict:
    """
    Check if rewards have converged based on recent history.
    
    Args:
        rewards_history: List of reward values over training
        window: Number of recent values to consider
        threshold: Relative change threshold for convergence
        
    Returns:
        Dictionary with convergence info
    """
    if len(rewards_history) < window * 2:
        return {"converged": False, "final_reward": rewards_history[-1] if rewards_history else 0}
    
    recent = rewards_history[-window:]
    previous = rewards_history[-2*window:-window]
    
    recent_mean = np.mean(recent)
    previous_mean = np.mean(previous)
    recent_std = np.std(recent)
    
    if abs(previous_mean) > 1e-6:
        relative_change = abs(recent_mean - previous_mean) / abs(previous_mean)
    else:
        relative_change = abs(recent_mean - previous_mean)
    
    converged = relative_change < threshold and recent_std < threshold
    
    return {
        "converged": converged,
        "final_reward": recent_mean,
        "reward_std": recent_std,
        "relative_change": relative_change,
    }


def run_single_simulation(
    n_samples: int,
    n_informative: int,
    n_fake: int,
    noise: float,
    sparsity_penalty: float,
    total_timesteps: int,
    random_state: int,
    verbose: int = 0,
):
    """
    Run a single simulation with both Bandit and Sequential MDP approaches.
    
    Returns:
        Dictionary with all results
    """
    # X is already standardized from create_synthetic_data_with_known_formula
    X_scaled, y, true_features, formula, coef_true, _ = create_synthetic_data_with_known_formula(
        n_samples=n_samples,
        n_informative=n_informative,
        n_fake=n_fake,
        noise=noise,
        random_state=random_state,
    )
    
    n_features = n_informative + n_fake
    
    # Split the already-standardized data
    X_temp, X_test_scaled, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state
    )
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=random_state
    )
    
    # X is already standardized, no need to re-standardize
    # But we still standardize y for numerical stability in training
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    results = {
        "n_samples": n_samples,
        "n_informative": n_informative,
        "n_fake": n_fake,
        "n_total_features": n_features,
        "noise": noise,
        "sparsity_penalty": sparsity_penalty,
        "total_timesteps": total_timesteps,
        "formula": formula,
        "true_features": sorted(true_features.tolist()),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulation: {n_samples} samples, {n_informative} informative, {n_fake} fake features")
        print(f"True formula: {formula}")
        print(f"True feature indices: {sorted(true_features)}")
        print(f"{'='*60}")
    
    if verbose:
        print("\n[1/4] Training Bandit MDP agent...")
    
    bandit_start = time.time()
    env_bandit = VariableSelectionEnv(
        X_train_scaled, y_train_scaled,
        sparsity_penalty=sparsity_penalty,
        reward_type="r2",
        use_cv=True,
        cv_folds=3,
        random_state=random_state,
    )
    
    agent_bandit = VariableSelectionPPO(
        env_bandit,
        learning_rate=3e-4,
        verbose=verbose,
        seed=random_state,
    )
    agent_bandit.train(total_timesteps=total_timesteps)
    
    bandit_features = agent_bandit.select_features(deterministic=True)
    bandit_runtime = time.time() - bandit_start
    
    bandit_eval = evaluate_selection(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        bandit_features, task="regression"
    )
    bandit_pr = compute_precision_recall(bandit_features, true_features, n_features)
    
    results["bandit_selected_features"] = sorted(bandit_features.tolist())
    results["bandit_n_selected"] = len(bandit_features)
    results["bandit_test_r2"] = bandit_eval["test_r2"]
    results["bandit_test_mse"] = bandit_eval["test_mse"]
    results["bandit_cv_r2_mean"] = bandit_eval["cv_r2_mean"]
    results["bandit_precision"] = bandit_pr["precision"]
    results["bandit_recall"] = bandit_pr["recall"]
    results["bandit_f1"] = bandit_pr["f1"]
    results["bandit_runtime_sec"] = bandit_runtime
    
    if verbose:
        print(f"  Bandit selected {len(bandit_features)} features: {sorted(bandit_features.tolist())}")
        print(f"  Bandit Test R²: {bandit_eval['test_r2']:.4f}, MSE: {bandit_eval['test_mse']:.4f}")
        print(f"  Bandit Precision: {bandit_pr['precision']:.3f}, Recall: {bandit_pr['recall']:.3f}, F1: {bandit_pr['f1']:.3f}")
        print(f"  Bandit Runtime: {bandit_runtime:.2f}s")
    
    if verbose:
        print("\n[2/4] Training Sequential MDP agent...")
    
    seq_start = time.time()
    env_seq = SequentialVariableSelectionEnv(
        X_train_scaled, y_train_scaled,
        sparsity_penalty=sparsity_penalty,
        reward_type="r2",
        use_cv=True,
        cv_folds=3,
        max_episode_steps=min(max(50, 2 * n_features), 200),
        random_state=random_state,
    )
    
    agent_seq = SequentialVariableSelectionPPO(
        env_seq,
        learning_rate=3e-4,
        gamma=0.0,
        verbose=verbose,
        seed=random_state,
    )
    agent_seq.train(total_timesteps=total_timesteps)
    
    seq_features = agent_seq.select_features(deterministic=True)
    seq_runtime = time.time() - seq_start
    
    seq_eval = evaluate_selection(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        seq_features, task="regression"
    )
    seq_pr = compute_precision_recall(seq_features, true_features, n_features)
    
    results["sequential_selected_features"] = sorted(seq_features.tolist())
    results["sequential_n_selected"] = len(seq_features)
    results["sequential_test_r2"] = seq_eval["test_r2"]
    results["sequential_test_mse"] = seq_eval["test_mse"]
    results["sequential_cv_r2_mean"] = seq_eval["cv_r2_mean"]
    results["sequential_precision"] = seq_pr["precision"]
    results["sequential_recall"] = seq_pr["recall"]
    results["sequential_f1"] = seq_pr["f1"]
    results["sequential_runtime_sec"] = seq_runtime
    
    if verbose:
        print(f"  Sequential selected {len(seq_features)} features: {sorted(seq_features.tolist())}")
        print(f"  Sequential Test R²: {seq_eval['test_r2']:.4f}, MSE: {seq_eval['test_mse']:.4f}")
        print(f"  Sequential Precision: {seq_pr['precision']:.3f}, Recall: {seq_pr['recall']:.3f}, F1: {seq_pr['f1']:.3f}")
        print(f"  Sequential Runtime: {seq_runtime:.2f}s")
    
    if verbose:
        print("\n[3/4] Running LassoCV baseline...")
    
    lasso_start = time.time()
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=2000)
    lasso.fit(X_train_scaled, y_train_scaled)
    lasso_features = np.where(np.abs(lasso.coef_) > 1e-6)[0]
    lasso_runtime = time.time() - lasso_start
    
    if len(lasso_features) > 0:
        lasso_eval = evaluate_selection(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            lasso_features, task="regression"
        )
        lasso_pr = compute_precision_recall(lasso_features, true_features, n_features)
    else:
        lasso_eval = {"test_r2": -np.inf, "test_mse": np.inf, "cv_r2_mean": -np.inf}
        lasso_pr = {"precision": 0, "recall": 0, "f1": 0}
    
    results["lasso_selected_features"] = sorted(lasso_features.tolist())
    results["lasso_n_selected"] = len(lasso_features)
    results["lasso_test_r2"] = lasso_eval["test_r2"]
    results["lasso_test_mse"] = lasso_eval["test_mse"]
    results["lasso_cv_r2_mean"] = lasso_eval.get("cv_r2_mean", -np.inf)
    results["lasso_precision"] = lasso_pr["precision"]
    results["lasso_recall"] = lasso_pr["recall"]
    results["lasso_f1"] = lasso_pr["f1"]
    results["lasso_runtime_sec"] = lasso_runtime
    
    if verbose:
        print(f"  LassoCV selected {len(lasso_features)} features")
        print(f"  LassoCV Precision: {lasso_pr['precision']:.3f}, Recall: {lasso_pr['recall']:.3f}, F1: {lasso_pr['f1']:.3f}")
        print(f"  LassoCV Runtime: {lasso_runtime:.2f}s")
    
    if verbose:
        print("\n[4/4] Running RFE baseline...")
    
    rfe_start = time.time()
    try:
        rfe = RFE(Ridge(alpha=1.0), n_features_to_select=n_informative)
        rfe.fit(X_train_scaled, y_train_scaled)
        rfe_features = np.where(rfe.get_support())[0]
        rfe_runtime = time.time() - rfe_start
        rfe_eval = evaluate_selection(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            rfe_features, task="regression"
        )
        rfe_pr = compute_precision_recall(rfe_features, true_features, n_features)
    except Exception as e:
        rfe_runtime = time.time() - rfe_start
        rfe_features = np.array([])
        rfe_eval = {"test_r2": -np.inf, "test_mse": np.inf, "cv_r2_mean": -np.inf}
        rfe_pr = {"precision": 0, "recall": 0, "f1": 0}
    
    results["rfe_selected_features"] = sorted(rfe_features.tolist())
    results["rfe_n_selected"] = len(rfe_features)
    results["rfe_test_r2"] = rfe_eval["test_r2"]
    results["rfe_test_mse"] = rfe_eval["test_mse"]
    results["rfe_cv_r2_mean"] = rfe_eval.get("cv_r2_mean", -np.inf)
    results["rfe_precision"] = rfe_pr["precision"]
    results["rfe_recall"] = rfe_pr["recall"]
    results["rfe_f1"] = rfe_pr["f1"]
    results["rfe_runtime_sec"] = rfe_runtime
    
    if verbose:
        print(f"  RFE selected {len(rfe_features)} features")
        print(f"  RFE Precision: {rfe_pr['precision']:.3f}, Recall: {rfe_pr['recall']:.3f}, F1: {rfe_pr['f1']:.3f}")
        print(f"  RFE Runtime: {rfe_runtime:.2f}s")
    
    all_features = np.arange(n_features)
    all_eval = evaluate_selection(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        all_features, task="regression"
    )
    results["all_features_test_r2"] = all_eval["test_r2"]
    results["all_features_test_mse"] = all_eval["test_mse"]
    
    return results


def run_simulation_suite(output_dir: str = "./results", verbose: int = 1, n_runs: int = 100):
    """
    Run a suite of simulations with different configurations.
    Each configuration is run n_runs times to compute statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    configs = [
        {"n_samples": 500, "n_informative": 5, "n_fake": 15, "noise": 2.0, "timesteps": 5000},
        {"n_samples": 1000, "n_informative": 8, "n_fake": 22, "noise": 2.0, "timesteps": 5000},
        {"n_samples": 2000, "n_informative": 10, "n_fake": 40, "noise": 2.0, "timesteps": 8000},
        {"n_samples": 5000, "n_informative": 15, "n_fake": 35, "noise": 2.0, "timesteps": 10000},
    ]
    
    all_results = []
    config_statistics = []
    
    print("="*70)
    print("RL Variable Selection Simulation Suite")
    print("="*70)
    print(f"Running {len(configs)} configurations × {n_runs} runs each = {len(configs) * n_runs} total simulations")
    print(f"Output directory: {output_path.absolute()}")
    print("="*70)
    
    for i, config in enumerate(configs):
        config_runs = []
        
        print(f"\n{'='*70}")
        print(f"[Configuration {i+1}/{len(configs)}]")
        print(f"  Samples: {config['n_samples']}, Informative: {config['n_informative']}, Fake: {config['n_fake']}, Noise: {config['noise']}")
        print(f"  Running {n_runs} iterations...")
        print("="*70)
        
        for run_idx in range(n_runs):
            if verbose or (run_idx + 1) % 10 == 0:
                print(f"  Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)
            
            results = run_single_simulation(
                n_samples=config["n_samples"],
                n_informative=config["n_informative"],
                n_fake=config["n_fake"],
                noise=config["noise"],
                sparsity_penalty=0.01,
                total_timesteps=config.get("timesteps", 10000),
                random_state=42 + i * 1000 + run_idx,
                verbose=0,  # Suppress per-run output
            )
            
            results["config_id"] = i + 1
            results["run_id"] = run_idx + 1
            config_runs.append(results)
            all_results.append(results)
            
            if verbose or (run_idx + 1) % 10 == 0:
                print(f"Bandit F1={results['bandit_f1']:.3f}, RFE F1={results['rfe_f1']:.3f}")
        
        # Compute statistics for this configuration
        stats = compute_config_statistics(config_runs, config, i + 1)
        config_statistics.append(stats)
        
        print(f"\n  Configuration {i+1} Summary:")
        print(f"    Bandit: F1={stats['bandit_f1_mean']:.3f}±{stats['bandit_f1_std']:.3f}, Precision={stats['bandit_precision_mean']:.3f}±{stats['bandit_precision_std']:.3f}")
        print(f"    Sequential: F1={stats['sequential_f1_mean']:.3f}±{stats['sequential_f1_std']:.3f}")
        print(f"    LassoCV: F1={stats['lasso_f1_mean']:.3f}±{stats['lasso_f1_std']:.3f}")
        print(f"    RFE: F1={stats['rfe_f1_mean']:.3f}±{stats['rfe_f1_std']:.3f}")
    
    # Save all individual run results
    all_runs_rows = []
    for r in all_results:
        for method in ["bandit", "sequential", "lasso", "rfe"]:
            row = {
                "config_id": r["config_id"],
                "run_id": r.get("run_id", 1),
                "n_samples": r["n_samples"],
                "n_informative": r["n_informative"],
                "n_fake": r["n_fake"],
                "noise": r["noise"],
                "method": method.capitalize(),
                "n_selected": r[f"{method}_n_selected"],
                "test_r2": r[f"{method}_test_r2"],
                "test_mse": r[f"{method}_test_mse"],
                "precision": r[f"{method}_precision"],
                "recall": r[f"{method}_recall"],
                "f1": r[f"{method}_f1"],
                "runtime_sec": r[f"{method}_runtime_sec"],
            }
            all_runs_rows.append(row)
    
    all_runs_df = pd.DataFrame(all_runs_rows)
    all_runs_df.to_csv(output_path / "simulation_all_runs.csv", index=False)
    
    # Save statistics summary
    stats_df = pd.DataFrame(config_statistics)
    stats_df.to_csv(output_path / "simulation_statistics.csv", index=False)
    
    # Generate the new report with statistics
    generate_markdown_report_with_stats(config_statistics, all_results, output_path, n_runs)
    
    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - {output_path / 'simulation_all_runs.csv'} (all {len(all_results)} individual runs)")
    print(f"  - {output_path / 'simulation_statistics.csv'} (mean±std per config)")
    print(f"  - {output_path / 'simulation_report.md'}")
    
    return stats_df, config_statistics


def compute_config_statistics(config_runs: list, config: dict, config_id: int) -> dict:
    """Compute mean and std statistics for a configuration across multiple runs."""
    stats = {
        "config_id": config_id,
        "n_samples": config["n_samples"],
        "n_informative": config["n_informative"],
        "n_fake": config["n_fake"],
        "n_total_features": config["n_informative"] + config["n_fake"],
        "noise": config["noise"],
        "timesteps": config.get("timesteps", 10000),
        "n_runs": len(config_runs),
    }
    
    # Collect metrics for each method
    for method in ["bandit", "sequential", "lasso", "rfe"]:
        for metric in ["n_selected", "test_r2", "test_mse", "precision", "recall", "f1", "runtime_sec"]:
            key = f"{method}_{metric}"
            values = [r[key] for r in config_runs if key in r]
            if values:
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
    
    # All features baseline
    stats["all_features_test_r2_mean"] = np.mean([r["all_features_test_r2"] for r in config_runs])
    stats["all_features_test_r2_std"] = np.std([r["all_features_test_r2"] for r in config_runs])
    
    return stats


def generate_markdown_report_with_stats(config_statistics: list, all_results: list, output_path: Path, n_runs: int):
    """Generate a detailed markdown report with statistics from multiple runs."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# RL Variable Selection Simulation Report

**Generated:** {timestamp}

**Simulation Settings:** {len(config_statistics)} configurations × {n_runs} runs each = {len(config_statistics) * n_runs} total simulations

## Overview

This report summarizes the results of running reinforcement learning-based variable selection
on synthetic datasets with known ground truth. We compare two RL approaches (Bandit MDP and 
Sequential MDP) against traditional baseline methods (LassoCV and RFE).

Each configuration was run **{n_runs} times** with different random seeds to compute mean ± standard deviation.

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
"""
    
    for s in config_statistics:
        md_content += f"| {s['config_id']} | {s['n_samples']} | {s['n_informative']} | {s['n_fake']} | {s['n_total_features']} | {s['noise']} | {s['timesteps']} |\n"
    
    md_content += f"""

## Results by Configuration (Mean ± Std over {n_runs} runs)

"""
    
    for s in config_statistics:
        md_content += f"""### Configuration {s['config_id']}

**Dataset Settings:**
- Samples: {s['n_samples']}
- Informative features: {s['n_informative']}
- Fake (noise) features: {s['n_fake']}
- Total features: {s['n_total_features']}
- Noise level: {s['noise']}
- Training timesteps: {s['timesteps']}

**Results (Mean ± Std over {n_runs} runs):**

| Method | Features | Test R² | Test MSE | Precision | Recall | F1 | Runtime (s) |
|--------|----------|---------|----------|-----------|--------|-----|-------------|
| Bandit MDP | {s['bandit_n_selected_mean']:.1f}±{s['bandit_n_selected_std']:.1f} | {s['bandit_test_r2_mean']:.3f}±{s['bandit_test_r2_std']:.3f} | {s['bandit_test_mse_mean']:.3f}±{s['bandit_test_mse_std']:.3f} | {s['bandit_precision_mean']:.3f}±{s['bandit_precision_std']:.3f} | {s['bandit_recall_mean']:.3f}±{s['bandit_recall_std']:.3f} | {s['bandit_f1_mean']:.3f}±{s['bandit_f1_std']:.3f} | {s['bandit_runtime_sec_mean']:.1f}±{s['bandit_runtime_sec_std']:.1f} |
| Sequential MDP | {s['sequential_n_selected_mean']:.1f}±{s['sequential_n_selected_std']:.1f} | {s['sequential_test_r2_mean']:.3f}±{s['sequential_test_r2_std']:.3f} | {s['sequential_test_mse_mean']:.3f}±{s['sequential_test_mse_std']:.3f} | {s['sequential_precision_mean']:.3f}±{s['sequential_precision_std']:.3f} | {s['sequential_recall_mean']:.3f}±{s['sequential_recall_std']:.3f} | {s['sequential_f1_mean']:.3f}±{s['sequential_f1_std']:.3f} | {s['sequential_runtime_sec_mean']:.1f}±{s['sequential_runtime_sec_std']:.1f} |
| LassoCV | {s['lasso_n_selected_mean']:.1f}±{s['lasso_n_selected_std']:.1f} | {s['lasso_test_r2_mean']:.3f}±{s['lasso_test_r2_std']:.3f} | {s['lasso_test_mse_mean']:.3f}±{s['lasso_test_mse_std']:.3f} | {s['lasso_precision_mean']:.3f}±{s['lasso_precision_std']:.3f} | {s['lasso_recall_mean']:.3f}±{s['lasso_recall_std']:.3f} | {s['lasso_f1_mean']:.3f}±{s['lasso_f1_std']:.3f} | {s['lasso_runtime_sec_mean']:.1f}±{s['lasso_runtime_sec_std']:.1f} |
| RFE | {s['rfe_n_selected_mean']:.1f}±{s['rfe_n_selected_std']:.1f} | {s['rfe_test_r2_mean']:.3f}±{s['rfe_test_r2_std']:.3f} | {s['rfe_test_mse_mean']:.3f}±{s['rfe_test_mse_std']:.3f} | {s['rfe_precision_mean']:.3f}±{s['rfe_precision_std']:.3f} | {s['rfe_recall_mean']:.3f}±{s['rfe_recall_std']:.3f} | {s['rfe_f1_mean']:.3f}±{s['rfe_f1_std']:.3f} | {s['rfe_runtime_sec_mean']:.1f}±{s['rfe_runtime_sec_std']:.1f} |
| All Features | {s['n_total_features']} | {s['all_features_test_r2_mean']:.3f}±{s['all_features_test_r2_std']:.3f} | - | {s['n_informative']/s['n_total_features']:.3f} | 1.000 | {2*(s['n_informative']/s['n_total_features'])/(1+s['n_informative']/s['n_total_features']):.3f} | - |

---

"""
    
    # Overall summary across all configurations
    md_content += f"""## Overall Summary (Averaged Across All {len(config_statistics)} Configurations)

| Method | Avg Features | Avg R² | Avg Precision | Avg Recall | Avg F1 | Avg Runtime (s) |
|--------|--------------|--------|---------------|------------|--------|-----------------|
"""
    
    for method in ["bandit", "sequential", "lasso", "rfe"]:
        avg_features = np.mean([s[f"{method}_n_selected_mean"] for s in config_statistics])
        avg_r2 = np.mean([s[f"{method}_test_r2_mean"] for s in config_statistics])
        avg_precision = np.mean([s[f"{method}_precision_mean"] for s in config_statistics])
        avg_recall = np.mean([s[f"{method}_recall_mean"] for s in config_statistics])
        avg_f1 = np.mean([s[f"{method}_f1_mean"] for s in config_statistics])
        avg_runtime = np.mean([s[f"{method}_runtime_sec_mean"] for s in config_statistics])
        
        method_name = {"bandit": "Bandit MDP", "sequential": "Sequential MDP", "lasso": "LassoCV", "rfe": "RFE"}[method]
        md_content += f"| {method_name} | {avg_features:.1f} | {avg_r2:.3f} | {avg_precision:.3f} | {avg_recall:.3f} | {avg_f1:.3f} | {avg_runtime:.1f} |\n"
    
    md_content += f"""
## Convergence Analysis

Since the RL methods (Bandit and Sequential MDP) use **fixed timestep stopping**, convergence 
is determined by observing the stability of results across multiple runs:

| Method | F1 Std (across runs) | Interpretation |
|--------|---------------------|----------------|
"""
    
    for s in config_statistics:
        bandit_converged = "✓ Converged" if s['bandit_f1_std'] < 0.1 else "⚠ High variance"
        seq_converged = "✓ Converged" if s['sequential_f1_std'] < 0.1 else "⚠ High variance"
        md_content += f"| Config {s['config_id']} - Bandit | {s['bandit_f1_std']:.3f} | {bandit_converged} |\n"
        md_content += f"| Config {s['config_id']} - Sequential | {s['sequential_f1_std']:.3f} | {seq_converged} |\n"
    
    md_content += """

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
"""
    
    with open(output_path / "simulation_report.md", "w") as f:
        f.write(md_content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RL Variable Selection Simulations")
    parser.add_argument("--n_runs", type=int, default=100, help="Number of runs per configuration")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()
    
    stats_df, config_statistics = run_simulation_suite(
        output_dir=args.output_dir,
        verbose=args.verbose,
        n_runs=args.n_runs,
    )
    
    print("\n" + "="*70)
    print("Statistics Summary (Mean ± Std):")
    print("="*70)
    print(stats_df.to_string(index=False))
