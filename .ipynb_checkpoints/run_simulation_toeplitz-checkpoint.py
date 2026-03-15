#!/usr/bin/env python3
"""
Run the Toeplitz simulation design for regression variable selection.
Self-contained: does not import from run_simulation.py.

Design:
  - (n, p_total, p_true): (1000, 50, 0), (1000, 50, 5), (1000, 50, 20), (10000, 200, 20)
  - SNR: 0.5, 1, 2  (Var(X@beta) / sigma^2)
  - Toeplitz correlation rho: 0, 0.5, 0.9

No p > n. All runs are regression only.
"""

import argparse
import time
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from data_sim import generate_toeplitz_regression
from env_bandit import VariableSelectionEnv
from env_sequential import SequentialVariableSelectionEnv
from agent_bandit import VariableSelectionPPO
from agent_sequential import SequentialVariableSelectionPPO
from evaluate import (
    evaluate_selection,
    compute_precision_recall,
    _lasso_by_criterion,
    _forward_backward_selection_by_criterion,
    _mcmc_metropolis_variable_selection,
)


def _record_selection(
    results: dict,
    name: str,
    features: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    true_features: np.ndarray,
    n_features: int,
    runtime: float,
    random_state: Optional[int] = None,
) -> None:
    """Store selected_features, n_selected, test_r2, test_mse, precision, recall, f1, runtime_sec for a method.
    Null model (empty features) is evaluated correctly by evaluate_selection (intercept-only: test_mse=Var(y), test_r2=0)."""
    eval_res = evaluate_selection(X_test, y_test, features, task="regression", random_state=random_state)
    pr = compute_precision_recall(features, true_features, n_features)
    results[f"{name}_selected_features"] = sorted(features.tolist())
    results[f"{name}_n_selected"] = len(features)
    results[f"{name}_test_r2"] = eval_res["test_r2"]
    results[f"{name}_test_mse"] = eval_res["test_mse"]
    results[f"{name}_precision"] = pr["precision"]
    results[f"{name}_recall"] = pr["recall"]
    results[f"{name}_f1"] = pr["f1"]
    results[f"{name}_runtime_sec"] = runtime


def _run_select_and_record(
    results: dict,
    name: str,
    select_fn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    true_features: np.ndarray,
    n_features: int,
    random_state: Optional[int] = None,
) -> None:
    """Run select_fn(); on exception use empty selection. Record results via _record_selection."""
    t0 = time.time()
    try:
        features = np.asarray(select_fn())
    except Exception:
        features = np.array([])
    runtime = time.time() - t0
    _record_selection(results, name, features, X_test, y_test, true_features, n_features, runtime, random_state=random_state)


def run_single_simulation_from_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    true_features: np.ndarray,
    total_timesteps: int,
    random_state: int,
    verbose: int = 0,
    reward_type: str = "cv_rmse",
    sequential_gammas: Optional[list] = None,
) -> dict:
    """
    Run one simulation from pre-split (X_train, y_train), (X_test, y_test) and true feature indices.
    X_train/X_test should already be standardized; y_train/y_test are used as-is (no scaling).
    Runs Bandit MDP, Sequential MDP, LassoCV, Backward selection, and All-features baseline; returns a results dict.
    reward_type: For regression use 'cv_rmse', 'aic', 'bic', or 'bayes_factor'.
    Baselines (Lasso, Adaptive Lasso, Forward, Backward) use the same criterion as reward_type (cv_rmse->cv).
    sequential_gammas: list of gamma values for Sequential MDP (default [0.3]).
    """
    if sequential_gammas is None:
        sequential_gammas = [0.3]
    n_features = X_train.shape[1]
    n_informative = len(true_features)
    selection_criterion = "cv" if reward_type == "cv_rmse" else reward_type

    results = {
        "n_samples": X_train.shape[0],
        "n_informative": n_informative,
        "n_fake": n_features - n_informative,
        "n_total_features": n_features,
        "true_features": sorted(true_features.tolist()),
        "total_timesteps": total_timesteps,
        "reward_type": reward_type,
    }

    # Bandit
    t0 = time.time()
    env_bandit = VariableSelectionEnv(
        X_train, y_train,
        reward_type=reward_type, cv_folds=5, random_state=random_state,
    )
    agent_bandit = VariableSelectionPPO(env_bandit, learning_rate=3e-4, verbose=verbose, seed=random_state)
    agent_bandit.train(total_timesteps=total_timesteps)
    bandit_features = agent_bandit.select_features(deterministic=True)
    _record_selection(results, "bandit", bandit_features, X_test, y_test, true_features, n_features, time.time() - t0, random_state=random_state)

    # Sequential: run for each gamma in sequential_gammas (random starts + improvement bonus)
    env_seq = SequentialVariableSelectionEnv(
        X_train, y_train,
        reward_type=reward_type, cv_folds=5,
        max_episode_steps=min(max(50, 2 * n_features), 500),
        random_start_probability=0.4,
        improvement_bonus_coef=0.2,
        random_state=random_state,
    )
    for gamma in sequential_gammas:
        t0 = time.time()
        agent_seq = SequentialVariableSelectionPPO(
            env_seq, learning_rate=3e-4, gamma=gamma, ent_coef=0.02,
            verbose=verbose, seed=random_state,
        )
        agent_seq.train(total_timesteps=int(total_timesteps * 2))
        seq_features = agent_seq.select_features(deterministic=True)
        _record_selection(results, f"sequential_g{gamma}", seq_features, X_test, y_test, true_features, n_features, time.time() - t0, random_state=random_state)

    # Baselines (same criteria as compare_with_baselines)
    _run_select_and_record(
        results, "lasso",
        lambda: _lasso_by_criterion(X_train, y_train, task="regression", criterion=selection_criterion, cv=5, random_state=random_state),
        X_test, y_test, true_features, n_features, random_state=random_state,
    )
    _run_select_and_record(
        results, "adaptive_lasso",
        lambda: _lasso_by_criterion(X_train, y_train, task="regression", criterion=selection_criterion, cv=5, adaptive=True, random_state=random_state),
        X_test, y_test, true_features, n_features, random_state=random_state,
    )
    _run_select_and_record(
        results, "forward",
        lambda: _forward_backward_selection_by_criterion(X_train, y_train, task="regression", criterion=selection_criterion, cv=5, direction="forward", random_state=random_state),
        X_test, y_test, true_features, n_features, random_state=random_state,
    )
    _run_select_and_record(
        results, "backward",
        lambda: _forward_backward_selection_by_criterion(X_train, y_train, task="regression", criterion=selection_criterion, cv=5, direction="backward", random_state=random_state),
        X_test, y_test, true_features, n_features, random_state=random_state,
    )
    if selection_criterion == "bayes_factor":
        _run_select_and_record(
            results, "mcmc",
            lambda: _mcmc_metropolis_variable_selection(X_train, y_train, n_iter=total_timesteps, random_state=random_state),
            X_test, y_test, true_features, n_features, random_state=random_state,
        )

    # All features
    all_eval = evaluate_selection(X_test, y_test, np.arange(n_features), task="regression", random_state=random_state)
    results["all_features_test_r2"] = all_eval["test_r2"]
    results["all_features_test_mse"] = all_eval["test_mse"]

    return results


# Design grid
DESIGN_N_P_PTRUE = [
    # (1000, 50, 0),
    # (1000, 50, 5),
    # (1000, 50, 20),
    (10000, 200, 20),
]
SNR_VALUES = [0.5, 1.0, 2.0]
RHO_VALUES = [0.0, 0.5, 0.9]


def main():
    parser = argparse.ArgumentParser(description="Toeplitz design simulation for variable selection")
    parser.add_argument("--output_dir", type=str, default="./results_toeplitz", help="Output directory")
    parser.add_argument("--n_runs", type=int, default=1, help="Replicates per design cell")
    parser.add_argument("--total_timesteps", type=int, default=10000, help="PPO timesteps per run")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity (0 or 1)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--reward_type",
        type=str,
        default="all",
        choices=("all", "cv_rmse", "aic", "bic", "bayes_factor"),
        help="Reward for RL envs: 'all' (default, loop over all four), cv_rmse, aic, bic, or bayes_factor",
    )
    parser.add_argument(
        "--sequential_gammas",
        type=str,
        default="0.3",
        help="Comma-separated gamma values for Sequential MDP (e.g. '0.3' or '0,0.3,0.9,1'). Default: 0.3",
    )
    args = parser.parse_args()
    sequential_gammas = [float(x.strip()) for x in args.sequential_gammas.split(",")]

    reward_types = ("cv_rmse", "aic", "bic", "bayes_factor") if args.reward_type == "all" else (args.reward_type,)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    total_cells = args.n_runs * len(DESIGN_N_P_PTRUE) * len(SNR_VALUES) * len(RHO_VALUES) * len(reward_types)
    run_count = 0

    for rep in range(args.n_runs):
        design_id = 0
        for (n, p_total, p_true) in DESIGN_N_P_PTRUE:
            for snr in SNR_VALUES:
                for rho in RHO_VALUES:
                    design_id += 1
                    for reward_type in reward_types:
                        run_count += 1
                        rt_offset = {"cv_rmse": 0, "aic": 1, "bic": 2, "bayes_factor": 3}[reward_type]
                        seed = args.seed + design_id * 1000 + rep * 100 + rt_offset
                        if args.verbose:
                            print(f"[{run_count}/{total_cells}] rep={rep+1}, n={n}, p={p_total}, p_true={p_true}, snr={snr}, rho={rho}, reward_type={reward_type}")

                        # Generate data
                        X, y, true_features, beta, sigma = generate_toeplitz_regression(
                            n=n, p_total=p_total, p_true=p_true, snr=snr, rho=rho, random_state=seed
                        )
                        # Train/test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=args.test_size, random_state=seed
                        )

                        # Standardize X (use train mean/std for test)
                        scaler_X = StandardScaler()
                        X_train_scaled = scaler_X.fit_transform(X_train)
                        X_test_scaled = scaler_X.transform(X_test)

                        # Run pipeline
                        res = run_single_simulation_from_data(
                            X_train_scaled,
                            y_train,
                            X_test_scaled,
                            y_test,
                            true_features,
                            total_timesteps=args.total_timesteps,
                            random_state=seed,
                            verbose=args.verbose,
                            reward_type=reward_type,
                            sequential_gammas=sequential_gammas,
                        )

                        # One row per run with design + metrics
                        row = {
                            "design_id": design_id,
                            "run_id": rep + 1,
                            "n": n,
                            "p_total": p_total,
                            "p_true": p_true,
                            "snr": snr,
                            "rho": rho,
                            "sigma_true": sigma,
                            "total_timesteps": args.total_timesteps,
                            "reward_type": res["reward_type"],
                        }
                        _methods = ["bandit"] + [f"sequential_g{g}" for g in sequential_gammas] + ["lasso", "adaptive_lasso", "forward", "backward", "mcmc"]
                        for method in _methods:
                            for suffix in ["n_selected", "test_r2", "test_mse", "precision", "recall", "f1", "runtime_sec"]:
                                row[f"{method}_{suffix}"] = res.get(f"{method}_{suffix}")
                        row["all_features_test_r2"] = res["all_features_test_r2"]
                        row["all_features_test_mse"] = res["all_features_test_mse"]
                        rows.append(row)

                    # Save after each rep (outermost loop)
                    df = pd.DataFrame(rows)
                    out_csv = output_path / "toeplitz_design_results.csv"
                    df.to_csv(out_csv, index=False)

        print(f"Saved {len(rows)} runs to {out_csv}")
        # Summary: mean ± std by (n, p_total, p_true, snr, rho, reward_type)
        group_cols = ["n", "p_total", "p_true", "snr", "rho", "reward_type"]
        agg_spec = {}
        _methods = ["bandit"] + [f"sequential_g{g}" for g in sequential_gammas] + ["lasso", "adaptive_lasso", "forward", "backward", "mcmc"]
        for method in _methods:
            for m in ["n_selected", "test_r2", "test_mse", "f1", "runtime_sec"]:
                col = f"{method}_{m}"
                agg_spec[f"{col}_mean"] = (col, "mean")
                agg_spec[f"{col}_std"] = (col, "std")
        agg_spec["all_features_test_r2_mean"] = ("all_features_test_r2", "mean")
        agg_spec["all_features_test_r2_std"] = ("all_features_test_r2", "std")
        summary = df.groupby(group_cols, as_index=False).agg(**agg_spec)
        summary_csv = output_path / "toeplitz_design_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"Saved summary to {summary_csv}")
    return df, summary


if __name__ == "__main__":
    main()
