"""
Comparison script for bandit vs sequential MDP variable selection.

This script trains and evaluates both approaches for comparison.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.datasets import (
    make_regression,
    make_classification,
    load_diabetes,
    fetch_california_housing,
    load_breast_cancer,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from env_bandit import VariableSelectionEnv
from agent_bandit import VariableSelectionPPO
from env_sequential import SequentialVariableSelectionEnv
from agent_sequential import SequentialVariableSelectionPPO
from evaluate import evaluate_selection, compare_with_baselines


def load_data(
    dataset_name: str,
    task: str = "regression",
    n_samples: int = None,
    n_features: int = None,
    noise: float = 10.0,
):
    """Load or generate dataset. task: 'regression' or 'classification'."""
    if task == "regression":
        if dataset_name == "synthetic":
            n_samples = n_samples or 200
            n_features = n_features or 50
            n_informative = max(1, n_features // 10)
            X, y = make_regression(
                n_samples=n_samples, n_features=n_features,
                n_informative=n_informative, noise=noise, random_state=42,
            )
            return X, y
        elif dataset_name == "diabetes":
            data = load_diabetes()
            return data.data, data.target
        elif dataset_name == "california":
            data = fetch_california_housing()
            return data.data, data.target
        raise ValueError(f"Unknown regression dataset: {dataset_name}")
    else:
        if dataset_name == "synthetic":
            n_samples = n_samples or 200
            n_features = n_features or 50
            n_informative = max(1, n_features // 10)
            X, y = make_classification(
                n_samples=n_samples, n_features=n_features,
                n_informative=n_informative, n_redundant=min(5, n_features // 5),
                random_state=42,
            )
            return X, y
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
            return data.data, data.target
        raise ValueError(f"Unknown classification dataset: {dataset_name}")


def train_and_evaluate_bandit(
    X_train, y_train, X_test, y_test,
    task="regression",
    sparsity_penalty=0.01,
    reward_type=None,
    total_timesteps=10000,
    learning_rate=3e-4,
):
    """Train and evaluate bandit approach."""
    if reward_type is None:
        reward_type = "r2" if task == "regression" else "accuracy"
    print("\n" + "="*60)
    print("BANDIT APPROACH (One-step MDP)")
    print("="*60)
    
    env = VariableSelectionEnv(
        X_train, y_train,
        task=task,
        sparsity_penalty=sparsity_penalty,
        reward_type=reward_type,
        use_cv=True, cv_folds=3, random_state=42,
    )
    
    # Create agent
    agent = VariableSelectionPPO(
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        seed=42,
    )
    
    # Train
    print(f"\nTraining bandit agent for {total_timesteps} timesteps...")
    agent.train(total_timesteps=total_timesteps, log_interval=10)
    
    # Select features
    selected_features = agent.select_features(deterministic=True)
    
    results = evaluate_selection(
        X_train, y_train, X_test, y_test, selected_features, task=task
    )
    return selected_features, results, agent, env


def train_and_evaluate_sequential(
    X_train, y_train, X_test, y_test,
    task="regression",
    sparsity_penalty=0.01,
    reward_type=None,
    total_timesteps=10000,
    learning_rate=3e-4,
    max_episode_steps=None,
    action_type="toggle",
):
    """Train and evaluate sequential MDP approach."""
    if reward_type is None:
        reward_type = "r2" if task == "regression" else "accuracy"
    print("\n" + "="*60)
    print("SEQUENTIAL MDP APPROACH (Multi-step, gamma=0)")
    print("="*60)
    
    env = SequentialVariableSelectionEnv(
        X_train, y_train,
        task=task,
        sparsity_penalty=sparsity_penalty,
        reward_type=reward_type,
        use_cv=True, cv_folds=3,
        max_episode_steps=max_episode_steps,
        action_type=action_type, random_state=42,
    )
    
    # Create agent
    agent = SequentialVariableSelectionPPO(
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.0,  # No future discounting
        verbose=1,
        seed=42,
    )
    
    # Train
    print(f"\nTraining sequential agent for {total_timesteps} timesteps...")
    agent.train(total_timesteps=total_timesteps, log_interval=10)
    
    # Select features (runs full episode)
    selected_features = agent.select_features(deterministic=True)
    
    # Evaluate
    results = evaluate_selection(
        X_train, y_train, X_test, y_test, selected_features, task=task
    )
    return selected_features, results, agent, env


def main():
    """Main comparison pipeline."""
    parser = argparse.ArgumentParser(
        description="Compare bandit vs sequential MDP variable selection"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset: regression: synthetic, diabetes, california; classification: synthetic, breast_cancer",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of samples (for synthetic data)",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=50,
        help="Number of features (for synthetic data)",
    )
    parser.add_argument(
        "--sparsity_penalty",
        type=float,
        default=0.01,
        help="Sparsity penalty coefficient",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default=None,
        help="Reward: regression: r2, mse; classification: accuracy, f1_weighted, roc_auc",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=10000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Max steps per episode for sequential (default: auto, scales with n_features)",
    )
    parser.add_argument(
        "--action_type",
        type=str,
        default="toggle",
        choices=["toggle", "add_remove"],
        help="Action type for sequential (toggle or add_remove)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_comparison",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading dataset: {args.dataset} (task={args.task})")
    X, y = load_data(
        args.dataset, task=args.task,
        n_samples=args.n_samples, n_features=args.n_features,
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    if args.task == "regression":
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    bandit_features, bandit_results, bandit_agent, bandit_env = train_and_evaluate_bandit(
        X_train, y_train, X_test, y_test,
        task=args.task,
        sparsity_penalty=args.sparsity_penalty,
        reward_type=args.reward_type,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
    )
    
    print("\nBandit Results:")
    print(f"  Selected features: {len(bandit_features)}")
    if args.task == "regression":
        print(f"  Test R²: {bandit_results['test_r2']:.4f}, MSE: {bandit_results['test_mse']:.4f}")
        print(f"  CV R²: {bandit_results['cv_r2_mean']:.4f} ± {bandit_results['cv_r2_std']:.4f}")
    else:
        print(f"  Test accuracy: {bandit_results['test_accuracy']:.4f}, F1: {bandit_results['test_f1']:.4f}")
        print(f"  CV accuracy: {bandit_results['cv_accuracy_mean']:.4f} ± {bandit_results['cv_accuracy_std']:.4f}")
    print(f"  Feature indices: {bandit_features.tolist()}")
    
    seq_features, seq_results, seq_agent, seq_env = train_and_evaluate_sequential(
        X_train, y_train, X_test, y_test,
        task=args.task,
        sparsity_penalty=args.sparsity_penalty,
        reward_type=args.reward_type,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        max_episode_steps=args.max_episode_steps,
        action_type=args.action_type,
    )
    
    print("\nSequential MDP Results:")
    print(f"  Selected features: {len(seq_features)}")
    if args.task == "regression":
        print(f"  Test R²: {seq_results['test_r2']:.4f}, MSE: {seq_results['test_mse']:.4f}")
        print(f"  CV R²: {seq_results['cv_r2_mean']:.4f} ± {seq_results['cv_r2_std']:.4f}")
    else:
        print(f"  Test accuracy: {seq_results['test_accuracy']:.4f}, F1: {seq_results['test_f1']:.4f}")
        print(f"  CV accuracy: {seq_results['cv_accuracy_mean']:.4f} ± {seq_results['cv_accuracy_std']:.4f}")
    print(f"  Feature indices: {seq_features.tolist()}")
    
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINES")
    print("="*60)
    
    bandit_comparison = compare_with_baselines(
        X_train, y_train, X_test, y_test, bandit_features, task=args.task
    )
    print("\nBandit vs Baselines:")
    print(bandit_comparison.to_string(index=False))
    
    seq_comparison = compare_with_baselines(
        X_train, y_train, X_test, y_test, seq_features, task=args.task
    )
    print("\nSequential vs Baselines:")
    print(seq_comparison.to_string(index=False))
    
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    if args.task == "regression":
        comparison_summary = pd.DataFrame({
            "Method": ["Bandit (One-step)", "Sequential MDP (Multi-step)"],
            "n_features": [len(bandit_features), len(seq_features)],
            "test_r2": [bandit_results["test_r2"], seq_results["test_r2"]],
            "test_mse": [bandit_results["test_mse"], seq_results["test_mse"]],
            "cv_r2_mean": [bandit_results["cv_r2_mean"], seq_results["cv_r2_mean"]],
            "cv_r2_std": [bandit_results["cv_r2_std"], seq_results["cv_r2_std"]],
        })
    else:
        comparison_summary = pd.DataFrame({
            "Method": ["Bandit (One-step)", "Sequential MDP (Multi-step)"],
            "n_features": [len(bandit_features), len(seq_features)],
            "test_accuracy": [bandit_results["test_accuracy"], seq_results["test_accuracy"]],
            "test_f1": [bandit_results["test_f1"], seq_results["test_f1"]],
            "cv_accuracy_mean": [bandit_results["cv_accuracy_mean"], seq_results["cv_accuracy_mean"]],
            "cv_accuracy_std": [bandit_results["cv_accuracy_std"], seq_results["cv_accuracy_std"]],
        })
    print(comparison_summary.to_string(index=False))
    
    # Save results
    comparison_summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    bandit_comparison.to_csv(output_dir / "bandit_vs_baselines.csv", index=False)
    seq_comparison.to_csv(output_dir / "sequential_vs_baselines.csv", index=False)
    
    np.save(output_dir / "bandit_features.npy", bandit_features)
    np.save(output_dir / "sequential_features.npy", seq_features)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
