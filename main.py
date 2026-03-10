"""
Main training script for reinforcement learning-based variable selection.

This script provides a complete pipeline for training and evaluating the RL
variable selection method on various datasets.
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
from sklearn.metrics import mean_squared_error, r2_score

from env_bandit import VariableSelectionEnv
from agent_bandit import VariableSelectionPPO
from evaluate import evaluate_selection, compare_with_baselines, plot_selection_history


def load_data(
    dataset_name: str,
    task: str = "regression",
    n_samples: int = None,
    n_features: int = None,
    noise: float = 10.0,
):
    """
    Load or generate dataset.
    Returns X, y. For classification, y is integer labels.
    """
    if task == "regression":
        if dataset_name == "synthetic":
            n_samples = n_samples or 200
            n_features = n_features or 50
            n_informative = max(1, n_features // 10)
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                noise=noise,
                random_state=42,
            )
            return X, y
        elif dataset_name == "diabetes":
            data = load_diabetes()
            return data.data, data.target
        elif dataset_name == "california":
            data = fetch_california_housing()
            return data.data, data.target
        else:
            raise ValueError(f"Unknown regression dataset: {dataset_name}")
    else:
        if dataset_name == "synthetic":
            n_samples = n_samples or 200
            n_features = n_features or 50
            n_informative = max(1, n_features // 10)
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=min(5, n_features // 5),
                random_state=42,
            )
            return X, y
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
            return data.data, data.target
        else:
            raise ValueError(f"Unknown classification dataset: {dataset_name}. Use 'synthetic' or 'breast_cancer'.")


def train_rl_agent(
    X_train,
    y_train,
    X_val,
    y_val,
    task: str = "regression",
    reward_type: str = None,
    total_timesteps: int = 10000,
    learning_rate: float = 3e-4,
    verbose: int = 1,
    save_path: str = None,
):
    """
    Train RL agent for variable selection (bandit problem).
    task: 'regression' or 'classification'. reward_type default: 'cv_rmse' / 'cv_auc'.
    """
    if reward_type is None:
        reward_type = "cv_rmse" if task == "regression" else "cv_auc"
    env = VariableSelectionEnv(
        X_train,
        y_train,
        task=task,
        reward_type=reward_type,
        cv_folds=5,
        random_state=42,
    )
    
    # Create agent
    agent = VariableSelectionPPO(
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=verbose,
        seed=42,
    )
    
    # Train agent
    print(f"Training RL agent for {total_timesteps} timesteps...")
    agent.train(total_timesteps=total_timesteps, log_interval=10)
    
    # Save if path provided
    if save_path:
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    
    # Evaluate on validation set periodically during training
    # (For simplicity, we'll evaluate once at the end)
    # In practice, you might want to evaluate periodically and save best model
    
    return agent, env


def evaluate_agent(
    agent,
    env,
    X_train,
    y_train,
    X_test,
    y_test,
    task: str = "regression",
    deterministic: bool = True,
):
    """Evaluate trained agent: select features, then evaluate on test set with task-appropriate metrics."""
    selected_features = agent.select_features(deterministic=deterministic)
    results = evaluate_selection(
        X_test, y_test, selected_features, task=task
    )
    return selected_features, results


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for variable selection"
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
        help="Dataset: for regression use synthetic, diabetes, california; for classification use synthetic, breast_cancer",
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
        "--reward_type",
        type=str,
        default=None,
        help="Reward: regression: cv_rmse, aic, bic, bayes_factor; classification: cv_auc, aic, bic",
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
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Path to save trained model",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading dataset: {args.dataset} (task={args.task})")
    X, y = load_data(
        args.dataset,
        task=args.task,
        n_samples=args.n_samples,
        n_features=args.n_features,
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    # Standardize features only (always)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    # Standardize target only for regression
    if args.task == "regression":
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train RL agent
    agent, env = train_rl_agent(
        X_train, y_train, X_val, y_val,
        task=args.task,
        reward_type=args.reward_type,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        save_path=args.save_model,
    )
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_features, val_results = evaluate_agent(
        agent, env, X_train, y_train, X_val, y_val, task=args.task
    )
    print(f"Validation Results:")
    print(f"  Selected features: {len(val_features)}")
    if args.task == "regression":
        print(f"  Test R²: {val_results['test_r2']:.4f}, MSE: {val_results['test_mse']:.4f}")
    else:
        print(f"  Test F1: {val_results['test_f1']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_features, test_results = evaluate_agent(
        agent, env, X_train, y_train, X_test, y_test, task=args.task
    )
    print(f"Test Results:")
    print(f"  Selected features: {len(test_features)}")
    if args.task == "regression":
        print(f"  Test R²: {test_results['test_r2']:.4f}, MSE: {test_results['test_mse']:.4f}")
    else:
        print(f"  Test F1: {test_results['test_f1']:.4f}")
    print(f"  Selected feature indices: {test_features.tolist()}")
    
    # Compare with baselines
    print("\nComparing with baseline methods...")
    comparison_df = compare_with_baselines(
        X_train, y_train, X_test, y_test, test_features, task=args.task
    )
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    # Save results
    comparison_df.to_csv(output_dir / "comparison_results.csv", index=False)
    
    # Save selected features
    np.save(output_dir / "selected_features.npy", test_features)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
