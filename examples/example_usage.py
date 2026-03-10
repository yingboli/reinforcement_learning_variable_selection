"""
Example usage script for RL-based variable selection.

This script demonstrates how to use the RL variable selection method
on a synthetic dataset.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env_bandit import VariableSelectionEnv
from agent_bandit import VariableSelectionPPO
from evaluate import evaluate_selection, compare_with_baselines


def main():
    """Example usage of RL variable selection."""
    
    # Generate synthetic data
    print("Generating synthetic dataset...")
    X, y = make_regression(
        n_samples=200,
        n_features=50,
        n_informative=5,  # Only 5 features are truly important
        noise=10.0,
        random_state=42,
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Create environment (bandit problem - one step per episode)
    print("\nCreating RL environment...")
    env = VariableSelectionEnv(
        X_train,
        y_train,
        reward_type="cv_rmse",
        cv_folds=5,
        random_state=42,
    )
    
    # Create agent
    print("Creating PPO agent...")
    agent = VariableSelectionPPO(
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        seed=42,
    )
    
    # Train agent
    print("\nTraining agent...")
    agent.train(total_timesteps=10000, log_interval=10)
    
    # Select features
    print("\nSelecting features...")
    selected_features = agent.select_features(deterministic=True)
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Evaluate selection
    print("\nEvaluating selection...")
    results = evaluate_selection(X_test, y_test, selected_features)
    print(f"Test R²: {results['test_r2']:.4f}")
    print(f"Test MSE: {results['test_mse']:.4f}")
    
    # Compare with baselines
    print("\nComparing with baseline methods...")
    comparison_df = compare_with_baselines(
        X_train, y_train, X_test, y_test, selected_features
    )
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
