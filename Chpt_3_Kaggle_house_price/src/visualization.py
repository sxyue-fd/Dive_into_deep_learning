"""
Copyright (c) 2025
此项目基于 MIT License 进行许可
有关详细信息，请参阅项目根目录中的 LICENSE 文件
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

def create_training_plots(config, session_id, train_losses, val_losses, fold):
    """Create training progress visualization
    
    Args:
        config: Configuration dictionary
        session_id: Training session ID (not used for file organization)
        train_losses: List of training losses
        val_losses: List of validation losses
        fold: Current fold number
    """
    # Setup output directory
    vis_dir = Path(config['paths']['visualization_dir'])
    vis_dir.mkdir(parents=True, exist_ok=True)
    output_path = vis_dir / 'training_progress.png'
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss
    plt.plot(epochs, train_losses, label='Training Loss', color='#1f77b4')
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training Progress - Fold {fold + 1}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Training plot updated: {output_path}")

def plot_predictions(predictions, actual, output_dir):
    """Create prediction analysis plots
    
    Args:
        predictions: Model predictions
        actual: Actual values
        output_dir: Output directory path
    """
    output_path = output_dir / 'predictions_analysis.png'
    
    plt.figure(figsize=(12, 5))
    
    # Predictions vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(actual, predictions, alpha=0.5)
    min_val = min(min(actual), min(predictions))
    max_val = max(max(actual), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Actual')
    plt.legend()
    
    # Residuals
    plt.subplot(1, 2, 2)
    residuals = predictions - actual
    plt.scatter(actual, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Prediction analysis updated: {output_path}")

def plot_feature_importance(dataset, output_dir, n_features=10):
    """Create feature importance visualization
    
    Args:
        dataset: HousePriceDataset instance
        output_dir: Output directory path
        n_features: Number of top features to show
    """
    output_path = output_dir / 'feature_importance.png'
    
    train_data = pd.read_csv(dataset.train_file)
    
    if not dataset.numerical_features:
        return
        
    # Calculate correlations with target
    correlations = train_data[dataset.numerical_features].corrwith(train_data['SalePrice'])
    correlations = correlations.sort_values(ascending=False)
    
    # Plot top feature correlations
    plt.figure(figsize=(10, 6))
    correlations[:n_features].plot(kind='bar')
    plt.title('Top Feature Correlations with Price')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Feature importance plot updated: {output_path}")

def create_model_evaluation(config, session_id, predictions, actual, dataset):
    """Create all evaluation plots for a model
    
    Args:
        config: Configuration dictionary
        session_id: Training session ID (not used for file organization)
        predictions: Model predictions
        actual: Actual values
        dataset: HousePriceDataset instance
    """
    vis_dir = Path(config['paths']['visualization_dir'])
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Predictions analysis
    plot_predictions(predictions, actual, vis_dir)
    
    # Feature importance
    plot_feature_importance(dataset, vis_dir)