"""
Model evaluation and visualization module.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

from src.utils import get_figures_dir, get_models_dir, load_pickle, ensure_dir
from src.train import prepare_features_and_targets

logger = logging.getLogger(__name__)


def evaluate_regression_model(model: Any, X: pd.DataFrame, y: pd.Series,
                              model_name: str = "model") -> Dict[str, float]:
    """
    Evaluate a regression model.
    
    Args:
        model: Trained regression model
        X: Feature matrix
        y: True target values
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    logger.info(f"{model_name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    
    return metrics, y_pred


def evaluate_classification_model(model: Any, X: pd.DataFrame, y: pd.Series,
                                  model_name: str = "model",
                                  scaler: Optional[Any] = None) -> Dict[str, float]:
    """
    Evaluate a classification model.
    
    Args:
        model: Trained classification model
        X: Feature matrix
        y: True binary labels
        model_name: Name of the model
        scaler: Optional scaler for preprocessing
        
    Returns:
        Dictionary of evaluation metrics
    """
    if scaler is not None:
        X = scaler.transform(X)
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
    
    return metrics, y_pred, y_pred_proba


def plot_feature_importance(model: Any, feature_names: list, 
                           model_name: str, output_dir: Path,
                           top_n: int = 20) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
        output_dir: Directory to save plot
        top_n: Number of top features to show
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"{model_name} does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Features - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_file = output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to {output_file}")


def plot_correlation_heatmap(features_df: pd.DataFrame, output_dir: Path,
                            max_features: int = 50) -> None:
    """
    Plot correlation heatmap of features.
    
    Args:
        features_df: DataFrame with features
        output_dir: Directory to save plot
        max_features: Maximum number of features to include
    """
    # Select numeric columns only
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    # Exclude fingerprint columns for readability
    feature_cols = [col for col in numeric_cols 
                   if not col.startswith('morgan_') and not col.startswith('maccs_')]
    
    if len(feature_cols) > max_features:
        # Select top features by variance
        variances = features_df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(max_features).index.tolist()
    
    corr_matrix = features_df[feature_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    output_file = output_dir / "correlation_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correlation heatmap to {output_file}")


def plot_roc_curves(models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                   scalers: Optional[Dict[str, Any]] = None,
                   output_dir: Path = None) -> None:
    """
    Plot ROC curves for classification models.
    
    Args:
        models: Dictionary of model names and models
        X: Feature matrix
        y: True binary labels
        scalers: Optional dictionary of scalers
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if scalers and name in scalers:
            X_scaled = scalers[name].transform(X)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X)[:, 1]
        
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Classification Models')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_file = output_dir / "roc_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curves to {output_file}")


def plot_predicted_vs_actual(models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                            output_dir: Path, scalers: Optional[Dict[str, Any]] = None) -> None:
    """
    Plot predicted vs actual values for regression models.
    
    Args:
        models: Dictionary of model names and models
        X: Feature matrix
        y: True target values
        output_dir: Directory to save plot
        scalers: Optional dictionary of scalers
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        # Apply scaler if available
        if scalers and name in scalers:
            X_scaled = scalers[name].transform(X)
            y_pred = model.predict(X_scaled)
        else:
            y_pred = model.predict(X)
        
        axes[idx].scatter(y, y_pred, alpha=0.6)
        axes[idx].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[idx].set_xlabel('Actual')
        axes[idx].set_ylabel('Predicted')
        axes[idx].set_title(f'{name}')
        axes[idx].grid(alpha=0.3)
        
        r2 = r2_score(y, y_pred)
        axes[idx].text(0.05, 0.95, f'R² = {r2:.3f}', 
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / "predicted_vs_actual.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved predicted vs actual plot to {output_file}")


def evaluate_all_models(features_df: pd.DataFrame,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Evaluate all trained models and generate visualizations.
    
    Args:
        features_df: DataFrame with features and targets
        test_size: Proportion of data for testing
        random_state: Random seed
        output_dir: Directory to save figures
        
    Returns:
        Dictionary of evaluation results
    """
    if output_dir is None:
        output_dir = get_figures_dir()
    ensure_dir(output_dir)
    
    # Prepare data
    X, y_reg, y_clf = prepare_features_and_targets(features_df)
    
    results = {}
    
    # Split data
    if y_reg is not None and not y_reg.isna().all():
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X, y_reg, test_size=test_size, random_state=random_state
        )
    else:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = None, None, None, None
    
    if y_clf is not None and not y_clf.isna().all():
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X, y_clf, test_size=test_size, random_state=random_state
        )
    else:
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = None, None, None, None
    
    # Load models
    models_dir = get_models_dir()
    
    # Evaluate regression models (simple linear regression)
    if X_test_reg is not None:
        reg_models = {}
        reg_results = {}
        
        for model_name in ['linear_regressor']:
            model_path = models_dir / f"{model_name}.pkl"
            if model_path.exists():
                model = load_pickle(str(model_path))
                reg_models[model_name] = model
                
                # Handle scaler for linear regression
                scaler_path = models_dir / "linear_regressor_scaler.pkl"
                if scaler_path.exists():
                    scaler = load_pickle(str(scaler_path))
                    X_test_scaled = scaler.transform(X_test_reg)
                    metrics, y_pred = evaluate_regression_model(
                        model, X_test_scaled, y_test_reg, model_name
                    )
                else:
                    metrics, y_pred = evaluate_regression_model(
                        model, X_test_reg, y_test_reg, model_name
                    )
                
                reg_results[model_name] = metrics
        
        results['regression'] = reg_results
        
        # Generate regression plots
        if reg_models:
            reg_scalers = {}
            scaler_path = models_dir / "linear_regressor_scaler.pkl"
            if scaler_path.exists():
                reg_scalers['linear_regressor'] = load_pickle(str(scaler_path))
            plot_predicted_vs_actual(reg_models, X_test_reg, y_test_reg, output_dir, 
                                    reg_scalers if reg_scalers else None)
            
            # Feature importance for linear models (coefficients)
            feature_names = load_pickle(str(models_dir / "feature_names.pkl"))
            for name, model in reg_models.items():
                if hasattr(model, 'coef_'):
                    # Plot coefficient importance for linear models
                    coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                    indices = np.argsort(np.abs(coef))[::-1][:20]
                    
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(20), coef[indices])
                    plt.yticks(range(20), [feature_names[i] for i in indices])
                    plt.xlabel('Coefficient Value')
                    plt.title(f'Top 20 Feature Coefficients - {name}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    
                    output_file = output_dir / f"feature_coefficients_{name.lower().replace(' ', '_')}.png"
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved feature coefficients plot to {output_file}")
    
    # Evaluate classification models (simple logistic regression)
    if X_test_clf is not None:
        clf_models = {}
        clf_scalers = {}
        clf_results = {}
        
        for model_name in ['logistic_classifier']:
            model_path = models_dir / f"{model_name}.pkl"
            if model_path.exists():
                model = load_pickle(str(model_path))
                clf_models[model_name] = model
                
                # Handle scaler for logistic regression
                scaler_path = models_dir / "logistic_classifier_scaler.pkl"
                if scaler_path.exists():
                    scaler = load_pickle(str(scaler_path))
                    clf_scalers[model_name] = scaler
                    X_test_scaled = scaler.transform(X_test_clf)
                    metrics, y_pred, y_pred_proba = evaluate_classification_model(
                        model, X_test_scaled, y_test_clf, model_name, scaler=scaler
                    )
                else:
                    metrics, y_pred, y_pred_proba = evaluate_classification_model(
                        model, X_test_clf, y_test_clf, model_name
                    )
                
                clf_results[model_name] = metrics
        
        results['classification'] = clf_results
        
        # Generate classification plots
        if clf_models:
            plot_roc_curves(clf_models, X_test_clf, y_test_clf, 
                          clf_scalers if clf_scalers else None, output_dir)
            
            # Feature importance for linear models (coefficients)
            feature_names = load_pickle(str(models_dir / "feature_names.pkl"))
            for name, model in clf_models.items():
                if hasattr(model, 'coef_'):
                    # Plot coefficient importance for logistic regression
                    coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                    indices = np.argsort(np.abs(coef))[::-1][:20]
                    
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(20), coef[indices])
                    plt.yticks(range(20), [feature_names[i] for i in indices])
                    plt.xlabel('Coefficient Value')
                    plt.title(f'Top 20 Feature Coefficients - {name}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    
                    output_file = output_dir / f"feature_coefficients_{name.lower().replace(' ', '_')}.png"
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved feature coefficients plot to {output_file}")
    
    # Generate correlation heatmap
    plot_correlation_heatmap(features_df, output_dir)
    
    return results


