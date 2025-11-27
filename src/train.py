"""
Model training module for regression and classification.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

from src.utils import get_models_dir, save_pickle, ensure_dir

logger = logging.getLogger(__name__)


def prepare_features_and_targets(features_df: pd.DataFrame,
                                 target_col: str = 'target',
                                 binary_target_col: str = 'target_binary') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare feature matrix and target vectors.
    
    Args:
        features_df: DataFrame with features and targets
        target_col: Name of continuous target column
        binary_target_col: Name of binary target column
        
    Returns:
        Tuple of (X, y_regression, y_classification)
    """
    # Exclude non-feature columns
    exclude_cols = ['target', 'target_binary', 'compound_id', 'smiles', 
                   'aggregation_reduction', 'is_effective']
    
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X = features_df[feature_cols].copy()
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Get targets
    y_reg = features_df[target_col].copy() if target_col in features_df.columns else None
    y_clf = features_df[binary_target_col].copy() if binary_target_col in features_df.columns else None
    
    logger.info(f"Feature matrix shape: {X.shape}")
    if y_reg is not None:
        logger.info(f"Regression target: {len(y_reg)} samples")
    if y_clf is not None:
        logger.info(f"Classification target: {len(y_clf)} samples, "
                   f"positive class: {y_clf.sum()} ({y_clf.mean()*100:.1f}%)")
    
    return X, y_reg, y_clf


def train_regression_models(X: pd.DataFrame, 
                           y: pd.Series,
                           cv_folds: int = 5,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Train simple linear regression model.
    
    Args:
        X: Feature matrix
        y: Target vector
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary of trained models and their CV scores
    """
    logger.info("Training simple linear regression model...")
    
    models = {}
    cv_scores = {}
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Simple Linear Regression
    lr_reg = LinearRegression()
    lr_reg.fit(X_scaled, y)
    models['linear_regressor'] = lr_reg
    models['linear_regressor_scaler'] = scaler
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    lr_scores = cross_val_score(lr_reg, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
    cv_scores['linear_regressor'] = {
        'rmse': np.sqrt(-lr_scores),
        'mean_rmse': np.sqrt(-lr_scores.mean()),
        'std_rmse': np.sqrt(lr_scores.std())
    }
    logger.info(f"Linear Regressor CV RMSE: {cv_scores['linear_regressor']['mean_rmse']:.3f} ± "
               f"{cv_scores['linear_regressor']['std_rmse']:.3f}")
    
    return {'models': models, 'cv_scores': cv_scores}


def train_classification_models(X: pd.DataFrame,
                                y: pd.Series,
                                cv_folds: int = 5,
                                random_state: int = 42) -> Dict[str, Any]:
    """
    Train simple logistic regression model for drug detection.
    
    Args:
        X: Feature matrix
        y: Binary target vector (effective vs ineffective)
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary of trained models and their CV scores
    """
    logger.info("Training simple logistic regression model for drug detection...")
    
    models = {}
    cv_scores = {}
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Simple Logistic Regression
    lr_clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'
    )
    lr_clf.fit(X_scaled, y)
    models['logistic_classifier'] = lr_clf
    models['logistic_classifier_scaler'] = scaler
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    lr_scores = cross_val_score(lr_clf, X_scaled, y, cv=cv, scoring='roc_auc')
    cv_scores['logistic_classifier'] = {
        'roc_auc': lr_scores,
        'mean_roc_auc': lr_scores.mean(),
        'std_roc_auc': lr_scores.std()
    }
    logger.info(f"Logistic Classifier CV ROC-AUC: {cv_scores['logistic_classifier']['mean_roc_auc']:.3f} ± "
               f"{cv_scores['logistic_classifier']['std_roc_auc']:.3f}")
    
    return {'models': models, 'cv_scores': cv_scores}


def train_all_models(features_df: pd.DataFrame,
                    target_col: str = 'target',
                    binary_target_col: str = 'target_binary',
                    output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Train all regression and classification models.
    
    Args:
        features_df: DataFrame with features and targets
        target_col: Name of continuous target column
        binary_target_col: Name of binary target column
        output_dir: Directory to save models
        
    Returns:
        Dictionary containing all models and scores
    """
    if output_dir is None:
        output_dir = get_models_dir()
    ensure_dir(output_dir)
    
    # Prepare data
    X, y_reg, y_clf = prepare_features_and_targets(features_df, target_col, binary_target_col)
    
    results = {
        'feature_names': list(X.columns),
        'n_samples': len(X),
        'n_features': len(X.columns)
    }
    
    # Train regression models
    if y_reg is not None and not y_reg.isna().all():
        reg_results = train_regression_models(X, y_reg)
        results['regression'] = reg_results
        
        # Save regression models
        for name, model in reg_results['models'].items():
            if name.endswith('_scaler'):
                continue
            model_path = output_dir / f"{name}.pkl"
            save_pickle(model, str(model_path))
            logger.info(f"Saved {name} to {model_path}")
        
        # Save scalers if they exist
        if 'linear_regressor_scaler' in reg_results['models']:
            scaler_path = output_dir / "linear_regressor_scaler.pkl"
            save_pickle(reg_results['models']['linear_regressor_scaler'], str(scaler_path))
    
    # Train classification models
    if y_clf is not None and not y_clf.isna().all():
        clf_results = train_classification_models(X, y_clf)
        results['classification'] = clf_results
        
        # Save classification models
        for name, model in clf_results['models'].items():
            if name.endswith('_scaler'):
                continue
            model_path = output_dir / f"{name}.pkl"
            save_pickle(model, str(model_path))
            logger.info(f"Saved {name} to {model_path}")
        
        # Save scalers if they exist
        if 'logistic_classifier_scaler' in clf_results['models']:
            scaler_path = output_dir / "logistic_classifier_scaler.pkl"
            save_pickle(clf_results['models']['logistic_classifier_scaler'], str(scaler_path))
    
    # Save feature names for later use
    feature_names_path = output_dir / "feature_names.pkl"
    save_pickle(results['feature_names'], str(feature_names_path))
    
    return results


