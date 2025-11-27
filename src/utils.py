"""
Utility functions for the HSP90 inhibitor screening project.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load an object from a pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def get_data_dir(subdir: str = "") -> Path:
    """
    Get path to data directory.
    
    Args:
        subdir: Subdirectory within data (e.g., 'raw', 'processed')
        
    Returns:
        Path to data directory
    """
    root = get_project_root()
    if subdir:
        return root / "data" / subdir
    return root / "data"


def get_models_dir() -> Path:
    """
    Get path to models directory.
    
    Returns:
        Path to models directory
    """
    return get_project_root() / "models"


def get_figures_dir() -> Path:
    """
    Get path to figures directory.
    
    Returns:
        Path to figures directory
    """
    return get_project_root() / "figures"


def get_output_dir() -> Path:
    """
    Get path to output directory.
    
    Returns:
        Path to output directory
    """
    return get_project_root() / "output"


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                default: float = 0.0) -> np.ndarray:
    """
    Safely divide two arrays, handling division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        default: Default value for division by zero
        
    Returns:
        Result array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result = np.where(np.isfinite(result), result, default)
    return result


def remove_outliers(df: pd.DataFrame, column: str, 
                   method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame column.
    
    Args:
        df: Input DataFrame
        column: Column name to filter
        method: Method to use ('iqr' or 'zscore')
        factor: Factor for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < factor]
    
    else:
        raise ValueError(f"Unknown method: {method}")



