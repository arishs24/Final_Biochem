"""
Data preprocessing module for merging HSP90 and α-synuclein datasets.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs

from src.utils import get_data_dir, ensure_dir

logger = logging.getLogger(__name__)


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two SMILES strings.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        Tanimoto similarity (0-1), or 0.0 if invalid
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        logger.debug(f"Error calculating similarity: {e}")
        return 0.0


def merge_datasets(hsp90_df: pd.DataFrame, 
                  alpha_syn_df: pd.DataFrame,
                  pink1_df: Optional[pd.DataFrame] = None,
                  similarity_threshold: float = 0.7) -> pd.DataFrame:
    """
    Merge HSP90, α-synuclein, and PINK1 datasets using SMILES matching.
    
    First tries exact SMILES match, then falls back to similarity matching.
    
    Args:
        hsp90_df: DataFrame with HSP90 inhibitor data
        alpha_syn_df: DataFrame with α-synuclein aggregation data
        pink1_df: Optional DataFrame with PINK1 data
        similarity_threshold: Minimum Tanimoto similarity for matching (default: 0.7)
        
    Returns:
        Merged DataFrame
    """
    logger.info("Merging datasets (HSP90 + α-synuclein + PINK1)...")
    
    # Ensure SMILES columns exist
    if 'smiles' not in hsp90_df.columns:
        raise ValueError("HSP90 dataset must have 'smiles' column")
    if 'smiles' not in alpha_syn_df.columns:
        raise ValueError("α-synuclein dataset must have 'smiles' column")
    
    # Clean SMILES (remove whitespace, standardize)
    hsp90_df = hsp90_df.copy()
    alpha_syn_df = alpha_syn_df.copy()
    
    hsp90_df['smiles'] = hsp90_df['smiles'].astype(str).str.strip()
    alpha_syn_df['smiles'] = alpha_syn_df['smiles'].astype(str).str.strip()
    
    # Remove invalid SMILES
    hsp90_df = hsp90_df[hsp90_df['smiles'] != 'nan']
    alpha_syn_df = alpha_syn_df[alpha_syn_df['smiles'] != 'nan']
    
    # Step 1: Merge HSP90 and α-synuclein first
    logger.info("Merging HSP90 and α-synuclein datasets...")
    merged_exact = pd.merge(
        hsp90_df,
        alpha_syn_df,
        on='smiles',
        how='inner',
        suffixes=('_hsp90', '_alpha_syn')
    )
    
    logger.info(f"Found {len(merged_exact)} exact SMILES matches between HSP90 and α-synuclein")
    
    # Step 2: Merge with PINK1 data if available
    if pink1_df is not None and len(pink1_df) > 0:
        pink1_df = pink1_df.copy()
        pink1_df['smiles'] = pink1_df['smiles'].astype(str).str.strip()
        pink1_df = pink1_df[pink1_df['smiles'] != 'nan']
        
        logger.info("Merging with PINK1 data...")
        merged_exact = pd.merge(
            merged_exact,
            pink1_df,
            on='smiles',
            how='left',
            suffixes=('', '_pink1')
        )
        logger.info(f"PINK1 data merged: {merged_exact['pink1_ic50'].notna().sum()} compounds with PINK1 data")
    
    # Step 2: Similarity matching for remaining compounds
    logger.info("Attempting similarity-based matching...")
    
    # Get compounds not matched exactly
    matched_smiles = set(merged_exact['smiles'].unique())
    hsp90_unmatched = hsp90_df[~hsp90_df['smiles'].isin(matched_smiles)].copy()
    alpha_syn_unmatched = alpha_syn_df[~alpha_syn_df['smiles'].isin(matched_smiles)].copy()
    
    similarity_matches = []
    
    for idx_hsp90, row_hsp90 in hsp90_unmatched.iterrows():
        best_match = None
        best_similarity = 0.0
        
        for idx_alpha, row_alpha in alpha_syn_unmatched.iterrows():
            similarity = calculate_tanimoto_similarity(
                row_hsp90['smiles'],
                row_alpha['smiles']
            )
            
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = (idx_hsp90, idx_alpha, similarity)
        
        if best_match:
            idx_hsp90, idx_alpha, sim = best_match
            # Merge the rows
            merged_row = {**row_hsp90.to_dict(), **row_alpha.to_dict()}
            merged_row['smiles'] = row_hsp90['smiles']  # Use HSP90 SMILES
            merged_row['similarity_score'] = sim
            similarity_matches.append(merged_row)
            
            # Remove matched α-syn compound to avoid duplicate matches
            alpha_syn_unmatched = alpha_syn_unmatched.drop(idx_alpha)
    
    if similarity_matches:
        merged_similarity = pd.DataFrame(similarity_matches)
        logger.info(f"Found {len(merged_similarity)} similarity-based matches")
        
        # Combine exact and similarity matches
        merged = pd.concat([merged_exact, merged_similarity], ignore_index=True)
    else:
        merged = merged_exact
    
    logger.info(f"Total merged compounds: {len(merged)}")
    
    # Add similarity_score = 1.0 for exact matches if not present
    if 'similarity_score' not in merged.columns:
        merged['similarity_score'] = 1.0
    else:
        merged['similarity_score'] = merged['similarity_score'].fillna(1.0)
    
    return merged


def clean_merged_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare merged dataset for feature engineering.
    
    Args:
        merged_df: Merged DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning merged data...")
    
    df = merged_df.copy()
    
    # Remove rows with missing critical columns
    required_cols = ['smiles', 'aggregation_reduction']
    df = df.dropna(subset=required_cols)
    
    # Handle missing kinetic parameters
    if 'k_on' in df.columns:
        df['k_on'] = pd.to_numeric(df['k_on'], errors='coerce')
        df['k_on'] = df['k_on'].fillna(df['k_on'].median())
    
    if 'k_off' in df.columns:
        df['k_off'] = pd.to_numeric(df['k_off'], errors='coerce')
        df['k_off'] = df['k_off'].fillna(df['k_off'].median())
    
    # Ensure aggregation_reduction is numeric
    df['aggregation_reduction'] = pd.to_numeric(df['aggregation_reduction'], errors='coerce')
    
    # Remove extreme outliers in aggregation_reduction
    Q1 = df['aggregation_reduction'].quantile(0.25)
    Q3 = df['aggregation_reduction'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    before = len(df)
    df = df[(df['aggregation_reduction'] >= lower_bound) & 
            (df['aggregation_reduction'] <= upper_bound)]
    after = len(df)
    
    if before != after:
        logger.info(f"Removed {before - after} outliers from aggregation_reduction")
    
    # Create binary target (effective vs ineffective)
    # Threshold: > 20% reduction is considered effective
    df['is_effective'] = (df['aggregation_reduction'] > 20).astype(int)
    
    logger.info(f"Cleaned dataset: {len(df)} compounds")
    logger.info(f"Effective compounds: {df['is_effective'].sum()} ({df['is_effective'].mean()*100:.1f}%)")
    
    return df


def preprocess_pipeline(hsp90_df: pd.DataFrame, 
                       alpha_syn_df: pd.DataFrame,
                       pink1_df: Optional[pd.DataFrame] = None,
                       output_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        hsp90_df: HSP90 dataset
        alpha_syn_df: α-synuclein dataset
        pink1_df: Optional PINK1 dataset
        output_file: Path to save merged CSV
        
    Returns:
        Processed and merged DataFrame
    """
    # Merge datasets
    merged = merge_datasets(hsp90_df, alpha_syn_df, pink1_df)
    
    # Clean data
    cleaned = clean_merged_data(merged)
    
    # Save to file
    if output_file is None:
        output_file = get_data_dir("processed") / "merged.csv"
    
    ensure_dir(output_file.parent)
    cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved merged data to {output_file}")
    
    return cleaned


