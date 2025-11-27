"""
Molecular feature engineering module using RDKit.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

from src.utils import get_data_dir, save_pickle, safe_divide

logger = logging.getLogger(__name__)


def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 512) -> Optional[np.ndarray]:
    """
    Compute Morgan (ECFP) fingerprint for a SMILES string.
    Simplified: using 512 bits instead of 2048 for faster computation.
    
    Args:
        smiles: SMILES string
        radius: Fingerprint radius
        n_bits: Number of bits (reduced to 512)
        
    Returns:
        NumPy array of fingerprint bits, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        logger.debug(f"Error computing Morgan fingerprint for {smiles}: {e}")
        return None


def compute_maccs_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """
    Compute MACCS fingerprint for a SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        NumPy array of MACCS bits, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        from rdkit.Chem import MACCSkeys
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,))  # MACCS has 167 bits
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        logger.debug(f"Error computing MACCS fingerprint for {smiles}: {e}")
        return None


def compute_molecular_descriptors(smiles: str) -> Dict[str, Any]:
    """
    Compute molecular descriptors for a SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of descriptor names and values
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'heteroatom_count': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        }
        
        return descriptors
    except Exception as e:
        logger.debug(f"Error computing descriptors for {smiles}: {e}")
        return {}


def compute_kinetic_descriptors(row: pd.Series) -> Dict[str, float]:
    """
    Compute kinetic descriptors from k_on and k_off.
    
    Args:
        row: DataFrame row with k_on and k_off columns
        
    Returns:
        Dictionary of kinetic descriptors
    """
    descriptors = {}
    
    if 'k_on' in row and pd.notna(row['k_on']):
        k_on = float(row['k_on'])
        descriptors['k_on'] = k_on
        descriptors['log_kon'] = np.log10(max(k_on, 1e-10))
    else:
        descriptors['k_on'] = np.nan
        descriptors['log_kon'] = np.nan
    
    if 'k_off' in row and pd.notna(row['k_off']):
        k_off = float(row['k_off'])
        descriptors['k_off'] = k_off
        descriptors['log_koff'] = np.log10(max(k_off, 1e-10))
        
        # Residence time = 1 / k_off
        if k_off > 0:
            descriptors['residence_time'] = 1.0 / k_off
        else:
            descriptors['residence_time'] = np.nan
    else:
        descriptors['k_off'] = np.nan
        descriptors['log_koff'] = np.nan
        descriptors['residence_time'] = np.nan
    
    # Affinity (Kd = k_off / k_on)
    if 'k_on' in descriptors and 'k_off' in descriptors:
        if pd.notna(descriptors['k_on']) and pd.notna(descriptors['k_off']):
            if descriptors['k_on'] > 0:
                descriptors['kd'] = descriptors['k_off'] / descriptors['k_on']
                descriptors['log_kd'] = np.log10(max(descriptors['kd'], 1e-12))
            else:
                descriptors['kd'] = np.nan
                descriptors['log_kd'] = np.nan
        else:
            descriptors['kd'] = np.nan
            descriptors['log_kd'] = np.nan
    else:
        descriptors['kd'] = np.nan
        descriptors['log_kd'] = np.nan
    
    return descriptors


def compute_pink1_descriptors(row: pd.Series) -> Dict[str, float]:
    """
    Compute PINK1-related descriptors.
    
    Args:
        row: DataFrame row with PINK1 data
        
    Returns:
        Dictionary of PINK1 descriptors
    """
    descriptors = {}
    
    if 'pink1_ic50' in row and pd.notna(row['pink1_ic50']):
        pink1_ic50 = float(row['pink1_ic50'])
        descriptors['pink1_ic50'] = pink1_ic50
        descriptors['pink1_ic50_log'] = np.log10(max(pink1_ic50, 1e-6))
    else:
        descriptors['pink1_ic50'] = np.nan
        descriptors['pink1_ic50_log'] = np.nan
    
    if 'pink1_activation_score' in row and pd.notna(row['pink1_activation_score']):
        descriptors['pink1_activation_score'] = float(row['pink1_activation_score'])
    else:
        descriptors['pink1_activation_score'] = np.nan
    
    return descriptors


def featurize_compound(smiles: str, kinetic_data: Optional[pd.Series] = None, 
                      use_fingerprints: bool = True) -> Dict[str, Any]:
    """
    Compute all features for a single compound (simplified version).
    
    Args:
        smiles: SMILES string
        kinetic_data: Optional Series with k_on, k_off, PINK1 data, etc.
        use_fingerprints: Whether to include fingerprints (default: True, but simplified)
        
    Returns:
        Dictionary of all features
    """
    features = {}
    
    # Molecular descriptors (always include)
    mol_desc = compute_molecular_descriptors(smiles)
    features.update(mol_desc)
    
    # Very simplified fingerprints: just summary statistics
    if use_fingerprints:
        morgan_fp = compute_morgan_fingerprint(smiles, n_bits=256)  # Even smaller
        if morgan_fp is not None:
            features['morgan_sum'] = int(morgan_fp.sum())
            features['morgan_mean'] = float(morgan_fp.mean())
            features['morgan_max'] = int(morgan_fp.max())
            features['morgan_std'] = float(morgan_fp.std())
    
    # Kinetic descriptors
    if kinetic_data is not None:
        kinetic_desc = compute_kinetic_descriptors(kinetic_data)
        features.update(kinetic_desc)
        
        # PINK1 descriptors
        pink1_desc = compute_pink1_descriptors(kinetic_data)
        features.update(pink1_desc)
    
    return features


def featurize_dataset(df: pd.DataFrame, 
                     smiles_col: str = 'smiles',
                     target_col: str = 'aggregation_reduction',
                     binary_target_col: str = 'is_effective') -> pd.DataFrame:
    """
    Featurize an entire dataset.
    
    Args:
        df: DataFrame with SMILES and target columns
        smiles_col: Name of SMILES column
        target_col: Name of continuous target column
        binary_target_col: Name of binary target column
        
    Returns:
        DataFrame with features and targets
    """
    logger.info(f"Featurizing {len(df)} compounds...")
    
    all_features = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        
        if pd.isna(smiles) or not isinstance(smiles, str):
            continue
        
        # Compute features (simplified version)
        features = featurize_compound(smiles, row, use_fingerprints=True)
        
        if not features:
            continue
        
        # Add targets
        if target_col in row:
            features['target'] = row[target_col]
        
        if binary_target_col in row:
            features['target_binary'] = row[binary_target_col]
        
        # Add compound identifier
        if 'compound_id' in row:
            features['compound_id'] = row['compound_id']
        else:
            features['compound_id'] = f'compound_{idx}'
        
        features['smiles'] = smiles
        
        all_features.append(features)
        valid_indices.append(idx)
    
    logger.info(f"Successfully featurized {len(all_features)} compounds")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Fill NaN values in numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
    
    logger.info(f"Feature matrix shape: {features_df.shape}")
    logger.info(f"Number of features: {len(features_df.columns)}")
    
    return features_df


def save_features(features_df: pd.DataFrame, output_file: Optional[Path] = None) -> None:
    """
    Save feature matrix to pickle file.
    
    Args:
        features_df: DataFrame with features
        output_file: Path to save file
    """
    if output_file is None:
        output_file = get_data_dir("processed") / "features.pkl"
    
    save_pickle(features_df, str(output_file))
    logger.info(f"Saved features to {output_file}")


def load_features(input_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Load feature matrix from pickle file.
    
    Args:
        input_file: Path to pickle file
        
    Returns:
        DataFrame with features
    """
    from src.utils import load_pickle
    
    if input_file is None:
        input_file = get_data_dir("processed") / "features.pkl"
    
    features_df = load_pickle(str(input_file))
    logger.info(f"Loaded features from {input_file}")
    return features_df

