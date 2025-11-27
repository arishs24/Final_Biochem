"""
Chemical screening module for new HSP90 inhibitors from ChEMBL.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from chembl_webresource_client.new_client import new_client

from src.utils import get_models_dir, get_output_dir, load_pickle, ensure_dir
from src.featurize import featurize_compound, compute_molecular_descriptors
from src.train import prepare_features_and_targets

logger = logging.getLogger(__name__)


def check_toxicity_smarts(smiles: str) -> Dict[str, Any]:
    """
    Check for toxic substructures using SMARTS patterns.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with toxicity flags and risk score
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'toxicity_risk': 'high', 'toxicity_score': 1.0, 'alerts': ['invalid_smiles']}
    except:
        return {'toxicity_risk': 'high', 'toxicity_score': 1.0, 'alerts': ['invalid_smiles']}
    
    # Common toxic SMARTS patterns
    toxic_smarts = {
        'aldehyde': '[CX3H1](=O)',
        'epoxide': 'C1OC1',
        'nitro_aromatic': '[$(c-[N+](=O)[O-])]',
        'azo': '[N+](=O)[O-]',
        'thiol': '[SH]',
        'alkyl_halide': '[Cl,Br,I][CX4]',
        'michael_acceptor': '[C,c]=[C,c]-[C,c](=O)',
        'quinone': '[C,c]1=[C,c][C,c](=O)[C,c](=O)[C,c]=[C,c]1',
        'furan': 'c1occc1',
        'thiophene': 'c1cscc1',
    }
    
    alerts = []
    risk_score = 0.0
    
    for alert_name, smarts in toxic_smarts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            alerts.append(alert_name)
            risk_score += 0.1
    
    # Determine risk level
    if risk_score >= 0.3:
        risk_level = 'high'
    elif risk_score >= 0.1:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    return {
        'toxicity_risk': risk_level,
        'toxicity_score': min(risk_score, 1.0),
        'alerts': alerts
    }


def download_chembl_hsp90_inhibitors(n_compounds: int = 50) -> pd.DataFrame:
    """
    Download HSP90 inhibitors from ChEMBL database.
    
    Args:
        n_compounds: Number of compounds to retrieve
        
    Returns:
        DataFrame with ChEMBL compound data
    """
    logger.info(f"Downloading {n_compounds} HSP90 inhibitors from ChEMBL...")
    
    try:
        # Search for HSP90 target
        target = new_client.target
        target_query = target.search('HSP90')
        
        if not target_query:
            logger.warning("Could not find HSP90 target in ChEMBL, using synthetic data")
            return _create_synthetic_chembl_data(n_compounds)
        
        # Get the first HSP90 target (usually HSP90AA1)
        hsp90_target = target_query[0]
        target_chembl_id = hsp90_target['target_chembl_id']
        
        logger.info(f"Found HSP90 target: {target_chembl_id}")
        
        # Get activities for this target
        activity = new_client.activity
        activities = activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type='IC50',
            standard_relation='='
        ).only(['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units'])
        
        compounds = []
        seen_smiles = set()
        
        for act in activities:
            if len(compounds) >= n_compounds:
                break
            
            smiles = act.get('canonical_smiles')
            if not smiles or smiles in seen_smiles:
                continue
            
            seen_smiles.add(smiles)
            
            compounds.append({
                'chembl_id': act.get('molecule_chembl_id', ''),
                'smiles': smiles,
                'ic50': act.get('standard_value'),
                'ic50_units': act.get('standard_units', 'nM')
            })
        
        if not compounds:
            logger.warning("No compounds found in ChEMBL, using synthetic data")
            return _create_synthetic_chembl_data(n_compounds)
        
        df = pd.DataFrame(compounds)
        logger.info(f"Downloaded {len(df)} compounds from ChEMBL")
        return df
        
    except Exception as e:
        logger.warning(f"Error downloading from ChEMBL: {e}. Using synthetic data.")
        return _create_synthetic_chembl_data(n_compounds)


def _create_synthetic_chembl_data(n_compounds: int) -> pd.DataFrame:
    """
    Create synthetic ChEMBL-like data for demonstration.
    
    Args:
        n_compounds: Number of compounds to generate
        
    Returns:
        DataFrame with synthetic compound data
    """
    import numpy as np
    
    # Example HSP90 inhibitor SMILES
    base_smiles = [
        "CC(C)OC(=O)Nc1ccc(C(=O)Nc2ccc(C(F)(F)F)cc2)cc1",
        "COc1cc2c(cc1OC)OC(=O)C(=C2O)C(=O)Nc1ccc(C(F)(F)F)cc1",
        "CC1=C(C(=O)Nc2ccc(C(F)(F)F)cc2)OC(=O)c2ccccc21",
        "CN(C)c1ccc(C(=O)Nc2ccc(C(=O)Nc3ccc(C(F)(F)F)cc3)cc2)cc1",
        "CC(C)OC(=O)Nc1ccc(C(=O)Nc2ccc(C(F)(F)F)cc2)cc1",
    ]
    
    np.random.seed(42)
    compounds = []
    
    for i in range(n_compounds):
        smiles = base_smiles[i % len(base_smiles)]
        # Add variation
        if i > len(base_smiles):
            if np.random.random() > 0.5:
                smiles = smiles.replace('C', 'CC', 1)
        
        compounds.append({
            'chembl_id': f'CHEMBL_{1000000 + i}',
            'smiles': smiles,
            'ic50': np.random.uniform(0.1, 1000),
            'ic50_units': 'nM'
        })
    
    return pd.DataFrame(compounds)


def screen_compounds(compounds_df: pd.DataFrame,
                    models_dir: Optional[Path] = None,
                    output_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Screen compounds using trained models.
    
    Args:
        compounds_df: DataFrame with SMILES and compound info
        models_dir: Directory containing trained models
        output_file: Path to save screening results
        
    Returns:
        DataFrame with predictions
    """
    if models_dir is None:
        models_dir = get_models_dir()
    
    if output_file is None:
        output_file = get_output_dir() / "screening_results.csv"
    
    ensure_dir(output_file.parent)
    
    logger.info(f"Screening {len(compounds_df)} compounds...")
    
    # Load feature names
    feature_names_path = models_dir / "feature_names.pkl"
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    
    feature_names = load_pickle(str(feature_names_path))
    
    # Load simple models
    reg_model_path = models_dir / "linear_regressor.pkl"
    clf_model_path = models_dir / "logistic_classifier.pkl"
    
    if not reg_model_path.exists():
        raise FileNotFoundError(f"Regression model not found at {reg_model_path}")
    if not clf_model_path.exists():
        raise FileNotFoundError(f"Classification model not found at {clf_model_path}")
    
    reg_model = load_pickle(str(reg_model_path))
    clf_model = load_pickle(str(clf_model_path))
    
    # Load scalers
    reg_scaler_path = models_dir / "linear_regressor_scaler.pkl"
    clf_scaler_path = models_dir / "logistic_classifier_scaler.pkl"
    
    reg_scaler = load_pickle(str(reg_scaler_path)) if reg_scaler_path.exists() else None
    clf_scaler = load_pickle(str(clf_scaler_path)) if clf_scaler_path.exists() else None
    
    # Featurize compounds
    results = []
    
    for idx, row in compounds_df.iterrows():
        smiles = row['smiles']
        
        if pd.isna(smiles) or not isinstance(smiles, str):
            continue
        
        # Compute features (simplified version)
        features = featurize_compound(smiles, row, use_fingerprints=True)
        
        if not features:
            continue
        
        # Create feature vector matching training features
        feature_vector = []
        for feat_name in feature_names:
            if feat_name in features:
                value = features[feat_name]
                # Handle NaN values
                if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    feature_vector.append(0.0)
                else:
                    feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
        
        X = np.array([feature_vector])
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Fill any remaining NaN values
        X_df = X_df.fillna(0.0)
        
        # Scale features if scaler exists
        if reg_scaler is not None:
            X_reg_scaled = reg_scaler.transform(X_df)
        else:
            X_reg_scaled = X_df
        
        if clf_scaler is not None:
            X_clf_scaled = clf_scaler.transform(X_df)
        else:
            X_clf_scaled = X_df
        
        # Ensure no NaN values remain
        X_reg_scaled = np.nan_to_num(X_reg_scaled, nan=0.0)
        X_clf_scaled = np.nan_to_num(X_clf_scaled, nan=0.0)
        
        # Make predictions
        pred_aggregation_reduction = reg_model.predict(X_reg_scaled)[0]
        pred_proba = clf_model.predict_proba(X_clf_scaled)[0]
        pred_effective = clf_model.predict(X_clf_scaled)[0]
        
        # Get top features from coefficients
        if hasattr(clf_model, 'coef_'):
            coef = clf_model.coef_[0] if clf_model.coef_.ndim > 1 else clf_model.coef_
            top_indices = np.argsort(np.abs(coef))[::-1][:5]
            top_features = [feature_names[i] for i in top_indices]
        else:
            top_features = []
        
        # Check toxicity
        toxicity = check_toxicity_smarts(smiles)
        
        # Store results
        result = {
            'smiles': smiles,
            'chembl_id': row.get('chembl_id', f'compound_{idx}'),
            'predicted_aggregation_reduction': pred_aggregation_reduction,
            'predicted_effectiveness': 'effective' if pred_effective == 1 else 'ineffective',
            'effectiveness_probability': pred_proba[1],
            'toxicity_risk': toxicity['toxicity_risk'],
            'toxicity_score': toxicity['toxicity_score'],
            'toxicity_alerts': ', '.join(toxicity['alerts']) if toxicity['alerts'] else 'none',
            'top_features': ', '.join(top_features[:3]) if top_features else 'N/A'
        }
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Sort by predicted effectiveness probability
    results_df = results_df.sort_values('effectiveness_probability', ascending=False)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved screening results to {output_file}")
    logger.info(f"Screened {len(results_df)} compounds")
    logger.info(f"Predicted effective: {(results_df['predicted_effectiveness'] == 'effective').sum()}")
    
    return results_df


def run_screening(n_compounds: int = 50,
                 output_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Complete screening pipeline: download compounds and screen them.
    
    Args:
        n_compounds: Number of compounds to download from ChEMBL
        output_file: Path to save results
        
    Returns:
        DataFrame with screening results
    """
    # Download compounds
    compounds_df = download_chembl_hsp90_inhibitors(n_compounds)
    
    # Screen compounds
    results_df = screen_compounds(compounds_df, output_file=output_file)
    
    return results_df


