"""
Streamlit interface for HSP90 and PINK1 Inhibitor Screening Project
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_models_dir, get_figures_dir, get_output_dir, get_data_dir, load_pickle

# Page configuration
st.set_page_config(
    page_title="HSP90 & PINK1 Drug Screening",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean CSS with good readability
st.markdown("""
<style>
    /* Simple light background */
    .stApp {
        background: #f5f7fa;
    }
    
    /* Simple card style */
    .modern-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Simple animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 0.4s ease-out;
    }
    
    /* Simple button */
    .stButton > button {
        background: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #5568d3;
    }
    
    /* Simple text input */
    .stTextInput > div > div > input {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        outline: none;
    }
    
    /* Simple sidebar */
    .css-1d391kg {
        background: #ffffff;
    }
    
    /* Code blocks - simple */
    code {
        background: #f7fafc !important;
        color: #2d3748 !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 4px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Text colors - simple and readable */
    p, li, div {
        color: #2d3748 !important;
    }
    
    h1, h2, h3 {
        color: #1a202c !important;
    }
    
    /* Simple success/error */
    .stSuccess {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        border-radius: 4px;
        padding: 1rem;
    }
    
    .stError {
        background: #fff5f5;
        border-left: 4px solid #f56565;
        border-radius: 4px;
        padding: 1rem;
    }
    
    .stWarning {
        background: #fffaf0;
        border-left: 4px solid #ed8936;
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px;
        background: white;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


SUMMARY_METADATA_COLUMNS = [
    'compound_id',
    'smiles',
    'chembl_id',
    'source',
    'notes',
    'pink1_ic50',
    'pink1_activation_score',
    'toxicity_risk',
    'toxicity_score',
    'toxicity_alerts',
]


@st.cache_data
def load_models():
    """Load trained models and feature names."""
    models_dir = get_models_dir()
    
    try:
        reg_model = load_pickle(str(models_dir / "linear_regressor.pkl"))
        clf_model = load_pickle(str(models_dir / "logistic_classifier.pkl"))
        feature_names = load_pickle(str(models_dir / "feature_names.pkl"))
        
        reg_scaler = None
        clf_scaler = None
        if (models_dir / "linear_regressor_scaler.pkl").exists():
            reg_scaler = load_pickle(str(models_dir / "linear_regressor_scaler.pkl"))
        if (models_dir / "logistic_classifier_scaler.pkl").exists():
            clf_scaler = load_pickle(str(models_dir / "logistic_classifier_scaler.pkl"))
        
        return reg_model, clf_model, feature_names, reg_scaler, clf_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None


@st.cache_data
def load_screening_results():
    """Load screening results."""
    results_file = get_output_dir() / "screening_results.csv"
    if results_file.exists():
        return pd.read_csv(results_file)
    return None


@st.cache_data
def load_feature_library():
    """Load precomputed training features for demo/testing."""
    features_path = get_data_dir("processed") / "features.pkl"
    if features_path.exists():
        try:
            return load_pickle(str(features_path))
        except Exception:
            return None
    return None


def build_feature_matrix(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Align arbitrary descriptor frames to the trained feature schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=feature_names)
    reindexed = df.reindex(columns=feature_names)
    numeric = reindexed.apply(pd.to_numeric, errors='coerce')
    return numeric.fillna(0.0)


def get_top_features(clf_model, feature_names, top_n: int = 5) -> List[str]:
    """Return globally important features from the classification model."""
    if hasattr(clf_model, 'coef_') and clf_model.coef_ is not None:
        coef = clf_model.coef_[0] if clf_model.coef_.ndim > 1 else clf_model.coef_
        indices = np.argsort(np.abs(coef))[::-1][:top_n]
        return [feature_names[i] for i in indices]
    return []


def predict_from_feature_rows(rows_df: pd.DataFrame,
                              reg_model,
                              clf_model,
                              feature_names: List[str],
                              reg_scaler,
                              clf_scaler) -> Tuple[pd.DataFrame, List[str], int]:
    """Run inference on descriptor rows that already contain RDKit features."""
    if rows_df is None or rows_df.empty:
        return pd.DataFrame(), [], len(feature_names)
    
    missing = [feat for feat in feature_names if feat not in rows_df.columns]
    feature_matrix = build_feature_matrix(rows_df, feature_names)
    feature_values = feature_matrix.to_numpy(dtype=float, copy=False)
    
    if reg_scaler is not None:
        X_reg = reg_scaler.transform(feature_values)
    else:
        X_reg = feature_values
    
    if clf_scaler is not None:
        X_clf = clf_scaler.transform(feature_values)
    else:
        X_clf = feature_values
    
    reg_preds = reg_model.predict(X_reg)
    clf_probs = clf_model.predict_proba(X_clf)
    clf_preds = clf_model.predict(X_clf)
    
    summary_cols = [col for col in SUMMARY_METADATA_COLUMNS if col in rows_df.columns]
    summary_df = rows_df.reset_index(drop=True)[summary_cols].copy() if summary_cols else pd.DataFrame(index=range(len(rows_df)))
    summary_df['predicted_aggregation_reduction'] = reg_preds
    summary_df['predicted_effectiveness'] = np.where(clf_preds == 1, 'effective', 'ineffective')
    summary_df['effectiveness_probability'] = clf_probs[:, 1]
    
    top_features = get_top_features(clf_model, feature_names)
    summary_df['top_features'] = [top_features] * len(summary_df)
    
    return summary_df, top_features, len(missing)


def format_feature_name(feat_name):
    """Format feature names to be more readable."""
    if not feat_name or not isinstance(feat_name, str):
        return str(feat_name)
    
    # Common molecular descriptor mappings
    descriptor_map = {
        'molecular_weight': 'Molecular Weight',
        'logp': 'LogP (Lipophilicity)',
        'tpsa': 'TPSA (Topological Polar Surface Area)',
        'h_bond_donors': 'H-Bond Donors',
        'h_bond_acceptors': 'H-Bond Acceptors',
        'rotatable_bonds': 'Rotatable Bonds',
        'aromatic_rings': 'Aromatic Rings',
        'heteroatom_count': 'Heteroatom Count',
        'k_on': 'Binding Rate (k_on)',
        'k_off': 'Dissociation Rate (k_off)',
        'residence_time': 'Residence Time',
        'pink1_ic50': 'PINK1 IC50',
        'pink1_activation_score': 'PINK1 Activation Score',
    }
    
    # Check if it's a known descriptor
    feat_lower = feat_name.lower()
    if feat_lower in descriptor_map:
        return descriptor_map[feat_lower]
    
    # Handle log-transformed features
    if feat_name.startswith('log_'):
        base_name = feat_name.replace('log_', '')
        formatted = format_feature_name(base_name)
        return f"Log({formatted})"
    
    # Handle Morgan fingerprints
    if feat_name.startswith('morgan_'):
        bit_num = feat_name.replace('morgan_', '')
        try:
            int(bit_num)
            return f"Morgan Fingerprint Bit {bit_num}"
        except ValueError:
            return f"Morgan Fingerprint: {bit_num}"
    
    # Handle MACCS fingerprints
    if feat_name.startswith('maccs_'):
        bit_num = feat_name.replace('maccs_', '')
        try:
            int(bit_num)
            return f"MACCS Fingerprint Bit {bit_num}"
        except ValueError:
            return f"MACCS Fingerprint: {bit_num}"
    
    # Handle fingerprint summary statistics
    if feat_name.startswith('morgan_fp_'):
        stat = feat_name.replace('morgan_fp_', '')
        stat_map = {'sum': 'Sum', 'mean': 'Mean', 'max': 'Max', 'std': 'Std Dev'}
        return f"Morgan Fingerprint {stat_map.get(stat, stat.capitalize())}"
    
    # Replace underscores with spaces and capitalize
    if '_' in feat_name:
        parts = feat_name.split('_')
        formatted_parts = []
        for part in parts:
            if part.isupper() and len(part) > 1:
                formatted_parts.append(part)
            elif part.lower() in ['ic50', 'ic', 'ec50', 'ki']:
                formatted_parts.append(part.upper())
            else:
                formatted_parts.append(part.capitalize())
        return ' '.join(formatted_parts)
    
    # Just capitalize first letter
    return feat_name.capitalize()


def render_prediction_ui(result: Dict[str, Any]) -> None:
    """Render the rich prediction view used across the app."""
    if not result:
        st.info("No prediction available.")
        return
    
    prob_pct = result.get('effectiveness_probability', 0) * 100
    toxicity_risk = result.get('toxicity_risk') or 'not provided'
    toxicity_score = result.get('toxicity_score')
    toxicity_alerts = result.get('toxicity_alerts') or 'not provided'
    if isinstance(toxicity_score, float) and np.isnan(toxicity_score):
        toxicity_score = None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="modern-card fade-in" style="text-align: center;
                    background: #f0f4ff;
                    border-left: 4px solid #667eea;
                    padding: 1.5rem;">
            <p style="color: #2d3748; font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 600;">
                Predicted Aggregation Reduction
            </p>
            <h2 style="color: #667eea; font-size: 2.5rem; margin: 0.5rem 0; font-weight: 700;">
                {result.get('predicted_aggregation_reduction', 0):.1f}%
            </h2>
            <p style="color: #4a5568; font-size: 0.85rem; margin: 0;">
                Expected reduction in α-synuclein aggregation
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        effectiveness = result.get('predicted_effectiveness', 'ineffective')
        effectiveness_color = "#48bb78" if effectiveness == 'effective' else "#f56565"
        effectiveness_icon = "✅" if effectiveness == 'effective' else "❌"
        bg_color = "#f0fff4" if effectiveness == 'effective' else "#fff5f5"
        st.markdown(f"""
        <div class="modern-card fade-in" style="text-align: center;
                    background: {bg_color};
                    border-left: 4px solid {effectiveness_color};
                    padding: 1.5rem;">
            <p style="color: #2d3748; font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 600;">
                Effectiveness
            </p>
            <h2 style="color: {effectiveness_color}; font-size: 2rem; margin: 0.5rem 0; font-weight: 700;">
                {effectiveness_icon} {effectiveness.title()}
            </h2>
            <p style="color: #4a5568; font-size: 0.85rem; margin: 0;">
                Binary classification result
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="modern-card fade-in" style="text-align: center;
                    background: #f0f9ff;
                    border-left: 4px solid #4facfe;
                    padding: 1.5rem;">
            <p style="color: #2d3748; font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 600;">
                Confidence Score
            </p>
            <h2 style="color: #4facfe; font-size: 2.5rem; margin: 0.5rem 0; font-weight: 700;">
                {prob_pct:.1f}%
            </h2>
            <p style="color: #4a5568; font-size: 0.85rem; margin: 0;">
                Probability of effectiveness
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="modern-card fade-in" style="background: #f7fafc;
                    border-left: 4px solid #667eea;
                    margin-top: 1rem;">
            <h3 style="color: #1a202c; margin-top: 0; font-weight: 700;">📋 Detailed Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        smiles = result.get('smiles') or 'not provided'
        st.markdown(f"""
        <div class="modern-card fade-in" style="padding: 1.5rem; margin-top: 0.5rem;">
            <p style="color: #2d3748; margin: 0.8rem 0;">
                <strong style="color: #1a202c;">SMILES:</strong><br>
                <code style="background: #f7fafc; color: #2d3748; padding: 0.5rem; border-radius: 6px; display: block; margin-top: 0.5rem; word-break: break-all; font-size: 0.9rem; border: 1px solid #e2e8f0;">{smiles}</code>
            </p>
            <p style="color: #2d3748; margin: 0.8rem 0;">
                <strong style="color: #1a202c;">Toxicity Risk:</strong> 
                <span style="background: {'#48bb78' if toxicity_risk == 'low' else '#ed8936' if toxicity_risk == 'medium' else '#f56565' if toxicity_risk == 'high' else '#a0aec0'}; 
                             color: white; 
                             padding: 0.3rem 0.8rem; 
                             border-radius: 6px; 
                             font-weight: 600;
                             font-size: 0.9rem;">
                    {toxicity_risk.upper() if isinstance(toxicity_risk, str) else toxicity_risk}
                </span>
            </p>
            <p style="color: #2d3748; margin: 0.8rem 0;">
                <strong style="color: #1a202c;">Toxicity Score:</strong> {toxicity_score if toxicity_score is not None else 'not provided'}
            </p>
            <p style="color: #2d3748; margin: 0.8rem 0;">
                <strong style="color: #1a202c;">Toxicity Alerts:</strong> 
                {toxicity_alerts if toxicity_alerts and toxicity_alerts != 'none' else 'None detected ✅'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card fade-in" style="background: #f7fafc;
                    border-left: 4px solid #4facfe;
                    margin-top: 1rem;">
            <h3 style="color: #1a202c; margin-top: 0; font-weight: 700;">🔑 Top Contributing Features</h3>
        </div>
        """, unsafe_allow_html=True)
        
        top_features = result.get('top_features') or []
        if top_features:
            for i, feat in enumerate(top_features, 1):
                formatted_feat = format_feature_name(feat)
                st.markdown(f"""
                <div style="background: white; 
                            padding: 1rem; 
                            border-radius: 8px; 
                            margin: 0.6rem 0;
                            border-left: 4px solid #4facfe;
                            border: 1px solid #e2e8f0;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <span style="color: #4facfe; font-weight: 700; font-size: 1.1rem; margin-right: 0.5rem;">{i}.</span> 
                    <span style="color: #1a202c; font-size: 1rem; font-weight: 500; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">{formatted_feat}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No feature importance available.")
    
    st.markdown("---")
    
    interpretation_container = st.container()
    effectiveness = result.get('predicted_effectiveness', 'ineffective')
    reduction = result.get('predicted_aggregation_reduction', 0.0)
    
    if effectiveness == 'effective':
        interpretation_container.markdown(f"""
        <div class="modern-card fade-in" style="background: #f0fff4;
                    border-left: 4px solid #48bb78;
                    margin-top: 1rem;">
            <h3 style="color: #1a202c; margin-top: 0; font-size: 1.4rem; font-weight: 700;">💡 Interpretation</h3>
            <p style="color: #2d3748; font-size: 1.05rem; line-height: 1.8; margin-bottom: 0.8rem;">
                ✅ This compound is predicted to be <strong style="color: #48bb78;">effective</strong> at reducing α-synuclein aggregation.
            </p>
            <ul style="color: #2d3748; font-size: 1rem; line-height: 2;">
                <li>Expected reduction: <strong>{reduction:.1f}%</strong></li>
                <li>Confidence: <strong>{prob_pct:.1f}%</strong></li>
                <li>Toxicity risk: <strong>{toxicity_risk.upper() if isinstance(toxicity_risk, str) else toxicity_risk}</strong></li>
            </ul>
            <p style="color: #2d3748; font-size: 1rem; margin-top: 1rem; font-weight: 600; padding: 1rem; background: #e6ffed; border-radius: 6px;">
                🎉 This compound shows promise and may be worth testing experimentally!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        interpretation_container.markdown(f"""
        <div class="modern-card fade-in" style="background: #fff5f5;
                    border-left: 4px solid #f56565;
                    margin-top: 1rem;">
            <h3 style="color: #1a202c; margin-top: 0; font-size: 1.4rem; font-weight: 700;">💡 Interpretation</h3>
            <p style="color: #2d3748; font-size: 1.05rem; line-height: 1.8; margin-bottom: 0.8rem;">
                ❌ This compound is predicted to be <strong style="color: #f56565;">ineffective</strong> at reducing α-synuclein aggregation.
            </p>
            <ul style="color: #2d3748; font-size: 1rem; line-height: 2;">
                <li>Expected reduction: <strong>{reduction:.1f}%</strong></li>
                <li>Confidence: <strong>{prob_pct:.1f}%</strong></li>
                <li>Toxicity risk: <strong>{toxicity_risk.upper() if isinstance(toxicity_risk, str) else toxicity_risk}</strong></li>
            </ul>
            <p style="color: #2d3748; font-size: 1rem; margin-top: 1rem; font-weight: 600; padding: 1rem; background: #ffe5e5; border-radius: 6px;">
                ⚠️ This compound may not be suitable for further development.
            </p>
        </div>
        """, unsafe_allow_html=True)
def main():
    """Main Streamlit app."""
    
    # Simple header
    st.markdown("""
    <div class="fade-in" style="background: white;
                padding: 2rem; 
                border-radius: 12px; 
                margin-bottom: 2rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border: 1px solid #e2e8f0;
                text-align: center;">
        <h1 style="color: #667eea;
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin: 0;">
            🧬 HSP90 & PINK1 Drug Screening
        </h1>
        <p style="color: #4a5568; 
                  font-size: 1.1rem; 
                  margin-top: 0.5rem;">
            Machine Learning Pipeline for Parkinson's Disease Drug Discovery
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple sidebar navigation
    st.sidebar.markdown("""
    <div style="background: white;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                text-align: center;
                border: 1px solid #e2e8f0;">
        <h2 style="color: #667eea; margin: 0; font-size: 1.3rem; font-weight: 700;">🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Choose a section:",
        ["🏠 Overview", "📊 Results & Visualizations", "🔬 Interactive Testing", "📈 Model Information", "📋 Screening Results"],
        label_visibility="collapsed"
    )
    
    # Load models
    reg_model, clf_model, feature_names, reg_scaler, clf_scaler = load_models()
    
    if reg_model is None or clf_model is None or feature_names is None:
        st.error("⚠️ Models not found! Please run the pipeline first: `python run_pipeline.py`")
        st.info("The pipeline will train the models and generate all necessary files.")
        return
    
    # Page routing
    if page == "🏠 Overview":
        show_overview()
    elif page == "📊 Results & Visualizations":
        show_visualizations()
    elif page == "🔬 Interactive Testing":
        show_interactive_testing(reg_model, clf_model, feature_names, reg_scaler, clf_scaler)
    elif page == "📈 Model Information":
        show_model_info(reg_model, clf_model, feature_names)
    elif page == "📋 Screening Results":
        show_screening_results()


def show_overview():
    """Show project overview."""
    st.markdown("""
    <div class="fade-in" style="background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                padding: 1.5rem 2rem; 
                border-radius: 20px; 
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                border: 1px solid rgba(255,255,255,0.3);">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin: 0; 
                    font-size: 2.5rem; 
                    font-weight: 800;
                    letter-spacing: -0.5px;">
            📖 Project Overview
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="modern-card fade-in">
        <h3 style="color: #1a202c; margin-top: 0; font-size: 1.5rem; font-weight: 700;">
            🎯 What is this project?
        </h3>
        <p style="color: #2d3748; font-size: 1.05rem; line-height: 1.8;">
            This is a machine learning pipeline that predicts which drugs (HSP90 and PINK1 modulators) 
            are most effective at reducing <strong style="color: #667eea;">α-synuclein aggregation</strong> - 
            a key pathological process in Parkinson's disease.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="modern-card fade-in" style="text-align: center; 
                    background: #f7fafc;
                    border-left: 4px solid #667eea;">
            <h2 style="color: #667eea; font-size: 1.8rem; margin: 0.5rem 0; font-weight: 700;">🧬 HSP90</h2>
            <p style="color: #2d3748; font-size: 0.95rem; line-height: 1.6;">
                Molecular chaperone involved in protein folding. HSP90 inhibitors may help reduce protein aggregation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card fade-in" style="text-align: center;
                    background: #f7fafc;
                    border-left: 4px solid #4facfe;">
            <h2 style="color: #4facfe; font-size: 1.8rem; margin: 0.5rem 0; font-weight: 700;">⚡ PINK1</h2>
            <p style="color: #2d3748; font-size: 0.95rem; line-height: 1.6;">
                Mitochondrial kinase involved in quality control. PINK1 activation protects against α-synuclein aggregation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="modern-card fade-in" style="text-align: center;
                    background: #f7fafc;
                    border-left: 4px solid #48bb78;">
            <h2 style="color: #48bb78; font-size: 1.8rem; margin: 0.5rem 0; font-weight: 700;">🔬 α-Synuclein</h2>
            <p style="color: #2d3748; font-size: 0.95rem; line-height: 1.6;">
                Protein that aggregates in Parkinson's disease. Reducing aggregation is a therapeutic goal.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="modern-card fade-in">
        <h2 style="color: #667eea; margin: 0; font-size: 1.8rem; font-weight: 700;">
            🔄 Pipeline Steps
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    steps = [
        ("1️⃣ Data Loading", "Loads HSP90, α-synuclein, and PINK1 datasets from various sources"),
        ("2️⃣ Preprocessing", "Merges datasets and creates binary labels (effective vs ineffective)"),
        ("3️⃣ Feature Engineering", "Computes molecular descriptors, fingerprints, and kinetic features"),
        ("4️⃣ Model Training", "Trains Linear Regression and Logistic Regression models"),
        ("5️⃣ Model Evaluation", "Evaluates performance and generates visualizations"),
        ("6️⃣ Compound Screening", "Screens new compounds and predicts effectiveness")
    ]
    
    for step_num, description in steps:
        st.markdown(f"""
        <div class="modern-card fade-in" style="background: #f7fafc;
                    padding: 1rem 1.5rem; 
                    margin: 0.5rem 0;
                    border-left: 4px solid #667eea;">
            <p style="color: #2d3748; margin: 0; font-size: 1rem; line-height: 1.6;">
                <strong style="color: #667eea;">{step_num}</strong> - {description}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="modern-card fade-in">
        <h2 style="color: #667eea; margin: 0; font-size: 1.8rem; font-weight: 700;">
            🎯 Key Results
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Regression RMSE", "20.86", "± 3.52")
        st.caption("Lower is better - measures prediction error")
    
    with col2:
        st.metric("Classification ROC-AUC", "0.65", "± 0.02")
        st.caption("Higher is better - measures drug detection accuracy")
    
    st.markdown("""
    <div class="modern-card fade-in" style="background: #fffaf0;
                border-left: 4px solid #f6ad55;
                margin-top: 2rem;">
        <h3 style="color: #1a202c; margin-top: 0; font-size: 1.4rem; font-weight: 700;">💡 What do these results mean?</h3>
        <ul style="color: #2d3748; font-size: 1rem; line-height: 2;">
            <li><strong>RMSE of 20.86:</strong> On average, predictions are off by ~21 percentage points in aggregation reduction</li>
            <li><strong>ROC-AUC of 0.65:</strong> The model is 65% better than random at identifying effective drugs</li>
            <li><strong>Model Type:</strong> Simple Linear and Logistic Regression - easy to interpret and understand</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def show_visualizations():
    """Show all visualizations."""
    st.markdown("""
    <div class="modern-card fade-in" style="text-align: center;">
        <h1 style="color: #667eea; margin: 0; font-size: 2.5rem; font-weight: 700;">
            📊 Results & Visualizations
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    figures_dir = get_figures_dir()
    
    # Predicted vs Actual
    st.markdown("""
    <div class="modern-card fade-in" style="border-left: 4px solid #667eea;">
        <h2 style="color: #1a202c; margin-top: 0; font-weight: 700;">1️⃣ Model Performance - Predicted vs Actual</h2>
    </div>
    """, unsafe_allow_html=True)
    
    predicted_file = figures_dir / "predicted_vs_actual.png"
    if predicted_file.exists():
        st.image(str(predicted_file), width='stretch')
        st.markdown("""
        <div class="modern-card fade-in" style="background: #f7fafc;
                    margin-top: 1rem;">
            <p style="color: #1a202c; font-weight: 600; margin-bottom: 0.5rem;">💡 What this shows:</p>
            <p style="color: #2d3748; line-height: 1.8;">
                How well the Linear Regression model predicts aggregation reduction.<br>
                • Points close to the red diagonal line = good predictions<br>
                • R² value shows how much variance is explained<br>
                • Scatter indicates prediction error
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Figure not found. Run the pipeline to generate it.")
    
    st.markdown("---")
    
    # ROC Curves
    st.markdown("""
    <div class="modern-card fade-in" style="border-left: 4px solid #4facfe;">
        <h2 style="color: #1a202c; margin-top: 0; font-weight: 700;">2️⃣ Drug Detection Performance - ROC Curves</h2>
    </div>
    """, unsafe_allow_html=True)
    
    roc_file = figures_dir / "roc_curves.png"
    if roc_file.exists():
        st.image(str(roc_file), width='stretch')
        st.markdown("""
        <div class="modern-card fade-in" style="background: #f7fafc;
                    margin-top: 1rem;">
            <p style="color: #1a202c; font-weight: 600; margin-bottom: 0.5rem;">💡 What this shows:</p>
            <p style="color: #2d3748; line-height: 1.8;">
                How well the Logistic Regression distinguishes effective vs ineffective drugs.<br>
                • Curve above diagonal = better than random guessing<br>
                • AUC (Area Under Curve) = overall performance metric<br>
                • Higher AUC = better at identifying effective drugs
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Figure not found. Run the pipeline to generate it.")
    
    st.markdown("---")
    
    # Feature Coefficients
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3. Feature Importance (Regression)")
        reg_coef_file = figures_dir / "feature_coefficients_linear_regressor.png"
        if reg_coef_file.exists():
            st.image(str(reg_coef_file), width='stretch')
            st.caption("Top features that predict aggregation reduction")
        else:
            st.warning("Figure not found.")
    
    with col2:
        st.subheader("4. Feature Importance (Classification)")
        clf_coef_file = figures_dir / "feature_coefficients_logistic_classifier.png"
        if clf_coef_file.exists():
            st.image(str(clf_coef_file), width='stretch')
            st.caption("Top features that determine drug effectiveness")
        else:
            st.warning("Figure not found.")
    
    st.markdown("""
    **What these show:** Which molecular properties are most important for predictions.
    - Positive coefficients = increase effectiveness
    - Negative coefficients = decrease effectiveness
    - Larger bars = stronger influence
    """)
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.subheader("5. Feature Correlation Heatmap")
    heatmap_file = figures_dir / "correlation_heatmap.png"
    if heatmap_file.exists():
        st.image(str(heatmap_file), width='stretch')
        st.markdown("""
        **What this shows:** How molecular features relate to each other.
        - Red = positive correlation (features increase together)
        - Blue = negative correlation (features move in opposite directions)
        - White = no correlation
        - Helps identify redundant features
        """)
    else:
        st.warning("Figure not found. Run the pipeline to generate it.")


def show_interactive_testing(reg_model, clf_model, feature_names, reg_scaler, clf_scaler):
    """Show interactive testing interface that relies on precomputed descriptors."""
    st.markdown("""
    <div class="modern-card fade-in" style="text-align: center;">
        <h1 style="color: #667eea; margin: 0; font-size: 2.5rem; font-weight: 700;">
            🔬 Interactive Drug Testing
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="modern-card fade-in" style="background: #f0f4ff;
                border: 2px solid #667eea;
                margin-bottom: 2rem;">
        <h2 style="color: #667eea; margin-top: 0; font-size: 1.8rem; text-align: center; font-weight: 700;">
            🔬 Test Your Own Compounds (RDKit-free Deployment)
        </h2>
        <p style="color: #2d3748; font-size: 1.05rem; line-height: 1.7; text-align: center;">
            Streamlit Cloud cannot run RDKit (Python 3.13). Upload descriptor files that you generated locally
            with the provided script to make predictions here.
        </p>
        <div style="background: white; 
                    padding: 1.5rem; 
                    border-radius: 8px; 
                    margin-top: 1.5rem;
                    border: 1px solid #e2e8f0;">
            <p style="color: #1a202c; font-weight: 600; margin-bottom: 0.5rem;">📋 What you'll get:</p>
            <ul style="color: #2d3748; font-size: 0.95rem; line-height: 2;">
                <li>✅ Predicted aggregation reduction percentage</li>
                <li>🎯 Effectiveness classification (effective/ineffective)</li>
                <li>📊 Confidence score</li>
                <li>⚠️ Optional toxicity tags (if precomputed)</li>
                <li>🔑 Most important molecular features (global)</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(
        "Need descriptors? Run `python scripts/precompute_descriptors.py --input your_smiles.csv` "
        "locally, then upload the resulting CSV."
    )
    
    feature_library = load_feature_library()
    tab_library, tab_upload = st.tabs(["Use training feature library", "Upload descriptor CSV"])
    
    with tab_library:
        if feature_library is None or feature_library.empty:
            st.warning("Local feature library not found. Run `python run_pipeline.py` locally to generate it.")
        else:
            st.markdown("""
            <div class="modern-card fade-in" style="background: #ffffff; border-left: 4px solid #667eea;">
                <p style="color: #2d3748; font-size: 1rem; line-height: 1.7; margin: 0;">
                    Select any compound from the training feature matrix to preview the model outputs
                    without running RDKit inside Streamlit.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            library_df = feature_library.reset_index(drop=True)
            option_labels = []
            for idx, row in library_df.iterrows():
                compound_id = row.get('compound_id', f'compound_{idx}')
                smiles = row.get('smiles', '') or ''
                truncated_smiles = (smiles[:60] + "...") if len(smiles) > 60 else smiles
                option_labels.append(f"{compound_id} | {truncated_smiles}")
            
            selected_label = st.selectbox("Select a compound from the library", option_labels)
            selected_index = option_labels.index(selected_label)
            selected_rows = library_df.iloc[[selected_index]]
            
            if st.button("🔍 Predict selected compound", type="primary"):
                with st.spinner("Loading precomputed descriptors..."):
                    predictions_df, top_features, missing = predict_from_feature_rows(
                        selected_rows, reg_model, clf_model, feature_names, reg_scaler, clf_scaler
                    )
                
                if predictions_df.empty:
                    st.error("No descriptors available for the selected compound.")
                else:
                    if missing > 0:
                        st.warning(f"{missing} trained features were missing. "
                                   "Rerun the local descriptor script to keep schemas in sync.")
                    st.success("✅ Prediction complete!")
                    render_prediction_ui(predictions_df.iloc[0].to_dict())
    
    with tab_upload:
        st.markdown("""
        <div class="modern-card fade-in" style="background: #ffffff; border-left: 4px solid #4facfe;">
            <p style="color: #2d3748; font-size: 1rem; line-height: 1.7; margin: 0;">
                Upload descriptor CSV files produced by <code>scripts/precompute_descriptors.py</code>.
                The file must include every feature used during training (stored in <code>models/feature_names.pkl</code>).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload descriptor CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                uploaded_df = None
            
            if uploaded_df is not None:
                st.success(f"Loaded {len(uploaded_df)} rows from the uploaded file.")
                st.dataframe(uploaded_df.head(), use_container_width=True)
                
                if st.button("🚀 Run predictions on uploaded descriptors", type="primary"):
                    with st.spinner("Running inference with uploaded descriptors..."):
                        predictions_df, top_features, missing = predict_from_feature_rows(
                            uploaded_df, reg_model, clf_model, feature_names, reg_scaler, clf_scaler
                        )
                    
                    if predictions_df.empty:
                        st.error("No valid descriptor rows found. Ensure the CSV contains the trained feature columns.")
                    else:
                        if missing > 0:
                            st.warning(
                                f"{missing} trained features were missing. "
                                "The model filled them with 0. Ensure you used the latest descriptor script."
                            )
                        st.success(f"Generated predictions for {len(predictions_df)} compounds.")
                        
                        render_prediction_ui(predictions_df.iloc[0].to_dict())
                        
                        summary_cols = [
                            col for col in [
                                'compound_id',
                                'smiles',
                                'predicted_aggregation_reduction',
                                'predicted_effectiveness',
                                'effectiveness_probability',
                                'toxicity_risk',
                                'toxicity_score'
                            ] if col in predictions_df.columns
                        ]
                        
                        st.subheader("Prediction summary (downloadable)")
                        st.dataframe(
                            predictions_df[summary_cols],
                            use_container_width=True,
                            height=300
                        )
                        
                        csv_data = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download predictions as CSV",
                            data=csv_data,
                            file_name="custom_predictions.csv",
                            mime="text/csv"
                        )


def show_model_info(reg_model, clf_model, feature_names):
    """Show model information."""
    st.markdown("""
    <div class="modern-card fade-in" style="text-align: center;">
        <h1 style="color: #667eea; margin: 0; font-size: 2.5rem; font-weight: 700;">
            📈 Model Information
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="modern-card fade-in" style="background: #f0f4ff;
                    border-left: 4px solid #667eea;">
            <h2 style="color: #667eea; margin-top: 0; font-weight: 700;">📉 Linear Regression Model</h2>
            <p style="color: #2d3748; margin: 0.8rem 0;"><strong>Purpose:</strong> Predicts continuous aggregation reduction percentage</p>
            <p style="color: #2d3748; margin: 0.8rem 0;"><strong>Model Type:</strong> Linear Regression</p>
            <ul style="color: #2d3748; line-height: 2;">
                <li>Simple, interpretable</li>
                <li>Fast predictions</li>
                <li>Easy to understand</li>
            </ul>
            <p style="color: #2d3748; margin: 1rem 0 0.5rem 0;"><strong>Performance:</strong></p>
            <ul style="color: #2d3748; line-height: 2;">
                <li>RMSE: 20.86 ± 3.52</li>
                <li>MAE: 16.73</li>
                <li>R²: 0.042</li>
            </ul>
            <p style="color: #2d3748; margin-top: 1rem; line-height: 1.7;">
                <strong>What it does:</strong> Takes molecular features and outputs a predicted percentage reduction in α-synuclein aggregation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card fade-in" style="background: #f0f9ff;
                    border-left: 4px solid #4facfe;">
            <h2 style="color: #4facfe; margin-top: 0; font-weight: 700;">🎯 Logistic Regression Model</h2>
            <p style="color: #2d3748; margin: 0.8rem 0;"><strong>Purpose:</strong> Classifies drugs as effective or ineffective</p>
            <p style="color: #2d3748; margin: 0.8rem 0;"><strong>Model Type:</strong> Logistic Regression</p>
            <ul style="color: #2d3748; line-height: 2;">
                <li>Binary classification</li>
                <li>Provides probability scores</li>
                <li>Interpretable coefficients</li>
            </ul>
            <p style="color: #2d3748; margin: 1rem 0 0.5rem 0;"><strong>Performance:</strong></p>
            <ul style="color: #2d3748; line-height: 2;">
                <li>ROC-AUC: 0.65 ± 0.02</li>
                <li>Accuracy: 60.8%</li>
                <li>F1-Score: 0.746</li>
            </ul>
            <p style="color: #2d3748; margin-top: 1rem; line-height: 1.7;">
                <strong>What it does:</strong> Takes molecular features and outputs a probability that the drug is effective (>20% reduction).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="modern-card fade-in" style="background: #f0fff4;
                border-left: 4px solid #48bb78;">
        <h2 style="color: #48bb78; margin-top: 0; font-weight: 700;">🔧 Feature Engineering</h2>
        <p style="color: #2d3748; margin: 1rem 0; font-size: 1.05rem;">
            The models use <strong style="color: #667eea;">2,237 features</strong> computed from molecular structures:
        </p>
        <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; border: 1px solid #e2e8f0;">
            <p style="color: #1a202c; margin: 0.8rem 0;"><strong>1. Molecular Descriptors:</strong></p>
            <ul style="color: #2d3748; line-height: 2;">
                <li>Molecular weight, LogP, TPSA</li>
                <li>H-bond donors/acceptors</li>
                <li>Rotatable bonds, aromatic rings</li>
                <li>Heteroatom count</li>
            </ul>
            <p style="color: #1a202c; margin: 1rem 0 0.8rem 0;"><strong>2. Simplified Fingerprints:</strong></p>
            <ul style="color: #2d3748; line-height: 2;">
                <li>Morgan fingerprint summary statistics</li>
                <li>Sum, mean, max, std of fingerprint bits</li>
            </ul>
            <p style="color: #1a202c; margin: 1rem 0 0.8rem 0;"><strong>3. Kinetic Descriptors:</strong></p>
            <ul style="color: #2d3748; line-height: 2;">
                <li>k_on, k_off (HSP90 binding rates)</li>
                <li>Residence time</li>
                <li>Log-transformed values</li>
            </ul>
            <p style="color: #1a202c; margin: 1rem 0 0.8rem 0;"><strong>4. PINK1 Descriptors:</strong></p>
            <ul style="color: #2d3748; line-height: 2;">
                <li>PINK1 IC50</li>
                <li>PINK1 activation score</li>
                <li>Log-transformed values</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="modern-card fade-in" style="background: #fffaf0;
                border-left: 4px solid #f6ad55;">
        <h2 style="color: #f6ad55; margin-top: 0; font-weight: 700;">📊 Training Data</h2>
        <ul style="color: #2d3748; font-size: 1rem; line-height: 2;">
            <li><strong>Total compounds:</strong> 2,291</li>
            <li><strong>Effective compounds:</strong> 1,556 (67.9%)</li>
            <li><strong>Ineffective compounds:</strong> 735 (32.1%)</li>
            <li><strong>Features:</strong> 2,237</li>
            <li><strong>Training method:</strong> 5-fold cross-validation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def show_screening_results():
    """Show screening results."""
    st.markdown("""
    <div class="modern-card fade-in" style="text-align: center;">
        <h1 style="color: #667eea; margin: 0; font-size: 2.5rem; font-weight: 700;">
            📋 Screening Results
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    results = load_screening_results()
    
    if results is None or len(results) == 0:
        st.warning("No screening results found. Run the pipeline to generate screening results.")
        return
    
    st.markdown(f"**Total compounds screened:** {len(results)}")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        effective_count = (results['predicted_effectiveness'] == 'effective').sum()
        st.metric("Predicted Effective", effective_count)
    
    with col2:
        avg_reduction = results['predicted_aggregation_reduction'].mean()
        st.metric("Avg. Reduction", f"{avg_reduction:.1f}%")
    
    with col3:
        avg_prob = results['effectiveness_probability'].mean() * 100
        st.metric("Avg. Confidence", f"{avg_prob:.1f}%")
    
    with col4:
        low_tox = (results['toxicity_risk'] == 'low').sum()
        st.metric("Low Toxicity", low_tox)
    
    st.markdown("---")
    
    # Filter options
    st.subheader("🔍 Filter Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filter_effective = st.checkbox("Show only effective compounds", value=False)
        min_reduction = st.slider("Minimum aggregation reduction (%)", 0, 100, 0)
    
    with col2:
        filter_toxicity = st.selectbox("Toxicity risk filter", ["All", "Low", "Medium", "High"])
        min_confidence = st.slider("Minimum confidence (%)", 0, 100, 0)
    
    # Apply filters
    filtered_results = results.copy()
    
    if filter_effective:
        filtered_results = filtered_results[filtered_results['predicted_effectiveness'] == 'effective']
    
    filtered_results = filtered_results[filtered_results['predicted_aggregation_reduction'] >= min_reduction]
    filtered_results = filtered_results[filtered_results['effectiveness_probability'] * 100 >= min_confidence]
    
    if filter_toxicity != "All":
        filtered_results = filtered_results[filtered_results['toxicity_risk'] == filter_toxicity.lower()]
    
    st.markdown(f"**Filtered results:** {len(filtered_results)} compounds")
    
    # Display table
    if len(filtered_results) > 0:
        # Sort by effectiveness probability
        filtered_results = filtered_results.sort_values('effectiveness_probability', ascending=False)
        
        # Select columns to display
        display_cols = ['smiles', 'predicted_aggregation_reduction', 'predicted_effectiveness', 
                       'effectiveness_probability', 'toxicity_risk']
        
        st.dataframe(
            filtered_results[display_cols].rename(columns={
                'smiles': 'SMILES',
                'predicted_aggregation_reduction': 'Predicted Reduction (%)',
                'predicted_effectiveness': 'Effectiveness',
                'effectiveness_probability': 'Confidence',
                'toxicity_risk': 'Toxicity Risk'
            }),
            width='stretch',
            height=400
        )
        
        # Download button
        csv = filtered_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Results (CSV)",
            data=csv,
            file_name="filtered_screening_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No compounds match the selected filters.")


if __name__ == "__main__":
    main()

