"""
Streamlit interface for HSP90 and PINK1 Inhibitor Screening Project
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_models_dir, get_figures_dir, get_output_dir, get_data_dir, load_pickle
from src.featurize import featurize_compound
from src.screen import check_toxicity_smarts

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


def predict_compound(smiles, reg_model, clf_model, feature_names, reg_scaler, clf_scaler):
    """Predict effectiveness for a single compound."""
    try:
        # Featurize compound
        features = featurize_compound(smiles, use_fingerprints=True)
        
        if not features:
            return None, "Invalid SMILES string"
        
        # Create feature vector
        feature_vector = []
        for feat_name in feature_names:
            if feat_name in features:
                value = features[feat_name]
                if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    feature_vector.append(0.0)
                else:
                    feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
        
        X = np.array([feature_vector])
        X_df = pd.DataFrame(X, columns=feature_names)
        X_df = X_df.fillna(0.0)
        
        # Scale features
        if reg_scaler is not None:
            X_reg_scaled = reg_scaler.transform(X_df)
        else:
            X_reg_scaled = X_df
        
        if clf_scaler is not None:
            X_clf_scaled = clf_scaler.transform(X_df)
        else:
            X_clf_scaled = X_df
        
        # Ensure no NaN
        X_reg_scaled = np.nan_to_num(X_reg_scaled, nan=0.0)
        X_clf_scaled = np.nan_to_num(X_clf_scaled, nan=0.0)
        
        # Make predictions
        pred_aggregation_reduction = reg_model.predict(X_reg_scaled)[0]
        pred_proba = clf_model.predict_proba(X_clf_scaled)[0]
        pred_effective = clf_model.predict(X_clf_scaled)[0]
        
        # Get top features
        if hasattr(clf_model, 'coef_'):
            coef = clf_model.coef_[0] if clf_model.coef_.ndim > 1 else clf_model.coef_
            top_indices = np.argsort(np.abs(coef))[::-1][:5]
            top_features = [feature_names[i] for i in top_indices]
        else:
            top_features = []
        
        # Check toxicity
        toxicity = check_toxicity_smarts(smiles)
        
        return {
            'smiles': smiles,
            'predicted_aggregation_reduction': pred_aggregation_reduction,
            'predicted_effectiveness': 'effective' if pred_effective == 1 else 'ineffective',
            'effectiveness_probability': pred_proba[1],
            'toxicity_risk': toxicity['toxicity_risk'],
            'toxicity_score': toxicity['toxicity_score'],
            'toxicity_alerts': ', '.join(toxicity['alerts']) if toxicity['alerts'] else 'none',
            'top_features': top_features
        }, None
        
    except Exception as e:
        return None, str(e)


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
    """Show interactive testing interface."""
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
            🔬 Test Your Own Compounds!
        </h2>
        <p style="color: #2d3748; font-size: 1.05rem; line-height: 1.7; text-align: center;">
            Enter a SMILES string (chemical structure) below to get predictions for drug effectiveness.
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
                <li>⚠️ Toxicity risk assessment</li>
                <li>🔑 Most important molecular features</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Example SMILES
    example_smiles = [
        "CC(C)OC(=O)Nc1ccc(C(=O)Nc2ccc(C(F)(F)F)cc2)cc1",
        "COc1cc2c(cc1OC)OC(=O)C(=C2O)C(=O)Nc1ccc(C(F)(F)F)cc1",
        "CC1=C(C(=O)Nc2ccc(C(F)(F)F)cc2)OC(=O)c2ccccc21"
    ]
    
    st.subheader("Enter SMILES String")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Type manually", "Select example"]
    )
    
    if input_method == "Select example":
        selected_example = st.selectbox("Choose an example:", example_smiles)
        smiles_input = st.text_input("SMILES String:", value=selected_example)
    else:
        smiles_input = st.text_input("SMILES String:", placeholder="e.g., CC(C)OC(=O)Nc1ccc(C(=O)Nc2ccc(C(F)(F)F)cc2)cc1")
    
    if st.button("🔍 Predict", type="primary"):
        if not smiles_input:
            st.error("Please enter a SMILES string")
        else:
            with st.spinner("Computing features and making predictions..."):
                result, error = predict_compound(smiles_input, reg_model, clf_model, feature_names, reg_scaler, clf_scaler)
            
            if error:
                st.error(f"Error: {error}")
            elif result:
                st.success("✅ Prediction complete!")
                
                # Display results with modern styling
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
                            {result['predicted_aggregation_reduction']:.1f}%
                        </h2>
                        <p style="color: #4a5568; font-size: 0.85rem; margin: 0;">
                            Expected reduction in α-synuclein aggregation
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    effectiveness_color = "#48bb78" if result['predicted_effectiveness'] == 'effective' else "#f56565"
                    effectiveness_icon = "✅" if result['predicted_effectiveness'] == 'effective' else "❌"
                    bg_color = "#f0fff4" if result['predicted_effectiveness'] == 'effective' else "#fff5f5"
                    st.markdown(f"""
                    <div class="modern-card fade-in" style="text-align: center;
                                background: {bg_color};
                                border-left: 4px solid {effectiveness_color};
                                padding: 1.5rem;">
                        <p style="color: #2d3748; font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 600;">
                            Effectiveness
                        </p>
                        <h2 style="color: {effectiveness_color}; font-size: 2rem; margin: 0.5rem 0; font-weight: 700;">
                            {effectiveness_icon} {result['predicted_effectiveness'].title()}
                        </h2>
                        <p style="color: #4a5568; font-size: 0.85rem; margin: 0;">
                            Binary classification result
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    prob_pct = result['effectiveness_probability'] * 100
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
                
                # Detailed results with modern styling
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="modern-card fade-in" style="background: #f7fafc;
                                border-left: 4px solid #667eea;
                                margin-top: 1rem;">
                        <h3 style="color: #1a202c; margin-top: 0; font-weight: 700;">📋 Detailed Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="modern-card fade-in" style="padding: 1.5rem; margin-top: 0.5rem;">
                        <p style="color: #2d3748; margin: 0.8rem 0;">
                            <strong style="color: #1a202c;">SMILES:</strong><br>
                            <code style="background: #f7fafc; color: #2d3748; padding: 0.5rem; border-radius: 6px; display: block; margin-top: 0.5rem; word-break: break-all; font-size: 0.9rem; border: 1px solid #e2e8f0;">{result['smiles']}</code>
                        </p>
                        <p style="color: #2d3748; margin: 0.8rem 0;">
                            <strong style="color: #1a202c;">Toxicity Risk:</strong> 
                            <span style="background: {'#48bb78' if result['toxicity_risk'] == 'low' else '#ed8936' if result['toxicity_risk'] == 'medium' else '#f56565'}; 
                                         color: white; 
                                         padding: 0.3rem 0.8rem; 
                                         border-radius: 6px; 
                                         font-weight: 600;
                                         font-size: 0.9rem;">
                                {result['toxicity_risk'].upper()}
                            </span>
                        </p>
                        <p style="color: #2d3748; margin: 0.8rem 0;">
                            <strong style="color: #1a202c;">Toxicity Score:</strong> {result['toxicity_score']:.2f}
                        </p>
                        <p style="color: #2d3748; margin: 0.8rem 0;">
                            <strong style="color: #1a202c;">Toxicity Alerts:</strong> 
                            {result['toxicity_alerts'] if result['toxicity_alerts'] != 'none' else 'None detected ✅'}
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
                    
                    if result['top_features']:
                        # Display features in a cleaner way - avoid code-like appearance
                        for i, feat in enumerate(result['top_features'], 1):
                            formatted_feat = format_feature_name(feat)
                            # Use Streamlit's native markdown to avoid HTML rendering issues
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
                        st.info("No feature importance available")
                
                # Interpretation with modern styling
                st.markdown("---")
                
                if result['predicted_effectiveness'] == 'effective':
                    st.markdown(f"""
                    <div class="modern-card fade-in" style="background: #f0fff4;
                                border-left: 4px solid #48bb78;
                                margin-top: 1rem;">
                        <h3 style="color: #1a202c; margin-top: 0; font-size: 1.4rem; font-weight: 700;">💡 Interpretation</h3>
                        <p style="color: #2d3748; font-size: 1.05rem; line-height: 1.8; margin-bottom: 0.8rem;">
                            ✅ This compound is predicted to be <strong style="color: #48bb78;">effective</strong> at reducing α-synuclein aggregation.
                        </p>
                        <ul style="color: #2d3748; font-size: 1rem; line-height: 2;">
                            <li>Expected reduction: <strong>{result['predicted_aggregation_reduction']:.1f}%</strong></li>
                            <li>Confidence: <strong>{prob_pct:.1f}%</strong></li>
                            <li>Toxicity risk: <strong>{result['toxicity_risk'].upper()}</strong></li>
                        </ul>
                        <p style="color: #2d3748; font-size: 1rem; margin-top: 1rem; font-weight: 600; padding: 1rem; background: #e6ffed; border-radius: 6px;">
                            🎉 This compound shows promise and may be worth testing experimentally!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="modern-card fade-in" style="background: #fff5f5;
                                border-left: 4px solid #f56565;
                                margin-top: 1rem;">
                        <h3 style="color: #1a202c; margin-top: 0; font-size: 1.4rem; font-weight: 700;">💡 Interpretation</h3>
                        <p style="color: #2d3748; font-size: 1.05rem; line-height: 1.8; margin-bottom: 0.8rem;">
                            ❌ This compound is predicted to be <strong style="color: #f56565;">ineffective</strong> at reducing α-synuclein aggregation.
                        </p>
                        <ul style="color: #2d3748; font-size: 1rem; line-height: 2;">
                            <li>Expected reduction: <strong>{result['predicted_aggregation_reduction']:.1f}%</strong></li>
                            <li>Confidence: <strong>{prob_pct:.1f}%</strong></li>
                            <li>Toxicity risk: <strong>{result['toxicity_risk'].upper()}</strong></li>
                        </ul>
                        <p style="color: #2d3748; font-size: 1rem; margin-top: 1rem; font-weight: 600; padding: 1rem; background: #ffe5e5; border-radius: 6px;">
                            ⚠️ This compound may not be suitable for further development.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="modern-card fade-in" style="background: #f7fafc;
                border-left: 4px solid #667eea;
                margin-top: 2rem;">
        <h3 style="color: #1a202c; margin-top: 0; font-size: 1.4rem; font-weight: 700;">📚 About SMILES</h3>
        <p style="color: #2d3748; font-size: 1rem; line-height: 1.7;">
            <strong>SMILES</strong> (Simplified Molecular Input Line Entry System) is a notation for describing chemical structures.
        </p>
        <div style="background: white; 
                    padding: 1.5rem; 
                    border-radius: 8px; 
                    margin: 1rem 0;
                    border: 1px solid #e2e8f0;">
            <p style="color: #1a202c; font-weight: 600; margin-bottom: 0.5rem;">💡 Examples:</p>
            <ul style="color: #2d3748; font-size: 0.95rem; line-height: 2;">
                <li><strong>Simple molecules:</strong> 
                    <code style="background: #f7fafc; color: #2d3748; padding: 0.3rem 0.6rem; border-radius: 4px; border: 1px solid #e2e8f0;">C</code> (methane), 
                    <code style="background: #f7fafc; color: #2d3748; padding: 0.3rem 0.6rem; border-radius: 4px; border: 1px solid #e2e8f0;">CC</code> (ethane)
                </li>
                <li><strong>Complex drugs:</strong> 
                    <code style="background: #f7fafc; color: #2d3748; padding: 0.3rem 0.6rem; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 0.85rem; word-break: break-all;">CC(C)OC(=O)Nc1ccc(C(=O)Nc2ccc(C(F)(F)F)cc2)cc1</code>
                </li>
            </ul>
        </div>
        <p style="color: #2d3748; font-size: 1rem;">
            🌐 You can find SMILES strings for compounds on websites like 
            <a href="https://pubchem.ncbi.nlm.nih.gov/" target="_blank" style="color: #667eea; font-weight: 600;">PubChem</a> or 
            <a href="https://www.ebi.ac.uk/chembl/" target="_blank" style="color: #667eea; font-weight: 600;">ChEMBL</a>.
        </p>
    </div>
    """, unsafe_allow_html=True)


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

