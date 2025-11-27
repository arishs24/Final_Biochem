# Streamlit Interface Guide

## Quick Start

1. **Install Streamlit** (if not already installed):
   ```bash
   pip install streamlit
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** - Streamlit will automatically open at `http://localhost:8501`

## Features

The Streamlit interface includes 5 main sections:

### 🏠 Overview
- Project introduction and background
- Pipeline steps explanation
- Key results summary
- What the metrics mean

### 📊 Results & Visualizations
- **Predicted vs Actual** - Shows model prediction accuracy
- **ROC Curves** - Shows drug detection performance
- **Feature Importance** - Shows which molecular properties matter most
- **Correlation Heatmap** - Shows relationships between features

### 🔬 Interactive Testing
- **Test your own compounds!**
- Enter SMILES strings to get predictions
- See detailed results including:
  - Predicted aggregation reduction
  - Effectiveness classification
  - Confidence scores
  - Toxicity assessment
  - Top contributing features

### 📈 Model Information
- Detailed model descriptions
- Performance metrics
- Feature engineering explanation
- Training data statistics

### 📋 Screening Results
- View all screened compounds
- Filter by effectiveness, toxicity, confidence
- Download filtered results as CSV
- Summary statistics

## Usage Tips

1. **Before running the app**, make sure you've run the pipeline at least once:
   ```bash
   python run_pipeline.py
   ```
   This generates the models and figures needed by the app.

2. **For interactive testing**, you can:
   - Type SMILES manually
   - Select from example SMILES
   - Test multiple compounds by running predictions multiple times

3. **SMILES strings** can be found on:
   - PubChem (https://pubchem.ncbi.nlm.nih.gov/)
   - ChEMBL (https://www.ebi.ac.uk/chembl/)
   - Or use the example SMILES provided in the app

## Troubleshooting

- **Models not found**: Run `python run_pipeline.py` first
- **Figures not showing**: Make sure the pipeline completed successfully
- **SMILES errors**: Check that the SMILES string is valid (no typos, proper format)

## Customization

You can customize the app by editing `app.py`:
- Change colors/styles in the CSS section
- Add more example SMILES
- Modify the layout
- Add additional visualizations

