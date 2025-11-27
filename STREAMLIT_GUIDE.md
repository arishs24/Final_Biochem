# Streamlit Interface Guide

## Quick Start

1. **Install Streamlit dependencies** (descriptor-only environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** - Streamlit will automatically open at `http://localhost:8501`

4. **(Optional) Precompute descriptor files locally** – to test new compounds you must first run:
   ```bash
   # Run this locally with RDKit installed
   python scripts/precompute_descriptors.py --input data/raw/custom_smiles.csv
   ```
   Upload the resulting CSV on the "Interactive Testing" page.

## Features

The Streamlit interface includes 6 main sections:

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

### 🧊 3D Structure
- Plotly-based STL viewer for the uploaded HSP90 model (`data/5XR9-ribbon-vis_NIH3D.stl`)
- Drag to rotate, scroll to zoom, double-click to reset the camera
- Displays triangle count, centroid coordinates, and the source path

### 🔬 Interactive Testing
- **Upload descriptor CSVs**
- Descriptors must be precomputed locally with RDKit using `scripts/precompute_descriptors.py`
- See detailed results including:
  - Predicted aggregation reduction
  - Effectiveness classification
  - Confidence scores
  - Toxicity assessment (if included in your upload)
  - Top contributing features (global)

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

2. **3D viewer tips**:
   - Use the Plotly toolbar (top-right) to switch between orthographic/perspective views
   - Click the camera icon to save PNG snapshots for slides or lab notes

3. **Interactive testing workflow**:
   - Collect SMILES in a CSV (e.g., `data/raw/custom_smiles.csv`)
   - Run `python scripts/precompute_descriptors.py --input your_file.csv`
   - Upload the generated `custom_descriptors.csv` to the Streamlit app
   - Review predictions and download the results table

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

