# HSP90 and PINK1 Inhibitor Screening for α-Synuclein Aggregation Reduction

## Project Purpose

This project implements a simplified machine learning pipeline to predict which HSP90 (Heat Shock Protein 90) and PINK1 (PTEN-induced kinase 1) modulators are most effective at reducing α-synuclein aggregation, a key pathological process in Parkinson's disease.

## Biochemical Rationale

### HSP90, PINK1, and α-Synuclein Aggregation

- **HSP90** is a molecular chaperone that plays a crucial role in protein folding and stability. It has been implicated in neurodegenerative diseases, including Parkinson's disease.

- **PINK1** (PTEN-induced kinase 1) is a mitochondrial kinase involved in mitochondrial quality control. Mutations in PINK1 are associated with early-onset Parkinson's disease. PINK1 activation can protect against α-synuclein aggregation.

- **α-Synuclein** is a presynaptic protein that aggregates into toxic fibrils in Parkinson's disease. The aggregation process is a hallmark of the disease pathology.

- **Therapeutic Hypothesis**: HSP90 inhibitors and PINK1 modulators may modulate α-synuclein aggregation by:
  - Altering protein folding pathways (HSP90)
  - Improving mitochondrial quality control (PINK1)
  - Affecting chaperone-mediated clearance
  - Modulating cellular stress responses
  - Directly interfering with aggregation kinetics

### Computational Approach

This project uses machine learning to:
1. Learn relationships between HSP90 inhibitor chemical structures and their effects on α-synuclein aggregation
2. Predict effectiveness of new compounds before experimental testing
3. Identify key molecular features that correlate with therapeutic efficacy

## Dataset Sources

### 1. HSP90-SPR Dataset
- **Source**: HITS (Heidelberg Institute for Theoretical Studies)
- **URL**: https://kbbox.h-its.org/toolbox/data/hsp90-spr/
- **Contents**:
  - ~140 HSP90 inhibitors
  - SMILES strings
  - Kinetic parameters (k_on, k_off)
  - Affinity metadata

### 2. α-Synuclein Aggregation Data
- **Source**: Nature Parkinson's Disease Journal
- **URL**: https://www.nature.com/articles/s41531-024-00830-y
- **Contents**:
  - α-syn aggregation levels
  - Effects of small-molecule modulators
  - Biochemical response variables

### 3. PINK1 Data
- **Source**: ChEMBL database
- **Contents**:
  - PINK1 inhibition/activation data
  - IC50 values for PINK1 modulators
  - PINK1 activation scores

## Model Description (Simplified)

### Feature Engineering
- **Molecular Descriptors** (RDKit):
  - Very simplified Morgan fingerprints (summary statistics only: sum, mean, max, std)
  - Molecular weight, LogP, TPSA
  - H-bond donor/acceptor counts
  - Rotatable bonds, aromatic rings, heteroatoms

- **Kinetic Descriptors**:
  - k_on, k_off (HSP90 binding)
  - Residence time, log(k_off), log(k_on)

- **PINK1 Descriptors**:
  - PINK1 IC50 (inhibition concentration)
  - PINK1 activation score
  - log(PINK1 IC50)

### Machine Learning Models (Simple & Straightforward)

#### Regression Models (Continuous Target)
- **Linear Regression** - Simple, interpretable model for predicting aggregation reduction

#### Classification Models (Binary Target - Drug Detection)
- **Logistic Regression** - Simple, straightforward model for detecting effective vs ineffective drugs

### Evaluation Metrics
- **Regression**: MAE, RMSE, R²
- **Classification**: Accuracy, F1-score, ROC-AUC

## Usage Instructions

### Prerequisites

The project now separates the local RDKit workflow from the deployable Streamlit app:

- **Local pipeline (with RDKit, dataset merging, descriptor generation, and screening)**  
  ```bash
  pip install -r requirements-local.txt
  ```
  RDKit wheels are not available on Streamlit Cloud, so install locally (conda works well):
  ```bash
  conda install -c conda-forge rdkit
  ```

- **Deployable Streamlit app (no RDKit, descriptors are precomputed offline)**  
  ```bash
  pip install -r requirements.txt
  ```

Keeping two environments prevents the cloud deployment from pulling in RDKit while still letting you run the full cheminformatics pipeline on your laptop.

### Running the Pipeline

Execute the complete workflow:

```bash
python run_pipeline.py
```

This will:
1. Download datasets from the specified URLs
2. Process and merge the data
3. Generate molecular features (locally with RDKit)
4. Train regression and classification models
5. Evaluate model performance
6. Screen new compounds from ChEMBL

### Precomputing descriptors for Streamlit uploads

Interactive testing on Streamlit Cloud relies on descriptor files generated offline. Use the helper script to convert any CSV containing SMILES into the feature schema used during training:

```bash
python scripts/precompute_descriptors.py \
    --input data/raw/custom_smiles.csv \
    --output-csv data/processed/custom_descriptors.csv
```

The resulting CSV (and optional pickle) can be uploaded to the Streamlit app for inference. You can pass `--metadata-cols column_a column_b` if you want to carry additional annotations (e.g., `source`, `notes`) into the deployment file.

### Running the Streamlit Interface

Launch the interactive web interface:

```bash
streamlit run app.py
```

The Streamlit interface provides:
- 📊 **Visualizations** - All plots and figures
- 🔬 **Interactive Testing** - Upload descriptor CSVs that were precomputed locally
- 📈 **Model Information** - Detailed model descriptions
- 📋 **Screening Results** - View and filter screened compounds
- 🏠 **Overview** - Complete project explanation

Open your browser to `http://localhost:8501` after running the command.

### Individual Module Usage

You can also run individual components:

```python
# Data loading
from src.data_loader import load_hsp90_data, load_alpha_syn_data
hsp90_df = load_hsp90_data()
alpha_syn_df = load_alpha_syn_data()

# Feature engineering
from src.featurize import featurize_dataset
features = featurize_dataset(alpha_syn_df)

# Model training
from src.train import train_all_models
train_results = train_all_models(features)

# Evaluation
from src.evaluate import evaluate_all_models
evaluate_all_models(features)

# Screening
from src.screen import screen_compounds, download_chembl_hsp90_inhibitors
compounds_df = download_chembl_hsp90_inhibitors(n_compounds=50)
screen_results = screen_compounds(compounds_df)
```

## Project Structure

```
project_root/
  data/
    raw/              # Downloaded raw datasets
    processed/        # Processed and merged data
  src/
    data_loader.py    # Dataset downloading and parsing
    preprocess.py     # Data cleaning and merging
    featurize.py      # Molecular descriptor computation
    train.py          # Model training
    evaluate.py       # Model evaluation and visualization
    screen.py         # Compound screening
    utils.py          # Helper functions
  models/             # Trained model files
  figures/            # Generated plots and visualizations
  output/             # Screening results and predictions
  run_pipeline.py     # Main execution script
  README.md
```

## Dependencies

### Local pipeline (RDKit-enabled)

- Python 3.10+
- RDKit 2022.09+ (conda recommended)
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- requests >= 2.31.0
- beautifulsoup4 >= 4.12.0
- chembl-webresource-client >= 0.10.8

### Streamlit deployment (descriptor-only)

- Python 3.13 (Streamlit Cloud default)
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- streamlit >= 1.28.0
- joblib >= 1.3.0
- scipy >= 1.11.0

## Output Files

- `/data/processed/merged.csv`: Merged dataset with SMILES and aggregation data
- `/data/processed/features.pkl`: Feature matrix with all molecular descriptors
- `/models/*.pkl`: Trained model files
- `/figures/*.png`: Evaluation plots (feature importance, ROC curves, etc.)
- `/output/screening_results.csv`: Predictions for new compounds

## Notes

- The pipeline automatically downloads data from the specified URLs
- If datasets are not available, the code includes fallback mechanisms
- All models are saved for future use
- Screening results include toxicity risk assessment using SMARTS patterns

## License

This project is for research purposes only.


