"""
Main pipeline runner for HSP90 inhibitor screening project.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, get_data_dir, get_models_dir, get_figures_dir, get_output_dir
from src.data_loader import load_all_data
from src.preprocess import preprocess_pipeline
from src.featurize import featurize_dataset, save_features, load_features
from src.train import train_all_models
from src.evaluate import evaluate_all_models
from src.screen import run_screening

logger = logging.getLogger(__name__)


def main():
    """Execute the complete pipeline."""
    # Setup logging
    setup_logging("INFO")
    logger.info("=" * 80)
    logger.info("HSP90 Inhibitor Screening Pipeline")
    logger.info("=" * 80)
    
    try:
        # Step 1: Data Loading
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Datasets (HSP90 + α-synuclein + PINK1)")
        logger.info("=" * 80)
        hsp90_df, alpha_syn_df, pink1_df = load_all_data()
        logger.info(f"HSP90 dataset: {len(hsp90_df)} compounds")
        logger.info(f"α-synuclein dataset: {len(alpha_syn_df)} compounds")
        logger.info(f"PINK1 dataset: {len(pink1_df)} compounds")
        
        # Step 2: Preprocessing
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Preprocessing and Merging Datasets")
        logger.info("=" * 80)
        merged_df = preprocess_pipeline(hsp90_df, alpha_syn_df, pink1_df)
        logger.info(f"Merged dataset: {len(merged_df)} compounds")
        
        # Step 3: Feature Engineering
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 80)
        
        # Check if features already exist
        features_file = get_data_dir("processed") / "features.pkl"
        if features_file.exists():
            logger.info("Loading existing features...")
            features_df = load_features()
        else:
            logger.info("Computing molecular features...")
            features_df = featurize_dataset(merged_df)
            save_features(features_df)
        
        logger.info(f"Feature matrix shape: {features_df.shape}")
        
        # Step 4: Model Training
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Training Models")
        logger.info("=" * 80)
        
        # Check if models already exist
        models_dir = get_models_dir()
        if (models_dir / "xgboost_regressor.pkl").exists() and \
           (models_dir / "xgboost_classifier.pkl").exists():
            logger.info("Models already exist. Skipping training.")
            logger.info("To retrain, delete models in /models/ directory")
        else:
            train_results = train_all_models(features_df)
            logger.info("Model training completed")
        
        # Step 5: Model Evaluation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Evaluating Models")
        logger.info("=" * 80)
        eval_results = evaluate_all_models(features_df)
        logger.info("Model evaluation completed")
        logger.info(f"Figures saved to {get_figures_dir()}")
        
        # Step 6: Compound Screening
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Screening New Compounds")
        logger.info("=" * 80)
        screening_results = run_screening(n_compounds=50)
        logger.info(f"Screening completed. Results saved to {get_output_dir() / 'screening_results.csv'}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Processed compounds: {len(merged_df)}")
        logger.info(f"Features computed: {features_df.shape[1]} (simplified)")
        logger.info(f"Models trained: Linear Regression + Logistic Regression (simple & straightforward)")
        logger.info(f"Compounds screened: {len(screening_results)}")
        logger.info(f"\nOutput directories:")
        logger.info(f"  - Data: {get_data_dir()}")
        logger.info(f"  - Models: {get_models_dir()}")
        logger.info(f"  - Figures: {get_figures_dir()}")
        logger.info(f"  - Results: {get_output_dir()}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


