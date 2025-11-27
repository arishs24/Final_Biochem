"""
Utility script to precompute molecular descriptors locally using RDKit.

This script stays local so that the Streamlit deployment can operate without
RDKit while still using descriptor files computed offline.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.featurize import featurize_dataset  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir,
    get_data_dir,
    save_pickle,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute RDKit descriptors locally for deployment."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=get_data_dir("raw") / "custom_smiles.csv",
        help="CSV file with at least a 'smiles' column.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=get_data_dir("processed") / "custom_descriptors.csv",
        help="Where to save the descriptor matrix (CSV) for uploading to Streamlit.",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=get_data_dir("processed") / "custom_descriptors.pkl",
        help="Optional pickle output that mirrors the CSV file.",
    )
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="Column name that contains SMILES strings.",
    )
    parser.add_argument(
        "--target-col",
        default="aggregation_reduction",
        help="Continuous target column (optional).",
    )
    parser.add_argument(
        "--binary-target-col",
        default="is_effective",
        help="Binary target column (optional).",
    )
    parser.add_argument(
        "--metadata-cols",
        nargs="*",
        default=None,
        help="Optional list of additional columns to carry over into the descriptor file.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging("INFO")
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    logging.info("Loading input molecules from %s", args.input)
    input_df = pd.read_csv(args.input)

    if args.smiles_col not in input_df.columns:
        raise ValueError(f"Input file must include a '{args.smiles_col}' column.")

    descriptors_df = featurize_dataset(
        input_df,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        binary_target_col=args.binary_target_col,
    )

    # Re-attach optional metadata
    if args.metadata_cols:
        metadata_cols = [
            col for col in args.metadata_cols if col in input_df.columns and col != args.smiles_col
        ]
        if metadata_cols:
            metadata_df = (
                input_df[[args.smiles_col] + metadata_cols]
                .drop_duplicates(subset=[args.smiles_col])
                .rename(columns={args.smiles_col: "smiles"})
            )
            descriptors_df = descriptors_df.merge(metadata_df, on="smiles", how="left")

    ensure_dir(args.output_csv.parent)
    descriptors_df.to_csv(args.output_csv, index=False)
    logging.info("Saved descriptor CSV to %s", args.output_csv)

    if args.output_pkl:
        ensure_dir(args.output_pkl.parent)
        save_pickle(descriptors_df, str(args.output_pkl))
        logging.info("Saved descriptor pickle to %s", args.output_pkl)

    logging.info("Descriptor generation complete. Rows: %d", len(descriptors_df))


if __name__ == "__main__":
    main()

