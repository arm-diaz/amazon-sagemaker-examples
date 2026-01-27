"""
SageMaker Training Script for Demand Forecasting
=================================================
Uses AutoGluon-TimeSeries for training demand forecast models.
This script is executed inside a SageMaker training container.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# AutoGluon imports
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Data paths (set by SageMaker)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    
    # Model hyperparameters
    parser.add_argument("--prediction-length", type=int, default=30)
    parser.add_argument("--freq", type=str, default="D")
    parser.add_argument("--presets", type=str, default="medium_quality")
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--eval-metric", type=str, default="MASE")
    
    # Column names
    parser.add_argument("--target-column", type=str, default="target")
    parser.add_argument("--item-id-column", type=str, default="item_id")
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    
    # Advanced settings
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models")
    parser.add_argument("--quantile-levels", type=str, default="0.1,0.25,0.5,0.75,0.9")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--enable-ensemble", type=str, default="true")
    parser.add_argument("--static-features", type=str, default=None)
    parser.add_argument("--known-covariates", type=str, default=None)
    
    return parser.parse_args()


def load_data(data_path: str, args) -> TimeSeriesDataFrame:
    """
    Load and convert data to TimeSeriesDataFrame format.
    """
    logger.info(f"Loading data from {data_path}")
    
    # Find CSV file(s)
    data_dir = Path(data_path)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")
    
    # Load and concatenate all CSVs
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows from {len(csv_files)} files")
    
    # Parse timestamp
    df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column])
    
    # Convert to TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column=args.item_id_column,
        timestamp_column=args.timestamp_column
    )
    
    logger.info(f"Created TimeSeriesDataFrame with {ts_df.num_items} items")
    logger.info(f"Time range: {ts_df.index.get_level_values('timestamp').min()} to "
                f"{ts_df.index.get_level_values('timestamp').max()}")
    
    return ts_df


def get_model_list(models_str: str):
    """Parse comma-separated model list."""
    if models_str is None or models_str.strip() == "":
        return None  # Use all available models
    
    models = [m.strip() for m in models_str.split(",")]
    
    # Map common names to AutoGluon model names
    model_mapping = {
        "deepar": "DeepAR",
        "ets": "ETS",
        "arima": "ARIMA",
        "theta": "Theta",
        "autoets": "AutoETS",
        "chronos": "Chronos",
        "tft": "TemporalFusionTransformer",
        "temporal_fusion_transformer": "TemporalFusionTransformer",
        "recursive_tabular": "RecursiveTabular",
        "direct_tabular": "DirectTabular",
        "naive": "Naive",
        "seasonal_naive": "SeasonalNaive",
        "average": "Average",
    }
    
    return [model_mapping.get(m.lower(), m) for m in models]


def train(args):
    """Main training function."""
    logger.info("Starting training...")
    logger.info(f"Arguments: {args}")
    
    # Load training data
    train_data = load_data(args.train, args)
    
    # Load validation data if provided
    val_data = None
    if args.validation and os.path.exists(args.validation):
        val_data = load_data(args.validation, args)
        logger.info("Loaded validation data")
    
    # Parse quantile levels
    quantile_levels = [float(q) for q in args.quantile_levels.split(",")]
    
    # Parse model list
    models = get_model_list(args.models)
    
    # Parse static features and known covariates
    static_features = None
    if args.static_features:
        static_features = [f.strip() for f in args.static_features.split(",")]
    
    known_covariates = None
    if args.known_covariates:
        known_covariates = [f.strip() for f in args.known_covariates.split(",")]
    
    # Configure predictor
    predictor_kwargs = {
        "path": args.model_dir,
        "prediction_length": args.prediction_length,
        "freq": args.freq,
        "target": args.target_column,
        "eval_metric": args.eval_metric,
        "quantile_levels": quantile_levels,
    }
    
    if known_covariates:
        predictor_kwargs["known_covariates_names"] = known_covariates
    
    logger.info(f"Creating TimeSeriesPredictor with config: {predictor_kwargs}")
    predictor = TimeSeriesPredictor(**predictor_kwargs)
    
    # Configure fit arguments
    fit_kwargs = {
        "train_data": train_data,
        "presets": args.presets,
        "time_limit": args.time_limit,
        "enable_ensemble": args.enable_ensemble.lower() == "true",
    }
    
    if val_data is not None:
        fit_kwargs["tuning_data"] = val_data
    
    if models:
        fit_kwargs["hyperparameters"] = {model: {} for model in models}
        logger.info(f"Training with specified models: {models}")
    
    if args.num_gpus > 0:
        fit_kwargs["num_gpus"] = args.num_gpus
    
    # Train the model
    logger.info("Starting model training...")
    predictor.fit(**fit_kwargs)
    
    # Log training results
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    
    # Leaderboard
    leaderboard = predictor.leaderboard(train_data)
    logger.info(f"\nModel Leaderboard:\n{leaderboard.to_string()}")
    
    # Feature importance (if available)
    try:
        importance = predictor.feature_importance(train_data)
        if importance is not None and len(importance) > 0:
            logger.info(f"\nFeature Importance:\n{importance.to_string()}")
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
    
    # Save training metrics
    metrics = {
        "leaderboard": leaderboard.to_dict(),
        "best_model": predictor.model_best,
        "prediction_length": args.prediction_length,
        "freq": args.freq,
        "num_items": train_data.num_items,
    }
    
    metrics_path = os.path.join(args.output_data_dir, "metrics.json")
    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Evaluate on validation data if available
    if val_data is not None:
        logger.info("\nEvaluating on validation data...")
        evaluation = predictor.evaluate(val_data)
        logger.info(f"Validation metrics: {evaluation}")
        
        eval_path = os.path.join(args.output_data_dir, "evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(evaluation, f, indent=2, default=str)
    
    logger.info(f"\nModel saved to {args.model_dir}")
    return predictor


def model_fn(model_dir: str):
    """
    Load model for inference (called by SageMaker inference container).
    """
    logger.info(f"Loading model from {model_dir}")
    predictor = TimeSeriesPredictor.load(model_dir)
    return predictor


if __name__ == "__main__":
    args = parse_args()
    
    # Validate required arguments
    if args.train is None:
        raise ValueError("Training data path is required (--train)")
    
    train(args)
