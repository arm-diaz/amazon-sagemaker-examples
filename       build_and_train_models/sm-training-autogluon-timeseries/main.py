#!/usr/bin/env python3
"""
Demand Forecasting Pipeline - End-to-End Workflow
==================================================

This script demonstrates the complete workflow for training and deploying
a demand forecasting model using Amazon SageMaker and AutoGluon.

Usage:
    # Local testing with sample data
    python main.py --mode local --generate-data

    # Full SageMaker training and deployment
    python main.py --mode sagemaker --train-path data/train.csv

    # Just train locally (no SageMaker)
    python main.py --mode local-train --train-path data/train.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_preparation import DemandDataProcessor, prepare_forecast_data
from utils import (
    load_config,
    generate_demand_data,
    evaluate_forecasts,
    plot_forecast,
    check_aws_credentials
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_local_training(train_data: pd.DataFrame, config: dict) -> "TimeSeriesPredictor":
    """
    Train model locally using AutoGluon (no SageMaker).
    Good for development and testing.
    """
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    
    logger.info("Starting local training with AutoGluon...")
    
    # Convert to TimeSeriesDataFrame
    train_data["timestamp"] = pd.to_datetime(train_data["timestamp"])
    
    ts_data = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    
    logger.info(f"Training data: {ts_data.num_items} items, {len(ts_data)} total rows")
    
    # Create predictor
    predictor = TimeSeriesPredictor(
        path="models/local_model",
        prediction_length=config["model"]["prediction_length"],
        freq=config["data"]["freq"],
        target="target",
        eval_metric=config["evaluation"]["metrics"][0],
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    
    # Train
    predictor.fit(
        train_data=ts_data,
        presets=config["model"]["presets"],
        time_limit=config["model"]["time_limit"],
    )
    
    # Show leaderboard
    leaderboard = predictor.leaderboard(ts_data)
    logger.info(f"\nModel Leaderboard:\n{leaderboard}")
    
    return predictor


def run_local_inference(
    predictor: "TimeSeriesPredictor",
    test_data: pd.DataFrame
) -> pd.DataFrame:
    """Run inference locally."""
    from autogluon.timeseries import TimeSeriesDataFrame
    
    test_data["timestamp"] = pd.to_datetime(test_data["timestamp"])
    
    ts_data = TimeSeriesDataFrame.from_data_frame(
        test_data,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    
    predictions = predictor.predict(ts_data)
    
    return predictions.reset_index()


def run_sagemaker_pipeline(
    train_path: str,
    config: dict,
    deploy: bool = True
) -> dict:
    """
    Full SageMaker training and deployment pipeline.
    """
    from sagemaker_pipeline import DemandForecastingPipeline
    
    # Check AWS credentials
    if not check_aws_credentials():
        raise RuntimeError("AWS credentials not configured")
    
    # Initialize pipeline
    pipeline = DemandForecastingPipeline(
        role=config["aws"].get("role_arn"),
        region=config["aws"]["region"],
        bucket=config["aws"].get("s3_bucket"),
        prefix=config["aws"]["s3_prefix"]
    )
    
    # Prepare hyperparameters
    hyperparameters = {
        "prediction_length": config["model"]["prediction_length"],
        "freq": config["data"]["freq"],
        "presets": config["model"]["presets"],
        "time_limit": config["model"]["time_limit"],
        "eval_metric": config["evaluation"]["metrics"][0],
        "quantile_levels": ",".join(map(str, config["inference"]["quantiles"])),
    }
    
    if config["data"].get("static_features"):
        hyperparameters["static_features"] = ",".join(config["data"]["static_features"])
    
    if config["data"].get("known_covariates"):
        hyperparameters["known_covariates"] = ",".join(config["data"]["known_covariates"])
    
    # Train
    logger.info("Starting SageMaker training job...")
    training_job = pipeline.train(
        train_data=train_path,
        hyperparameters=hyperparameters,
        instance_type=config["training"]["instance_type"],
        instance_count=config["training"]["instance_count"],
        volume_size=config["training"]["volume_size"],
        max_runtime=config["training"]["max_runtime"],
        wait=True
    )
    
    result = {
        "training_job": training_job,
        "model_data": pipeline.model_data,
    }
    
    # Deploy if requested
    if deploy:
        logger.info("Deploying model to endpoint...")
        predictor = pipeline.deploy(
            endpoint_name=config["inference"]["endpoint_name"],
            instance_type=config["inference"]["instance_type"],
            instance_count=config["inference"]["instance_count"],
            wait=True
        )
        
        result["endpoint_name"] = pipeline.endpoint_name
        result["predictor"] = predictor
    
    return result


def demo_workflow():
    """
    Demonstrate the complete workflow with sample data.
    """
    logger.info("=" * 60)
    logger.info("DEMAND FORECASTING DEMO")
    logger.info("=" * 60)
    
    # Step 1: Generate sample data
    logger.info("\n[Step 1] Generating sample demand data...")
    
    df = generate_demand_data(
        n_items=20,
        n_days=365,
        start_date="2024-01-01",
        seed=42
    )
    
    # Step 2: Prepare data
    logger.info("\n[Step 2] Preparing data...")
    
    processor = DemandDataProcessor(
        item_id_column="item_id",
        timestamp_column="timestamp",
        target_column="target",
        freq="D"
    )
    
    # Validate
    validation = processor.validate_time_series(df)
    logger.info(f"Validation: {validation['num_items']} items, "
                f"{validation['total_rows']} rows")
    
    # Split data
    train_df, test_df = processor.train_test_split(df, prediction_length=30)
    
    # Save
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    
    logger.info(f"Saved train ({len(train_df)} rows) and test ({len(test_df)} rows)")
    
    # Step 3: Train model locally
    logger.info("\n[Step 3] Training model locally...")
    
    config = {
        "data": {"freq": "D"},
        "model": {
            "prediction_length": 30,
            "presets": "fast_training",  # Use fast for demo
            "time_limit": 120,  # 2 minutes
        },
        "evaluation": {"metrics": ["MASE"]}
    }
    
    predictor = run_local_training(train_df, config)
    
    # Step 4: Generate predictions
    logger.info("\n[Step 4] Generating forecasts...")
    
    # Use training data for forecasting (predict next 30 days)
    predictions = run_local_inference(predictor, train_df)
    
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Prediction columns: {list(predictions.columns)}")
    
    # Step 5: Evaluate
    logger.info("\n[Step 5] Evaluating forecasts...")
    
    if len(test_df) > 0:
        # Merge predictions with actuals
        eval_df = test_df.merge(
            predictions,
            on=["item_id", "timestamp"],
            how="inner"
        )
        
        if len(eval_df) > 0:
            metrics = evaluate_forecasts(
                test_df,
                predictions,
                target_column="target",
                item_id_column="item_id"
            )
            
            logger.info("Evaluation Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
    
    # Step 6: Visualize (if matplotlib available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        logger.info("\n[Step 6] Creating visualizations...")
        
        sample_item = train_df["item_id"].unique()[0]
        
        fig = plot_forecast(
            historical=train_df,
            predictions=predictions,
            item_id=sample_item,
            actuals=test_df if len(test_df) > 0 else None,
            title=f"Demand Forecast - {sample_item}"
        )
        
        fig.savefig("forecast_plot.png", dpi=150, bbox_inches="tight")
        logger.info("Saved forecast plot to forecast_plot.png")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)
    
    return predictor, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Demand Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run demo with sample data
    python main.py --mode demo

    # Train locally with your data
    python main.py --mode local-train --train-path data/train.csv

    # Full SageMaker pipeline
    python main.py --mode sagemaker --train-path data/train.csv --deploy

    # Generate sample data only
    python main.py --mode generate --output data/
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "local-train", "sagemaker", "generate"],
        default="demo",
        help="Execution mode"
    )
    
    parser.add_argument("--train-path", type=str, help="Path to training data CSV")
    parser.add_argument("--test-path", type=str, help="Path to test data CSV")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--deploy", action="store_true", help="Deploy model after training")
    parser.add_argument("--n-items", type=int, default=50, help="Number of items for data generation")
    parser.add_argument("--n-days", type=int, default=730, help="Number of days for data generation")
    
    args = parser.parse_args()
    
    # Load config if exists
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(args.config)
    else:
        config = {
            "data": {
                "freq": "D",
                "item_id_column": "item_id",
                "timestamp_column": "timestamp",
                "target_column": "target",
                "date_format": "%Y-%m-%d"
            },
            "model": {
                "prediction_length": 30,
                "presets": "medium_quality",
                "time_limit": 3600,
            },
            "evaluation": {"metrics": ["MASE", "RMSE", "MAPE"]},
            "aws": {
                "region": "us-east-1",
                "s3_prefix": "demand-forecasting"
            },
            "training": {
                "instance_type": "ml.m5.xlarge",
                "instance_count": 1,
                "volume_size": 50,
                "max_runtime": 86400
            },
            "inference": {
                "endpoint_name": f"demand-forecast-{datetime.now().strftime('%Y%m%d')}",
                "instance_type": "ml.m5.large",
                "instance_count": 1,
                "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9]
            }
        }
    
    # Execute based on mode
    if args.mode == "demo":
        demo_workflow()
    
    elif args.mode == "generate":
        df = generate_demand_data(
            n_items=args.n_items,
            n_days=args.n_days,
            seed=42
        )
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_dir / "demand_data.csv", index=False)
        logger.info(f"Generated data saved to {output_dir / 'demand_data.csv'}")
    
    elif args.mode == "local-train":
        if not args.train_path:
            raise ValueError("--train-path required for local-train mode")
        
        train_df = pd.read_csv(args.train_path)
        predictor = run_local_training(train_df, config)
        
        if args.test_path:
            test_df = pd.read_csv(args.test_path)
            predictions = run_local_inference(predictor, train_df)
            
            metrics = evaluate_forecasts(
                test_df,
                predictions,
                target_column="target"
            )
            
            logger.info(f"Evaluation metrics: {metrics}")
    
    elif args.mode == "sagemaker":
        if not args.train_path:
            raise ValueError("--train-path required for sagemaker mode")
        
        result = run_sagemaker_pipeline(
            args.train_path,
            config,
            deploy=args.deploy
        )
        
        logger.info(f"SageMaker pipeline result: {result}")


if __name__ == "__main__":
    main()
