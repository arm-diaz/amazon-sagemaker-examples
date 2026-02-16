#!/usr/bin/env python3
"""
Fuel Delivery Optimization Pipeline - End-to-End Workflow
==========================================================

This script demonstrates the complete workflow for training and deploying
a fuel delivery consumption forecasting model using Amazon SageMaker and
AutoGluon, then generating ML-driven delivery recommendations.

Usage:
    # Local demo with sample data
    python main.py --mode demo

    # Train locally with your data
    python main.py --mode local-train --train-path data/train.csv

    # Full SageMaker training and deployment
    python main.py --mode sagemaker --train-path data/train.csv --deploy

    # Generate sample data only
    python main.py --mode generate --output data/
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

from data_preparation import FuelDeliveryDataProcessor, prepare_forecast_data
from utils import (
    load_config,
    generate_fuel_delivery_data,
    evaluate_forecasts,
    plot_consumption_forecast,
    check_aws_credentials,
    calculate_business_metrics,
)
from optimization import DeliveryConstraints, DeliveryOptimizer, DeliveryScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_local_training(train_data: pd.DataFrame, config: dict) -> "TimeSeriesPredictor":
    """Train model locally using AutoGluon (no SageMaker)."""
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    logger.info("Starting local training with AutoGluon...")

    train_data["reading_date"] = pd.to_datetime(train_data["reading_date"])

    ts_data = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column="tank_id",
        timestamp_column="reading_date",
    )

    logger.info(f"Training data: {ts_data.num_items} tanks, {len(ts_data)} total rows")

    predictor = TimeSeriesPredictor(
        path="models/local_model",
        prediction_length=config["model"]["prediction_length"],
        freq=config["data"]["freq"],
        target="daily_consumption",
        eval_metric=config["evaluation"]["metrics"][0],
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
    )

    predictor.fit(
        train_data=ts_data,
        presets=config["model"]["presets"],
        time_limit=config["model"]["time_limit"],
    )

    leaderboard = predictor.leaderboard(ts_data)
    logger.info(f"\nModel Leaderboard:\n{leaderboard}")

    return predictor


def run_local_inference(
    predictor: "TimeSeriesPredictor",
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run inference locally."""
    from autogluon.timeseries import TimeSeriesDataFrame

    test_data["reading_date"] = pd.to_datetime(test_data["reading_date"])

    ts_data = TimeSeriesDataFrame.from_data_frame(
        test_data,
        id_column="tank_id",
        timestamp_column="reading_date",
    )

    predictions = predictor.predict(ts_data)
    return predictions.reset_index()


def run_sagemaker_pipeline(
    train_path: str,
    config: dict,
    deploy: bool = True,
) -> dict:
    """Full SageMaker training and deployment pipeline."""
    from sagemaker_pipeline import FuelDeliveryPipeline

    if not check_aws_credentials():
        raise RuntimeError("AWS credentials not configured")

    pipeline = FuelDeliveryPipeline(
        role=config["aws"].get("role_arn"),
        region=config["aws"]["region"],
        bucket=config["aws"].get("s3_bucket"),
        prefix=config["aws"]["s3_prefix"],
    )

    hyperparameters = {
        "prediction_length": config["model"]["prediction_length"],
        "freq": config["data"]["freq"],
        "presets": config["model"]["presets"],
        "time_limit": config["model"]["time_limit"],
        "eval_metric": config["evaluation"]["metrics"][0],
        "quantile_levels": ",".join(map(str, config["inference"]["quantiles"])),
        "target_column": "daily_consumption",
        "item_id_column": "tank_id",
        "timestamp_column": "reading_date",
    }

    if config["data"].get("static_features"):
        hyperparameters["static_features"] = ",".join(config["data"]["static_features"])

    if config["data"].get("known_covariates"):
        hyperparameters["known_covariates"] = ",".join(config["data"]["known_covariates"])

    logger.info("Starting SageMaker training job...")
    training_job = pipeline.train(
        train_data=train_path,
        hyperparameters=hyperparameters,
        instance_type=config["training"]["instance_type"],
        instance_count=config["training"]["instance_count"],
        volume_size=config["training"]["volume_size"],
        max_runtime=config["training"]["max_runtime"],
        wait=True,
    )

    result = {
        "training_job": training_job,
        "model_data": pipeline.model_data,
    }

    if deploy:
        logger.info("Deploying model to endpoint...")
        predictor = pipeline.deploy(
            endpoint_name=config["inference"]["endpoint_name"],
            instance_type=config["inference"]["instance_type"],
            instance_count=config["inference"]["instance_count"],
            wait=True,
        )
        result["endpoint_name"] = pipeline.endpoint_name
        result["predictor"] = predictor

    return result


def demo_workflow():
    """Demonstrate the complete workflow with sample data."""
    logger.info("=" * 60)
    logger.info("FUEL DELIVERY OPTIMIZATION DEMO")
    logger.info("=" * 60)

    # Step 1: Generate sample data
    logger.info("\n[Step 1] Generating sample fuel delivery data...")

    df = generate_fuel_delivery_data(
        n_tanks=20,
        n_days=365,
        start_date="2024-01-01",
        seed=42,
    )

    logger.info(f"Generated {len(df)} rows for {df['tank_id'].nunique()} tanks")

    # Step 2: Prepare data
    logger.info("\n[Step 2] Preparing data...")

    processor = FuelDeliveryDataProcessor(
        item_id_column="tank_id",
        timestamp_column="reading_date",
        target_column="daily_consumption",
        freq="D",
    )

    validation = processor.validate_time_series(df)
    logger.info(f"Validation: {validation['num_tanks']} tanks, "
                f"{validation['total_rows']} rows")

    train_df, test_df = processor.train_test_split(df, prediction_length=10)

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
            "prediction_length": 10,
            "presets": "fast_training",
            "time_limit": 120,
        },
        "evaluation": {"metrics": ["MASE"]},
    }

    predictor = run_local_training(train_df, config)

    # Step 4: Generate predictions
    logger.info("\n[Step 4] Generating consumption forecasts...")

    predictions = run_local_inference(predictor, train_df)

    logger.info(f"Generated {len(predictions)} predictions")

    # Step 5: Evaluate
    logger.info("\n[Step 5] Evaluating forecasts...")

    if len(test_df) > 0:
        metrics = evaluate_forecasts(
            test_df, predictions,
            target_column="daily_consumption",
            item_id_column="tank_id",
            timestamp_column="reading_date",
        )
        logger.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    # Step 6: Delivery optimization
    logger.info("\n[Step 6] Generating delivery recommendations...")

    constraints = DeliveryConstraints()
    optimizer = DeliveryOptimizer(constraints)

    sample_tank = train_df["tank_id"].unique()[0]
    tank_data = train_df[train_df["tank_id"] == sample_tank]
    tank_predictions = predictions[predictions["item_id"] == sample_tank]

    if len(tank_predictions) > 0 and "current_level" in tank_data.columns:
        current_level = tank_data["current_level"].iloc[-1]
        tank_capacity = tank_data["tank_capacity"].iloc[-1]

        rec = optimizer.calculate_optimal_delivery(
            tank_id=sample_tank,
            current_level=current_level,
            tank_capacity=tank_capacity,
            forecast=tank_predictions,
        )

        logger.info(f"\nDelivery Recommendation for {sample_tank}:")
        logger.info(f"  Urgency: {rec.urgency}")
        logger.info(f"  Volume: {rec.volume:.0f} gallons")
        logger.info(f"  Fill %: {rec.fill_pct:.1%}")
        logger.info(f"  Coverage: {rec.coverage_days:.0f} days")
        logger.info(f"  Reasoning: {rec.reasoning}")

        # Compare with heuristic
        hist_avg = tank_data["daily_consumption"].mean()
        comparison = optimizer.compare_with_heuristic(
            tank_id=sample_tank,
            current_level=current_level,
            tank_capacity=tank_capacity,
            forecast=tank_predictions,
            historical_avg_consumption=hist_avg,
        )

        logger.info(f"\nML vs Heuristic Comparison:")
        logger.info(f"  ML Volume: {comparison['ml_volume']:.0f} gal | "
                    f"Heuristic: {comparison['heuristic_volume']:.0f} gal")
        logger.info(f"  Volume Savings: {comparison['volume_savings']:.0f} gal "
                    f"({comparison['volume_savings_pct']:.1f}%)")

    # Step 7: Visualize
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        logger.info("\n[Step 7] Creating visualizations...")

        fig = plot_consumption_forecast(
            historical=train_df,
            predictions=predictions,
            tank_id=sample_tank,
            actuals=test_df if len(test_df) > 0 else None,
            title=f"Consumption Forecast - {sample_tank}",
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
        description="Fuel Delivery Optimization Pipeline",
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
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "local-train", "sagemaker", "generate"],
        default="demo",
        help="Execution mode",
    )

    parser.add_argument("--train-path", type=str, help="Path to training data CSV")
    parser.add_argument("--test-path", type=str, help="Path to test data CSV")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--deploy", action="store_true", help="Deploy model after training")
    parser.add_argument("--n-tanks", type=int, default=50, help="Number of tanks for data generation")
    parser.add_argument("--n-days", type=int, default=365, help="Number of days for data generation")

    args = parser.parse_args()

    # Load config if exists
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(args.config)
    else:
        config = {
            "data": {
                "freq": "D",
                "item_id_column": "tank_id",
                "timestamp_column": "reading_date",
                "target_column": "daily_consumption",
                "date_format": "%Y-%m-%d",
            },
            "model": {
                "prediction_length": 10,
                "presets": "medium_quality",
                "time_limit": 3600,
            },
            "evaluation": {"metrics": ["MASE", "RMSE", "MAPE"]},
            "aws": {
                "region": "us-east-1",
                "s3_prefix": "fuel-delivery-optimization",
            },
            "training": {
                "instance_type": "ml.m5.xlarge",
                "instance_count": 1,
                "volume_size": 50,
                "max_runtime": 86400,
            },
            "inference": {
                "endpoint_name": f"fuel-delivery-forecast-{datetime.now().strftime('%Y%m%d')}",
                "instance_type": "ml.m5.large",
                "instance_count": 1,
                "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
            },
        }

    if args.mode == "demo":
        demo_workflow()

    elif args.mode == "generate":
        df = generate_fuel_delivery_data(
            n_tanks=args.n_tanks,
            n_days=args.n_days,
            seed=42,
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / "fuel_delivery_data.csv", index=False)
        logger.info(f"Generated data saved to {output_dir / 'fuel_delivery_data.csv'}")

    elif args.mode == "local-train":
        if not args.train_path:
            raise ValueError("--train-path required for local-train mode")

        train_df = pd.read_csv(args.train_path)
        predictor = run_local_training(train_df, config)

        if args.test_path:
            test_df = pd.read_csv(args.test_path)
            predictions = run_local_inference(predictor, train_df)

            metrics = evaluate_forecasts(
                test_df, predictions,
                target_column="daily_consumption",
                item_id_column="tank_id",
                timestamp_column="reading_date",
            )
            logger.info(f"Evaluation metrics: {metrics}")

    elif args.mode == "sagemaker":
        if not args.train_path:
            raise ValueError("--train-path required for sagemaker mode")

        result = run_sagemaker_pipeline(
            args.train_path,
            config,
            deploy=args.deploy,
        )
        logger.info(f"SageMaker pipeline result: {result}")


if __name__ == "__main__":
    main()
