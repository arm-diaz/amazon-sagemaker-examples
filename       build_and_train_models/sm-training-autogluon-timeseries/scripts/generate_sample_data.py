#!/usr/bin/env python3
"""
Generate Sample Fuel Delivery Data
=====================================
Creates synthetic fuel delivery data for testing the optimization pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import generate_fuel_delivery_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate sample fuel delivery data")
    parser.add_argument("--tanks", type=int, default=50, help="Number of tanks")
    parser.add_argument("--days", type=int, default=365, help="Number of days")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start date")
    parser.add_argument("--output", type=str, default="data/fuel_delivery_data.csv", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", action="store_true", help="Split into train/test")
    parser.add_argument("--test-days", type=int, default=10, help="Days for test set")

    args = parser.parse_args()

    # Generate data
    df = generate_fuel_delivery_data(
        n_tanks=args.tanks,
        n_days=args.days,
        start_date=args.start_date,
        seed=args.seed,
    )

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.split:
        cutoff_date = pd.to_datetime(df["reading_date"]).max() - pd.Timedelta(days=args.test_days)

        train_df = df[pd.to_datetime(df["reading_date"]) <= cutoff_date]
        test_df = df[pd.to_datetime(df["reading_date"]) > cutoff_date]

        train_path = output_path.parent / "train.csv"
        test_path = output_path.parent / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved training data: {train_path} ({len(train_df):,} rows)")
        logger.info(f"Saved test data: {test_path} ({len(test_df):,} rows)")
    else:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved data to {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("FUEL DELIVERY DATA SUMMARY")
    print("=" * 50)
    print(f"\nTanks: {df['tank_id'].nunique()}")
    print(f"Date range: {df['reading_date'].min()} to {df['reading_date'].max()}")
    print(f"Total rows: {len(df):,}")

    print(f"\nConsumption (gal/day) by tank type:")
    print(df.groupby("tank_type")["daily_consumption"].agg(["mean", "std", "min", "max"]).round(2))

    print(f"\nTank capacities: {sorted(df['tank_capacity'].unique())}")
    print(f"Delivery events: {(df['delivery_volume'] > 0).sum():,}")
    print(f"Holiday rate: {df['is_holiday'].mean():.1%}")


if __name__ == "__main__":
    main()
