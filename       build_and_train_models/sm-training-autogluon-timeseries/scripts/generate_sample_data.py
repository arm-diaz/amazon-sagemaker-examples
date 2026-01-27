#!/usr/bin/env python3
"""
Generate Sample Demand Data
============================
Creates synthetic demand data for testing the forecasting pipeline.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_realistic_demand(
    n_products: int = 50,
    n_days: int = 730,  # 2 years
    start_date: str = "2023-01-01",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic demand data with various patterns.
    
    Features:
    - Multiple product categories with different demand patterns
    - Weekly, monthly, and yearly seasonality
    - Trend (growing/declining products)
    - Promotional effects
    - Holiday spikes
    - Weather effects (simulated)
    - Stockout events (zero demand)
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    
    # Define product categories with different characteristics
    categories = {
        "GROCERY": {"base": 150, "trend": 0.0002, "weekly_amp": 0.15, "promo_lift": 1.3},
        "ELECTRONICS": {"base": 30, "trend": 0.0005, "weekly_amp": 0.25, "promo_lift": 1.5},
        "APPAREL": {"base": 50, "trend": 0.0001, "weekly_amp": 0.3, "promo_lift": 1.4},
        "HOME": {"base": 40, "trend": 0.0003, "weekly_amp": 0.2, "promo_lift": 1.35},
        "SEASONAL": {"base": 60, "trend": 0.0, "weekly_amp": 0.2, "promo_lift": 1.25},
    }
    
    records = []
    
    for prod_idx in range(n_products):
        # Assign category
        cat_name = list(categories.keys())[prod_idx % len(categories)]
        cat = categories[cat_name]
        
        item_id = f"PROD_{prod_idx:05d}"
        
        # Product-specific variations
        base_demand = cat["base"] * np.random.uniform(0.5, 1.5)
        trend = cat["trend"] * np.random.uniform(-1, 2)
        phase_shift = np.random.uniform(0, 2 * np.pi)
        
        for day_idx, date in enumerate(dates):
            # Base + Trend
            demand = base_demand * (1 + trend * day_idx)
            
            # Weekly pattern (higher weekends for some, lower for others)
            dow = date.dayofweek
            if cat_name in ["GROCERY", "HOME"]:
                # Weekend spike
                weekly = cat["weekly_amp"] * (1 if dow >= 5 else -0.5)
            else:
                # Weekday higher
                weekly = cat["weekly_amp"] * (-0.3 if dow >= 5 else 0.2)
            
            demand *= (1 + weekly)
            
            # Monthly pattern (payday effect)
            dom = date.day
            if dom <= 5 or dom >= 25:
                demand *= 1.1
            
            # Yearly seasonality
            doy = date.dayofyear
            yearly = 0.2 * np.sin(2 * np.pi * (doy - 80) / 365 + phase_shift)
            demand *= (1 + yearly)
            
            # Strong seasonality for SEASONAL category
            if cat_name == "SEASONAL":
                # Holiday season (Nov-Dec)
                if date.month in [11, 12]:
                    demand *= 2.0
                # Summer peak
                elif date.month in [6, 7, 8]:
                    demand *= 1.4
            
            # Random promotions (~10% of days)
            is_promo = np.random.random() < 0.10
            if is_promo:
                demand *= cat["promo_lift"]
            
            # Holiday effects
            is_holiday = False
            # Christmas period
            if date.month == 12 and date.day >= 20:
                demand *= 1.5
                is_holiday = True
            # Black Friday (4th Thursday of November)
            elif date.month == 11 and date.day >= 22 and date.day <= 28 and dow == 4:
                demand *= 2.5
                is_holiday = True
            # Major holidays (simplified)
            elif (date.month, date.day) in [(1, 1), (7, 4), (11, 11)]:
                demand *= 1.3
                is_holiday = True
            
            # Weather effect (simulated)
            weather_score = np.sin(2 * np.pi * doy / 365)  # -1 to 1
            if cat_name in ["APPAREL", "SEASONAL"]:
                demand *= (1 + 0.1 * weather_score)
            
            # Random noise
            noise = np.random.normal(0, 0.1 * base_demand)
            demand += noise
            
            # Occasional stockouts (zero demand)
            if np.random.random() < 0.01:
                demand = 0
            
            # Ensure non-negative and round
            demand = max(0, round(demand))
            
            # Price (varies by category and with some random daily variation)
            base_price = 10 + prod_idx * 0.5 + np.random.uniform(-1, 1)
            price = base_price * (0.9 if is_promo else 1.0)
            
            records.append({
                "item_id": item_id,
                "timestamp": date.strftime("%Y-%m-%d"),
                "target": int(demand),
                "category": cat_name,
                "price": round(price, 2),
                "is_promotion": int(is_promo),
                "is_holiday": int(is_holiday),
                "day_of_week": dow,
                "month": date.month,
            })
    
    df = pd.DataFrame(records)
    
    logger.info(f"Generated {len(df):,} records")
    logger.info(f"Products: {n_products}")
    logger.info(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    logger.info(f"Categories: {df['category'].value_counts().to_dict()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample demand data")
    parser.add_argument("--products", type=int, default=50, help="Number of products")
    parser.add_argument("--days", type=int, default=730, help="Number of days")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--output", type=str, default="data/demand_data.csv", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", action="store_true", help="Split into train/test")
    parser.add_argument("--test-days", type=int, default=30, help="Days for test set")
    
    args = parser.parse_args()
    
    # Generate data
    df = generate_realistic_demand(
        n_products=args.products,
        n_days=args.days,
        start_date=args.start_date,
        seed=args.seed
    )
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.split:
        # Split into train/test
        cutoff_date = pd.to_datetime(df["timestamp"]).max() - pd.Timedelta(days=args.test_days)
        
        train_df = df[pd.to_datetime(df["timestamp"]) <= cutoff_date]
        test_df = df[pd.to_datetime(df["timestamp"]) > cutoff_date]
        
        train_path = output_path.parent / "train.csv"
        test_path = output_path.parent / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved training data: {train_path} ({len(train_df):,} rows)")
        logger.info(f"Saved test data: {test_path} ({len(test_df):,} rows)")
    else:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved data to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(f"\nTarget (demand) statistics:")
    print(df["target"].describe())
    
    print(f"\nDemand by category:")
    print(df.groupby("category")["target"].agg(["mean", "std", "min", "max"]))
    
    print(f"\nPromotion rate: {df['is_promotion'].mean():.1%}")
    print(f"Holiday rate: {df['is_holiday'].mean():.1%}")
    print(f"Zero demand rate: {(df['target'] == 0).mean():.1%}")


if __name__ == "__main__":
    main()
