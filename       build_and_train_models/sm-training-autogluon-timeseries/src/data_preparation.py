"""
Data Preparation Module for Demand Forecasting
===============================================
Handles loading, validation, and preprocessing of time series data
for use with AutoGluon-TimeSeries and SageMaker.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DemandDataProcessor:
    """Processor for demand forecasting data preparation."""
    
    def __init__(
        self,
        item_id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target_column: str = "target",
        freq: str = "D",
        date_format: str = "%Y-%m-%d"
    ):
        self.item_id_column = item_id_column
        self.timestamp_column = timestamp_column
        self.target_column = target_column
        self.freq = freq
        self.date_format = date_format
        
    def load_data(
        self,
        filepath: Union[str, Path],
        static_features: Optional[List[str]] = None,
        known_covariates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data from CSV file with validation.
        
        Args:
            filepath: Path to CSV file
            static_features: List of static feature column names
            known_covariates: List of known covariate column names
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = [self.item_id_column, self.timestamp_column, self.target_column]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Parse timestamps
        df[self.timestamp_column] = pd.to_datetime(
            df[self.timestamp_column], 
            format=self.date_format
        )
        
        # Sort by item and timestamp
        df = df.sort_values([self.item_id_column, self.timestamp_column])
        
        # Validate features if specified
        if static_features:
            missing_static = set(static_features) - set(df.columns)
            if missing_static:
                logger.warning(f"Missing static features: {missing_static}")
                
        if known_covariates:
            missing_covariates = set(known_covariates) - set(df.columns)
            if missing_covariates:
                logger.warning(f"Missing known covariates: {missing_covariates}")
        
        logger.info(f"Loaded {len(df)} rows, {df[self.item_id_column].nunique()} items")
        return df
    
    def validate_time_series(self, df: pd.DataFrame) -> Dict:
        """
        Validate time series data quality.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_rows": len(df),
            "num_items": df[self.item_id_column].nunique(),
            "date_range": (
                df[self.timestamp_column].min(),
                df[self.timestamp_column].max()
            ),
            "missing_values": {},
            "negative_targets": 0,
            "zero_targets": 0,
            "gaps": []
        }
        
        # Check for missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                results["missing_values"][col] = missing
        
        # Check target values
        results["negative_targets"] = (df[self.target_column] < 0).sum()
        results["zero_targets"] = (df[self.target_column] == 0).sum()
        
        # Check for time gaps per item
        for item_id in df[self.item_id_column].unique()[:10]:  # Sample check
            item_df = df[df[self.item_id_column] == item_id]
            dates = item_df[self.timestamp_column].sort_values()
            
            expected_freq = pd.infer_freq(dates)
            if expected_freq and expected_freq != self.freq:
                results["gaps"].append({
                    "item": item_id,
                    "expected": self.freq,
                    "inferred": expected_freq
                })
        
        return results
    
    def fill_missing_timestamps(
        self,
        df: pd.DataFrame,
        fill_method: str = "zero"
    ) -> pd.DataFrame:
        """
        Fill missing timestamps in time series.
        
        Args:
            df: Input DataFrame
            fill_method: Method to fill missing values ('zero', 'ffill', 'interpolate')
            
        Returns:
            DataFrame with complete time series
        """
        logger.info(f"Filling missing timestamps with method: {fill_method}")
        
        all_items = []
        date_range = pd.date_range(
            start=df[self.timestamp_column].min(),
            end=df[self.timestamp_column].max(),
            freq=self.freq
        )
        
        for item_id in df[self.item_id_column].unique():
            item_df = df[df[self.item_id_column] == item_id].copy()
            item_df = item_df.set_index(self.timestamp_column)
            
            # Reindex to complete date range
            item_df = item_df.reindex(date_range)
            item_df[self.item_id_column] = item_id
            
            # Fill missing targets
            if fill_method == "zero":
                item_df[self.target_column] = item_df[self.target_column].fillna(0)
            elif fill_method == "ffill":
                item_df[self.target_column] = item_df[self.target_column].ffill().bfill()
            elif fill_method == "interpolate":
                item_df[self.target_column] = item_df[self.target_column].interpolate()
            
            item_df = item_df.reset_index()
            item_df = item_df.rename(columns={"index": self.timestamp_column})
            all_items.append(item_df)
        
        return pd.concat(all_items, ignore_index=True)
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based time features."""
        df = df.copy()
        ts = df[self.timestamp_column]
        
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_month"] = ts.dt.day
        df["week_of_year"] = ts.dt.isocalendar().week
        df["month"] = ts.dt.month
        df["quarter"] = ts.dt.quarter
        df["year"] = ts.dt.year
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
        df["is_month_start"] = ts.dt.is_month_start.astype(int)
        df["is_month_end"] = ts.dt.is_month_end.astype(int)
        
        return df
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        lags: List[int] = [7, 14, 28]
    ) -> pd.DataFrame:
        """Add lagged target features."""
        df = df.copy()
        
        for lag in lags:
            df[f"lag_{lag}"] = df.groupby(self.item_id_column)[self.target_column].shift(lag)
        
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [7, 14, 28]
    ) -> pd.DataFrame:
        """Add rolling statistics features."""
        df = df.copy()
        
        for window in windows:
            rolling = df.groupby(self.item_id_column)[self.target_column].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1)
            )
            df[f"rolling_mean_{window}"] = rolling.mean()
            df[f"rolling_std_{window}"] = rolling.std()
            df[f"rolling_min_{window}"] = rolling.min()
            df[f"rolling_max_{window}"] = rolling.max()
        
        return df
    
    def prepare_autogluon_format(
        self,
        df: pd.DataFrame,
        static_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert DataFrame to AutoGluon TimeSeriesDataFrame format.
        
        AutoGluon expects:
        - MultiIndex with (item_id, timestamp)
        - 'target' column
        - Optional static features
        """
        logger.info("Converting to AutoGluon format")
        
        # Select required columns
        columns = [self.item_id_column, self.timestamp_column, self.target_column]
        if static_features:
            columns.extend(static_features)
        
        ag_df = df[columns].copy()
        
        # Rename to standard names
        ag_df = ag_df.rename(columns={
            self.item_id_column: "item_id",
            self.timestamp_column: "timestamp",
            self.target_column: "target"
        })
        
        return ag_df
    
    def train_test_split(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        num_test_windows: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets for time series.
        
        The test set contains the last `prediction_length * num_test_windows`
        timestamps for each item.
        """
        logger.info(f"Splitting data with prediction_length={prediction_length}")
        
        test_length = prediction_length * num_test_windows
        
        train_dfs = []
        test_dfs = []
        
        for item_id in df[self.item_id_column].unique():
            item_df = df[df[self.item_id_column] == item_id].sort_values(
                self.timestamp_column
            )
            
            if len(item_df) <= test_length:
                logger.warning(f"Item {item_id} has insufficient data, skipping test split")
                train_dfs.append(item_df)
                continue
            
            train_dfs.append(item_df.iloc[:-test_length])
            test_dfs.append(item_df.iloc[-test_length:])
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        
        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        return train_df, test_df
    
    def save_to_s3(
        self,
        df: pd.DataFrame,
        bucket: str,
        key: str
    ) -> str:
        """Save DataFrame to S3."""
        import boto3
        from io import StringIO
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        
        s3_path = f"s3://{bucket}/{key}"
        logger.info(f"Saved data to {s3_path}")
        return s3_path


def prepare_forecast_data(
    train_path: str,
    config: Dict,
    output_dir: str = "prepared_data"
) -> Tuple[str, str]:
    """
    Main function to prepare data for forecasting.
    
    Args:
        train_path: Path to raw training data
        config: Configuration dictionary
        output_dir: Directory to save prepared data
        
    Returns:
        Tuple of (train_path, test_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = DemandDataProcessor(
        item_id_column=config["data"]["item_id_column"],
        timestamp_column=config["data"]["timestamp_column"],
        target_column=config["data"]["target_column"],
        freq=config["data"]["freq"],
        date_format=config["data"]["date_format"]
    )
    
    # Load and validate
    df = processor.load_data(
        train_path,
        static_features=config["data"].get("static_features"),
        known_covariates=config["data"].get("known_covariates")
    )
    
    validation = processor.validate_time_series(df)
    logger.info(f"Validation results: {validation}")
    
    # Fill missing timestamps
    df = processor.fill_missing_timestamps(df, fill_method="zero")
    
    # Add features (optional, AutoGluon handles this automatically)
    # df = processor.add_time_features(df)
    
    # Convert to AutoGluon format
    df = processor.prepare_autogluon_format(
        df,
        static_features=config["data"].get("static_features")
    )
    
    # Split data
    train_df, test_df = processor.train_test_split(
        df,
        prediction_length=config["model"]["prediction_length"]
    )
    
    # Save
    train_output = output_dir / "train.csv"
    test_output = output_dir / "test.csv"
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    logger.info(f"Saved prepared data to {output_dir}")
    return str(train_output), str(test_output)


if __name__ == "__main__":
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    prepare_forecast_data("data/raw_demand.csv", config)
