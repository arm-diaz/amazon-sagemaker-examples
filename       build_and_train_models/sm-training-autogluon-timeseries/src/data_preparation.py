"""
Data Preparation Module for Fuel Delivery Optimization
========================================================
Handles loading, validation, feature engineering, and preprocessing
of tank-level consumption time series for AutoGluon-TimeSeries and SageMaker.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
import logging

from calendar_us import get_us_holidays

logger = logging.getLogger(__name__)


class FuelDeliveryDataProcessor:
    """Processor for fuel delivery consumption data."""

    def __init__(
        self,
        item_id_column: str = "tank_id",
        timestamp_column: str = "reading_date",
        target_column: str = "daily_consumption",
        freq: str = "D",
        date_format: str = "%Y-%m-%d",
    ):
        self.item_id_column = item_id_column
        self.timestamp_column = timestamp_column
        self.target_column = target_column
        self.freq = freq
        self.date_format = date_format

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        filepath: Union[str, Path],
        static_features: Optional[List[str]] = None,
        known_covariates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load data from CSV file with validation."""
        logger.info(f"Loading data from {filepath}")

        df = pd.read_csv(filepath)

        required_cols = [self.item_id_column, self.timestamp_column, self.target_column]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df[self.timestamp_column] = pd.to_datetime(
            df[self.timestamp_column], format=self.date_format
        )
        df = df.sort_values([self.item_id_column, self.timestamp_column])

        if static_features:
            missing_static = set(static_features) - set(df.columns)
            if missing_static:
                logger.warning(f"Missing static features: {missing_static}")

        if known_covariates:
            missing_cov = set(known_covariates) - set(df.columns)
            if missing_cov:
                logger.warning(f"Missing known covariates: {missing_cov}")

        logger.info(
            f"Loaded {len(df)} rows, {df[self.item_id_column].nunique()} tanks"
        )
        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_time_series(self, df: pd.DataFrame) -> Dict:
        """Validate time series data quality."""
        results = {
            "total_rows": len(df),
            "num_tanks": df[self.item_id_column].nunique(),
            "date_range": (
                df[self.timestamp_column].min(),
                df[self.timestamp_column].max(),
            ),
            "missing_values": {},
            "negative_targets": 0,
            "zero_targets": 0,
            "gaps": [],
        }

        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                results["missing_values"][col] = int(missing)

        results["negative_targets"] = int((df[self.target_column] < 0).sum())
        results["zero_targets"] = int((df[self.target_column] == 0).sum())

        for tank_id in df[self.item_id_column].unique()[:10]:
            tank_df = df[df[self.item_id_column] == tank_id]
            dates = tank_df[self.timestamp_column].sort_values()
            expected_freq = pd.infer_freq(dates)
            if expected_freq and expected_freq != self.freq:
                results["gaps"].append(
                    {"tank": tank_id, "expected": self.freq, "inferred": expected_freq}
                )

        return results

    # ------------------------------------------------------------------
    # Missing-data handling
    # ------------------------------------------------------------------

    def fill_missing_timestamps(
        self, df: pd.DataFrame, fill_method: str = "zero"
    ) -> pd.DataFrame:
        """Fill missing timestamps in each tank's time series."""
        logger.info(f"Filling missing timestamps with method: {fill_method}")

        all_tanks = []
        date_range = pd.date_range(
            start=df[self.timestamp_column].min(),
            end=df[self.timestamp_column].max(),
            freq=self.freq,
        )

        for tank_id in df[self.item_id_column].unique():
            tank_df = df[df[self.item_id_column] == tank_id].copy()
            tank_df = tank_df.set_index(self.timestamp_column)
            tank_df = tank_df.reindex(date_range)
            tank_df[self.item_id_column] = tank_id

            if fill_method == "zero":
                tank_df[self.target_column] = tank_df[self.target_column].fillna(0)
            elif fill_method == "ffill":
                tank_df[self.target_column] = (
                    tank_df[self.target_column].ffill().bfill()
                )
            elif fill_method == "interpolate":
                tank_df[self.target_column] = tank_df[self.target_column].interpolate()

            tank_df = tank_df.reset_index().rename(
                columns={"index": self.timestamp_column}
            )
            all_tanks.append(tank_df)

        return pd.concat(all_tanks, ignore_index=True)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def add_time_features(
        self, df: pd.DataFrame, state: Optional[str] = None
    ) -> pd.DataFrame:
        """Add calendar and US-holiday features."""
        df = df.copy()
        ts = pd.to_datetime(df[self.timestamp_column])

        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_month"] = ts.dt.day
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["month"] = ts.dt.month
        df["quarter"] = ts.dt.quarter
        df["year"] = ts.dt.year
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
        df["is_month_start"] = ts.dt.is_month_start.astype(int)
        df["is_month_end"] = ts.dt.is_month_end.astype(int)

        # Season (US-centric)
        df["season"] = ts.dt.month.map(
            {
                12: "winter", 1: "winter", 2: "winter",
                3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer",
                9: "fall", 10: "fall", 11: "fall",
            }
        )

        # US holidays
        years = sorted(ts.dt.year.unique())
        holiday_df = get_us_holidays(years=years, state=state)
        holiday_dates = set(holiday_df["date"].dt.date)
        holiday_map = dict(
            zip(holiday_df["date"].dt.date, holiday_df["holiday_name"])
        )

        df["is_holiday"] = ts.dt.date.map(lambda d: int(d in holiday_dates))
        df["holiday_name"] = ts.dt.date.map(lambda d: holiday_map.get(d))
        df["days_to_next_holiday"] = ts.map(
            lambda d: _days_to_next(d, holiday_df)
        )

        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Add lagged target features (lags 1, 2, 7, 14, 21)."""
        if lags is None:
            lags = [1, 2, 7, 14, 21]
        df = df.copy()
        for lag in lags:
            df[f"lag_{lag}"] = df.groupby(self.item_id_column)[
                self.target_column
            ].shift(lag)
        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Add rolling mean, std, and trend slope over windows 7, 14, 30."""
        if windows is None:
            windows = [7, 14, 30]
        df = df.copy()

        for window in windows:
            grp = df.groupby(self.item_id_column)[self.target_column]

            shifted = grp.transform(lambda x: x.shift(1))
            rolling = shifted.rolling(window=window, min_periods=1)

            df[f"rolling_mean_{window}"] = rolling.mean()
            df[f"rolling_std_{window}"] = rolling.std()

            # Trend slope: linear regression coefficient over the window
            df[f"rolling_slope_{window}"] = grp.transform(
                lambda x: x.shift(1)
                .rolling(window=window, min_periods=2)
                .apply(_linear_slope, raw=True)
            )

        return df

    def add_tank_state_features(
        self,
        df: pd.DataFrame,
        capacity_column: str = "tank_capacity",
        level_column: str = "current_level",
        reorder_pct: float = 0.25,
    ) -> pd.DataFrame:
        """Add tank-state features: capacity utilization, days of inventory, distance to reorder."""
        df = df.copy()

        if capacity_column in df.columns and level_column in df.columns:
            df["capacity_utilization"] = df[level_column] / df[capacity_column]

            # Estimated days of inventory remaining
            safe_consumption = df[self.target_column].replace(0, np.nan)
            df["days_of_inventory"] = df[level_column] / safe_consumption

            # Distance to reorder point (gallons above reorder level)
            reorder_level = df[capacity_column] * reorder_pct
            df["distance_to_reorder"] = df[level_column] - reorder_level
        else:
            logger.info(
                "Skipping tank state features: "
                f"'{capacity_column}' or '{level_column}' column not found."
            )

        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spike/dip flags, volatility, and acceleration."""
        df = df.copy()
        grp = df.groupby(self.item_id_column)[self.target_column]

        rolling_mean = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        rolling_std = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())

        safe_std = rolling_std.replace(0, np.nan)
        z_score = (df[self.target_column] - rolling_mean) / safe_std

        df["consumption_spike"] = (z_score > 2).astype(int)
        df["consumption_dip"] = (z_score < -2).astype(int)

        # Volatility: rolling coefficient of variation (std / mean)
        safe_mean = rolling_mean.replace(0, np.nan)
        df["volatility_7d"] = rolling_std / safe_mean

        # Acceleration: change in daily consumption vs prior day
        df["consumption_accel"] = grp.transform(lambda x: x.diff().diff())

        return df

    def build_feature_pipeline(
        self, df: pd.DataFrame, state: Optional[str] = None
    ) -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        logger.info("Running feature engineering pipeline...")
        df = self.add_time_features(df, state=state)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_tank_state_features(df)
        df = self.add_derived_features(df)
        logger.info(f"Feature pipeline complete. Columns: {len(df.columns)}")
        return df

    # ------------------------------------------------------------------
    # AutoGluon conversion
    # ------------------------------------------------------------------

    def prepare_autogluon_format(
        self,
        df: pd.DataFrame,
        static_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert DataFrame to AutoGluon TimeSeriesDataFrame-compatible format."""
        logger.info("Converting to AutoGluon format")

        columns = [self.item_id_column, self.timestamp_column, self.target_column]
        if static_features:
            columns.extend([c for c in static_features if c in df.columns])

        ag_df = df[columns].copy()
        ag_df = ag_df.rename(
            columns={
                self.item_id_column: "item_id",
                self.timestamp_column: "timestamp",
                self.target_column: "target",
            }
        )
        return ag_df

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------

    def train_test_split(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        num_test_windows: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets for time series."""
        logger.info(f"Splitting data with prediction_length={prediction_length}")

        test_length = prediction_length * num_test_windows
        train_dfs, test_dfs = [], []

        for tank_id in df[self.item_id_column].unique():
            tank_df = df[df[self.item_id_column] == tank_id].sort_values(
                self.timestamp_column
            )
            if len(tank_df) <= test_length:
                logger.warning(
                    f"Tank {tank_id} has insufficient data, skipping test split"
                )
                train_dfs.append(tank_df)
                continue
            train_dfs.append(tank_df.iloc[:-test_length])
            test_dfs.append(tank_df.iloc[-test_length:])

        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        return train_df, test_df

    # ------------------------------------------------------------------
    # S3
    # ------------------------------------------------------------------

    def save_to_s3(self, df: pd.DataFrame, bucket: str, key: str) -> str:
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


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _linear_slope(y: np.ndarray) -> float:
    """Compute OLS slope over an array of values."""
    n = len(y)
    if n < 2 or np.all(np.isnan(y)):
        return np.nan
    x = np.arange(n, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    x, y = x[mask], y[mask]
    return np.polyfit(x, y, 1)[0]


def _days_to_next(date: pd.Timestamp, holiday_df: pd.DataFrame) -> int:
    future = holiday_df.loc[holiday_df["date"] >= date, "date"]
    if future.empty:
        return 365
    return (future.iloc[0] - date).days


# ------------------------------------------------------------------
# Convenience entry-point
# ------------------------------------------------------------------

def prepare_forecast_data(
    train_path: str,
    config: Dict,
    output_dir: str = "prepared_data",
) -> Tuple[str, str]:
    """Prepare data for forecasting (CLI / script helper)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = FuelDeliveryDataProcessor(
        item_id_column=config["data"]["item_id_column"],
        timestamp_column=config["data"]["timestamp_column"],
        target_column=config["data"]["target_column"],
        freq=config["data"]["freq"],
        date_format=config["data"]["date_format"],
    )

    df = processor.load_data(
        train_path,
        static_features=config["data"].get("static_features"),
        known_covariates=config["data"].get("known_covariates"),
    )

    validation = processor.validate_time_series(df)
    logger.info(f"Validation results: {validation}")

    df = processor.fill_missing_timestamps(df, fill_method="zero")
    df = processor.prepare_autogluon_format(
        df, static_features=config["data"].get("static_features")
    )

    train_df, test_df = processor.train_test_split(
        df, prediction_length=config["model"]["prediction_length"]
    )

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

    prepare_forecast_data("data/raw_data.csv", config)
