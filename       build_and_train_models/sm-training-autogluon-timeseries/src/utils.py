"""
Utility Functions for Demand Forecasting
=========================================
Helper functions for configuration, evaluation, and visualization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, config_path: str = "config/config.yaml"):
    """Save configuration to YAML file."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
    """
    Mean Absolute Scaled Error.
    
    MASE < 1: Better than naive forecast
    MASE = 1: Same as naive forecast
    MASE > 1: Worse than naive forecast
    """
    n = len(y_train)
    d = np.abs(np.diff(y_train, n=seasonality)).sum() / (n - seasonality)
    
    if d == 0:
        return np.inf
    
    errors = np.abs(y_true - y_pred)
    return np.mean(errors) / d


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Quantile loss (pinball loss)."""
    errors = y_true - y_pred
    return np.mean(np.maximum(q * errors, (q - 1) * errors))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction interval coverage."""
    return np.mean((y_true >= lower) & (y_true <= upper))


def evaluate_forecasts(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    target_column: str = "target",
    item_id_column: str = "item_id"
) -> Dict[str, float]:
    """
    Evaluate forecast accuracy across multiple metrics.
    
    Args:
        actuals: DataFrame with actual values
        predictions: DataFrame with predictions
        target_column: Name of target column
        item_id_column: Name of item ID column
        
    Returns:
        Dictionary of metric names and values
    """
    # Merge actuals and predictions
    merged = actuals.merge(
        predictions,
        on=[item_id_column, "timestamp"],
        suffixes=("_actual", "_pred")
    )
    
    y_true = merged[f"{target_column}_actual"].values
    y_pred = merged["mean"].values if "mean" in merged.columns else merged[f"{target_column}_pred"].values
    
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }
    
    # Add quantile metrics if available
    for col in merged.columns:
        if col.startswith("0."):
            q = float(col)
            metrics[f"quantile_loss_{col}"] = quantile_loss(y_true, merged[col].values, q)
    
    # Add coverage if prediction intervals available
    if "0.1" in merged.columns and "0.9" in merged.columns:
        metrics["coverage_80"] = coverage(
            y_true,
            merged["0.1"].values,
            merged["0.9"].values
        )
    
    return metrics


# ============================================================================
# Visualization
# ============================================================================

def plot_forecast(
    historical: pd.DataFrame,
    predictions: pd.DataFrame,
    item_id: str,
    timestamp_column: str = "timestamp",
    target_column: str = "target",
    actuals: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> Any:
    """
    Plot historical data and forecasts for a single item.
    
    Args:
        historical: Historical time series data
        predictions: Forecast predictions
        item_id: Item to plot
        timestamp_column: Timestamp column name
        target_column: Target column name
        actuals: Optional actual values for forecast period
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to item
    hist = historical[historical["item_id"] == item_id].sort_values(timestamp_column)
    pred = predictions[predictions["item_id"] == item_id].sort_values(timestamp_column)
    
    # Plot historical
    ax.plot(
        hist[timestamp_column],
        hist[target_column],
        label="Historical",
        color="blue",
        linewidth=1.5
    )
    
    # Plot mean prediction
    pred_col = "mean" if "mean" in pred.columns else "0.5"
    ax.plot(
        pred[timestamp_column],
        pred[pred_col],
        label="Forecast",
        color="red",
        linewidth=1.5
    )
    
    # Plot prediction intervals if available
    if "0.1" in pred.columns and "0.9" in pred.columns:
        ax.fill_between(
            pred[timestamp_column],
            pred["0.1"],
            pred["0.9"],
            alpha=0.2,
            color="red",
            label="80% Interval"
        )
    
    if "0.25" in pred.columns and "0.75" in pred.columns:
        ax.fill_between(
            pred[timestamp_column],
            pred["0.25"],
            pred["0.75"],
            alpha=0.3,
            color="red",
            label="50% Interval"
        )
    
    # Plot actuals if provided
    if actuals is not None:
        act = actuals[actuals["item_id"] == item_id].sort_values(timestamp_column)
        ax.scatter(
            act[timestamp_column],
            act[target_column],
            label="Actuals",
            color="green",
            s=20,
            zorder=5
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.set_title(title or f"Demand Forecast - {item_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(
    leaderboard: pd.DataFrame,
    metric: str = "score_val",
    figsize: Tuple[int, int] = (10, 6)
) -> Any:
    """Plot model performance comparison."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric
    lb = leaderboard.sort_values(metric)
    
    bars = ax.barh(lb["model"], lb[metric])
    
    # Color best model
    best_idx = lb[metric].idxmin() if "error" in metric.lower() else lb[metric].idxmax()
    for i, bar in enumerate(bars):
        if lb.index[i] == best_idx:
            bar.set_color("green")
        else:
            bar.set_color("steelblue")
    
    ax.set_xlabel(metric)
    ax.set_title("Model Performance Comparison")
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# Data Generation
# ============================================================================

def generate_demand_data(
    n_items: int = 30,
    n_days: int = 365,
    start_date: str = "2024-01-01",
    base_demand: float = 500,
    seed: int = 42
) -> pd.DataFrame:
    """Generate demand data with realistic seasonal patterns and feature columns."""
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    records = []
    
    categories = ["GROCERY", "ELECTRONICS", "APPAREL", "HOME", "SEASONAL"]
    
    for item_idx in range(n_items):
        item_id = f"SKU_{item_idx:04d}"
        category = categories[item_idx % len(categories)]
        
        # Item-specific base demand (no strong trend)
        item_base = base_demand * np.random.uniform(0.5, 1.5)
        base_price = 10 + item_idx * 2
        
        for day_idx, date in enumerate(dates):
            demand = item_base
            
            # Day of week effect
            dow = date.dayofweek
            is_weekend = 1 if dow >= 5 else 0
            weekly = 0.25 * np.sin(2 * np.pi * dow / 7)
            demand *= (1 + weekly)
            
            # Monthly effect
            dom = date.day
            is_month_start = 1 if dom <= 5 else 0
            is_month_end = 1 if dom >= 25 else 0
            monthly = 0.15 * np.sin(2 * np.pi * dom / 30)
            demand *= (1 + monthly)
            
            # Yearly seasonality
            doy = date.dayofyear
            yearly = 0.4 * np.sin(2 * np.pi * (doy - 80) / 365)
            demand *= (1 + yearly)
            
            # Quarter
            quarter = date.quarter
            
            # Holiday detection
            is_holiday = 0
            holiday_name = None
            
            if date.month == 12 and date.day >= 15:
                is_holiday = 1
                holiday_name = "christmas"
                demand *= 1.8
            elif date.month == 11 and date.day >= 20 and date.day <= 30:
                is_holiday = 1
                holiday_name = "black_friday"
                demand *= 1.6
            elif date.month == 2 and date.day >= 10 and date.day <= 14:
                is_holiday = 1
                holiday_name = "valentines"
                demand *= 1.3
            elif date.month == 7 and date.day == 4:
                is_holiday = 1
                holiday_name = "july_4th"
                demand *= 1.2
            
            # Promotion (random, affects price and demand)
            is_promotion = 1 if np.random.random() < 0.08 else 0
            if is_promotion:
                demand *= np.random.uniform(1.3, 1.6)
            
            # Price (lower during promotions)
            price = base_price * (0.8 if is_promotion else 1.0) + np.random.normal(0, 0.5)
            
            # Add noise
            noise = np.random.normal(0, 0.1 * item_base)
            demand = max(0, demand + noise)
            
            records.append({
                "item_id": item_id,
                "timestamp": date.strftime("%Y-%m-%d"),
                "target": round(demand, 2),
                # Feature columns
                "category": category,
                "price": round(price, 2),
                "is_promotion": is_promotion,
                "is_holiday": is_holiday,
                "holiday_name": holiday_name,
                "is_weekend": is_weekend,
                "is_month_start": is_month_start,
                "is_month_end": is_month_end,
                "day_of_week": dow,
                "day_of_month": dom,
                "day_of_year": doy,
                "week_of_year": date.isocalendar()[1],
                "month": date.month,
                "quarter": quarter,
                "year": date.year,
            })
    
    return pd.DataFrame(records)


# ============================================================================
# AWS Utilities
# ============================================================================

def get_sagemaker_role() -> str:
    """Get SageMaker execution role from environment or IAM."""
    import os
    
    # Try environment variable first
    role = os.environ.get("SAGEMAKER_ROLE")
    if role:
        return role
    
    # Try to get from SageMaker session
    try:
        from sagemaker import get_execution_role
        return get_execution_role()
    except Exception:
        pass
    
    # Try IAM
    try:
        import boto3
        iam = boto3.client("iam")
        roles = iam.list_roles()["Roles"]
        
        for role in roles:
            if "SageMaker" in role["RoleName"]:
                return role["Arn"]
    except Exception:
        pass
    
    raise ValueError(
        "Could not determine SageMaker role. "
        "Set SAGEMAKER_ROLE environment variable or run from SageMaker."
    )


def check_aws_credentials() -> bool:
    """Verify AWS credentials are configured."""
    try:
        import boto3
        sts = boto3.client("sts")
        sts.get_caller_identity()
        return True
    except Exception as e:
        logger.error(f"AWS credentials check failed: {e}")
        return False


def estimate_training_cost(
    instance_type: str,
    hours: float,
    instance_count: int = 1
) -> float:
    """
    Estimate SageMaker training cost.
    
    Note: Prices are approximate and region-dependent.
    """
    # Approximate hourly prices (USD, us-east-1)
    prices = {
        "ml.m5.large": 0.115,
        "ml.m5.xlarge": 0.23,
        "ml.m5.2xlarge": 0.461,
        "ml.m5.4xlarge": 0.922,
        "ml.c5.xlarge": 0.204,
        "ml.c5.2xlarge": 0.408,
        "ml.p3.2xlarge": 3.825,
        "ml.g4dn.xlarge": 0.736,
    }
    
    price_per_hour = prices.get(instance_type, 0.5)
    total_cost = price_per_hour * hours * instance_count
    
    return round(total_cost, 2)
