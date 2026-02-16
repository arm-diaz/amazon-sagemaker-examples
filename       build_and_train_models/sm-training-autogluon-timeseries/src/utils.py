"""
Utility Functions for Fuel Delivery Optimization
==================================================
Helper functions for configuration, evaluation, data generation,
and visualization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

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
    """Mean Absolute Scaled Error."""
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
    target_column: str = "daily_consumption",
    item_id_column: str = "tank_id",
    timestamp_column: str = "reading_date",
) -> Dict[str, float]:
    """Evaluate forecast accuracy across multiple metrics."""
    merged = actuals.merge(
        predictions,
        left_on=[item_id_column, timestamp_column],
        right_on=["item_id", "timestamp"] if "item_id" in predictions.columns else [item_id_column, timestamp_column],
        suffixes=("_actual", "_pred"),
    )

    actual_col = f"{target_column}_actual" if f"{target_column}_actual" in merged.columns else target_column
    y_true = merged[actual_col].values
    y_pred = merged["mean"].values if "mean" in merged.columns else merged[f"{target_column}_pred"].values

    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }

    for col in merged.columns:
        if col.startswith("0."):
            q = float(col)
            metrics[f"quantile_loss_{col}"] = quantile_loss(y_true, merged[col].values, q)

    if "0.1" in merged.columns and "0.9" in merged.columns:
        metrics["coverage_80"] = coverage(y_true, merged["0.1"].values, merged["0.9"].values)

    return metrics


def calculate_business_metrics(
    recommendations: List[Dict],
    actuals: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Calculate business-level metrics from delivery recommendations.

    Args:
        recommendations: List of delivery recommendation dicts.
        actuals: Optional DataFrame with actual consumption to detect stockouts.

    Returns:
        Dictionary with avg_fill_rate, emergency_rate, etc.
    """
    if not recommendations:
        return {}

    fill_rates = [r.get("fill_pct", 0) for r in recommendations]
    urgencies = [r.get("urgency", "normal") for r in recommendations]

    metrics = {
        "total_deliveries": len(recommendations),
        "avg_fill_pct": float(np.mean(fill_rates)),
        "emergency_rate": sum(1 for u in urgencies if u == "emergency") / len(urgencies),
        "avg_volume": float(np.mean([r.get("volume", 0) for r in recommendations])),
        "total_volume": float(np.sum([r.get("volume", 0) for r in recommendations])),
    }

    coverage_days = [r.get("coverage_days", 0) for r in recommendations]
    if coverage_days:
        metrics["avg_coverage_days"] = float(np.mean(coverage_days))

    return metrics


# ============================================================================
# Data Generation
# ============================================================================

def generate_fuel_delivery_data(
    n_tanks: int = 50,
    n_days: int = 365,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic fuel delivery data with realistic patterns.

    Tank types:
        - residential: 2-8 gal/day, 275-gal tank
        - commercial:  8-25 gal/day, 1000-gal tank
        - industrial:  25-60 gal/day, 5000-gal tank

    Patterns applied:
        - US seasonal (cold winters => higher heating fuel demand)
        - Day-of-week variation
        - US holiday effects (Thanksgiving/Christmas spikes for residential)
        - Physically consistent current_level with simulated deliveries
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    tank_specs = {
        "residential": {"base_range": (2, 8), "capacity": 275},
        "commercial": {"base_range": (8, 25), "capacity": 1000},
        "industrial": {"base_range": (25, 60), "capacity": 5000},
    }
    tank_types = list(tank_specs.keys())

    # Pre-compute US holidays for the date range
    import holidays as _holidays
    years = sorted(set(d.year for d in dates))
    us_holidays = _holidays.US(years=years)
    holiday_dates = {pd.Timestamp(d): n for d, n in us_holidays.items()}

    records = []

    for tank_idx in range(n_tanks):
        tank_id = f"TANK_{tank_idx:04d}"
        tank_type = tank_types[tank_idx % len(tank_types)]
        spec = tank_specs[tank_type]

        base_low, base_high = spec["base_range"]
        tank_capacity = spec["capacity"]
        base_consumption = rng.uniform(base_low, base_high)

        # Start with a random fill level (50-95%)
        current_level = tank_capacity * rng.uniform(0.50, 0.95)
        reorder_level = tank_capacity * 0.25

        for day_idx, date in enumerate(dates):
            consumption = base_consumption

            # --- Seasonal pattern (US: cold winters increase heating fuel) ---
            doy = date.dayofyear
            # Peak heating demand in Jan (~day 15), trough in Jul (~day 196)
            seasonal = 0.35 * np.cos(2 * np.pi * (doy - 15) / 365)
            if tank_type == "residential":
                seasonal *= 1.5  # residential more affected by heating
            elif tank_type == "industrial":
                seasonal *= 0.5  # industrial steadier
            consumption *= (1 + seasonal)

            # --- Day-of-week ---
            dow = date.dayofweek
            if tank_type == "commercial":
                # Lower on weekends
                consumption *= 0.6 if dow >= 5 else 1.1
            elif tank_type == "industrial":
                consumption *= 0.4 if dow >= 5 else 1.15

            # --- Holiday effects ---
            is_holiday = 0
            holiday_name = None
            ts = pd.Timestamp(date)
            if ts in holiday_dates:
                is_holiday = 1
                holiday_name = holiday_dates[ts]

            # Thanksgiving/Christmas: residential spikes, commercial dips
            if date.month == 11 and date.day >= 22 and date.day <= 28:
                if tank_type == "residential":
                    consumption *= 1.30
                elif tank_type == "commercial":
                    consumption *= 0.70
            elif date.month == 12 and date.day >= 20:
                if tank_type == "residential":
                    consumption *= 1.40
                elif tank_type == "commercial":
                    consumption *= 0.65
            elif is_holiday and tank_type == "commercial":
                consumption *= 0.80

            # --- Noise ---
            noise = rng.normal(0, 0.10 * base_consumption)
            consumption = max(0.0, consumption + noise)

            # --- Update tank level ---
            current_level -= consumption

            # Simulate delivery when level drops below reorder
            delivery_volume = 0.0
            if current_level <= reorder_level:
                target_fill = tank_capacity * 0.85
                delivery_volume = target_fill - current_level
                current_level += delivery_volume

            current_level = np.clip(current_level, 0, tank_capacity)

            records.append({
                "tank_id": tank_id,
                "reading_date": date.strftime("%Y-%m-%d"),
                "daily_consumption": round(consumption, 2),
                "tank_type": tank_type,
                "tank_capacity": tank_capacity,
                "current_level": round(current_level, 2),
                "delivery_volume": round(delivery_volume, 2),
                "is_holiday": is_holiday,
                "holiday_name": holiday_name,
                "is_weekend": int(dow >= 5),
                "day_of_week": dow,
                "month": date.month,
                "year": date.year,
            })

    return pd.DataFrame(records)


# ============================================================================
# Visualization
# ============================================================================

def plot_consumption_forecast(
    historical: pd.DataFrame,
    predictions: pd.DataFrame,
    tank_id: str,
    timestamp_column: str = "reading_date",
    target_column: str = "daily_consumption",
    actuals: Optional[pd.DataFrame] = None,
    reorder_level: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> Any:
    """Plot historical consumption and forecasts for a single tank."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    id_col = "item_id" if "item_id" in historical.columns else "tank_id"
    hist = historical[historical[id_col] == tank_id].sort_values(timestamp_column)

    pred_id_col = "item_id" if "item_id" in predictions.columns else "tank_id"
    pred_ts_col = "timestamp" if "timestamp" in predictions.columns else timestamp_column
    pred = predictions[predictions[pred_id_col] == tank_id].sort_values(pred_ts_col)

    ax.plot(pd.to_datetime(hist[timestamp_column]), hist[target_column],
            label="Historical", color="blue", linewidth=1.5)

    pred_col = "mean" if "mean" in pred.columns else "0.5"
    ax.plot(pd.to_datetime(pred[pred_ts_col]), pred[pred_col],
            label="Forecast", color="red", linewidth=1.5)

    if "0.1" in pred.columns and "0.9" in pred.columns:
        ax.fill_between(pd.to_datetime(pred[pred_ts_col]),
                        pred["0.1"], pred["0.9"],
                        alpha=0.2, color="red", label="80% Interval")

    if "0.25" in pred.columns and "0.75" in pred.columns:
        ax.fill_between(pd.to_datetime(pred[pred_ts_col]),
                        pred["0.25"], pred["0.75"],
                        alpha=0.3, color="red", label="50% Interval")

    if actuals is not None:
        act_id_col = "item_id" if "item_id" in actuals.columns else "tank_id"
        act_ts_col = "timestamp" if "timestamp" in actuals.columns else timestamp_column
        act = actuals[actuals[act_id_col] == tank_id].sort_values(act_ts_col)
        act_target = "target" if "target" in act.columns else target_column
        ax.scatter(pd.to_datetime(act[act_ts_col]), act[act_target],
                   label="Actuals", color="green", s=20, zorder=5)

    if reorder_level is not None:
        ax.axhline(y=reorder_level, color="orange", linestyle="--",
                   alpha=0.7, label="Reorder Level")

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Consumption (gallons)")
    ax.set_title(title or f"Consumption Forecast - {tank_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_delivery_schedule(
    schedule: List[Dict],
    figsize: Tuple[int, int] = (14, 6),
) -> Any:
    """Plot a delivery schedule as a Gantt-style chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    tanks = sorted(set(s["tank_id"] for s in schedule))
    tank_idx = {t: i for i, t in enumerate(tanks)}

    colors = {"emergency": "red", "urgent": "orange", "normal": "steelblue", "low": "gray"}

    for rec in schedule:
        y = tank_idx[rec["tank_id"]]
        urgency = rec.get("urgency", "normal")
        color = colors.get(urgency, "steelblue")
        ax.barh(y, width=1, left=pd.Timestamp(rec["delivery_date"]).toordinal(),
                height=0.6, color=color, alpha=0.8, edgecolor="white")

    ax.set_yticks(range(len(tanks)))
    ax.set_yticklabels(tanks)
    ax.set_xlabel("Date")
    ax.set_title("Delivery Schedule")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig


def plot_tank_inventory_projection(
    tank_id: str,
    current_level: float,
    tank_capacity: float,
    forecast_consumption: pd.DataFrame,
    delivery_date: Optional[str] = None,
    delivery_volume: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> Any:
    """Project tank inventory level based on forecasted consumption."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    ts_col = "timestamp" if "timestamp" in forecast_consumption.columns else "reading_date"
    pred_col = "mean" if "mean" in forecast_consumption.columns else "0.5"

    dates = pd.to_datetime(forecast_consumption[ts_col])
    consumption = forecast_consumption[pred_col].values

    # Project level day by day
    levels = [current_level]
    for c in consumption:
        next_level = levels[-1] - c
        levels.append(max(0, next_level))
    level_dates = list(dates) + [dates.iloc[-1] + pd.Timedelta(days=1)] if len(dates) > 0 else []

    ax.plot(level_dates[:len(levels)], levels, color="blue", linewidth=2, label="Projected Level")
    ax.axhline(y=tank_capacity * 0.25, color="orange", linestyle="--",
               alpha=0.7, label="Reorder Level (25%)")
    ax.axhline(y=tank_capacity * 0.10, color="red", linestyle="--",
               alpha=0.7, label="Emergency Level (10%)")
    ax.axhline(y=tank_capacity, color="green", linestyle=":", alpha=0.5, label="Tank Capacity")

    if delivery_date and delivery_volume:
        delivery_ts = pd.Timestamp(delivery_date)
        ax.axvline(x=delivery_ts, color="green", linewidth=2, alpha=0.7)
        ax.annotate(f"Delivery: {delivery_volume:.0f} gal",
                    xy=(delivery_ts, current_level * 0.8),
                    fontsize=9, color="green", fontweight="bold")

    ax.fill_between(level_dates[:len(levels)], 0, levels, alpha=0.1, color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tank Level (gallons)")
    ax.set_title(f"Inventory Projection - {tank_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_heuristic_vs_ml_comparison(
    comparison: Dict,
    figsize: Tuple[int, int] = (12, 5),
) -> Any:
    """Bar chart comparing heuristic vs ML delivery metrics."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Volume comparison
    labels = ["Heuristic", "ML"]
    volumes = [comparison.get("heuristic_volume", 0), comparison.get("ml_volume", 0)]
    colors = ["#ff7f0e", "#1f77b4"]
    axes[0].bar(labels, volumes, color=colors)
    axes[0].set_ylabel("Delivery Volume (gallons)")
    axes[0].set_title("Delivery Volume Comparison")

    # Coverage comparison
    coverages = [comparison.get("heuristic_coverage_days", 0),
                 comparison.get("ml_coverage_days", 0)]
    axes[1].bar(labels, coverages, color=colors)
    axes[1].set_ylabel("Coverage (days)")
    axes[1].set_title("Coverage Days Comparison")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_comparison(
    leaderboard: pd.DataFrame,
    metric: str = "score_val",
    figsize: Tuple[int, int] = (10, 6),
) -> Any:
    """Plot model performance comparison."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    lb = leaderboard.sort_values(metric)
    bars = ax.barh(lb["model"], lb[metric])

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
# AWS Utilities
# ============================================================================

def get_sagemaker_role() -> str:
    """Get SageMaker execution role from environment or IAM."""
    import os

    role = os.environ.get("SAGEMAKER_ROLE")
    if role:
        return role

    try:
        from sagemaker import get_execution_role
        return get_execution_role()
    except Exception:
        pass

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
    instance_count: int = 1,
) -> float:
    """Estimate SageMaker training cost (approximate, us-east-1)."""
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
