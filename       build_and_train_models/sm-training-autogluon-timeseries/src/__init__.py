"""
Fuel Delivery Optimization - Source Package
=============================================
ML-driven delivery optimization using AutoGluon-TimeSeries and Amazon SageMaker.
"""

from data_preparation import FuelDeliveryDataProcessor
from optimization import DeliveryConstraints, DeliveryOptimizer, DeliveryScheduler, DeliveryRecommendation
from backtesting import BacktestEngine, BacktestResults
from calendar_us import get_us_holidays, is_holiday_us, add_holiday_features
from utils import (
    generate_fuel_delivery_data,
    evaluate_forecasts,
    calculate_business_metrics,
    load_config,
    plot_consumption_forecast,
    plot_delivery_schedule,
    plot_tank_inventory_projection,
    plot_heuristic_vs_ml_comparison,
)

__all__ = [
    # Data processing
    "FuelDeliveryDataProcessor",
    # Optimization
    "DeliveryConstraints",
    "DeliveryOptimizer",
    "DeliveryScheduler",
    "DeliveryRecommendation",
    # Backtesting
    "BacktestEngine",
    "BacktestResults",
    # Calendar
    "get_us_holidays",
    "is_holiday_us",
    "add_holiday_features",
    # Utils
    "generate_fuel_delivery_data",
    "evaluate_forecasts",
    "calculate_business_metrics",
    "load_config",
    "plot_consumption_forecast",
    "plot_delivery_schedule",
    "plot_tank_inventory_projection",
    "plot_heuristic_vs_ml_comparison",
]
