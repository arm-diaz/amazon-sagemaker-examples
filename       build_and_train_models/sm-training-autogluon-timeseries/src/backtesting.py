"""
Backtesting Engine
===================
Walk-forward simulation comparing ML-driven delivery recommendations
against the heuristic baseline (avg_consumption * 1.20).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from optimization import DeliveryConstraints, DeliveryOptimizer


# ============================================================================
# Result container
# ============================================================================

@dataclass
class BacktestResults:
    """Aggregated backtesting results."""
    ml_recommendations: List[Dict] = field(default_factory=list)
    heuristic_recommendations: List[Dict] = field(default_factory=list)
    daily_states: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        """Return a summary of ML vs heuristic performance."""
        return {
            "ml_total_deliveries": len(self.ml_recommendations),
            "heuristic_total_deliveries": len(self.heuristic_recommendations),
            **self.metrics,
        }


# ============================================================================
# Backtest engine
# ============================================================================

class BacktestEngine:
    """
    Walk-forward backtest that simulates delivery decisions day-by-day.

    For each window:
      1. Train/load a model on history up to the window start.
      2. Forecast the next `prediction_length` days.
      3. Generate ML recommendation via DeliveryOptimizer.
      4. Generate heuristic recommendation (avg * 1.20).
      5. Simulate tank-level evolution for both strategies.
      6. Record daily states and deliveries.
    """

    def __init__(
        self,
        constraints: Optional[DeliveryConstraints] = None,
        prediction_length: int = 10,
    ):
        self.constraints = constraints or DeliveryConstraints()
        self.prediction_length = prediction_length
        self.optimizer = DeliveryOptimizer(self.constraints)

    def run(
        self,
        df: pd.DataFrame,
        tank_id: str,
        tank_capacity: float,
        start_level: Optional[float] = None,
        step_days: int = 10,
        item_id_column: str = "tank_id",
        timestamp_column: str = "reading_date",
        target_column: str = "daily_consumption",
    ) -> BacktestResults:
        """
        Run backtest on a single tank's historical data.

        Instead of actually re-training a model at each step (expensive),
        this method uses the *actual* future values as a stand-in for the
        forecast.  This isolates the optimisation logic from model quality.

        Args:
            df: Full historical DataFrame for the tank.
            tank_id: Tank identifier.
            tank_capacity: Tank capacity in gallons.
            start_level: Starting tank level (defaults to 85% capacity).
            step_days: Number of days to advance the window.
        """
        tank_df = (
            df[df[item_id_column] == tank_id]
            .sort_values(timestamp_column)
            .reset_index(drop=True)
        )

        if start_level is None:
            start_level = tank_capacity * 0.85

        n = len(tank_df)
        results = BacktestResults()

        ml_level = start_level
        heuristic_level = start_level

        window_start = 0

        while window_start + self.prediction_length <= n:
            window_end = window_start + self.prediction_length
            window = tank_df.iloc[window_start:window_end].copy()
            actual_consumption = window[target_column].values

            # Historical average for heuristic
            history = tank_df.iloc[:window_start]
            hist_avg = history[target_column].mean() if len(history) > 0 else actual_consumption.mean()

            # Build a fake "forecast" DataFrame from actuals (perfect foresight)
            forecast_df = window[[timestamp_column, target_column]].copy()
            forecast_df = forecast_df.rename(columns={
                timestamp_column: "timestamp",
                target_column: "mean",
            })
            # Simulate quantiles: p10 = mean*0.8, p90 = mean*1.2
            forecast_df["0.1"] = forecast_df["mean"] * 0.8
            forecast_df["0.9"] = forecast_df["mean"] * 1.2

            # --- ML recommendation ---
            ml_rec = self.optimizer.calculate_optimal_delivery(
                tank_id=tank_id,
                current_level=ml_level,
                tank_capacity=tank_capacity,
                forecast=forecast_df,
            )
            if ml_rec.volume > 0:
                results.ml_recommendations.append(ml_rec.to_dict())

            # --- Heuristic recommendation ---
            heuristic_total = hist_avg * self.prediction_length * 1.20
            heuristic_volume = max(
                0,
                min(
                    tank_capacity * self.constraints.target_fill_pct - heuristic_level,
                    heuristic_total,
                ),
            )
            h_level_pct = heuristic_level / tank_capacity
            if h_level_pct <= self.constraints.reorder_pct or h_level_pct <= self.constraints.emergency_pct:
                if heuristic_volume >= self.constraints.min_delivery_gallons:
                    results.heuristic_recommendations.append({
                        "tank_id": tank_id,
                        "volume": round(heuristic_volume, 1),
                        "delivery_date": str(window[timestamp_column].iloc[0])[:10],
                        "urgency": "emergency" if h_level_pct <= self.constraints.emergency_pct else "urgent",
                    })
                    heuristic_level += heuristic_volume

            # Add ML delivery volume
            ml_level += ml_rec.volume

            # Simulate consumption over the window
            for day_idx, consumption in enumerate(actual_consumption):
                ml_level = max(0, ml_level - consumption)
                heuristic_level = max(0, heuristic_level - consumption)

                results.daily_states.append({
                    "date": str(window[timestamp_column].iloc[day_idx])[:10],
                    "ml_level": round(ml_level, 1),
                    "heuristic_level": round(heuristic_level, 1),
                    "actual_consumption": round(consumption, 2),
                    "ml_stockout": int(ml_level <= 0),
                    "heuristic_stockout": int(heuristic_level <= 0),
                })

            window_start += step_days

        # Aggregate metrics
        states = pd.DataFrame(results.daily_states)
        if len(states) > 0:
            results.metrics = {
                "ml_stockout_days": int(states["ml_stockout"].sum()),
                "heuristic_stockout_days": int(states["heuristic_stockout"].sum()),
                "ml_avg_level": float(states["ml_level"].mean()),
                "heuristic_avg_level": float(states["heuristic_level"].mean()),
                "ml_total_delivered": sum(
                    r["volume"] for r in results.ml_recommendations
                ),
                "heuristic_total_delivered": sum(
                    r["volume"] for r in results.heuristic_recommendations
                ),
            }

            ml_fills = [r.get("fill_pct", 0) for r in results.ml_recommendations]
            if ml_fills:
                results.metrics["ml_avg_fill_pct"] = float(np.mean(ml_fills))

            h_fills = []
            for r in results.heuristic_recommendations:
                vol = r.get("volume", 0)
                # approximate fill
                h_fills.append(vol / tank_capacity if tank_capacity > 0 else 0)
            if h_fills:
                results.metrics["heuristic_avg_fill_pct"] = float(np.mean(h_fills))

            emergency_ml = sum(
                1 for r in results.ml_recommendations if r.get("urgency") == "emergency"
            )
            emergency_h = sum(
                1 for r in results.heuristic_recommendations if r.get("urgency") == "emergency"
            )
            ml_total = max(len(results.ml_recommendations), 1)
            h_total = max(len(results.heuristic_recommendations), 1)
            results.metrics["ml_emergency_rate"] = emergency_ml / ml_total
            results.metrics["heuristic_emergency_rate"] = emergency_h / h_total

        return results
