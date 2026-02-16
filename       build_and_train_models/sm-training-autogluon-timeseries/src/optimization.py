"""
Delivery Optimization Module
==============================
Translates consumption forecasts into actionable delivery recommendations.
Compares ML-driven recommendations against the simple heuristic baseline
(avg_consumption * 1.20).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class DeliveryConstraints:
    """Physical and operational constraints for deliveries."""
    min_delivery_gallons: float = 100
    max_delivery_gallons: float = 3000
    safety_buffer_days: int = 2
    target_fill_pct: float = 0.85
    reorder_pct: float = 0.25
    emergency_pct: float = 0.10
    max_deliveries_per_day: int = 20
    lead_time_days: int = 1

    @classmethod
    def from_config(cls, config: Dict) -> "DeliveryConstraints":
        opt = config.get("optimization", {})
        return cls(
            min_delivery_gallons=opt.get("min_delivery_gallons", 100),
            max_delivery_gallons=opt.get("max_delivery_gallons", 3000),
            safety_buffer_days=opt.get("safety_buffer_days", 2),
            target_fill_pct=opt.get("target_fill_pct", 0.85),
            reorder_pct=opt.get("reorder_pct", 0.25),
            emergency_pct=opt.get("emergency_pct", 0.10),
            max_deliveries_per_day=opt.get("max_deliveries_per_day", 20),
            lead_time_days=opt.get("lead_time_days", 1),
        )


@dataclass
class DeliveryRecommendation:
    """A single delivery recommendation for one tank."""
    tank_id: str
    delivery_date: str
    volume: float
    fill_pct: float
    coverage_days: float
    urgency: str  # "emergency", "urgent", "normal", "low"
    confidence: float
    reasoning: str
    current_level: float = 0.0
    tank_capacity: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "tank_id": self.tank_id,
            "delivery_date": self.delivery_date,
            "volume": round(self.volume, 1),
            "fill_pct": round(self.fill_pct, 3),
            "coverage_days": round(self.coverage_days, 1),
            "urgency": self.urgency,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "current_level": round(self.current_level, 1),
            "tank_capacity": round(self.tank_capacity, 1),
        }


# ============================================================================
# Delivery Optimizer
# ============================================================================

class DeliveryOptimizer:
    """
    Compute optimal delivery volume and timing for a single tank using
    ML consumption forecasts (p90 for conservative estimates).
    """

    def __init__(self, constraints: Optional[DeliveryConstraints] = None):
        self.constraints = constraints or DeliveryConstraints()

    def calculate_optimal_delivery(
        self,
        tank_id: str,
        current_level: float,
        tank_capacity: float,
        forecast: pd.DataFrame,
        delivery_date: Optional[str] = None,
    ) -> DeliveryRecommendation:
        """
        Calculate the optimal delivery for a single tank.

        Args:
            tank_id: Tank identifier.
            current_level: Current tank level in gallons.
            tank_capacity: Tank capacity in gallons.
            forecast: DataFrame with consumption forecast (columns: timestamp/mean/0.9).
            delivery_date: Planned delivery date (defaults to lead_time_days from now).
        """
        c = self.constraints

        # Use p90 (conservative) if available, else mean
        if "0.9" in forecast.columns:
            daily_consumption = forecast["0.9"].values
        else:
            col = "mean" if "mean" in forecast.columns else forecast.columns[-1]
            daily_consumption = forecast[col].values

        total_consumption = float(np.sum(daily_consumption))
        avg_daily = float(np.mean(daily_consumption)) if len(daily_consumption) > 0 else 0.0

        # Urgency based on current level
        level_pct = current_level / tank_capacity if tank_capacity > 0 else 0
        if level_pct <= c.emergency_pct:
            urgency = "emergency"
        elif level_pct <= c.reorder_pct:
            urgency = "urgent"
        elif level_pct <= 0.40:
            urgency = "normal"
        else:
            urgency = "low"

        # Target volume: fill to target_fill_pct
        target_level = tank_capacity * c.target_fill_pct
        volume_needed = target_level - current_level

        # Add safety buffer
        if avg_daily > 0:
            safety_gallons = avg_daily * c.safety_buffer_days
            volume_needed += safety_gallons

        # Clamp to constraints
        volume = np.clip(volume_needed, c.min_delivery_gallons, c.max_delivery_gallons)
        volume = min(volume, tank_capacity - current_level)  # Can't overfill

        if volume < c.min_delivery_gallons and urgency not in ("emergency", "urgent"):
            volume = 0.0

        # Fill percentage after delivery
        fill_pct = (current_level + volume) / tank_capacity if tank_capacity > 0 else 0

        # Days of coverage
        coverage_days = (current_level + volume) / avg_daily if avg_daily > 0 else 999

        # Confidence: based on forecast spread
        if "0.1" in forecast.columns and "0.9" in forecast.columns:
            spread = (forecast["0.9"] - forecast["0.1"]).mean()
            mean_val = forecast["mean"].mean() if "mean" in forecast.columns else avg_daily
            confidence = max(0, 1 - (spread / mean_val)) if mean_val > 0 else 0.5
        else:
            confidence = 0.5

        # Delivery date
        if delivery_date is None:
            ts_col = "timestamp" if "timestamp" in forecast.columns else forecast.columns[0]
            if len(forecast) > c.lead_time_days:
                delivery_date = str(forecast[ts_col].iloc[c.lead_time_days])[:10]
            elif len(forecast) > 0:
                delivery_date = str(forecast[ts_col].iloc[0])[:10]
            else:
                delivery_date = "unknown"

        # Reasoning
        reasoning = (
            f"Tank at {level_pct:.0%} capacity ({current_level:.0f}/{tank_capacity:.0f} gal). "
            f"Forecasted {total_consumption:.0f} gal consumption over {len(daily_consumption)} days "
            f"(avg {avg_daily:.1f} gal/day). "
            f"Recommend {volume:.0f} gal delivery to reach {fill_pct:.0%} fill, "
            f"providing ~{coverage_days:.0f} days coverage."
        )

        return DeliveryRecommendation(
            tank_id=tank_id,
            delivery_date=delivery_date,
            volume=volume,
            fill_pct=fill_pct,
            coverage_days=coverage_days,
            urgency=urgency,
            confidence=confidence,
            reasoning=reasoning,
            current_level=current_level,
            tank_capacity=tank_capacity,
        )

    def compare_with_heuristic(
        self,
        tank_id: str,
        current_level: float,
        tank_capacity: float,
        forecast: pd.DataFrame,
        historical_avg_consumption: float,
    ) -> Dict:
        """
        Compare ML-driven recommendation with the simple heuristic
        (avg_consumption * 1.20 safety factor).
        """
        ml_rec = self.calculate_optimal_delivery(
            tank_id, current_level, tank_capacity, forecast
        )

        # Heuristic: simple 20% safety margin on historical average
        prediction_length = len(forecast)
        heuristic_total = historical_avg_consumption * prediction_length * 1.20
        heuristic_volume = min(
            tank_capacity * self.constraints.target_fill_pct - current_level,
            heuristic_total,
        )
        heuristic_volume = max(0, heuristic_volume)

        avg_daily_heuristic = historical_avg_consumption
        heuristic_coverage = (
            (current_level + heuristic_volume) / avg_daily_heuristic
            if avg_daily_heuristic > 0
            else 999
        )
        heuristic_fill = (
            (current_level + heuristic_volume) / tank_capacity
            if tank_capacity > 0
            else 0
        )

        return {
            "tank_id": tank_id,
            "ml_volume": ml_rec.volume,
            "ml_coverage_days": ml_rec.coverage_days,
            "ml_fill_pct": ml_rec.fill_pct,
            "ml_urgency": ml_rec.urgency,
            "heuristic_volume": heuristic_volume,
            "heuristic_coverage_days": heuristic_coverage,
            "heuristic_fill_pct": heuristic_fill,
            "volume_savings": heuristic_volume - ml_rec.volume,
            "volume_savings_pct": (
                (heuristic_volume - ml_rec.volume) / heuristic_volume * 100
                if heuristic_volume > 0
                else 0
            ),
        }


# ============================================================================
# Delivery Scheduler
# ============================================================================

class DeliveryScheduler:
    """
    Multi-tank scheduling: prioritise tanks by urgency and respect
    daily delivery capacity limits.
    """

    def __init__(self, constraints: Optional[DeliveryConstraints] = None):
        self.constraints = constraints or DeliveryConstraints()
        self.optimizer = DeliveryOptimizer(self.constraints)

    def create_schedule(
        self,
        tank_states: List[Dict],
        forecasts: Dict[str, pd.DataFrame],
    ) -> List[DeliveryRecommendation]:
        """
        Build a delivery schedule across all tanks.

        Args:
            tank_states: List of dicts with keys tank_id, current_level, tank_capacity.
            forecasts: Mapping from tank_id to its forecast DataFrame.

        Returns:
            Sorted list of DeliveryRecommendation (highest urgency first).
        """
        recommendations: List[DeliveryRecommendation] = []

        for state in tank_states:
            tank_id = state["tank_id"]
            forecast = forecasts.get(tank_id)
            if forecast is None or forecast.empty:
                continue

            rec = self.optimizer.calculate_optimal_delivery(
                tank_id=tank_id,
                current_level=state["current_level"],
                tank_capacity=state["tank_capacity"],
                forecast=forecast,
            )

            if rec.volume > 0:
                recommendations.append(rec)

        # Sort by urgency priority
        urgency_order = {"emergency": 0, "urgent": 1, "normal": 2, "low": 3}
        recommendations.sort(key=lambda r: (urgency_order.get(r.urgency, 9), -r.volume))

        # Enforce daily capacity limit
        daily_count: Dict[str, int] = {}
        scheduled: List[DeliveryRecommendation] = []

        for rec in recommendations:
            date_key = rec.delivery_date
            current = daily_count.get(date_key, 0)
            if current < self.constraints.max_deliveries_per_day:
                scheduled.append(rec)
                daily_count[date_key] = current + 1

        return scheduled
