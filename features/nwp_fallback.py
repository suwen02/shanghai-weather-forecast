# -*- coding: utf-8 -*-
"""Legacy 模型期间的确定性 NWP 共识 fallback。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from features.weather_condition import (
    precipitation_event_probabilities,
    summarize_daily_condition,
)


def _finite_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _finite_or_nan(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _temperature_center(row: pd.Series) -> Optional[float]:
    mean_value = _finite_or_nan(row.get("tmax_max_model_mean"))
    if np.isfinite(mean_value):
        return mean_value

    minimum = _finite_or_nan(row.get("tmax_max_model_min"))
    maximum = _finite_or_nan(row.get("tmax_max_model_max"))
    if np.isfinite(minimum) and np.isfinite(maximum):
        return (minimum + maximum) / 2.0
    if np.isfinite(minimum):
        return minimum
    if np.isfinite(maximum):
        return maximum
    return None


def build_nwp_consensus_fallback(
    det_df: pd.DataFrame,
    consensus: pd.DataFrame,
    report_date: date,
    horizon: int,
    precipitation_threshold: float,
    city_name: str,
    city_name_en: str,
    generated_at: Optional[str] = None,
) -> dict:
    """Format multi-model NWP consensus without pretending vote fractions are calibrated."""
    if consensus is None or consensus.empty:
        return {}

    current = consensus.copy()
    if "time" not in current.columns:
        return {}
    current["time"] = pd.to_datetime(current["time"], errors="coerce")
    current = current[current["time"].notna()].sort_values("time")
    current = current[current["time"].dt.date >= report_date].head(int(horizon))
    if current.empty:
        return {}

    raw = det_df.copy() if det_df is not None else pd.DataFrame()
    if not raw.empty and "time" in raw.columns:
        raw["time"] = pd.to_datetime(raw["time"], errors="coerce")

    output = {
        "generated_at": generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": city_name,
        "city_en": city_name_en,
        "source": "nwp_consensus_fallback",
        "calibrated": False,
        "nwp_training_aware": False,
        "rain_probability_basis": "deterministic_model_event_frequency",
        "rain_probability_threshold_mm": 1.0,
        "temperature": [],
        "precipitation": [],
        "conditions": [],
    }

    for lead_days, (_, row) in enumerate(current.iterrows(), start=0):
        target_time = pd.Timestamp(row["time"])
        center = _temperature_center(row)
        if center is None:
            continue

        t_std = max(0.0, _finite_float(row.get("tmax_max_model_std"), 0.0))
        t_min = _finite_float(row.get("tmax_max_model_min"), center)
        t_max = _finite_float(row.get("tmax_max_model_max"), center)
        p_mean = max(0.0, _finite_float(row.get("precip_model_mean"), 0.0))
        p_std = max(0.0, _finite_float(row.get("precip_model_std"), 0.0))

        events = precipitation_event_probabilities(raw, target_time)
        condition = summarize_daily_condition(raw, target_time)
        condition_row = {
            "date": target_time.date().isoformat(),
            "lead_days": lead_days,
            **condition,
        }
        output["conditions"].append(condition_row)

        temp_quantiles = {
            "p05": round(min(t_min, center - 1.64 * t_std), 2),
            "p25": round(center - 0.67 * t_std, 2),
            "p50": round(center, 2),
            "p75": round(center + 0.67 * t_std, 2),
            "p95": round(max(t_max, center + 1.64 * t_std), 2),
        }
        output["temperature"].append({
            "date": target_time.date().isoformat(),
            "lead_days": lead_days,
            "median": round(center, 2),
            "quantiles": temp_quantiles,
            "weather_code": condition.get("weather_code"),
            "condition_kind": condition.get("kind"),
            "confidence": "un-calibrated",
        })
        output["precipitation"].append({
            "date": target_time.date().isoformat(),
            "lead_days": lead_days,
            "expected_mm": round(p_mean, 2),
            "p_trace": events["p_trace"],
            "p_wet": events["p_wet"],
            "p_heavy": events["p_heavy"],
            "quantiles": {
                "p25": round(max(0.0, p_mean - 0.67 * p_std), 2),
                "p50": round(p_mean, 2),
                "p75": round(max(0.0, p_mean + 0.67 * p_std), 2),
                "p_rain": events["p_wet"],
            },
            "params": {
                "p_rain_occurrence": events["p_wet"],
                "p_trace": events["p_trace"],
                "p_heavy": events["p_heavy"],
            },
            "weather_code": condition.get("weather_code"),
            "condition_kind": condition.get("kind"),
            "confidence": "un-calibrated",
        })

    return output
