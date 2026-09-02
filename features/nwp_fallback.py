# -*- coding: utf-8 -*-
"""Legacy 模型期间的确定性 NWP 共识 fallback。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd


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
    """把当前多模型共识格式化为明确标注“未校准”的 7 天 fallback。"""
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
        "temperature": [],
        "precipitation": [],
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

        p_rain = 0.0
        if not raw.empty and "precipitation_sum" in raw.columns:
            day_raw = raw[raw["time"].dt.normalize() == target_time.normalize()]
            valid_precip = pd.to_numeric(
                day_raw["precipitation_sum"], errors="coerce"
            ).dropna()
            if len(valid_precip):
                p_rain = float((valid_precip >= precipitation_threshold).mean())

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
            "confidence": "un-calibrated",
        })
        output["precipitation"].append({
            "date": target_time.date().isoformat(),
            "lead_days": lead_days,
            "expected_mm": round(p_mean, 2),
            "quantiles": {
                "p25": round(max(0.0, p_mean - 0.67 * p_std), 2),
                "p50": round(p_mean, 2),
                "p75": round(max(0.0, p_mean + 0.67 * p_std), 2),
                "p_rain": round(p_rain, 4),
            },
            "params": {"p_rain_occurrence": round(p_rain, 4)},
            "confidence": "un-calibrated",
        })

    return output
