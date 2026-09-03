# -*- coding: utf-8 -*-
"""Dominant daily weather condition and precipitation-event consensus."""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}
STORM_CODES = {95, 96, 99}
SNOW_CODES = {71, 73, 75, 77, 85, 86}
FOG_CODES = {45, 48}

# In a tie, prefer the less severe primary state. Severe states still win when
# they have a strict plurality through the normal vote count.
TIE_PRIORITY = {
    "sunny": 0,
    "partly-cloudy": 1,
    "cloudy": 2,
    "fog": 3,
    "rain": 4,
    "snow": 5,
    "storm": 6,
}


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _code(value) -> Optional[int]:
    number = _finite(value)
    return int(number) if number is not None else None


def _day_rows(det_df: pd.DataFrame, target_time) -> pd.DataFrame:
    if det_df is None or det_df.empty or "time" not in det_df.columns:
        return pd.DataFrame()
    rows = det_df.copy()
    rows["time"] = pd.to_datetime(rows["time"], errors="coerce")
    target = pd.Timestamp(target_time).normalize()
    return rows[rows["time"].dt.normalize() == target].copy()


def _cloud_primary(cloud_cover, code: Optional[int]) -> str:
    cloud = _finite(cloud_cover)
    if cloud is not None:
        if cloud < 35.0:
            return "sunny"
        if cloud < 80.0:
            return "partly-cloudy"
        return "cloudy"
    if code == 0:
        return "sunny"
    if code in {1, 2}:
        return "partly-cloudy"
    if code == 3:
        return "cloudy"
    return "partly-cloudy"


def classify_model_condition(row: pd.Series) -> str:
    """Classify one model's dominant daily condition.

    Open-Meteo's daily WMO code is the most severe condition during the day,
    so a brief shower must not automatically become the primary daily state.
    """
    code = _code(row.get("weather_code"))
    precip_hours = _finite(row.get("precipitation_hours")) or 0.0
    precip_sum = _finite(row.get("precipitation_sum")) or 0.0

    if code in STORM_CODES:
        return "storm"
    if code in SNOW_CODES:
        return "snow"
    if code in FOG_CODES:
        return "fog"
    if code in RAIN_CODES:
        if precip_hours >= 6.0 or precip_sum >= 5.0:
            return "rain"
        return _cloud_primary(row.get("cloud_cover_mean"), code)
    return _cloud_primary(row.get("cloud_cover_mean"), code)


def precipitation_event_probabilities(det_df: pd.DataFrame, target_time) -> dict:
    """Return deterministic-model event frequencies for 0.1/1/10 mm/day."""
    rows = _day_rows(det_df, target_time)
    if rows.empty or "precipitation_sum" not in rows.columns:
        return {"p_trace": 0.0, "p_wet": 0.0, "p_heavy": 0.0, "model_count": 0}

    values = pd.to_numeric(rows["precipitation_sum"], errors="coerce").dropna()
    count = int(len(values))
    if count == 0:
        return {"p_trace": 0.0, "p_wet": 0.0, "p_heavy": 0.0, "model_count": 0}

    return {
        "p_trace": round(float((values >= 0.1).mean()), 4),
        "p_wet": round(float((values >= 1.0).mean()), 4),
        "p_heavy": round(float((values >= 10.0).mean()), 4),
        "model_count": count,
    }


def summarize_daily_condition(det_df: pd.DataFrame, target_time) -> dict:
    """Build a transparent cross-model dominant-condition summary."""
    rows = _day_rows(det_df, target_time)
    if rows.empty:
        return {
            "kind": "partly-cloudy",
            "secondary": None,
            "weather_code": None,
            "model_agreement": 0.0,
            "cloud_cover_mean": None,
            "model_count": 0,
        }

    kinds = [classify_model_condition(row) for _, row in rows.iterrows()]
    counts = Counter(kinds)
    max_count = max(counts.values())
    tied = [kind for kind, count in counts.items() if count == max_count]
    kind = min(tied, key=lambda item: TIE_PRIORITY.get(item, 99))

    probs = precipitation_event_probabilities(rows, pd.Timestamp(target_time))
    secondary = None
    if kind not in {"rain", "storm", "snow"} and probs["p_trace"] >= 0.4:
        secondary = "showers"

    codes = pd.to_numeric(rows.get("weather_code"), errors="coerce").dropna() if "weather_code" in rows.columns else pd.Series(dtype=float)
    weather_code = None
    if not codes.empty:
        modes = codes.astype(int).mode()
        if not modes.empty:
            weather_code = int(modes.iloc[0])

    clouds = pd.to_numeric(rows.get("cloud_cover_mean"), errors="coerce").dropna() if "cloud_cover_mean" in rows.columns else pd.Series(dtype=float)
    cloud_mean = round(float(clouds.mean()), 1) if not clouds.empty else None

    return {
        "kind": kind,
        "secondary": secondary,
        "weather_code": weather_code,
        "model_agreement": round(max_count / len(kinds), 4),
        "cloud_cover_mean": cloud_mean,
        "model_count": int(len(kinds)),
    }
