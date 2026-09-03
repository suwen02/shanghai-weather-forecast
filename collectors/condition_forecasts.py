# -*- coding: utf-8 -*-
"""Focused Open-Meteo collector for daily weather-condition signals."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

import pandas as pd

from collectors.open_meteo import OpenMeteoCollector
from config.settings import (
    API_ENDPOINTS,
    DETERMINISTIC_MODELS,
    SHANGHAI_LAT,
    SHANGHAI_LON,
    TIMEZONE,
)

CONDITION_DAILY_VARIABLES = (
    "weather_code",
    "cloud_cover_mean",
    "precipitation_hours",
    "precipitation_sum",
)


def collect_condition_forecasts(
    collector: Optional[OpenMeteoCollector] = None,
    target_date: Optional[date] = None,
    lat: float = SHANGHAI_LAT,
    lon: float = SHANGHAI_LON,
    models: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Collect seven daily condition-signal rows for every deterministic model."""
    collector = collector or OpenMeteoCollector()
    model_names = list(models or DETERMINISTIC_MODELS)
    frames = []

    for model in model_names:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(CONDITION_DAILY_VARIABLES),
            "models": model,
            "timezone": TIMEZONE,
            "forecast_days": 7,
        }
        if target_date is not None and target_date != date.today():
            params["start_date"] = target_date.isoformat()
            params["end_date"] = (pd.Timestamp(target_date) + pd.Timedelta(days=6)).date().isoformat()
            params.pop("forecast_days", None)

        try:
            data = collector._get(API_ENDPOINTS["deterministic"], params)
        except Exception:
            continue
        daily = data.get("daily") or {}
        if not daily or not daily.get("time"):
            continue
        frame = pd.DataFrame(daily)
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame = frame[frame["time"].notna()].copy()
        frame["model"] = model
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["time", "model", *CONDITION_DAILY_VARIABLES])
    return pd.concat(frames, ignore_index=True)
