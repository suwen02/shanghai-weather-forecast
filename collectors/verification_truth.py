# -*- coding: utf-8 -*-
"""Adapters for verified historical weather truth."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

import pandas as pd

from features.weather_condition import classify_model_condition


def ensure_past_date(valid_date: date, *, today: date | None = None) -> date:
    """Return ``valid_date`` only when it is strictly in the past."""
    reference = today or date.today()
    if valid_date >= reference:
        raise ValueError("verification truth is only available for past dates")
    return valid_date


def _iso_timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _number(row: pd.Series, name: str):
    value = row.get(name)
    if pd.isna(value):
        return None
    return float(value)


def _integer(row: pd.Series, name: str):
    value = row.get(name)
    if pd.isna(value):
        return None
    return int(value)


def daily_to_verification_rows(
    daily: pd.DataFrame,
    *,
    location: str,
    observed_at: datetime | str,
    source: str = "open_meteo_archive",
) -> List[Dict[str, Any]]:
    """Convert archived daily observations to ``weather_verifications`` rows."""
    if daily is None or daily.empty:
        return []

    frame = daily.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time")
    observed = _iso_timestamp(observed_at)

    rows: List[Dict[str, Any]] = []
    for _, item in frame.iterrows():
        payload = {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in item.to_dict().items()
            if key != "time"
        }
        rows.append(
            {
                "location": location,
                "valid_date": item["time"].date().isoformat(),
                "source": source,
                "observed_condition_kind": classify_model_condition(item),
                "observed_weather_code": _integer(item, "weather_code"),
                "observed_precipitation_mm": _number(item, "precipitation_sum"),
                "observed_temperature_max": _number(item, "temperature_2m_max"),
                "observed_temperature_min": _number(item, "temperature_2m_min"),
                "observed_temperature_mean": _number(item, "temperature_2m_mean"),
                "payload": payload,
                "observed_at": observed,
            }
        )

    return rows
