# -*- coding: utf-8 -*-
"""Convert one published forecast payload into immutable forecast-run rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping


def _iso_timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _by_date(items: Iterable[Mapping[str, Any]] | None) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for item in items or []:
        valid_date = item.get("date")
        if valid_date:
            result[str(valid_date)] = item
    return result


def _lead_for_date(*items: Mapping[str, Any] | None) -> int | None:
    for item in items:
        if item is None:
            continue
        lead = item.get("lead_days")
        if lead is not None:
            return int(lead)
    return None


def payload_to_forecast_run_rows(
    payload: Mapping[str, Any],
    location: str,
    run_key: str,
    issued_at: datetime | str,
) -> List[Dict[str, Any]]:
    """Join condition, temperature and precipitation records by valid date.

    The output is deterministic and contains one row per date present in any of
    the three forecast sections.  It mirrors the ``weather_forecast_runs``
    storage schema but performs no I/O.
    """

    conditions = _by_date(payload.get("conditions"))
    temperatures = _by_date(payload.get("temperature"))
    precipitation = _by_date(payload.get("precipitation"))
    valid_dates = sorted(set(conditions) | set(temperatures) | set(precipitation))
    issued = _iso_timestamp(issued_at)

    rows: List[Dict[str, Any]] = []
    for valid_date in valid_dates:
        condition = conditions.get(valid_date)
        temperature = temperatures.get(valid_date)
        precip = precipitation.get(valid_date)
        lead_days = _lead_for_date(condition, temperature, precip)

        row_payload = {
            "condition": dict(condition) if condition is not None else None,
            "temperature": dict(temperature) if temperature is not None else None,
            "precipitation": dict(precip) if precip is not None else None,
        }
        rows.append(
            {
                "location": location,
                "run_key": run_key,
                "issued_at": issued,
                "valid_date": valid_date,
                "lead_days": lead_days,
                "source": payload.get("source"),
                "model": payload.get("model"),
                "condition_kind": condition.get("kind") if condition else None,
                "condition_secondary": condition.get("secondary") if condition else None,
                "weather_code": condition.get("weather_code") if condition else None,
                "model_agreement": condition.get("model_agreement") if condition else None,
                "cloud_cover_mean": condition.get("cloud_cover_mean") if condition else None,
                "p_trace": precip.get("p_trace") if precip else None,
                "p_wet": precip.get("p_wet") if precip else None,
                "p_heavy": precip.get("p_heavy") if precip else None,
                "temperature_median": temperature.get("median") if temperature else None,
                "precipitation_expected_mm": precip.get("expected_mm") if precip else None,
                "payload": row_payload,
            }
        )

    return rows
