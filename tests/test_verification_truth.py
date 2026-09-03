from datetime import date, datetime, timezone

import pandas as pd
import pytest

from collectors.verification_truth import (
    collect_verification_truth,
    daily_to_verification_rows,
    ensure_past_date,
)


def test_daily_to_verification_rows_uses_observed_daily_values_and_condition_semantics():
    daily = pd.DataFrame(
        [
            {
                "time": "2026-09-01",
                "weather_code": 80,
                "cloud_cover_mean": 88.0,
                "precipitation_hours": 2.0,
                "precipitation_sum": 0.6,
                "temperature_2m_max": 31.2,
                "temperature_2m_min": 26.1,
                "temperature_2m_mean": 28.4,
            },
            {
                "time": "2026-09-02",
                "weather_code": 61,
                "cloud_cover_mean": 91.0,
                "precipitation_hours": 9.0,
                "precipitation_sum": 12.4,
                "temperature_2m_max": 29.0,
                "temperature_2m_min": 24.7,
                "temperature_2m_mean": 26.3,
            },
        ]
    )
    observed_at = datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)

    rows = daily_to_verification_rows(daily, location="shanghai", observed_at=observed_at)

    assert [row["valid_date"] for row in rows] == ["2026-09-01", "2026-09-02"]
    assert rows[0]["observed_condition_kind"] == "cloudy"
    assert rows[0]["observed_weather_code"] == 80
    assert rows[0]["observed_precipitation_mm"] == 0.6
    assert rows[0]["observed_temperature_mean"] == 28.4
    assert rows[1]["observed_condition_kind"] == "rain"
    assert rows[1]["observed_precipitation_mm"] == 12.4
    assert rows[1]["source"] == "open_meteo_archive"
    assert rows[1]["observed_at"] == "2026-09-03T00:00:00+00:00"


def test_ensure_past_date_rejects_today_and_future():
    today = date(2026, 9, 3)

    assert ensure_past_date(date(2026, 9, 2), today=today) == date(2026, 9, 2)
    with pytest.raises(ValueError):
        ensure_past_date(today, today=today)
    with pytest.raises(ValueError):
        ensure_past_date(date(2026, 9, 4), today=today)


class RecordingCollector:
    def __init__(self):
        self.calls = []

    def _get(self, url, params):
        self.calls.append((url, params))
        return {
            "daily": {
                "time": ["2026-09-02"],
                "weather_code": [61],
                "cloud_cover_mean": [92.0],
                "precipitation_hours": [8.0],
                "precipitation_sum": [6.2],
                "temperature_2m_max": [29.4],
                "temperature_2m_min": [24.8],
                "temperature_2m_mean": [26.7],
            }
        }


def test_collect_verification_truth_uses_archive_single_day_request():
    collector = RecordingCollector()
    observed_at = datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)

    rows = collect_verification_truth(
        date(2026, 9, 2),
        location="shanghai",
        observed_at=observed_at,
        today=date(2026, 9, 3),
        collector=collector,
    )

    assert len(collector.calls) == 1
    url, params = collector.calls[0]
    assert "archive-api.open-meteo.com" in url
    assert params["start_date"] == "2026-09-02"
    assert params["end_date"] == "2026-09-02"
    assert "weather_code" in params["daily"]
    assert "cloud_cover_mean" in params["daily"]
    assert rows[0]["valid_date"] == "2026-09-02"
    assert rows[0]["observed_condition_kind"] == "rain"
