import pandas as pd

from collectors.training_forecasts import normalize_previous_runs_hourly


def test_normalize_previous_runs_hourly_aggregates_daily_by_lead_and_model():
    payload = {
        "hourly": {
            "time": [
                "2026-08-01T00:00", "2026-08-01T12:00",
                "2026-08-02T00:00", "2026-08-02T12:00",
            ],
            "temperature_2m_previous_day1": [25.0, 33.0, 26.0, 34.0],
            "temperature_2m_previous_day2": [24.0, 31.0, 25.0, 32.0],
            "precipitation_previous_day1": [0.0, 2.0, 1.0, 3.0],
            "precipitation_previous_day2": [4.0, 6.0, 0.0, 2.0],
        }
    }

    result = normalize_previous_runs_hourly(payload, model="gfs_seamless", lead_days=(1, 2))

    assert result[["time", "forecast_lead_days", "model"]].to_dict("records") == [
        {"time": pd.Timestamp("2026-08-01"), "forecast_lead_days": 1, "model": "gfs_seamless"},
        {"time": pd.Timestamp("2026-08-01"), "forecast_lead_days": 2, "model": "gfs_seamless"},
        {"time": pd.Timestamp("2026-08-02"), "forecast_lead_days": 1, "model": "gfs_seamless"},
        {"time": pd.Timestamp("2026-08-02"), "forecast_lead_days": 2, "model": "gfs_seamless"},
    ]
    first = result.iloc[0]
    assert first["temperature_2m_max"] == 33.0
    assert first["temperature_2m_min"] == 25.0
    assert first["temperature_2m_mean"] == 29.0
    assert first["precipitation_sum"] == 2.0


def test_normalize_previous_runs_hourly_drops_unavailable_leads():
    payload = {
        "hourly": {
            "time": ["2026-08-01T00:00", "2026-08-01T12:00"],
            "temperature_2m_previous_day1": [25.0, 33.0],
            "precipitation_previous_day1": [0.0, 2.0],
            "temperature_2m_previous_day2": [None, None],
            "precipitation_previous_day2": [None, None],
        }
    }

    result = normalize_previous_runs_hourly(payload, model="short_model", lead_days=(1, 2))

    assert result["forecast_lead_days"].tolist() == [1]
