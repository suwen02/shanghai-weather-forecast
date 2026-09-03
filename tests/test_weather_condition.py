import pandas as pd

from features.weather_condition import (
    precipitation_event_probabilities,
    summarize_daily_condition,
)


def _rows(**overrides):
    base = {
        "time": pd.Timestamp("2026-09-03"),
        "model": "ecmwf_ifs025",
        "weather_code": 3,
        "cloud_cover_mean": 88.0,
        "precipitation_hours": 0.0,
        "precipitation_sum": 0.0,
    }
    rows = []
    for index, model in enumerate([
        "cma_grapes_global",
        "ecmwf_ifs025",
        "gfs_seamless",
        "icon_seamless",
        "jma_seamless",
    ]):
        row = dict(base)
        row["model"] = model
        for key, value in overrides.items():
            row[key] = value[index] if isinstance(value, list) else value
        rows.append(row)
    return pd.DataFrame(rows)


def test_brief_showers_do_not_turn_cloudy_day_into_rain_primary():
    det = _rows(
        weather_code=[61, 61, 80, 3, 3],
        cloud_cover_mean=[88, 92, 84, 90, 86],
        precipitation_hours=[2, 3, 1, 0, 0],
        precipitation_sum=[0.4, 0.7, 0.2, 0.0, 0.0],
    )

    condition = summarize_daily_condition(det, pd.Timestamp("2026-09-03"))

    assert condition["kind"] == "cloudy"
    assert condition["secondary"] == "showers"
    assert condition["model_count"] == 5
    assert 0.0 <= condition["model_agreement"] <= 1.0


def test_primary_weather_code_represents_dominant_kind_not_severe_raw_mode():
    det = _rows(
        weather_code=[61, 61, 61, 2, 3],
        cloud_cover_mean=[90, 92, 88, 85, 87],
        precipitation_hours=[1, 2, 1, 0, 0],
        precipitation_sum=[0.2, 0.3, 0.4, 0.0, 0.0],
    )

    condition = summarize_daily_condition(det, pd.Timestamp("2026-09-03"))

    assert condition["kind"] == "cloudy"
    assert condition["weather_code"] == 3
    assert condition["source_weather_code"] == 61


def test_sustained_material_rain_is_rain_primary():
    det = _rows(
        weather_code=[61, 63, 80, 61, 63],
        cloud_cover_mean=[90, 95, 88, 91, 89],
        precipitation_hours=[8, 10, 7, 9, 8],
        precipitation_sum=[8.0, 12.0, 6.0, 9.0, 7.0],
    )

    condition = summarize_daily_condition(det, pd.Timestamp("2026-09-03"))

    assert condition["kind"] == "rain"
    assert condition["secondary"] is None


def test_storm_and_snow_classes_are_preserved_when_they_dominate():
    storm = _rows(weather_code=[95, 95, 95, 3, 3], precipitation_hours=2, precipitation_sum=2.0)
    snow = _rows(weather_code=[71, 73, 75, 3, 3], precipitation_hours=4, precipitation_sum=4.0)

    assert summarize_daily_condition(storm, pd.Timestamp("2026-09-03"))["kind"] == "storm"
    assert summarize_daily_condition(snow, pd.Timestamp("2026-09-03"))["kind"] == "snow"


def test_precipitation_event_probabilities_separate_trace_wet_and_heavy():
    det = _rows(precipitation_sum=[0.0, 0.2, 1.2, 4.0, 12.0])

    probs = precipitation_event_probabilities(det, pd.Timestamp("2026-09-03"))

    assert probs == {
        "p_trace": 0.8,
        "p_wet": 0.6,
        "p_heavy": 0.2,
        "model_count": 5,
    }
