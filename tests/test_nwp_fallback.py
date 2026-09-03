import math
from datetime import date

import pandas as pd

from features.nwp_fallback import build_nwp_consensus_fallback


def test_fallback_uses_lead0_through_lead6_and_finite_values():
    dates = pd.date_range("2026-09-02", periods=7, freq="D")
    consensus = pd.DataFrame({
        "time": dates,
        "tmax_max_model_mean": [33.0, 34.0, float("nan"), 31.0, 30.0, 32.0, 35.0],
        "tmax_max_model_std": [1.0, float("nan"), 2.0, 1.5, 0.5, 1.0, 2.0],
        "tmax_max_model_min": [31.0, 32.0, 29.0, 29.0, 29.0, 30.0, 31.0],
        "tmax_max_model_max": [35.0, 36.0, 33.0, 33.0, 31.0, 34.0, 39.0],
        "precip_model_mean": [0.0, float("nan"), 5.0, 2.0, 0.0, 1.0, 3.0],
        "precip_model_std": [0.0, float("nan"), 2.0, 1.0, 0.0, 0.5, 1.0],
    })
    det = pd.DataFrame({
        "time": dates,
        "precipitation_sum": [0.0, 1.0, 6.0, 0.0, 0.2, 2.0, 4.0],
    })

    output = build_nwp_consensus_fallback(
        det_df=det,
        consensus=consensus,
        report_date=date(2026, 9, 2),
        horizon=7,
        precipitation_threshold=0.1,
        city_name="上海",
        city_name_en="Shanghai",
        generated_at="2026-09-02 12:00:00",
    )

    assert [row["lead_days"] for row in output["temperature"]] == list(range(7))
    assert [row["lead_days"] for row in output["precipitation"]] == list(range(7))
    assert output["source"] == "nwp_consensus_fallback"
    assert output["calibrated"] is False
    assert output["nwp_training_aware"] is False
    for section in ("temperature", "precipitation"):
        for row in output[section]:
            numeric_values = []
            for value in row.values():
                if isinstance(value, (int, float)):
                    numeric_values.append(float(value))
                elif isinstance(value, dict):
                    numeric_values.extend(float(v) for v in value.values() if isinstance(v, (int, float)))
            assert all(math.isfinite(value) for value in numeric_values)


def test_fallback_derives_temperature_center_when_model_mean_is_nan():
    consensus = pd.DataFrame({
        "time": pd.to_datetime(["2026-09-02"]),
        "tmax_max_model_mean": [float("nan")],
        "tmax_max_model_min": [30.0],
        "tmax_max_model_max": [34.0],
        "precip_model_mean": [0.0],
    })

    output = build_nwp_consensus_fallback(
        det_df=pd.DataFrame(),
        consensus=consensus,
        report_date=date(2026, 9, 2),
        horizon=7,
        precipitation_threshold=0.1,
        city_name="上海",
        city_name_en="Shanghai",
        generated_at="2026-09-02 12:00:00",
    )

    assert output["temperature"][0]["median"] == 32.0


def test_fallback_publishes_dominant_condition_and_separate_rain_events():
    target = pd.Timestamp("2026-09-03")
    consensus = pd.DataFrame({
        "time": [target],
        "tmax_max_model_mean": [30.0],
        "tmax_max_model_std": [1.0],
        "tmax_max_model_min": [29.0],
        "tmax_max_model_max": [31.0],
        "precip_model_mean": [1.5],
        "precip_model_std": [0.5],
    })
    det = pd.DataFrame({
        "time": [target] * 5,
        "model": ["cma", "ecmwf", "gfs", "icon", "jma"],
        "weather_code": [61, 61, 80, 3, 3],
        "cloud_cover_mean": [88, 92, 84, 90, 86],
        "precipitation_hours": [2, 3, 1, 0, 0],
        "precipitation_sum": [0.0, 0.2, 1.2, 4.0, 12.0],
    })

    output = build_nwp_consensus_fallback(
        det_df=det,
        consensus=consensus,
        report_date=date(2026, 9, 3),
        horizon=1,
        precipitation_threshold=0.1,
        city_name="上海",
        city_name_en="Shanghai",
        generated_at="2026-09-03 12:00:00",
    )

    condition = output["conditions"][0]
    precip = output["precipitation"][0]
    assert condition["kind"] == "cloudy"
    assert condition["secondary"] == "showers"
    assert precip["p_trace"] == 0.8
    assert precip["p_wet"] == 0.6
    assert precip["p_heavy"] == 0.2
    assert precip["params"]["p_rain_occurrence"] == 0.6
