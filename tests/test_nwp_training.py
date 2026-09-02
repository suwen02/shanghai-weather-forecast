import pandas as pd

from features.nwp_training import merge_historical_nwp_features


def test_merge_historical_nwp_features_preserves_daily_variation():
    observations = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01", "2026-08-02"]),
        "temperature_2m_max": [31.0, 32.0],
        "precipitation_sum": [0.0, 2.0],
    })
    forecasts = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01", "2026-08-02"]),
        "model": ["m1", "m1"],
        "temperature_2m_max": [30.0, 34.0],
        "temperature_2m_min": [24.0, 26.0],
        "precipitation_sum": [0.0, 5.0],
    })

    def builder(df):
        return pd.DataFrame({
            "time": df["time"],
            "tmax_max_model_mean": df["temperature_2m_max"],
        })

    result = merge_historical_nwp_features(observations, forecasts, builder)
    assert result["tmax_max_model_mean"].tolist() == [30.0, 34.0]
