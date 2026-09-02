import pandas as pd

from features.prediction_frame import build_forecast_scaffold


def test_seven_day_scaffold_preserves_nwp_divergence():
    history = pd.DataFrame({
        "time": pd.to_datetime(["2026-09-01"]),
        "temperature_2m_max_lag1d": [32.0],
    })
    dates = pd.date_range("2026-09-02", periods=7, freq="D")
    consensus = pd.DataFrame({
        "time": dates,
        "tmax_max_model_mean": [33.0, 34.5, 31.2, 29.8, 30.5, 32.1, 35.0],
        "precip_model_mean": [0.0, 1.2, 18.0, 5.0, 0.0, 0.4, 2.1],
    })

    result = build_forecast_scaffold(history, consensus, pd.DataFrame(), pd.DataFrame())

    assert len(result) == 7
    assert result["tmax_max_model_mean"].nunique() == 7
    assert result["temperature_2m_max_lag1d"].nunique() == 1


def test_forecast_sources_override_nan_columns_from_history():
    history = pd.DataFrame({
        "time": pd.to_datetime(["2026-09-01"]),
        "ens_spread": [float("nan")],
        "spatial_temp_mean": [float("nan")],
    })
    consensus = pd.DataFrame({
        "time": pd.to_datetime(["2026-09-02"]),
        "tmax_max_model_mean": [33.0],
    })
    ensemble = pd.DataFrame({
        "time": pd.to_datetime(["2026-09-02"]),
        "ens_spread": [2.5],
    })
    spatial = pd.DataFrame({
        "time": pd.to_datetime(["2026-09-02"]),
        "spatial_temp_mean": [31.4],
    })

    result = build_forecast_scaffold(history, consensus, ensemble, spatial)

    assert result.loc[0, "ens_spread"] == 2.5
    assert result.loc[0, "spatial_temp_mean"] == 31.4
