import pandas as pd

from features.prediction_frame import build_forecast_scaffold


def test_forecast_scaffold_uses_future_nwp_dates_not_history_rows():
    history = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-29", "2026-08-30"]),
        "temperature_2m_max_lag1d": [30.0, 31.0],
        "temperature_2m_max_rmean7d": [29.5, 30.0],
    })
    consensus = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-31", "2026-09-01"]),
        "tmax_max_model_mean": [32.0, 33.0],
    })
    result = build_forecast_scaffold(history, consensus, pd.DataFrame(), pd.DataFrame())
    assert list(result["time"].dt.strftime("%Y-%m-%d")) == ["2026-08-31", "2026-09-01"]
    assert list(result["temperature_2m_max_lag1d"]) == [31.0, 31.0]
    assert list(result["tmax_max_model_mean"]) == [32.0, 33.0]


def test_forecast_scaffold_merges_ensemble_and_spatial_by_forecast_date():
    history = pd.DataFrame({"time": pd.to_datetime(["2026-08-30"]), "x_lag1d": [1.0]})
    consensus = pd.DataFrame({"time": pd.to_datetime(["2026-08-31"]), "model_mean": [10.0]})
    ensemble = pd.DataFrame({"time": pd.to_datetime(["2026-08-31"]), "ens_spread": [2.0]})
    spatial = pd.DataFrame({"time": pd.to_datetime(["2026-08-31"]), "spatial_temp_mean": [31.0]})
    result = build_forecast_scaffold(history, consensus, ensemble, spatial)
    assert result.loc[0, "ens_spread"] == 2.0
    assert result.loc[0, "spatial_temp_mean"] == 31.0


def test_forecast_sources_override_nan_columns_copied_from_history():
    history = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-30"]),
        "ens_spread": [float("nan")],
        "spatial_temp_mean": [float("nan")],
    })
    consensus = pd.DataFrame({"time": pd.to_datetime(["2026-08-31"]), "model_mean": [10.0]})
    ensemble = pd.DataFrame({"time": pd.to_datetime(["2026-08-31"]), "ens_spread": [2.5]})
    spatial = pd.DataFrame({"time": pd.to_datetime(["2026-08-31"]), "spatial_temp_mean": [31.4]})
    result = build_forecast_scaffold(history, consensus, ensemble, spatial)
    assert result.loc[0, "ens_spread"] == 2.5
    assert result.loc[0, "spatial_temp_mean"] == 31.4
