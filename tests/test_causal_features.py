import pandas as pd

from features.nwp_aware_engineer import NwpAwareFeatureEngineer


def test_nwp_aware_rolling_features_exclude_current_day_target():
    engineer = NwpAwareFeatureEngineer()
    engineer.rolling_windows = [2]
    df = pd.DataFrame({
        "time": pd.date_range("2026-08-01", periods=3, freq="D"),
        "temperature_2m_max": [10.0, 20.0, 100.0],
    })

    result = engineer.add_rolling_features(
        df,
        target_cols=["temperature_2m_max"],
    )

    assert result.loc[2, "temperature_2m_max_rmean2d"] == 15.0
    assert result.loc[2, "temperature_2m_max_rmax2d"] == 20.0


def test_nwp_aware_yoy_feature_is_lag_only_not_current_target_difference():
    engineer = NwpAwareFeatureEngineer()
    dates = pd.date_range("2025-01-01", periods=366, freq="D")
    df = pd.DataFrame({
        "time": dates,
        "temperature_2m_max": list(range(366)),
        "precipitation_sum": [0.0] * 366,
    })

    result = engineer.add_yoy_features(df)

    assert result.loc[365, "temperature_2m_max_yoy"] == 0
    assert "temperature_2m_max_yoy_diff" not in result.columns
    assert "precipitation_sum_yoy_diff" not in result.columns


def test_training_feature_contract_keeps_only_servable_observation_features(monkeypatch):
    engineer = NwpAwareFeatureEngineer()
    observations = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01"]),
        "temperature_2m_max": [33.0],
        "precipitation_sum": [2.0],
    })
    previous_runs = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01"]),
        "forecast_lead_days": [0],
        "model": ["m1"],
        "temperature_2m_max": [32.0],
        "temperature_2m_min": [25.0],
        "precipitation_sum": [1.0],
    })

    def fake_base(self, historical):
        built = historical.copy()
        built["doy_sin"] = 0.5
        built["temperature_2m_max_lag1d"] = 31.0
        built["temperature_2m_max_rmean3d"] = 30.0
        built["temperature_2m_max_yoy"] = 29.0
        built["sat_vapor_pressure"] = 50.0
        built["rh_seasonal_anomaly"] = 4.0
        built["precipitation_probability_max"] = 80.0
        cols = [
            "doy_sin",
            "temperature_2m_max_lag1d",
            "temperature_2m_max_rmean3d",
            "temperature_2m_max_yoy",
            "sat_vapor_pressure",
            "rh_seasonal_anomaly",
            "precipitation_probability_max",
        ]
        return built, cols, "temperature_2m_max", "precipitation_sum"

    monkeypatch.setattr(engineer._base_engineer_type, "build_training_features", fake_base)
    _, cols, _, _ = engineer.build_training_features(observations, previous_runs)

    assert "doy_sin" in cols
    assert "temperature_2m_max_lag1d" in cols
    assert "temperature_2m_max_rmean3d" in cols
    assert "tmax_max_model_mean" in cols
    assert "temperature_2m_max_yoy" not in cols
    assert "sat_vapor_pressure" not in cols
    assert "rh_seasonal_anomaly" not in cols
    assert "precipitation_probability_max" not in cols


def test_lead_state_features_are_aligned_to_forecast_origin(monkeypatch):
    engineer = NwpAwareFeatureEngineer()
    observations = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01", "2026-08-02", "2026-08-03"]),
        "temperature_2m_max": [10.0, 20.0, 30.0],
        "precipitation_sum": [0.0, 0.0, 0.0],
    })
    previous_runs = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-03", "2026-08-03"]),
        "forecast_lead_days": [0, 1],
        "model": ["m1", "m1"],
        "temperature_2m_max": [31.0, 29.0],
        "temperature_2m_min": [24.0, 23.0],
        "precipitation_sum": [1.0, 2.0],
    })

    def fake_base(self, historical):
        built = historical.copy()
        built["doy_sin"] = [0.1, 0.2, 0.3]
        built["temperature_2m_max_lag1d"] = [float("nan"), 10.0, 20.0]
        built["temperature_2m_max_rmean2d"] = [float("nan"), 10.0, 15.0]
        return (
            built,
            ["doy_sin", "temperature_2m_max_lag1d", "temperature_2m_max_rmean2d"],
            "temperature_2m_max",
            "precipitation_sum",
        )

    monkeypatch.setattr(engineer._base_engineer_type, "build_training_features", fake_base)
    result, _, _, _ = engineer.build_training_features(observations, previous_runs)
    rows = result.sort_values("forecast_lead_days").reset_index(drop=True)

    assert rows.loc[0, "temperature_2m_max_lag1d"] == 20.0
    assert rows.loc[1, "temperature_2m_max_lag1d"] == 10.0
    assert rows.loc[0, "temperature_2m_max_rmean2d"] == 15.0
    assert rows.loc[1, "temperature_2m_max_rmean2d"] == 10.0
