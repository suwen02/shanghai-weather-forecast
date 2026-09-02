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


def test_nwp_aware_engineer_merges_before_base_feature_build(monkeypatch):
    from features.nwp_aware_engineer import NwpAwareFeatureEngineer

    engineer = NwpAwareFeatureEngineer()
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
    captured = {}

    def fake_consensus(df):
        return pd.DataFrame({
            "time": df["time"],
            "tmax_max_model_mean": df["temperature_2m_max"],
        })

    def fake_base(self, merged):
        captured["merged"] = merged.copy()
        return merged, ["tmax_max_model_mean"], "temperature_2m_max", "precipitation_sum"

    monkeypatch.setattr(engineer, "build_model_consensus_features", fake_consensus)
    monkeypatch.setattr(engineer._base_engineer_type, "build_training_features", fake_base)
    _, cols, _, _ = engineer.build_training_features(observations, forecasts)

    assert cols == ["tmax_max_model_mean"]
    assert captured["merged"]["tmax_max_model_mean"].tolist() == [30.0, 34.0]


def test_nwp_training_awareness_requires_model_features():
    from features.nwp_aware_engineer import NwpAwareFeatureEngineer

    assert NwpAwareFeatureEngineer.has_nwp_training_features(["doy_sin", "tmax_max_model_mean"])
    assert not NwpAwareFeatureEngineer.has_nwp_training_features(["doy_sin", "temperature_2m_max_lag1d"])


def test_nwp_aware_prediction_recomputes_calendar_features(monkeypatch):
    from features.nwp_aware_engineer import NwpAwareFeatureEngineer

    engineer = NwpAwareFeatureEngineer()
    history = pd.DataFrame({
        "time": pd.to_datetime(["2026-09-01"]),
        "state": [5.0],
    })
    det = pd.DataFrame({
        "time": pd.to_datetime(["2026-09-02", "2026-09-03"]),
        "model": ["m1", "m1"],
        "x": [1, 2],
    })

    monkeypatch.setattr(engineer, "build_model_consensus_features", lambda _: pd.DataFrame({
        "time": pd.to_datetime(["2026-09-02", "2026-09-03"]),
        "tmax_max_model_mean": [31.0, 35.0],
    }))
    monkeypatch.setattr(engineer, "build_ensemble_features", lambda _: pd.DataFrame())
    monkeypatch.setattr(engineer, "build_station_spatial_features", lambda _: pd.DataFrame())
    monkeypatch.setattr(engineer, "add_physical_features", lambda df: df)
    monkeypatch.setattr(engineer, "add_shanghai_features", lambda df: df)
    monkeypatch.setattr(engineer, "add_yoy_features", lambda df: df)
    monkeypatch.setattr(engineer, "add_lag_features", lambda df: df)
    monkeypatch.setattr(engineer, "add_rolling_features", lambda df: df)
    monkeypatch.setattr(
        engineer,
        "add_temporal_features",
        lambda df: df.assign(calendar_day=pd.to_datetime(df["time"]).dt.day),
    )

    result = engineer.build_prediction_features(det, pd.DataFrame(), pd.DataFrame(), history)

    assert result["calendar_day"].tolist() == [2, 3]
    assert result["tmax_max_model_mean"].tolist() == [31.0, 35.0]
