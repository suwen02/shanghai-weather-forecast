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
