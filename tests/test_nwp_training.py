import pandas as pd

from features.nwp_training import (
    build_lead_consensus_features,
    expand_observation_features_by_lead,
)


def test_build_lead_consensus_features_preserves_each_horizon():
    previous_runs = pd.DataFrame({
        "time": pd.to_datetime([
            "2026-08-01", "2026-08-01", "2026-08-01", "2026-08-01",
        ]),
        "forecast_lead_days": [0, 0, 1, 1],
        "model": ["m1", "m2", "m1", "m2"],
        "temperature_2m_max": [31.0, 33.0, 29.0, 31.0],
        "temperature_2m_min": [25.0, 27.0, 23.0, 25.0],
        "precipitation_sum": [0.0, 2.0, 8.0, 12.0],
    })

    result = build_lead_consensus_features(previous_runs)

    assert result["forecast_lead_days"].tolist() == [0, 1]
    assert result["tmax_max_model_mean"].tolist() == [32.0, 30.0]
    assert result["precip_model_mean"].tolist() == [1.0, 10.0]


def test_expand_observation_features_by_lead_keeps_causal_state_and_target():
    observation_features = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01"]),
        "temperature_2m_max": [33.0],
        "precipitation_sum": [4.0],
        "temperature_2m_max_lag1d": [31.0],
        "doy_sin": [0.5],
    })
    lead_consensus = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01", "2026-08-01"]),
        "forecast_lead_days": [0, 1],
        "tmax_max_model_mean": [32.0, 30.0],
        "precip_model_mean": [1.0, 10.0],
    })

    result = expand_observation_features_by_lead(observation_features, lead_consensus)

    assert len(result) == 2
    assert result["forecast_lead_days"].tolist() == [0, 1]
    assert result["temperature_2m_max"].tolist() == [33.0, 33.0]
    assert result["temperature_2m_max_lag1d"].tolist() == [31.0, 31.0]
    assert result["tmax_max_model_mean"].tolist() == [32.0, 30.0]


def test_nwp_aware_engineer_builds_observation_lags_before_lead_expansion(monkeypatch):
    from features.nwp_aware_engineer import NwpAwareFeatureEngineer

    engineer = NwpAwareFeatureEngineer()
    observations = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01"]),
        "temperature_2m_max": [33.0],
        "precipitation_sum": [4.0],
    })
    previous_runs = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01", "2026-08-01"]),
        "forecast_lead_days": [0, 1],
        "model": ["m1", "m1"],
        "temperature_2m_max": [32.0, 30.0],
        "temperature_2m_min": [26.0, 24.0],
        "precipitation_sum": [1.0, 10.0],
    })

    def fake_base(self, historical):
        built = historical.copy()
        built["temperature_2m_max_lag1d"] = 31.0
        self.feature_cols = ["temperature_2m_max_lag1d"]
        return built, list(self.feature_cols), "temperature_2m_max", "precipitation_sum"

    monkeypatch.setattr(engineer._base_engineer_type, "build_training_features", fake_base)
    result, cols, _, _ = engineer.build_training_features(observations, previous_runs)

    assert len(result) == 2
    assert result["temperature_2m_max_lag1d"].tolist() == [31.0, 31.0]
    assert result["forecast_lead_days"].tolist() == [0, 1]
    assert "forecast_lead_days" in cols
    assert "tmax_max_model_mean" in cols


def test_nwp_training_awareness_requires_model_and_lead_features():
    from features.nwp_aware_engineer import NwpAwareFeatureEngineer

    assert NwpAwareFeatureEngineer.has_nwp_training_features([
        "forecast_lead_days", "tmax_max_model_mean"
    ])
    assert not NwpAwareFeatureEngineer.has_nwp_training_features([
        "tmax_max_model_mean"
    ])
    assert not NwpAwareFeatureEngineer.has_nwp_training_features([
        "forecast_lead_days", "temperature_2m_max_lag1d"
    ])


def test_nwp_aware_prediction_recomputes_calendar_and_lead_features(monkeypatch):
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
    assert result["forecast_lead_days"].tolist() == [0, 1]
    assert result["tmax_max_model_mean"].tolist() == [31.0, 35.0]
