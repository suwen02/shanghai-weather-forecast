import pandas as pd

from evaluation.candidate_protocol import (
    build_previous_runs_baseline,
    ml_promotion_gate,
    temporal_date_holdout,
)


def test_temporal_holdout_keeps_all_leads_of_a_date_together():
    rows = []
    for day in pd.date_range("2026-01-01", periods=6, freq="D"):
        for lead in range(3):
            rows.append({"time": day, "forecast_lead_days": lead, "x": 1.0})
    frame = pd.DataFrame(rows)

    train, holdout = temporal_date_holdout(frame, holdout_days=2)

    assert set(pd.to_datetime(train["time"]).dt.date).isdisjoint(
        set(pd.to_datetime(holdout["time"]).dt.date)
    )
    assert sorted(pd.to_datetime(holdout["time"]).dt.date.unique()) == [
        pd.Timestamp("2026-01-05").date(),
        pd.Timestamp("2026-01-06").date(),
    ]
    assert holdout.groupby("time")["forecast_lead_days"].nunique().tolist() == [3, 3]


def test_previous_runs_baseline_uses_model_mean_temperature_and_wet_frequency():
    previous_runs = pd.DataFrame(
        [
            {"time": "2026-01-02", "forecast_lead_days": 1, "model": "a", "temperature_2m_max": 10.0, "precipitation_sum": 0.0},
            {"time": "2026-01-02", "forecast_lead_days": 1, "model": "b", "temperature_2m_max": 12.0, "precipitation_sum": 1.0},
            {"time": "2026-01-02", "forecast_lead_days": 1, "model": "c", "temperature_2m_max": 14.0, "precipitation_sum": 5.0},
            {"time": "2026-01-03", "forecast_lead_days": 2, "model": "a", "temperature_2m_max": 20.0, "precipitation_sum": 0.2},
            {"time": "2026-01-03", "forecast_lead_days": 2, "model": "b", "temperature_2m_max": 22.0, "precipitation_sum": 0.8},
        ]
    )

    baseline = build_previous_runs_baseline(previous_runs, wet_threshold_mm=1.0)

    first = baseline.iloc[0]
    assert first["temperature_baseline"] == 12.0
    assert first["p_wet_baseline"] == 2 / 3
    second = baseline.iloc[1]
    assert second["temperature_baseline"] == 21.0
    assert second["p_wet_baseline"] == 0.0


def test_ml_promotion_gate_requires_both_metrics_and_guards_each_lead():
    candidate = {
        "temperature_mae": 1.0,
        "wet_brier": 0.15,
        "by_lead": {
            0: {"temperature_mae": 0.8, "wet_brier": 0.10},
            1: {"temperature_mae": 1.2, "wet_brier": 0.20},
        },
    }
    baseline = {
        "temperature_mae": 1.2,
        "wet_brier": 0.18,
        "by_lead": {
            0: {"temperature_mae": 0.9, "wet_brier": 0.11},
            1: {"temperature_mae": 1.1, "wet_brier": 0.19},
        },
    }

    passed = ml_promotion_gate(candidate, baseline, max_temp_regression=0.15, max_brier_regression=0.02)
    assert passed["promote"] is True

    candidate["by_lead"][1]["temperature_mae"] = 1.4
    failed = ml_promotion_gate(candidate, baseline, max_temp_regression=0.15, max_brier_regression=0.02)
    assert failed["promote"] is False
    assert any("lead 1 temperature" in reason for reason in failed["reasons"])
