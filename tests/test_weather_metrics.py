import math

import pandas as pd

from evaluation.weather_metrics import (
    brier_score,
    condition_accuracy,
    interval_coverage,
    leadwise_metrics,
    macro_f1,
    pinball_loss,
    promotion_gate,
)


def test_condition_accuracy_and_macro_f1_are_multiclass_metrics():
    truth = ["cloudy", "cloudy", "rain", "sunny"]
    pred = ["cloudy", "rain", "rain", "sunny"]

    assert condition_accuracy(truth, pred) == 0.75
    assert math.isclose(macro_f1(truth, pred), (2 / 3 + 2 / 3 + 1.0) / 3, rel_tol=1e-9)


def test_probability_and_interval_metrics_have_expected_meaning():
    assert math.isclose(brier_score([0.8, 0.2], [1, 0]), 0.04, rel_tol=1e-9)
    assert interval_coverage([9, 19, 29], [11, 21, 31], [10, 25, 30]) == 2 / 3
    assert math.isclose(pinball_loss([10, 12], [9, 14], 0.5), 0.75, rel_tol=1e-9)


def test_leadwise_metrics_scores_condition_wet_event_and_temperature():
    forecasts = pd.DataFrame([
        {"location": "shanghai", "valid_date": "2026-09-03", "lead_days": 0, "condition_kind": "cloudy", "p_wet": 0.8, "temperature_median": 30.0},
        {"location": "shanghai", "valid_date": "2026-09-04", "lead_days": 1, "condition_kind": "rain", "p_wet": 0.7, "temperature_median": 31.0},
        {"location": "shanghai", "valid_date": "2026-09-05", "lead_days": 1, "condition_kind": "cloudy", "p_wet": 0.2, "temperature_median": 29.0},
    ])
    truth = pd.DataFrame([
        {"location": "shanghai", "valid_date": "2026-09-03", "observed_condition_kind": "cloudy", "observed_precipitation_mm": 2.0, "observed_temperature_max": 31.0},
        {"location": "shanghai", "valid_date": "2026-09-04", "observed_condition_kind": "cloudy", "observed_precipitation_mm": 0.0, "observed_temperature_max": 30.0},
        {"location": "shanghai", "valid_date": "2026-09-05", "observed_condition_kind": "cloudy", "observed_precipitation_mm": 0.5, "observed_temperature_max": 28.0},
    ])

    report = leadwise_metrics(forecasts, truth).set_index("lead_days")

    assert report.loc[0, "n"] == 1
    assert report.loc[0, "condition_accuracy"] == 1.0
    assert math.isclose(report.loc[0, "wet_brier"], 0.04, rel_tol=1e-9)
    assert report.loc[0, "temperature_mae"] == 1.0

    assert report.loc[1, "n"] == 2
    assert report.loc[1, "condition_accuracy"] == 0.5
    assert math.isclose(report.loc[1, "wet_brier"], (0.7**2 + 0.2**2) / 2, rel_tol=1e-9)
    assert report.loc[1, "temperature_mae"] == 1.0


def test_promotion_gate_requires_candidate_to_beat_every_baseline_on_core_metrics():
    candidate = {
        "condition_accuracy": 0.78,
        "condition_macro_f1": 0.74,
        "wet_brier": 0.13,
        "temperature_mae": 1.15,
    }
    baselines = {
        "consensus": {"condition_accuracy": 0.75, "condition_macro_f1": 0.70, "wet_brier": 0.15, "temperature_mae": 1.2},
        "best_match": {"condition_accuracy": 0.76, "condition_macro_f1": 0.72, "wet_brier": 0.14, "temperature_mae": 1.18},
    }

    decision = promotion_gate(candidate, baselines)
    assert decision["promote"] is True
    assert decision["failures"] == []

    candidate["wet_brier"] = 0.16
    decision = promotion_gate(candidate, baselines)
    assert decision["promote"] is False
    assert any("wet_brier" in failure for failure in decision["failures"])
