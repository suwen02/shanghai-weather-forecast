import pandas as pd

from evaluation.calibration_readiness import live_calibration_readiness


def _paired_frames(days_per_lead=20, include_leads=range(7), all_wet=False):
    forecasts = []
    observations = []
    base = pd.Timestamp("2026-01-01")
    for lead in include_leads:
        for offset in range(days_per_lead):
            valid = (base + pd.Timedelta(days=lead * 40 + offset)).date().isoformat()
            wet = True if all_wet else (offset % 2 == 0)
            forecasts.append({
                "location": "shanghai",
                "valid_date": valid,
                "lead_days": lead,
                "p_wet": 0.75 if wet else 0.2,
            })
            observations.append({
                "location": "shanghai",
                "valid_date": valid,
                "observed_precipitation_mm": 2.0 if wet else 0.0,
            })
    return pd.DataFrame(forecasts), pd.DataFrame(observations)


def test_live_calibration_readiness_requires_balanced_samples_for_every_lead():
    forecasts, observations = _paired_frames(days_per_lead=20)

    result = live_calibration_readiness(
        forecasts,
        observations,
        min_per_lead=20,
        min_wet_events=20,
        min_dry_events=20,
    )

    assert result["ready"] is True
    assert result["paired_samples"] == 140
    assert result["lead_counts"] == {0: 20, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20}
    assert result["wet_events"] == 70
    assert result["dry_events"] == 70
    assert result["failures"] == []


def test_live_calibration_readiness_rejects_missing_lead_and_one_class_history():
    forecasts, observations = _paired_frames(days_per_lead=20, include_leads=range(6), all_wet=True)

    result = live_calibration_readiness(
        forecasts,
        observations,
        min_per_lead=20,
        min_wet_events=20,
        min_dry_events=20,
    )

    assert result["ready"] is False
    assert any("lead 6" in failure for failure in result["failures"])
    assert any("dry events" in failure for failure in result["failures"])
