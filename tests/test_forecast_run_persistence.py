from datetime import datetime, timezone

from persistence.forecast_runs import payload_to_forecast_run_rows


def test_payload_to_forecast_run_rows_joins_condition_temp_and_precip_by_date():
    payload = {
        "source": "nwp_consensus_fallback",
        "conditions": [
            {"date": "2026-09-03", "lead_days": 0, "kind": "cloudy", "secondary": "showers", "weather_code": 3, "model_agreement": 0.7, "cloud_cover_mean": 88.0},
            {"date": "2026-09-04", "lead_days": 1, "kind": "partly-cloudy", "secondary": None, "weather_code": 2, "model_agreement": 0.6, "cloud_cover_mean": 67.0},
        ],
        "temperature": [
            {"date": "2026-09-03", "lead_days": 0, "median": 30.1},
            {"date": "2026-09-04", "lead_days": 1, "median": 31.2},
        ],
        "precipitation": [
            {"date": "2026-09-03", "lead_days": 0, "expected_mm": 1.4, "p_trace": 0.8, "p_wet": 0.4, "p_heavy": 0.1},
            {"date": "2026-09-04", "lead_days": 1, "expected_mm": 0.2, "p_trace": 0.3, "p_wet": 0.1, "p_heavy": 0.0},
        ],
    }
    issued_at = datetime(2026, 9, 3, 6, 0, tzinfo=timezone.utc)

    rows = payload_to_forecast_run_rows(
        payload,
        location="shanghai",
        run_key="fingerprint-abc",
        issued_at=issued_at,
    )

    assert len(rows) == 2
    assert rows[0]["location"] == "shanghai"
    assert rows[0]["run_key"] == "fingerprint-abc"
    assert rows[0]["valid_date"] == "2026-09-03"
    assert rows[0]["lead_days"] == 0
    assert rows[0]["condition_kind"] == "cloudy"
    assert rows[0]["condition_secondary"] == "showers"
    assert rows[0]["p_wet"] == 0.4
    assert rows[0]["temperature_median"] == 30.1
    assert rows[0]["precipitation_expected_mm"] == 1.4
    assert rows[0]["issued_at"] == "2026-09-03T06:00:00+00:00"


def test_payload_to_forecast_run_rows_is_deterministic_and_uses_union_of_dates():
    payload = {
        "source": "nwp_consensus_fallback",
        "conditions": [{"date": "2026-09-03", "lead_days": 0, "kind": "cloudy"}],
        "temperature": [{"date": "2026-09-04", "lead_days": 1, "median": 31.0}],
        "precipitation": [],
    }

    first = payload_to_forecast_run_rows(payload, "shanghai", "run-1", "2026-09-03T06:00:00Z")
    second = payload_to_forecast_run_rows(payload, "shanghai", "run-1", "2026-09-03T06:00:00Z")

    assert first == second
    assert [row["valid_date"] for row in first] == ["2026-09-03", "2026-09-04"]
    assert [row["lead_days"] for row in first] == [0, 1]
