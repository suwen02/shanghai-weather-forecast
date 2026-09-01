import json
from pathlib import Path

from src.realtime import RefreshStateStore, atomic_publish_json, snapshot_fingerprint


def test_snapshot_fingerprint_is_stable_and_changes_with_weather_data():
    a = {"hourly": {"time": ["2026-08-31T16:00"], "temperature_2m": [31.2]}}
    b = {"hourly": {"temperature_2m": [31.2], "time": ["2026-08-31T16:00"]}}
    c = {"hourly": {"time": ["2026-08-31T16:00"], "temperature_2m": [31.3]}}
    assert snapshot_fingerprint(a) == snapshot_fingerprint(b)
    assert snapshot_fingerprint(a) != snapshot_fingerprint(c)


def test_refresh_state_store_only_accepts_new_snapshot(tmp_path: Path):
    store = RefreshStateStore(tmp_path / "refresh_state.json")
    assert store.should_refresh("abc") is True
    store.mark_refreshed("abc", "2026-08-31T16:00:00+08:00")
    assert store.should_refresh("abc") is False
    assert store.should_refresh("def") is True


def test_atomic_publish_json_writes_versioned_and_latest_files(tmp_path: Path):
    payload = {"city": "上海", "data_as_of": "2026-08-31T16:00:00+08:00"}
    versioned = tmp_path / "predictions_20260831_160000.json"
    latest = tmp_path / "latest.json"
    atomic_publish_json(payload, versioned, latest)
    assert json.loads(versioned.read_text(encoding="utf-8")) == payload
    assert json.loads(latest.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_snapshot_fingerprint_ignores_transient_response_metadata():
    a = {
        "generationtime_ms": 1.2,
        "utc_offset_seconds": 28800,
        "hourly": {"time": ["2026-08-31T16:00"], "temperature_2m": [31.2]},
    }
    b = {
        "generationtime_ms": 9.9,
        "utc_offset_seconds": 28800,
        "hourly": {"time": ["2026-08-31T16:00"], "temperature_2m": [31.2]},
    }
    assert snapshot_fingerprint(a) == snapshot_fingerprint(b)

from datetime import datetime
from zoneinfo import ZoneInfo
from src.realtime import build_short_term_forecast


def test_build_short_term_forecast_returns_future_hours_and_freshness():
    snapshot = {
        "hourly": {
            "time": [
                "2026-08-31T14:00", "2026-08-31T15:00",
                "2026-08-31T16:00", "2026-08-31T17:00",
            ],
            "temperature_2m": [30.0, 30.5, 31.0, 31.2],
            "precipitation_probability": [10, 20, 30, 40],
            "precipitation": [0.0, 0.0, 0.1, 0.5],
            "wind_speed_10m": [7.0, 8.0, 9.0, 10.0],
        }
    }
    now = datetime(2026, 8, 31, 15, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    result = build_short_term_forecast(snapshot, now=now, timezone="Asia/Shanghai", horizon_hours=48)
    assert result["data_as_of"] == "2026-08-31T15:00:00+08:00"
    assert result["data_age_minutes"] == 30
    assert [x["time"] for x in result["hours"]] == ["2026-08-31T16:00:00+08:00", "2026-08-31T17:00:00+08:00"]
    assert result["hours"][0]["temperature_2m"] == 31.0

from src.realtime import fetch_latest_snapshot


class _FakeCollector:
    def __init__(self):
        self.calls = []

    def _get(self, url, params):
        self.calls.append((url, params))
        return {"hourly": {"time": []}}


def test_fetch_latest_snapshot_requests_past_and_future_hours(monkeypatch):
    import sys
    import types
    settings = types.ModuleType("config.settings")
    settings.API_ENDPOINTS = {"deterministic": "https://example.test/forecast"}
    settings.SHANGHAI_LAT = 31.2304
    settings.SHANGHAI_LON = 121.4737
    settings.TIMEZONE = "Asia/Shanghai"
    config = types.ModuleType("config")
    config.settings = settings
    monkeypatch.setitem(sys.modules, "config", config)
    monkeypatch.setitem(sys.modules, "config.settings", settings)

    collector = _FakeCollector()
    fetch_latest_snapshot(collector, past_hours=6, forecast_hours=48)
    _, params = collector.calls[0]
    assert params["past_hours"] == 6
    # Open-Meteo 的 forecast_hours 包含当前整点，因此多取1小时，确保发布48个未来小时。
    assert params["forecast_hours"] == 49
    assert params["models"] == "best_match"
    assert "temperature_2m" in params["hourly"]
    assert "precipitation_probability" in params["hourly"]
