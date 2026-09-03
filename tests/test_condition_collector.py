from datetime import date

from collectors.condition_forecasts import collect_condition_forecasts


class FakeCollector:
    def __init__(self):
        self.calls = []

    def _get(self, url, params):
        self.calls.append((url, params))
        return {
            "daily": {
                "time": ["2026-09-03", "2026-09-04"],
                "weather_code": [3, 61],
                "cloud_cover_mean": [88.0, 91.0],
                "precipitation_hours": [1.0, 8.0],
                "precipitation_sum": [0.2, 7.0],
            }
        }


def test_condition_collector_requests_daily_condition_signals_per_model():
    fake = FakeCollector()

    result = collect_condition_forecasts(
        collector=fake,
        target_date=date(2026, 9, 3),
        models=["ecmwf_ifs025", "cma_grapes_global"],
    )

    assert len(fake.calls) == 2
    assert set(result["model"]) == {"ecmwf_ifs025", "cma_grapes_global"}
    assert set(["weather_code", "cloud_cover_mean", "precipitation_hours", "precipitation_sum"]).issubset(result.columns)
    for _, params in fake.calls:
        requested = set(params["daily"].split(","))
        assert {"weather_code", "cloud_cover_mean", "precipitation_hours", "precipitation_sum"}.issubset(requested)
        assert params["forecast_days"] == 7
        assert params["timezone"] == "Asia/Shanghai"
