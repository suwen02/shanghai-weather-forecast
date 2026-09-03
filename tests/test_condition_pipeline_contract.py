from config.settings import DAILY_VARIABLES


def test_main_deterministic_collector_includes_condition_signals():
    required = {
        "weather_code",
        "cloud_cover_mean",
        "precipitation_hours",
        "precipitation_sum",
    }
    assert required.issubset(set(DAILY_VARIABLES))
