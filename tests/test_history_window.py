from features.history_window import required_history_days


def test_required_history_days_covers_yoy_lags_with_margin():
    days = required_history_days(
        lag_days=[1, 2, 7, 28],
        rolling_windows=[3, 7, 30, 90],
        yoy_days=365,
        safety_margin=35,
    )
    assert days == 400


def test_required_history_days_respects_larger_explicit_windows():
    assert required_history_days([1], [450], yoy_days=365, safety_margin=35) == 455
    assert required_history_days([500], [90], yoy_days=365, safety_margin=35) == 505
