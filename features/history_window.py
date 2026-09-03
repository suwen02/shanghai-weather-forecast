# -*- coding: utf-8 -*-
"""History-lookback sizing for causal online features."""

from __future__ import annotations

from typing import Iterable


def _max_or_zero(values: Iterable[int]) -> int:
    normalized = [max(0, int(value)) for value in values]
    return max(normalized) if normalized else 0


def required_history_days(
    lag_days: Iterable[int],
    rolling_windows: Iterable[int],
    yoy_days: int = 365,
    safety_margin: int = 35,
) -> int:
    """Return enough observations for YoY, lag and rolling features.

    Lag/rolling features only need a small buffer beyond their largest window,
    while the retained annual lag needs a wider margin for missing upstream
    dates and timezone/date-boundary effects. With the current feature config
    this intentionally resolves to 400 days.
    """
    yoy_requirement = max(0, int(yoy_days)) + max(0, int(safety_margin))
    lag_requirement = _max_or_zero(lag_days) + 5
    rolling_requirement = _max_or_zero(rolling_windows) + 5
    return max(yoy_requirement, lag_requirement, rolling_requirement)
