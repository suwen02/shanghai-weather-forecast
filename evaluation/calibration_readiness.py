# -*- coding: utf-8 -*-
"""Readiness checks for live probability calibration."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def live_calibration_readiness(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    expected_leads: Iterable[int] = range(7),
    min_per_lead: int = 20,
    min_wet_events: int = 20,
    min_dry_events: int = 20,
    wet_threshold_mm: float = 1.0,
) -> dict:
    """Assess whether verified live history is sufficient for wet-probability calibration."""
    failures = []
    expected = [int(lead) for lead in expected_leads]

    if forecasts is None or observations is None or forecasts.empty or observations.empty:
        failures.append("no paired forecast/verification history")
        return {
            "ready": False,
            "paired_samples": 0,
            "lead_counts": {},
            "wet_events": 0,
            "dry_events": 0,
            "failures": failures,
        }

    left = forecasts.copy()
    right = observations.copy()
    for frame in (left, right):
        frame["valid_date"] = pd.to_datetime(frame["valid_date"], errors="coerce").dt.date

    merged = left.merge(right, on=["location", "valid_date"], how="inner")
    merged["lead_days"] = pd.to_numeric(merged.get("lead_days"), errors="coerce")
    merged["p_wet"] = pd.to_numeric(merged.get("p_wet"), errors="coerce")
    merged["observed_precipitation_mm"] = pd.to_numeric(
        merged.get("observed_precipitation_mm"), errors="coerce"
    )
    merged = merged.dropna(subset=["lead_days", "p_wet", "observed_precipitation_mm"])
    merged["lead_days"] = merged["lead_days"].astype(int)

    lead_counts = {
        int(lead): int(count)
        for lead, count in merged.groupby("lead_days").size().sort_index().items()
    }
    for lead in expected:
        count = lead_counts.get(lead, 0)
        if count < min_per_lead:
            failures.append(f"lead {lead}: {count} samples < {min_per_lead}")

    wet = merged["observed_precipitation_mm"] >= float(wet_threshold_mm)
    wet_events = int(wet.sum())
    dry_events = int((~wet).sum())
    if wet_events < min_wet_events:
        failures.append(f"wet events: {wet_events} < {min_wet_events}")
    if dry_events < min_dry_events:
        failures.append(f"dry events: {dry_events} < {min_dry_events}")

    return {
        "ready": not failures,
        "paired_samples": int(len(merged)),
        "lead_counts": lead_counts,
        "wet_events": wet_events,
        "dry_events": dry_events,
        "failures": failures,
    }
