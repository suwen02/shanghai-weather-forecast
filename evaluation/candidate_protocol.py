# -*- coding: utf-8 -*-
"""Pure helpers for temporal holdout and ML candidate promotion."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def temporal_date_holdout(frame: pd.DataFrame, *, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on unique valid dates so all lead rows for one date stay together."""
    if frame is None or frame.empty:
        raise ValueError("frame must not be empty")
    if "time" not in frame.columns:
        raise ValueError("frame must contain time")
    if int(holdout_days) <= 0:
        raise ValueError("holdout_days must be positive")

    work = frame.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.dropna(subset=["time"]).sort_values(["time", "forecast_lead_days"])
    unique_dates = sorted(work["time"].dt.normalize().unique())
    if len(unique_dates) <= int(holdout_days):
        raise ValueError("holdout_days must leave at least one training date")

    holdout_dates = set(unique_dates[-int(holdout_days):])
    mask = work["time"].dt.normalize().isin(holdout_dates)
    train = work.loc[~mask].reset_index(drop=True)
    holdout = work.loc[mask].reset_index(drop=True)
    return train, holdout


def fit_feature_medians(frame: pd.DataFrame, feature_cols: Sequence[str]) -> dict[str, float]:
    """Fit deterministic numeric medians on the training slice only."""
    medians: dict[str, float] = {}
    for col in feature_cols:
        if col not in frame.columns:
            medians[col] = 0.0
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        median = values.median()
        medians[col] = 0.0 if pd.isna(median) else float(median)
    return medians


def apply_feature_medians(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    medians: Mapping[str, float],
) -> pd.DataFrame:
    """Apply train-fitted medians without consulting holdout statistics."""
    out = frame.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = float(medians.get(col, 0.0))
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
            float(medians.get(col, 0.0))
        )
    return out


def build_previous_runs_baseline(
    previous_runs: pd.DataFrame,
    *,
    wet_threshold_mm: float = 1.0,
) -> pd.DataFrame:
    """Build same-lead raw NWP baselines from the archived model runs."""
    if previous_runs is None or previous_runs.empty:
        return pd.DataFrame()
    required = {
        "time",
        "forecast_lead_days",
        "model",
        "temperature_2m_max",
        "precipitation_sum",
    }
    missing = required - set(previous_runs.columns)
    if missing:
        raise ValueError(f"previous_runs missing columns: {sorted(missing)}")

    work = previous_runs.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work["temperature_2m_max"] = pd.to_numeric(work["temperature_2m_max"], errors="coerce")
    work["precipitation_sum"] = pd.to_numeric(work["precipitation_sum"], errors="coerce")
    work = work.dropna(subset=["time", "forecast_lead_days"])
    work["wet_event"] = (work["precipitation_sum"] >= float(wet_threshold_mm)).astype(float)

    baseline = (
        work.groupby(["time", "forecast_lead_days"], as_index=False)
        .agg(
            temperature_baseline=("temperature_2m_max", "mean"),
            p_wet_baseline=("wet_event", "mean"),
            model_count=("model", "nunique"),
        )
        .sort_values(["time", "forecast_lead_days"])
        .reset_index(drop=True)
    )
    baseline["forecast_lead_days"] = baseline["forecast_lead_days"].astype(int)
    return baseline


def ml_promotion_gate(
    candidate: Mapping,
    baseline: Mapping,
    *,
    max_temp_regression: float = 0.15,
    max_brier_regression: float = 0.02,
) -> dict:
    """Require aggregate improvement while preventing material per-lead regressions."""
    reasons: list[str] = []

    candidate_temp = float(candidate.get("temperature_mae", float("inf")))
    baseline_temp = float(baseline.get("temperature_mae", float("inf")))
    candidate_brier = float(candidate.get("wet_brier", float("inf")))
    baseline_brier = float(baseline.get("wet_brier", float("inf")))

    if not candidate_temp < baseline_temp:
        reasons.append(
            f"aggregate temperature MAE {candidate_temp:.4f} is not better than baseline {baseline_temp:.4f}"
        )
    if not candidate_brier < baseline_brier:
        reasons.append(
            f"aggregate wet Brier {candidate_brier:.4f} is not better than baseline {baseline_brier:.4f}"
        )

    candidate_by_lead = candidate.get("by_lead", {}) or {}
    baseline_by_lead = baseline.get("by_lead", {}) or {}
    leads = sorted(set(candidate_by_lead) | set(baseline_by_lead), key=int)
    for lead in leads:
        c = candidate_by_lead.get(lead, candidate_by_lead.get(str(lead)))
        b = baseline_by_lead.get(lead, baseline_by_lead.get(str(lead)))
        if c is None or b is None:
            reasons.append(f"lead {lead} missing candidate or baseline metrics")
            continue
        c_temp = float(c.get("temperature_mae", float("inf")))
        b_temp = float(b.get("temperature_mae", float("inf")))
        c_brier = float(c.get("wet_brier", float("inf")))
        b_brier = float(b.get("wet_brier", float("inf")))
        if c_temp > b_temp + float(max_temp_regression):
            reasons.append(
                f"lead {lead} temperature MAE regression {c_temp:.4f} > {b_temp:.4f} + {max_temp_regression:.4f}"
            )
        if c_brier > b_brier + float(max_brier_regression):
            reasons.append(
                f"lead {lead} wet Brier regression {c_brier:.4f} > {b_brier:.4f} + {max_brier_regression:.4f}"
            )

    return {
        "promote": not reasons,
        "reasons": reasons,
        "candidate": dict(candidate),
        "baseline": dict(baseline),
        "max_temp_regression": float(max_temp_regression),
        "max_brier_regression": float(max_brier_regression),
    }
