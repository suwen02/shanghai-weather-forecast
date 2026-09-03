# -*- coding: utf-8 -*-
"""Pure forecast-evaluation metrics and model-promotion gates."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _paired(a: Iterable, b: Iterable):
    pairs = []
    for left, right in zip(a, b):
        if pd.isna(left) or pd.isna(right):
            continue
        pairs.append((left, right))
    return pairs


def condition_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    pairs = _paired(y_true, y_pred)
    if not pairs:
        return float("nan")
    return float(sum(actual == predicted for actual, predicted in pairs) / len(pairs))


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    pairs = _paired(y_true, y_pred)
    if not pairs:
        return float("nan")
    labels = sorted({value for pair in pairs for value in pair})
    scores = []
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in pairs)
        fp = sum(actual != label and predicted == label for actual, predicted in pairs)
        fn = sum(actual == label and predicted != label for actual, predicted in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(score)
    return float(np.mean(scores))


def brier_score(probabilities: Sequence, outcomes: Sequence) -> float:
    pairs = _paired(probabilities, outcomes)
    if not pairs:
        return float("nan")
    values = [(float(prob) - float(outcome)) ** 2 for prob, outcome in pairs]
    return float(np.mean(values))


def interval_coverage(lower: Sequence, upper: Sequence, truth: Sequence) -> float:
    triples = []
    for low, high, actual in zip(lower, upper, truth):
        if pd.isna(low) or pd.isna(high) or pd.isna(actual):
            continue
        triples.append((float(low), float(high), float(actual)))
    if not triples:
        return float("nan")
    return float(np.mean([low <= actual <= high for low, high, actual in triples]))


def pinball_loss(y_true: Sequence, y_pred: Sequence, quantile: float) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    pairs = _paired(y_true, y_pred)
    if not pairs:
        return float("nan")
    losses = []
    for actual, predicted in pairs:
        error = float(actual) - float(predicted)
        losses.append(max(quantile * error, (quantile - 1.0) * error))
    return float(np.mean(losses))


def _mae(predicted: Sequence, actual: Sequence) -> float:
    pairs = _paired(predicted, actual)
    if not pairs:
        return float("nan")
    return float(np.mean([abs(float(pred) - float(obs)) for pred, obs in pairs]))


def leadwise_metrics(forecasts: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Score issued forecasts against later observations, grouped by lead day."""
    if forecasts is None or observations is None or forecasts.empty or observations.empty:
        return pd.DataFrame(columns=[
            "lead_days", "n", "condition_accuracy", "condition_macro_f1",
            "wet_brier", "temperature_mae",
        ])

    left = forecasts.copy()
    right = observations.copy()
    for frame in (left, right):
        frame["valid_date"] = pd.to_datetime(frame["valid_date"], errors="coerce").dt.date

    merged = left.merge(right, on=["location", "valid_date"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=[
            "lead_days", "n", "condition_accuracy", "condition_macro_f1",
            "wet_brier", "temperature_mae",
        ])

    rows = []
    for lead_days, group in merged.groupby("lead_days", sort=True):
        wet_truth = (
            pd.to_numeric(group.get("observed_precipitation_mm"), errors="coerce") >= 1.0
        ).astype(float)
        rows.append({
            "lead_days": int(lead_days),
            "n": int(len(group)),
            "condition_accuracy": condition_accuracy(
                group.get("observed_condition_kind", pd.Series(dtype=object)),
                group.get("condition_kind", pd.Series(dtype=object)),
            ),
            "condition_macro_f1": macro_f1(
                group.get("observed_condition_kind", pd.Series(dtype=object)),
                group.get("condition_kind", pd.Series(dtype=object)),
            ),
            "wet_brier": brier_score(
                pd.to_numeric(group.get("p_wet"), errors="coerce"),
                wet_truth,
            ),
            "temperature_mae": _mae(
                pd.to_numeric(group.get("temperature_median"), errors="coerce"),
                pd.to_numeric(group.get("observed_temperature_max"), errors="coerce"),
            ),
        })
    return pd.DataFrame(rows)


def promotion_gate(
    candidate: Mapping[str, float],
    baselines: Mapping[str, Mapping[str, float]],
) -> dict:
    """Require the candidate to beat every supplied baseline on core metrics."""
    if not baselines:
        return {"promote": False, "failures": ["no baselines supplied"]}

    failures = []
    higher_is_better = ("condition_accuracy", "condition_macro_f1")
    lower_is_better = ("wet_brier", "temperature_mae")

    for metric in higher_is_better:
        value = candidate.get(metric)
        refs = [metrics.get(metric) for metrics in baselines.values()]
        refs = [float(item) for item in refs if item is not None and np.isfinite(item)]
        if value is None or not np.isfinite(value):
            failures.append(f"{metric}: candidate missing")
        elif refs and not float(value) > max(refs):
            failures.append(f"{metric}: {float(value):.6g} must exceed best baseline {max(refs):.6g}")

    for metric in lower_is_better:
        value = candidate.get(metric)
        refs = [metrics.get(metric) for metrics in baselines.values()]
        refs = [float(item) for item in refs if item is not None and np.isfinite(item)]
        if value is None or not np.isfinite(value):
            failures.append(f"{metric}: candidate missing")
        elif refs and not float(value) < min(refs):
            failures.append(f"{metric}: {float(value):.6g} must beat best baseline {min(refs):.6g}")

    return {"promote": not failures, "failures": failures}
