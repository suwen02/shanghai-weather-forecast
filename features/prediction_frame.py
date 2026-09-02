# -*- coding: utf-8 -*-
"""把历史状态与未来 NWP 数据组合成真正的未来预测行。"""

from __future__ import annotations

import pandas as pd


def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return df.copy()
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    return out.sort_values("time").drop_duplicates(subset=["time"], keep="last")


def build_forecast_scaffold(
    history_features: pd.DataFrame,
    consensus: pd.DataFrame,
    ensemble: pd.DataFrame,
    spatial: pd.DataFrame,
) -> pd.DataFrame:
    """以未来 NWP 日期为主表，并携带最近历史状态特征。"""
    history = _normalize_time(history_features)
    consensus = _normalize_time(consensus)
    ensemble = _normalize_time(ensemble)
    spatial = _normalize_time(spatial)

    if consensus.empty or "time" not in consensus.columns:
        return pd.DataFrame()

    last_history_time = history["time"].max() if not history.empty else None
    future = consensus.copy()
    if last_history_time is not None:
        future = future[future["time"] > last_history_time].copy()
    if future.empty:
        return pd.DataFrame()

    for extra in (ensemble, spatial):
        if not extra.empty and "time" in extra.columns:
            overlap = [c for c in extra.columns if c != "time" and c in future.columns]
            future = future.merge(extra, on="time", how="left", suffixes=("", "__new"))
            for col in overlap:
                new_col = f"{col}__new"
                if new_col in future.columns:
                    future[col] = future[new_col].combine_first(future[col])
                    future = future.drop(columns=[new_col])

    if not history.empty:
        latest = history.iloc[-1]
        state_cols = [c for c in history.columns if c != "time"]
        missing_state = {
            col: [latest[col]] * len(future)
            for col in state_cols
            if col not in future.columns
        }
        if missing_state:
            future = pd.concat(
                [future.reset_index(drop=True), pd.DataFrame(missing_state)],
                axis=1,
            )

    return future.sort_values("time").reset_index(drop=True)
