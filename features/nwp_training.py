# -*- coding: utf-8 -*-
"""历史 NWP 训练特征适配器。"""

from collections.abc import Callable

import pandas as pd


def merge_historical_nwp_features(
    observations: pd.DataFrame,
    historical_forecasts: pd.DataFrame,
    consensus_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """把按日期聚合的历史 NWP 共识特征左连接到观测训练行。"""
    out = observations.copy()
    if "time" not in out.columns:
        raise ValueError("observations must contain 'time'")

    out["time"] = pd.to_datetime(out["time"])
    if historical_forecasts is None or historical_forecasts.empty:
        return out

    consensus = consensus_builder(historical_forecasts.copy())
    if consensus is None or consensus.empty:
        return out
    if "time" not in consensus.columns:
        raise ValueError("consensus features must contain 'time'")

    consensus = consensus.copy()
    consensus["time"] = pd.to_datetime(consensus["time"])
    consensus = consensus.drop_duplicates(subset=["time"], keep="last")
    return out.merge(consensus, on="time", how="left")
