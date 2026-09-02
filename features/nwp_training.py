# -*- coding: utf-8 -*-
"""固定提前量 NWP 训练样本构造。"""

from __future__ import annotations

import pandas as pd


def build_lead_consensus_features(previous_runs: pd.DataFrame) -> pd.DataFrame:
    """按 valid date + lead day 构建与线上推理同名的多模型共识特征。"""
    if previous_runs is None or previous_runs.empty:
        return pd.DataFrame()
    required = {"time", "forecast_lead_days", "model"}
    missing = required - set(previous_runs.columns)
    if missing:
        raise ValueError(f"previous_runs missing columns: {sorted(missing)}")

    # 延迟导入，避免 nwp_training 与 FeatureEngineer 的模块依赖形成环。
    from features.engineer import FeatureEngineer

    frames = []
    source = previous_runs.copy()
    source["time"] = pd.to_datetime(source["time"])
    for lead, group in source.groupby("forecast_lead_days", sort=True):
        consensus = FeatureEngineer.build_model_consensus_features(group)
        if consensus.empty:
            continue
        consensus["forecast_lead_days"] = int(lead)
        frames.append(consensus)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["time", "forecast_lead_days"]).reset_index(drop=True)


def expand_observation_features_by_lead(
    observation_features: pd.DataFrame,
    lead_consensus: pd.DataFrame,
) -> pd.DataFrame:
    """把已计算好的因果观测特征按 1–7 天 lead 展开并合并 NWP。"""
    if observation_features is None or observation_features.empty:
        return pd.DataFrame()
    if lead_consensus is None or lead_consensus.empty:
        return pd.DataFrame()
    if "time" not in observation_features.columns or "time" not in lead_consensus.columns:
        raise ValueError("both frames must contain 'time'")

    obs = observation_features.copy()
    nwp = lead_consensus.copy()
    obs["time"] = pd.to_datetime(obs["time"])
    nwp["time"] = pd.to_datetime(nwp["time"])

    overlap = [
        col for col in nwp.columns
        if col != "time" and col in obs.columns
    ]
    if overlap:
        # NWP 共识列不应覆盖已存在的观测/目标列；重名通常意味着协议错误。
        raise ValueError(f"NWP feature columns overlap observation columns: {overlap}")

    result = obs.merge(nwp, on="time", how="inner", validate="one_to_many")
    return result.sort_values(["time", "forecast_lead_days"]).reset_index(drop=True)
