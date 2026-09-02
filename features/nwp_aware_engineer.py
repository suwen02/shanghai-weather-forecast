# -*- coding: utf-8 -*-
"""在现有 FeatureEngineer 上叠加固定提前量 NWP 训练特征。"""

from typing import Optional

import numpy as np
import pandas as pd

from features.engineer import FeatureEngineer
from features.nwp_training import (
    build_lead_consensus_features,
    expand_observation_features_by_lead,
)
from features.prediction_frame import build_forecast_scaffold


class NwpAwareFeatureEngineer(FeatureEngineer):
    """保证训练和在线推理使用同名、同提前量语义的 NWP 共识特征。"""

    _base_engineer_type = FeatureEngineer

    def build_training_features(
        self,
        historical_daily: pd.DataFrame,
        previous_runs: Optional[pd.DataFrame] = None,
    ):
        # 关键顺序：先在唯一的观测日期序列上计算 lag/rolling，
        # 再把每个 valid date 展开成 lead1..lead7，避免 shift 落到重复 lead 行上。
        observation_features, base_cols, temp_target, precip_target = (
            self._base_engineer_type.build_training_features(self, historical_daily)
        )
        if previous_runs is None or previous_runs.empty:
            self.feature_cols = list(base_cols)
            return observation_features, list(base_cols), temp_target, precip_target

        lead_consensus = build_lead_consensus_features(previous_runs)
        expanded = expand_observation_features_by_lead(
            observation_features,
            lead_consensus,
        )
        if expanded.empty:
            self.feature_cols = list(base_cols)
            return expanded, list(base_cols), temp_target, precip_target

        nwp_cols = [
            col for col in lead_consensus.columns
            if col != "time"
            and col not in {temp_target, precip_target}
            and lead_consensus[col].dtype in [np.float64, np.int64, float, int]
        ]
        feature_cols = list(dict.fromkeys([*base_cols, *nwp_cols]))
        feature_cols = [c for c in feature_cols if c in expanded.columns and expanded[c].notna().any()]
        self.feature_cols = feature_cols
        return expanded, feature_cols, temp_target, precip_target

    def build_prediction_features(
        self,
        det_df: pd.DataFrame,
        ens_df: pd.DataFrame,
        station_df: pd.DataFrame,
        recent_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """以未来 NWP 日期为预测主表，并重新计算目标日时间特征。"""
        base = recent_history.copy()
        if "time" not in base.columns:
            return pd.DataFrame()

        base["time"] = pd.to_datetime(base["time"])
        base = base.sort_values("time").reset_index(drop=True)
        base = self.add_temporal_features(base)
        base = self.add_physical_features(base)
        base = self.add_shanghai_features(base)
        base = self.add_yoy_features(base)
        base = self.add_lag_features(base)
        base = self.add_rolling_features(base)

        consensus = self.build_model_consensus_features(det_df)
        ensemble = self.build_ensemble_features(ens_df)
        spatial = self.build_station_spatial_features(station_df)

        future = build_forecast_scaffold(base, consensus, ensemble, spatial)
        if future.empty:
            return future

        # 时间/季节编码描述的是目标日期，不能复制最近观测日的编码。
        future = self.add_temporal_features(future)
        future = self.add_shanghai_features(future)

        if self.feature_cols:
            missing = {
                col: [0.0] * len(future)
                for col in self.feature_cols
                if col not in future.columns
            }
            if missing:
                future = pd.concat(
                    [future.reset_index(drop=True), pd.DataFrame(missing)],
                    axis=1,
                )
        return future

    @staticmethod
    def has_nwp_training_features(feature_cols) -> bool:
        names = set(feature_cols)
        return "forecast_lead_days" in names and any("_model_" in name for name in names)
