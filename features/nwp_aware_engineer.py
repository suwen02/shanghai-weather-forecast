# -*- coding: utf-8 -*-
"""在现有 FeatureEngineer 上叠加历史 NWP 共识训练特征。"""

from typing import Optional

import pandas as pd

from features.engineer import FeatureEngineer
from features.nwp_training import merge_historical_nwp_features
from features.prediction_frame import build_forecast_scaffold


class NwpAwareFeatureEngineer(FeatureEngineer):
    """保证训练和在线推理使用同名的 NWP 共识特征。"""

    _base_engineer_type = FeatureEngineer

    def build_training_features(
        self,
        historical_daily: pd.DataFrame,
        historical_forecasts: Optional[pd.DataFrame] = None,
    ):
        if historical_forecasts is None:
            historical_forecasts = pd.DataFrame()

        merged = merge_historical_nwp_features(
            historical_daily,
            historical_forecasts,
            self.build_model_consensus_features,
        )
        return self._base_engineer_type.build_training_features(self, merged)

    def build_prediction_features(
        self,
        det_df: pd.DataFrame,
        ens_df: pd.DataFrame,
        station_df: pd.DataFrame,
        recent_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """以未来 NWP 日期为预测主表，并重新计算未来目标日的日历特征。"""
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
        return any("_model_" in name for name in feature_cols)
