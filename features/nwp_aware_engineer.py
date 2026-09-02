# -*- coding: utf-8 -*-
"""在现有 FeatureEngineer 上叠加历史 NWP 共识训练特征。"""

from typing import Optional

import pandas as pd

from features.engineer import FeatureEngineer
from features.nwp_training import merge_historical_nwp_features


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

    @staticmethod
    def has_nwp_training_features(feature_cols) -> bool:
        return any("_model_" in name for name in feature_cols)
