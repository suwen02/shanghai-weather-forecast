# -*- coding: utf-8 -*-
"""在现有 FeatureEngineer 上叠加固定提前量、可上线的 NWP 训练特征。"""

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
    """保证训练和在线推理使用同名、同提前量、同可见性语义的特征。"""

    _base_engineer_type = FeatureEngineer

    _TARGET_DATE_FEATURES = {
        "doy_sin",
        "doy_cos",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
        "season",
        "is_weekend",
        "is_typhoon_season",
        "is_meiyu_season",
        "monsoon_indicator",
    }
    _ROLLING_MARKERS = ("_rmean", "_rstd", "_rmin", "_rmax")

    @classmethod
    def _is_origin_state_feature(cls, name: str) -> bool:
        """只允许在预报签发时已经可见的历史状态特征。"""
        if "_lag" in name and name.endswith("d"):
            return True
        return name.endswith("d") and any(marker in name for marker in cls._ROLLING_MARKERS)

    @classmethod
    def _is_causal_training_feature(cls, name: str) -> bool:
        """目标日期日历 + 签发时历史状态；拒绝当天观测派生特征。"""
        return name in cls._TARGET_DATE_FEATURES or cls._is_origin_state_feature(name)

    @staticmethod
    def add_yoy_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
        """保留纯 365 天滞后参考，不构造包含当天目标的同比差值。"""
        out = df.copy().sort_values(time_col).reset_index(drop=True)
        for col in ["temperature_2m_max", "precipitation_sum"]:
            if col in out.columns:
                out[f"{col}_yoy"] = out[col].shift(365)
        return out

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        target_cols=None,
        time_col: str = "time",
    ) -> pd.DataFrame:
        """滚动统计只使用前一日及更早观测，排除当天目标。"""
        out = df.copy()
        if target_cols is None:
            target_cols = [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
                "shortwave_radiation_sum",
            ]
        out = out.sort_values(time_col).reset_index(drop=True)
        available_cols = [c for c in target_cols if c in out.columns]
        for col in available_cols:
            history_only = out[col].shift(1)
            for window in self.rolling_windows:
                min_periods = max(1, window // 2)
                roll = history_only.rolling(window=window, min_periods=min_periods)
                out[f"{col}_rmean{window}d"] = roll.mean()
                out[f"{col}_rstd{window}d"] = roll.std()
                out[f"{col}_rmin{window}d"] = roll.min()
                out[f"{col}_rmax{window}d"] = roll.max()
        return out

    def _align_origin_state_by_lead(
        self,
        expanded: pd.DataFrame,
        observation_features: pd.DataFrame,
        state_cols: list[str],
    ) -> pd.DataFrame:
        """把 lag/rolling 状态对齐到 target_date - forecast_lead_days 的签发日。"""
        if expanded.empty or not state_cols:
            return expanded

        out = expanded.copy()
        out["time"] = pd.to_datetime(out["time"])
        base = observation_features[["time", *state_cols]].copy()
        base["time"] = pd.to_datetime(base["time"])
        base = base.sort_values("time").drop_duplicates("time", keep="last")

        for lead_value in sorted(out["forecast_lead_days"].dropna().unique()):
            lead = int(lead_value)
            mask = out["forecast_lead_days"].astype(int) == lead
            shifted = base.copy()
            if lead:
                shifted[state_cols] = shifted[state_cols].shift(lead)
            shifted = shifted.set_index("time")
            target_times = out.loc[mask, "time"]
            for col in state_cols:
                out.loc[mask, col] = target_times.map(shifted[col]).to_numpy()
        return out

    def build_training_features(
        self,
        historical_daily: pd.DataFrame,
        previous_runs: Optional[pd.DataFrame] = None,
    ):
        # 先在唯一观测日期上构造 history-only lag/rolling，再按固定 lead 展开。
        observation_features, base_cols, temp_target, precip_target = (
            self._base_engineer_type.build_training_features(self, historical_daily)
        )
        causal_base_cols = [
            col for col in base_cols
            if self._is_causal_training_feature(col)
        ]

        if previous_runs is None or previous_runs.empty:
            self.feature_cols = list(causal_base_cols)
            return observation_features, list(causal_base_cols), temp_target, precip_target

        lead_consensus = build_lead_consensus_features(previous_runs)
        expanded = expand_observation_features_by_lead(
            observation_features,
            lead_consensus,
        )
        if expanded.empty:
            self.feature_cols = list(causal_base_cols)
            return expanded, list(causal_base_cols), temp_target, precip_target

        state_cols = [
            col for col in causal_base_cols
            if self._is_origin_state_feature(col) and col in observation_features.columns
        ]
        expanded = self._align_origin_state_by_lead(
            expanded,
            observation_features,
            state_cols,
        )

        nwp_cols = [
            col for col in lead_consensus.columns
            if col != "time"
            and col not in {temp_target, precip_target}
            and pd.api.types.is_numeric_dtype(lead_consensus[col])
        ]
        feature_cols = list(dict.fromkeys([*causal_base_cols, *nwp_cols]))
        feature_cols = [
            c for c in feature_cols
            if c in expanded.columns and expanded[c].notna().any()
        ]
        self.feature_cols = feature_cols
        return expanded, feature_cols, temp_target, precip_target

    def build_prediction_features(
        self,
        det_df: pd.DataFrame,
        ens_df: pd.DataFrame,
        station_df: pd.DataFrame,
        recent_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """用当前签发时可见的历史状态 + 各目标日 NWP 构造未来行。"""
        base = recent_history.copy()
        if "time" not in base.columns or base.empty:
            return pd.DataFrame()

        base["time"] = pd.to_datetime(base["time"])
        base = base.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        last_observed = pd.Timestamp(base["time"].max()).normalize()
        origin_time = last_observed + pd.Timedelta(days=1)

        # 增加一个无目标值的“签发日”锚点，使 lag1=昨天，rolling 截止昨天。
        anchor = {col: np.nan for col in base.columns}
        anchor["time"] = origin_time
        state_frame = pd.concat([base, pd.DataFrame([anchor])], ignore_index=True, sort=False)
        state_frame = self.add_lag_features(state_frame)
        state_frame = self.add_rolling_features(state_frame)
        origin_row = state_frame.iloc[-1]

        history_state = pd.DataFrame({"time": [last_observed]})
        origin_state_cols = [
            col for col in state_frame.columns
            if self._is_origin_state_feature(col)
        ]
        for col in origin_state_cols:
            history_state[col] = origin_row[col]

        consensus = self.build_model_consensus_features(det_df)
        ensemble = self.build_ensemble_features(ens_df)
        spatial = self.build_station_spatial_features(station_df)

        future = build_forecast_scaffold(history_state, consensus, ensemble, spatial)
        if future.empty:
            return future

        # 这些特征只由目标日期决定，线上可精确重算。
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
