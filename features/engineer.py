# -*- coding: utf-8 -*-
"""
特征工程管线

构建约150个特征，包括：
1. 时间特征（周期性编码）
2. 滞后特征（1-14天）
3. 滚动窗口统计
4. 物理/气象导出特征
5. 多模型共识特征
6. 集合散度特征
7. 空间（跨站点）特征
8. 上海特色特征（季风、台风季、梅雨）

严格防止数据泄漏。
"""

import logging
from typing import Tuple, List, Optional, Dict

import pandas as pd
import numpy as np

from config.settings import (
    ML_CONFIG, LEAKED_FEATURES, DETERMINISTIC_MODELS,
    TYPHOON_SEASON_MONTHS, MEIYU_START_DOY, MEIYU_END_DOY,
    MONSOON_WET_MONTHS, MONSOON_DRY_MONTHS,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    特征工程管线

    负责将原始气象数据转换为机器学习模型可用的特征矩阵。
    """

    def __init__(self):
        self.lag_days = ML_CONFIG.lag_days
        self.rolling_windows = ML_CONFIG.rolling_windows
        self.feature_cols: List[str] = []

    # =========================================================================
    # 时间特征
    # =========================================================================
    @staticmethod
    def add_temporal_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
        """
        添加周期性时间编码特征

        - doy_sin/cos: 一年中的天数（周期性）
        - month_sin/cos: 月份
        - week_sin/cos: ISO周
        - season: 季节（0冬1春2夏3秋）
        - is_weekend: 是否周末

        Args:
            df: 输入DataFrame
            time_col: 时间列名

        Returns:
            添加了时间特征的DataFrame
        """
        df = df.copy()
        dt = pd.to_datetime(df[time_col])

        doy = dt.dt.dayofyear
        month = dt.dt.month
        week = dt.dt.isocalendar().week.astype(int)

        # 周期性编码
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["week_sin"] = np.sin(2 * np.pi * week / 52)
        df["week_cos"] = np.cos(2 * np.pi * week / 52)

        # 季节: 12-2冬(0), 3-5春(1), 6-8夏(2), 9-11秋(3)
        df["season"] = (month % 12 // 3).astype(int)
        df["is_weekend"] = dt.dt.weekday.isin([5, 6]).astype(int)

        return df

    # =========================================================================
    # 年同比特征
    # =========================================================================
    @staticmethod
    def add_yoy_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
        """
        添加年同比特征（去年同日温度/降水的滞后参考）

        帮助模型学习气候基线偏移。
        """
        df = df.copy()
        df = df.sort_values(time_col).reset_index(drop=True)

        for col in ["temperature_2m_max", "precipitation_sum"]:
            if col in df.columns:
                # 365-day lag ≈ same day last year
                df[f"{col}_yoy"] = df[col].shift(365)
                # difference from last year
                df[f"{col}_yoy_diff"] = df[col] - df[col].shift(365)

        return df

    # =========================================================================
    # 上海特色特征
    # =========================================================================
    @staticmethod
    def add_shanghai_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
        """
        添加上海特色气象特征

        - 台风季标志 (6-11月)
        - 梅雨季标志 (约6/12-7/12)
        - 东亚季风指标
        - 湿度异常

        Args:
            df: 输入DataFrame
            time_col: 时间列名

        Returns:
            添加了上海特征的DataFrame
        """
        df = df.copy()
        dt = pd.to_datetime(df[time_col])
        month = dt.dt.month
        doy = dt.dt.dayofyear

        # 台风季标志
        df["is_typhoon_season"] = month.isin(TYPHOON_SEASON_MONTHS).astype(int)

        # 梅雨季标志
        df["is_meiyu_season"] = (
            (doy >= MEIYU_START_DOY) & (doy <= MEIYU_END_DOY)
        ).astype(int)

        # 季风类型: 夏季风(1)、冬季风(-1)、过渡(0)
        df["monsoon_indicator"] = 0
        df.loc[month.isin(MONSOON_WET_MONTHS), "monsoon_indicator"] = 1
        df.loc[month.isin(MONSOON_DRY_MONTHS), "monsoon_indicator"] = -1

        # 季风湿度指标（如有湿度数据）
        if "relative_humidity_2m" in df.columns:
            # 季节性湿度异常
            month_mean_rh = df.groupby(month)["relative_humidity_2m"].transform("mean")
            df["rh_seasonal_anomaly"] = df["relative_humidity_2m"] - month_mean_rh

        return df

    # =========================================================================
    # 滞后特征
    # =========================================================================
    def add_lag_features(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        time_col: str = "time",
    ) -> pd.DataFrame:
        """
        添加滞后特征

        为指定变量创建1,2,3,5,7,14天的滞后值。

        Args:
            df: 输入DataFrame（需按时间排序）
            target_cols: 目标变量列名列表
            time_col: 时间列名

        Returns:
            添加了滞后特征的DataFrame
        """
        df = df.copy()
        if target_cols is None:
            target_cols = [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "precipitation_sum", "wind_speed_10m_max", "shortwave_radiation_sum",
            ]

        # 确保按时间排序
        df = df.sort_values(time_col).reset_index(drop=True)

        available_cols = [c for c in target_cols if c in df.columns]
        for col in available_cols:
            for lag in self.lag_days:
                df[f"{col}_lag{lag}d"] = df[col].shift(lag)

        return df

    # =========================================================================
    # 滚动窗口特征
    # =========================================================================
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        time_col: str = "time",
    ) -> pd.DataFrame:
        """
        添加滚动窗口统计特征

        对指定变量计算3,7,14,30天窗口的均值/标准差/最小值/最大值。

        Args:
            df: 输入DataFrame
            target_cols: 目标变量列名列表
            time_col: 时间列名

        Returns:
            添加了滚动特征的DataFrame
        """
        df = df.copy()
        if target_cols is None:
            target_cols = [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "precipitation_sum", "wind_speed_10m_max", "shortwave_radiation_sum",
            ]

        df = df.sort_values(time_col).reset_index(drop=True)

        available_cols = [c for c in target_cols if c in df.columns]
        for col in available_cols:
            for window in self.rolling_windows:
                min_periods = max(1, window // 2)
                roll = df[col].rolling(window=window, min_periods=min_periods)
                df[f"{col}_rmean{window}d"] = roll.mean()
                df[f"{col}_rstd{window}d"] = roll.std()
                df[f"{col}_rmin{window}d"] = roll.min()
                df[f"{col}_rmax{window}d"] = roll.max()

        return df

    # =========================================================================
    # 物理/气象导出特征
    # =========================================================================
    @staticmethod
    def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        添加物理导出特征

        - temp_range: 日温差
        - dewpoint_depression: 露点温度差
        - sat_vapor_pressure: 饱和水汽压(Magnus公式)
        - wind_chill: 风寒指数
        - heat_index: 高温指数
        - moisture_load: 湿度负荷
        - pressure_tendency: 气压趋势

        Returns:
            添加了物理特征的DataFrame
        """
        df = df.copy()

        # 日温差
        if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
            df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

        # 露点温度差（不稳定性指标）
        if "temperature_2m" in df.columns and "dew_point_2m" in df.columns:
            df["dewpoint_depression"] = df["temperature_2m"] - df["dew_point_2m"]
        elif "temperature_2m_mean" in df.columns and "dew_point_2m" in df.columns:
            df["dewpoint_depression"] = df["temperature_2m_mean"] - df["dew_point_2m"]

        # 饱和水汽压 (Magnus公式)
        temp_col = None
        for col in ["temperature_2m_mean", "temperature_2m", "temperature_2m_max"]:
            if col in df.columns:
                temp_col = col
                break
        if temp_col is not None:
            t = df[temp_col]
            df["sat_vapor_pressure"] = 6.1078 * np.exp(17.269 * t / (237.3 + t))

        # 风寒指数（T<10°C时）
        if temp_col and "wind_speed_10m_max" in df.columns:
            t = df[temp_col]
            ws = df["wind_speed_10m_max"] * 3.6  # m/s → km/h
            wc = (
                13.12 + 0.6215 * t - 11.37 * np.power(ws.clip(lower=1), 0.16)
                + 0.3965 * t * np.power(ws.clip(lower=1), 0.16)
            )
            df["wind_chill"] = np.where(t < 10, wc, t)

        # 高温指数（简化版，T>27°C时）
        if temp_col and "relative_humidity_2m" in df.columns:
            t = df[temp_col]
            rh = df["relative_humidity_2m"]
            # 简化Steadman公式
            hi = (
                -8.785 + 1.611 * t + 2.339 * rh
                - 0.1461 * t * rh - 0.01231 * t**2
                - 0.01642 * rh**2 + 0.002212 * t**2 * rh
                + 0.0007255 * t * rh**2 - 0.000003582 * t**2 * rh**2
            )
            df["heat_index"] = np.where(t > 27, hi, t)

        # 湿度负荷
        if temp_col and "relative_humidity_2m" in df.columns:
            df["moisture_load"] = df[temp_col] * df["relative_humidity_2m"] / 100

        # 气压趋势
        if "pressure_msl" in df.columns:
            df["pressure_tendency"] = df["pressure_msl"].diff()

        return df

    # =========================================================================
    # 多模型共识特征
    # =========================================================================
    @staticmethod
    def build_model_consensus_features(det_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建多模型共识特征

        对8个确定性模型的预报变量计算统计量：
        均值、标准差、最小值、最大值、极差、有效模型数。

        Args:
            det_df: 确定性预报DataFrame（长表，含model列）

        Returns:
            模型共识特征DataFrame（每天一行）
        """
        if det_df.empty or "model" not in det_df.columns:
            return pd.DataFrame()

        consensus_vars = [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        ]
        available_vars = [v for v in consensus_vars if v in det_df.columns]

        if not available_vars or "time" not in det_df.columns:
            return pd.DataFrame()

        result_frames = []
        for var in available_vars:
            prefix = var.replace("temperature_2m_", "tmax_" if "max" in var else "tmin_").replace(
                "precipitation_sum", "precip"
            )
            pivot = det_df.pivot_table(
                index="time", columns="model", values=var, aggfunc="first"
            )
            stats = pd.DataFrame(index=pivot.index)
            stats[f"{prefix}_model_mean"] = pivot.mean(axis=1)
            stats[f"{prefix}_model_std"] = pivot.std(axis=1)
            stats[f"{prefix}_model_min"] = pivot.min(axis=1)
            stats[f"{prefix}_model_max"] = pivot.max(axis=1)
            stats[f"{prefix}_model_range"] = pivot.max(axis=1) - pivot.min(axis=1)
            stats[f"{prefix}_model_count"] = pivot.notna().sum(axis=1)
            result_frames.append(stats)

        if not result_frames:
            return pd.DataFrame()

        consensus = pd.concat(result_frames, axis=1).reset_index()
        return consensus

    # =========================================================================
    # 集合散度特征
    # =========================================================================
    @staticmethod
    def build_ensemble_features(ens_summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建集合散度特征

        从集合预报统计摘要提取逐日特征。

        Args:
            ens_summary_df: 集合统计摘要DataFrame

        Returns:
            集合特征DataFrame
        """
        if ens_summary_df.empty:
            return pd.DataFrame()

        # 已经是逐日聚合的格式
        ens = ens_summary_df.copy()

        # 如果有date列用date，否则用time
        if "date" in ens.columns:
            ens = ens.rename(columns={"date": "time"})

        return ens

    # =========================================================================
    # 空间特征
    # =========================================================================
    @staticmethod
    def build_station_spatial_features(station_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建跨站点空间特征

        计算30个站点的温度/降水/湿度的均值、标准差和极差。

        Args:
            station_df: 多站点数据DataFrame

        Returns:
            空间特征DataFrame（每天一行）
        """
        if station_df.empty or "station_id" not in station_df.columns:
            return pd.DataFrame()

        # 构建时间列（如果逐小时数据，先聚合为逐日）
        df = station_df.copy()
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"]).dt.date

        spatial_vars = {
            "temperature_2m": "spatial_temp",
            "precipitation": "spatial_precip",
            "relative_humidity_2m": "spatial_rh",
        }

        # 如果是逐日数据用逐日变量名
        if "temperature_2m_max" in df.columns:
            spatial_vars = {
                "temperature_2m_max": "spatial_tmax",
                "temperature_2m_min": "spatial_tmin",
                "precipitation_sum": "spatial_precip",
            }

        result_frames = []
        date_col = "date" if "date" in df.columns else "time"

        for var, prefix in spatial_vars.items():
            if var not in df.columns:
                continue

            # 每站每天均值
            daily_station = df.groupby([date_col, "station_id"])[var].mean().reset_index()

            # 跨站点统计
            spatial = daily_station.groupby(date_col)[var].agg(
                **{
                    f"{prefix}_mean": "mean",
                    f"{prefix}_std": "std",
                    f"{prefix}_range": lambda x: x.max() - x.min(),
                }
            ).reset_index()
            result_frames.append(spatial)

        if not result_frames:
            return pd.DataFrame()

        merged = result_frames[0]
        for df_r in result_frames[1:]:
            merged = merged.merge(df_r, on=date_col, how="outer")

        if date_col == "date":
            merged["time"] = pd.to_datetime(merged["date"])
            merged = merged.drop(columns=["date"])

        return merged

    # =========================================================================
    # 训练特征构建主管线
    # =========================================================================
    def build_training_features(
        self, historical_daily: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], str, str]:
        """
        构建训练特征的完整管线

        对历史逐日数据执行：时间特征→物理特征→上海特色→滞后→滚动。

        Args:
            historical_daily: 历史逐日观测DataFrame

        Returns:
            (处理后的DataFrame, 特征列列表, 温度目标列名, 降水目标列名)
        """
        df = historical_daily.copy()

        # 确保time列存在
        if "time" not in df.columns:
            raise ValueError("历史数据必须包含'time'列")

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # 1. 时间特征
        df = self.add_temporal_features(df)
        logger.info("时间特征添加完成")

        # 2. 物理特征
        df = self.add_physical_features(df)
        logger.info("物理特征添加完成")

        # 3. 上海特色特征
        df = self.add_shanghai_features(df)
        logger.info("上海特色特征添加完成")

        # 3.5. 年同比特征
        df = self.add_yoy_features(df)
        logger.info("年同比特征添加完成")

        # 4. 滞后特征
        df = self.add_lag_features(df)
        logger.info("滞后特征添加完成")

        # 5. 滚动窗口特征
        df = self.add_rolling_features(df)
        logger.info("滚动窗口特征添加完成")

        # 确定特征列（排除目标、时间和泄漏特征）
        temp_target = "temperature_2m_max"
        precip_target = "precipitation_sum"

        exclude_cols = {
            "time", temp_target, precip_target,
            "model", "station_id", "station_name", "zone", "lat", "lon",
            "date",
        } | LEAKED_FEATURES

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, float, int]
        ]

        # 删除全NaN列
        valid_cols = [c for c in feature_cols if df[c].notna().any()]
        self.feature_cols = valid_cols

        logger.info(f"特征工程完成: {len(valid_cols)}个特征")
        return df, valid_cols, temp_target, precip_target

    # =========================================================================
    # 预测特征构建
    # =========================================================================
    def build_prediction_features(
        self,
        det_df: pd.DataFrame,
        ens_df: pd.DataFrame,
        station_df: pd.DataFrame,
        recent_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        构建预测时的特征矩阵

        合并确定性共识、集合散度、空间特征和历史滞后/滚动特征。

        Args:
            det_df: 确定性预报数据
            ens_df: 集合统计摘要
            station_df: 站点数据
            recent_history: 最近60天历史观测

        Returns:
            预测特征矩阵DataFrame
        """
        # 基础历史特征
        base = recent_history.copy()
        if "time" not in base.columns:
            return pd.DataFrame()

        base["time"] = pd.to_datetime(base["time"])
        base = base.sort_values("time").reset_index(drop=True)

        # 添加基础特征
        base = self.add_temporal_features(base)
        base = self.add_physical_features(base)
        base = self.add_shanghai_features(base)
        base = self.add_yoy_features(base)
        base = self.add_lag_features(base)
        base = self.add_rolling_features(base)

        # 多模型共识
        consensus = self.build_model_consensus_features(det_df)
        if not consensus.empty and "time" in consensus.columns:
            consensus["time"] = pd.to_datetime(consensus["time"])
            base = base.merge(consensus, on="time", how="left")

        # 集合散度
        ens_features = self.build_ensemble_features(ens_df)
        if not ens_features.empty and "time" in ens_features.columns:
            ens_features["time"] = pd.to_datetime(ens_features["time"])
            base = base.merge(ens_features, on="time", how="left", suffixes=("", "_ens"))

        # 空间特征
        spatial = self.build_station_spatial_features(station_df)
        if not spatial.empty and "time" in spatial.columns:
            spatial["time"] = pd.to_datetime(spatial["time"])
            base = base.merge(spatial, on="time", how="left")

        # 对齐特征列
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in base.columns:
                    base[col] = 0.0

        return base

    # =========================================================================
    # 工具方法
    # =========================================================================
    @staticmethod
    def remove_leaked_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        移除数据泄漏特征

        Args:
            df: 输入DataFrame

        Returns:
            移除泄漏特征后的DataFrame
        """
        leak_cols = [c for c in df.columns if c in LEAKED_FEATURES]
        if leak_cols:
            logger.info(f"移除{len(leak_cols)}个泄漏特征: {leak_cols}")
            df = df.drop(columns=leak_cols, errors="ignore")
        return df

    @staticmethod
    def impute_missing(
        df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        中位数填充缺失值

        Args:
            df: 输入DataFrame
            feature_cols: 特征列列表

        Returns:
            填充后的DataFrame
        """
        df = df.copy()
        for col in feature_cols:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
        return df
