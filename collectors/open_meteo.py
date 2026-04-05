# -*- coding: utf-8 -*-
"""
Open-Meteo 多模型数据采集器

支持以下数据源：
1. 确定性预报 (8个NWP模型)
2. 集合预报 (5个系统，共161个成员)
3. 历史观测存档 (5年)
4. 历史预报存档 (用于训练数据对齐)

所有API免费，无需密钥。
"""

import time
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    SHANGHAI_LAT, SHANGHAI_LON, TIMEZONE,
    DETERMINISTIC_MODELS, ENSEMBLE_MODELS, ENSEMBLE_HOURLY_VARIABLES,
    API_ENDPOINTS, HOURLY_VARIABLES, DAILY_VARIABLES,
    HTTP_RATE_LIMIT_DELAY, HTTP_TIMEOUT, HTTP_MAX_RETRIES, HTTP_USER_AGENT,
    RAW_DIR, ML_CONFIG,
)

logger = logging.getLogger(__name__)


class OpenMeteoCollector:
    """
    Open-Meteo API 数据采集器

    支持多模型确定性预报、集合预报统计和历史存档数据的采集。
    包含速率限制、超时控制和指数退避重试机制。
    """

    def __init__(self, rate_limit_delay: float = HTTP_RATE_LIMIT_DELAY):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": HTTP_USER_AGENT})

        # 配置重试策略
        retry_strategy = Retry(
            total=HTTP_MAX_RETRIES,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, url: str, params: Dict) -> Dict:
        """
        发送HTTP GET请求，带速率限制和错误处理

        Args:
            url: API端点URL
            params: 请求参数

        Returns:
            JSON响应字典
        """
        time.sleep(self.rate_limit_delay)
        try:
            resp = self.session.get(url, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {url}, 参数: {params}, 错误: {e}")
            raise

    # =========================================================================
    # 确定性预报采集
    # =========================================================================
    def collect_deterministic_forecasts(
        self,
        target_date: Optional[date] = None,
        lat: float = SHANGHAI_LAT,
        lon: float = SHANGHAI_LON,
    ) -> pd.DataFrame:
        """
        采集8个确定性NWP模型的预报数据

        每个模型返回7天逐小时+逐日预报。
        最终合并为一个宽表，列名前缀为模型名。

        Args:
            target_date: 目标日期，默认为今天
            lat: 纬度
            lon: 经度

        Returns:
            合并后的DataFrame，包含所有模型的预报数据
        """
        if target_date is None:
            target_date = date.today()

        all_frames = []
        for model in DETERMINISTIC_MODELS:
            try:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": ",".join(HOURLY_VARIABLES),
                    "daily": ",".join(DAILY_VARIABLES),
                    "models": model,
                    "timezone": TIMEZONE,
                    "past_days": 1,
                    "forecast_days": 7,
                }
                data = self._get(API_ENDPOINTS["deterministic"], params)

                # 解析逐日数据
                if "daily" in data and data["daily"]:
                    daily = data["daily"]
                    df_daily = pd.DataFrame(daily)
                    if "time" in df_daily.columns:
                        df_daily["time"] = pd.to_datetime(df_daily["time"])
                    df_daily["model"] = model
                    all_frames.append(df_daily)

                logger.info(f"确定性预报采集成功: {model}")
            except Exception as e:
                logger.warning(f"确定性预报采集失败: {model}, 错误: {e}")
                continue

        if not all_frames:
            logger.error("所有确定性预报模型采集失败")
            return pd.DataFrame()

        # 合并所有模型数据
        combined = pd.concat(all_frames, ignore_index=True)
        return combined

    def collect_deterministic_wide(
        self,
        target_date: Optional[date] = None,
        lat: float = SHANGHAI_LAT,
        lon: float = SHANGHAI_LON,
    ) -> pd.DataFrame:
        """
        采集确定性预报并转换为宽表格式

        每个模型的变量作为独立列，列名格式: {变量}_{模型名}

        Returns:
            宽表DataFrame，每行为一天
        """
        long_df = self.collect_deterministic_forecasts(target_date, lat, lon)
        if long_df.empty:
            return pd.DataFrame()

        # 转为宽表：每个模型每个变量一列
        pivot_frames = []
        for model_name, group in long_df.groupby("model"):
            group = group.set_index("time").drop(columns=["model"], errors="ignore")
            group = group.rename(columns={col: f"{col}_{model_name}" for col in group.columns})
            pivot_frames.append(group)

        if not pivot_frames:
            return pd.DataFrame()

        wide = pivot_frames[0]
        for df in pivot_frames[1:]:
            wide = wide.join(df, how="outer")

        wide = wide.reset_index()
        return wide

    # =========================================================================
    # 集合预报采集
    # =========================================================================
    def collect_ensemble_summary(
        self,
        lat: float = SHANGHAI_LAT,
        lon: float = SHANGHAI_LON,
    ) -> pd.DataFrame:
        """
        采集5个集合预报系统的统计摘要

        对每个变量计算：均值、标准差、最小值、最大值、
        P10、P25、中位数、P75、P90和成员数。

        Returns:
            集合统计摘要DataFrame
        """
        all_summaries = []

        for model_name, n_members in ENSEMBLE_MODELS.items():
            try:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": ",".join(ENSEMBLE_HOURLY_VARIABLES),
                    "models": model_name,
                    "timezone": TIMEZONE,
                    "forecast_days": 7,
                }
                data = self._get(API_ENDPOINTS["ensemble"], params)

                if "hourly" not in data or not data["hourly"]:
                    continue

                hourly = data["hourly"]
                times = pd.to_datetime(hourly.get("time", []))

                # 对每个变量处理成员数据
                for base_var in ENSEMBLE_HOURLY_VARIABLES:
                    # 收集所有成员数据
                    member_cols = []
                    for key, values in hourly.items():
                        if key.startswith(base_var) and key != "time":
                            member_cols.append(values)

                    if not member_cols:
                        # 尝试只有基础变量名的情况
                        if base_var in hourly:
                            member_cols.append(hourly[base_var])

                    if not member_cols:
                        continue

                    arr = np.array(member_cols, dtype=float).T  # (时间步, 成员数)

                    summary_data = {
                        "time": times,
                        f"ens_{base_var}_mean": np.nanmean(arr, axis=1),
                        f"ens_{base_var}_std": np.nanstd(arr, axis=1),
                        f"ens_{base_var}_min": np.nanmin(arr, axis=1),
                        f"ens_{base_var}_max": np.nanmax(arr, axis=1),
                        f"ens_{base_var}_p10": np.nanpercentile(arr, 10, axis=1),
                        f"ens_{base_var}_p25": np.nanpercentile(arr, 25, axis=1),
                        f"ens_{base_var}_median": np.nanmedian(arr, axis=1),
                        f"ens_{base_var}_p75": np.nanpercentile(arr, 75, axis=1),
                        f"ens_{base_var}_p90": np.nanpercentile(arr, 90, axis=1),
                        f"ens_{base_var}_n_members": [arr.shape[1]] * len(times),
                    }
                    df_var = pd.DataFrame(summary_data)
                    df_var["model"] = model_name
                    all_summaries.append(df_var)

                logger.info(f"集合预报统计采集成功: {model_name} ({n_members}个成员)")

            except Exception as e:
                logger.warning(f"集合预报采集失败: {model_name}, 错误: {e}")
                continue

        if not all_summaries:
            logger.error("所有集合预报系统采集失败")
            return pd.DataFrame()

        combined = pd.concat(all_summaries, ignore_index=True)

        # 按时间聚合跨模型统计
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        if "time" in combined.columns and numeric_cols:
            daily = combined.copy()
            daily["date"] = daily["time"].dt.date
            daily_agg = daily.groupby("date")[numeric_cols].mean().reset_index()
            daily_agg["date"] = pd.to_datetime(daily_agg["date"])
            return daily_agg

        return combined

    # =========================================================================
    # 历史观测数据采集
    # =========================================================================
    def collect_historical_data(
        self,
        start_date: date,
        end_date: date,
        lat: float = SHANGHAI_LAT,
        lon: float = SHANGHAI_LON,
    ) -> Dict[str, pd.DataFrame]:
        """
        采集历史观测存档数据（作为训练真值）

        按年分块采集以避免API限制。

        Args:
            start_date: 起始日期
            end_date: 结束日期
            lat: 纬度
            lon: 经度

        Returns:
            字典包含 'daily' 和 'hourly' DataFrame
        """
        daily_frames = []
        hourly_frames = []

        # 按年分块
        current_start = start_date
        while current_start < end_date:
            chunk_end = min(
                date(current_start.year, 12, 31),
                end_date
            )

            try:
                # 逐日数据
                params_daily = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": current_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "daily": ",".join(DAILY_VARIABLES),
                    "timezone": TIMEZONE,
                }
                data_daily = self._get(API_ENDPOINTS["archive"], params_daily)

                if "daily" in data_daily and data_daily["daily"]:
                    df = pd.DataFrame(data_daily["daily"])
                    if "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"])
                    daily_frames.append(df)

                # 逐小时数据
                params_hourly = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": current_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "hourly": ",".join(HOURLY_VARIABLES),
                    "timezone": TIMEZONE,
                }
                data_hourly = self._get(API_ENDPOINTS["archive"], params_hourly)

                if "hourly" in data_hourly and data_hourly["hourly"]:
                    df_h = pd.DataFrame(data_hourly["hourly"])
                    if "time" in df_h.columns:
                        df_h["time"] = pd.to_datetime(df_h["time"])
                    hourly_frames.append(df_h)

                logger.info(
                    f"历史数据采集成功: {current_start} ~ {chunk_end}"
                )

            except Exception as e:
                logger.warning(
                    f"历史数据采集失败: {current_start} ~ {chunk_end}, 错误: {e}"
                )

            # 下一年
            current_start = date(current_start.year + 1, 1, 1)

        result = {}
        if daily_frames:
            df_daily = pd.concat(daily_frames, ignore_index=True)
            df_daily = df_daily.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
            result["daily"] = df_daily
        else:
            result["daily"] = pd.DataFrame()

        if hourly_frames:
            df_hourly = pd.concat(hourly_frames, ignore_index=True)
            df_hourly = df_hourly.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
            result["hourly"] = df_hourly
        else:
            result["hourly"] = pd.DataFrame()

        return result

    def collect_historical_daily(
        self,
        years: int = 5,
        lat: float = SHANGHAI_LAT,
        lon: float = SHANGHAI_LON,
    ) -> pd.DataFrame:
        """
        采集指定年数的历史逐日数据

        便捷方法，自动计算起始日期。

        Args:
            years: 历史年数
            lat: 纬度
            lon: 经度

        Returns:
            历史逐日DataFrame
        """
        end_date = date.today() - timedelta(days=1)
        start_date = date(end_date.year - years, end_date.month, end_date.day)
        result = self.collect_historical_data(start_date, end_date, lat, lon)
        return result.get("daily", pd.DataFrame())

    # =========================================================================
    # 历史预报数据采集（用于训练数据对齐）
    # =========================================================================
    def collect_historical_forecasts(
        self,
        start_date: date,
        end_date: date,
        lat: float = SHANGHAI_LAT,
        lon: float = SHANGHAI_LON,
    ) -> pd.DataFrame:
        """
        采集历史NWP预报数据（用于构建训练集中的预报→观测对）

        Args:
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            历史预报DataFrame
        """
        all_frames = []

        # 按3个月分块
        current_start = start_date
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=89), end_date)

            for model in ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]:
                try:
                    params = {
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": current_start.isoformat(),
                        "end_date": chunk_end.isoformat(),
                        "daily": ",".join(DAILY_VARIABLES),
                        "models": model,
                        "timezone": TIMEZONE,
                    }
                    data = self._get(API_ENDPOINTS["historical_forecast"], params)

                    if "daily" in data and data["daily"]:
                        df = pd.DataFrame(data["daily"])
                        if "time" in df.columns:
                            df["time"] = pd.to_datetime(df["time"])
                        df["model"] = model
                        all_frames.append(df)

                except Exception as e:
                    logger.warning(
                        f"历史预报采集失败: {model} ({current_start}~{chunk_end}), 错误: {e}"
                    )
                    continue

            logger.info(f"历史预报采集完成: {current_start} ~ {chunk_end}")
            current_start = chunk_end + timedelta(days=1)

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        return combined


# =============================================================================
# 便捷管道函数
# =============================================================================

def run_daily_collection(collect_date: Optional[date] = None) -> Dict[str, str]:
    """
    执行每日数据采集

    采集确定性预报、集合预报摘要，保存为Parquet文件。

    Args:
        collect_date: 采集日期，默认今天

    Returns:
        保存的文件路径字典
    """
    if collect_date is None:
        collect_date = date.today()

    collector = OpenMeteoCollector()
    date_str = collect_date.strftime("%Y%m%d")
    saved_files = {}

    # 1. 确定性预报
    logger.info("开始采集确定性预报...")
    det_df = collector.collect_deterministic_forecasts(collect_date)
    if not det_df.empty:
        det_path = RAW_DIR / f"deterministic_{date_str}.parquet"
        det_df.to_parquet(det_path, index=False)
        saved_files["deterministic"] = str(det_path)
        logger.info(f"确定性预报已保存: {det_path} ({len(det_df)}行)")

    # 2. 集合预报统计
    logger.info("开始采集集合预报统计...")
    ens_df = collector.collect_ensemble_summary()
    if not ens_df.empty:
        ens_path = RAW_DIR / f"ensemble_summary_{date_str}.parquet"
        ens_df.to_parquet(ens_path, index=False)
        saved_files["ensemble"] = str(ens_path)
        logger.info(f"集合预报统计已保存: {ens_path} ({len(ens_df)}行)")

    return saved_files


def collect_training_history(years: int = 5) -> Dict[str, str]:
    """
    一次性采集训练用历史数据

    包含中心站点多年历史（逐日+逐小时）。

    Args:
        years: 历史年数

    Returns:
        保存的文件路径字典
    """
    collector = OpenMeteoCollector()
    end_date = date.today() - timedelta(days=1)
    start_date = date(end_date.year - years, end_date.month, end_date.day)

    saved_files = {}

    # 上海中心站历史数据
    logger.info(f"开始采集{years}年历史数据...")
    history = collector.collect_historical_data(start_date, end_date)

    if not history["daily"].empty:
        daily_path = RAW_DIR / f"historical_daily_{years}yr.parquet"
        history["daily"].to_parquet(daily_path, index=False)
        saved_files["daily"] = str(daily_path)
        logger.info(f"历史逐日数据已保存: {daily_path} ({len(history['daily'])}行)")

    if not history["hourly"].empty:
        hourly_path = RAW_DIR / f"historical_hourly_{years}yr.parquet"
        history["hourly"].to_parquet(hourly_path, index=False)
        saved_files["hourly"] = str(hourly_path)
        logger.info(f"历史逐小时数据已保存: {hourly_path} ({len(history['hourly'])}行)")

    return saved_files
