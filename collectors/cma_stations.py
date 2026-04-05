# -*- coding: utf-8 -*-
"""
CMA气象站数据采集器

通过Open-Meteo存档API采集上海及周边30个气象站的历史和实时数据。
用于构建空间特征和多站点交叉验证。
"""

import time
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    ALL_STATIONS, CITY_STATIONS, SURROUNDING_STATIONS,
    API_ENDPOINTS, HOURLY_VARIABLES, DAILY_VARIABLES,
    HTTP_RATE_LIMIT_DELAY, HTTP_TIMEOUT, HTTP_MAX_RETRIES, HTTP_USER_AGENT,
    RAW_DIR, TIMEZONE, WeatherStation,
)

logger = logging.getLogger(__name__)


class CMAStationCollector:
    """
    30个气象站数据采集器

    使用Open-Meteo API的经纬度参数，
    为每个站点获取独立的观测/预报数据。
    """

    def __init__(self, rate_limit_delay: float = HTTP_RATE_LIMIT_DELAY):
        self.rate_limit_delay = rate_limit_delay
        self.stations = ALL_STATIONS
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": HTTP_USER_AGENT})

        retry_strategy = Retry(
            total=HTTP_MAX_RETRIES,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, url: str, params: Dict) -> Dict:
        """发送HTTP GET请求，带速率限制"""
        time.sleep(self.rate_limit_delay)
        try:
            resp = self.session.get(url, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"站点API请求失败: {url}, 错误: {e}")
            raise

    # =========================================================================
    # 站点预报采集
    # =========================================================================
    def collect_station_forecasts(
        self,
        stations: Optional[List[WeatherStation]] = None,
    ) -> pd.DataFrame:
        """
        采集所有站点的当前预报数据

        使用best_match模型为每个站点获取3天逐小时预报。

        Args:
            stations: 站点列表，默认全部30个站点

        Returns:
            合并的站点预报DataFrame
        """
        if stations is None:
            stations = self.stations

        all_frames = []
        for station in stations:
            try:
                params = {
                    "latitude": station.lat,
                    "longitude": station.lon,
                    "hourly": ",".join(HOURLY_VARIABLES),
                    "models": "best_match",
                    "timezone": TIMEZONE,
                    "past_days": 1,
                    "forecast_days": 3,
                }
                data = self._get(API_ENDPOINTS["deterministic"], params)

                if "hourly" in data and data["hourly"]:
                    df = pd.DataFrame(data["hourly"])
                    if "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"])
                    df["station_id"] = station.station_id
                    df["station_name"] = station.name_cn
                    df["zone"] = station.zone
                    df["lat"] = station.lat
                    df["lon"] = station.lon
                    all_frames.append(df)

                logger.debug(f"站点预报采集成功: {station.name_cn}")

            except Exception as e:
                logger.warning(f"站点预报采集失败: {station.name_cn}, 错误: {e}")
                continue

        if not all_frames:
            logger.error("所有站点预报采集失败")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        logger.info(f"站点预报采集完成: {len(all_frames)}/{len(stations)}个站点")
        return combined

    # =========================================================================
    # 多站点历史数据采集
    # =========================================================================
    def collect_multi_station_historical(
        self,
        start_date: date,
        end_date: date,
        stations: Optional[List[WeatherStation]] = None,
    ) -> pd.DataFrame:
        """
        采集多站点历史逐日数据

        用于构建空间特征（跨站点温度/降水差异等）。

        Args:
            start_date: 起始日期
            end_date: 结束日期
            stations: 站点列表

        Returns:
            多站点历史DataFrame
        """
        if stations is None:
            stations = self.stations

        all_frames = []
        for station in stations:
            try:
                # 按年分块采集
                current_start = start_date
                station_frames = []

                while current_start < end_date:
                    chunk_end = min(
                        date(current_start.year, 12, 31),
                        end_date,
                    )

                    params = {
                        "latitude": station.lat,
                        "longitude": station.lon,
                        "start_date": current_start.isoformat(),
                        "end_date": chunk_end.isoformat(),
                        "daily": ",".join(DAILY_VARIABLES),
                        "timezone": TIMEZONE,
                    }
                    data = self._get(API_ENDPOINTS["archive"], params)

                    if "daily" in data and data["daily"]:
                        df = pd.DataFrame(data["daily"])
                        if "time" in df.columns:
                            df["time"] = pd.to_datetime(df["time"])
                        station_frames.append(df)

                    current_start = date(current_start.year + 1, 1, 1)

                if station_frames:
                    df_station = pd.concat(station_frames, ignore_index=True)
                    df_station["station_id"] = station.station_id
                    df_station["station_name"] = station.name_cn
                    df_station["zone"] = station.zone
                    df_station["lat"] = station.lat
                    df_station["lon"] = station.lon
                    all_frames.append(df_station)

                logger.info(f"站点历史数据采集成功: {station.name_cn}")

            except Exception as e:
                logger.warning(f"站点历史数据采集失败: {station.name_cn}, 错误: {e}")
                continue

        if not all_frames:
            logger.error("所有站点历史数据采集失败")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["time", "station_id"]
        ).sort_values(["time", "station_id"]).reset_index(drop=True)

        logger.info(
            f"多站点历史数据采集完成: {len(all_frames)}个站点, "
            f"{len(combined)}行数据"
        )
        return combined

    def collect_station_daily_summary(
        self,
        target_date: Optional[date] = None,
        stations: Optional[List[WeatherStation]] = None,
    ) -> pd.DataFrame:
        """
        采集所有站点的当日逐日摘要数据

        Args:
            target_date: 目标日期
            stations: 站点列表

        Returns:
            站点逐日摘要DataFrame
        """
        if target_date is None:
            target_date = date.today()
        if stations is None:
            stations = self.stations

        all_frames = []
        for station in stations:
            try:
                params = {
                    "latitude": station.lat,
                    "longitude": station.lon,
                    "daily": ",".join(DAILY_VARIABLES),
                    "models": "best_match",
                    "timezone": TIMEZONE,
                    "past_days": 1,
                    "forecast_days": 3,
                }
                data = self._get(API_ENDPOINTS["deterministic"], params)

                if "daily" in data and data["daily"]:
                    df = pd.DataFrame(data["daily"])
                    if "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"])
                    df["station_id"] = station.station_id
                    df["station_name"] = station.name_cn
                    df["zone"] = station.zone
                    all_frames.append(df)

            except Exception as e:
                logger.warning(f"站点摘要采集失败: {station.name_cn}, 错误: {e}")
                continue

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        return combined


# =============================================================================
# 便捷管道函数
# =============================================================================

def run_station_collection(collect_date: Optional[date] = None) -> Dict[str, str]:
    """
    执行站点数据采集

    Args:
        collect_date: 采集日期

    Returns:
        保存的文件路径字典
    """
    if collect_date is None:
        collect_date = date.today()

    collector = CMAStationCollector()
    date_str = collect_date.strftime("%Y%m%d")
    saved_files = {}

    # 站点预报
    logger.info("开始采集站点预报...")
    station_df = collector.collect_station_forecasts()
    if not station_df.empty:
        station_path = RAW_DIR / f"stations_{date_str}.parquet"
        station_df.to_parquet(station_path, index=False)
        saved_files["stations"] = str(station_path)
        logger.info(f"站点预报已保存: {station_path} ({len(station_df)}行)")

    return saved_files


def collect_station_history(years: int = 3) -> Dict[str, str]:
    """
    采集多站点历史数据

    Args:
        years: 历史年数

    Returns:
        保存的文件路径字典
    """
    collector = CMAStationCollector()
    end_date = date.today() - timedelta(days=1)
    start_date = date(end_date.year - years, end_date.month, end_date.day)

    saved_files = {}

    logger.info(f"开始采集{years}年多站点历史数据...")
    station_hist = collector.collect_multi_station_historical(start_date, end_date)

    if not station_hist.empty:
        path = RAW_DIR / f"historical_stations_{years}yr.parquet"
        station_hist.to_parquet(path, index=False)
        saved_files["stations"] = str(path)
        logger.info(
            f"多站点历史数据已保存: {path} ({len(station_hist)}行)"
        )

    return saved_files
