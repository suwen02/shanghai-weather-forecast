# -*- coding: utf-8 -*-
"""训练用 Open-Meteo Previous Runs 固定提前量数据采集与持久化。"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from collectors.open_meteo import OpenMeteoCollector
from config.settings import ML_CONFIG, RAW_DIR, SHANGHAI_LAT, SHANGHAI_LON, TIMEZONE

logger = logging.getLogger(__name__)

PREVIOUS_RUNS_ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_ARCHIVE_START = date(2024, 1, 1)
DEFAULT_TRAINING_MODELS: tuple[str, ...] = (
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
    "jma_seamless",
)
# 页面当前展示“今天 + 后 6 天”，因此训练 horizon 与 UI 对齐为 day0..day6。
DEFAULT_LEAD_DAYS: tuple[int, ...] = tuple(range(0, 7))


def _previous_run_variables(lead_days: Iterable[int]) -> list[str]:
    variables = []
    for lead in lead_days:
        variables.extend([
            f"temperature_2m_previous_day{int(lead)}",
            f"precipitation_previous_day{int(lead)}",
        ])
    return variables


def normalize_previous_runs_hourly(
    payload: dict,
    model: str,
    lead_days: Sequence[int] = DEFAULT_LEAD_DAYS,
) -> pd.DataFrame:
    """把 Previous Runs 小时字段规范化为 daily × lead × model 长表。"""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()

    timestamps = pd.to_datetime(pd.Series(times), errors="coerce")
    frames = []
    for lead in lead_days:
        temp_key = f"temperature_2m_previous_day{int(lead)}"
        precip_key = f"precipitation_previous_day{int(lead)}"
        temp_values = hourly.get(temp_key)
        precip_values = hourly.get(precip_key)
        if temp_values is None and precip_values is None:
            continue

        frame = pd.DataFrame({
            "timestamp": timestamps,
            "temperature": pd.to_numeric(
                pd.Series(temp_values if temp_values is not None else [None] * len(times)),
                errors="coerce",
            ),
            "precipitation": pd.to_numeric(
                pd.Series(precip_values if precip_values is not None else [None] * len(times)),
                errors="coerce",
            ),
        })
        frame = frame[frame["timestamp"].notna()].copy()
        if frame.empty or (frame["temperature"].isna().all() and frame["precipitation"].isna().all()):
            continue

        frame["time"] = frame["timestamp"].dt.normalize()
        daily = frame.groupby("time", as_index=False).agg(
            temperature_2m_max=("temperature", "max"),
            temperature_2m_min=("temperature", "min"),
            temperature_2m_mean=("temperature", "mean"),
            precipitation_sum=(
                "precipitation",
                lambda values: values.sum(min_count=1),
            ),
        )
        daily = daily.dropna(
            subset=["temperature_2m_max", "precipitation_sum"],
            how="all",
        )
        if daily.empty:
            continue
        daily["forecast_lead_days"] = int(lead)
        daily["model"] = model
        frames.append(daily)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["time", "forecast_lead_days", "model"]).reset_index(drop=True)


def collect_training_forecasts(
    years: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
    lead_days: Sequence[int] = DEFAULT_LEAD_DAYS,
    lat: float = SHANGHAI_LAT,
    lon: float = SHANGHAI_LON,
) -> Optional[Path]:
    """采集固定 day0–day6 Previous Runs 并保存为 parquet。"""
    years = years or ML_CONFIG.historical_years
    models = tuple(models or DEFAULT_TRAINING_MODELS)
    end_date = date.today() - timedelta(days=1)
    requested_start = date(end_date.year - years, end_date.month, end_date.day)
    start_date = max(requested_start, PREVIOUS_RUNS_ARCHIVE_START)

    collector = OpenMeteoCollector()
    all_frames = []
    current_start = start_date
    while current_start <= end_date:
        chunk_end = min(current_start + timedelta(days=89), end_date)
        for model in models:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": current_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(_previous_run_variables(lead_days)),
                "models": model,
                "timezone": TIMEZONE,
            }
            try:
                payload = collector._get(PREVIOUS_RUNS_ENDPOINT, params)
                normalized = normalize_previous_runs_hourly(payload, model, lead_days)
                if not normalized.empty:
                    all_frames.append(normalized)
            except Exception as exc:
                logger.warning(
                    "Previous Runs 采集失败: %s %s~%s: %s",
                    model,
                    current_start,
                    chunk_end,
                    exc,
                )
        current_start = chunk_end + timedelta(days=1)

    if not all_frames:
        return None

    df = pd.concat(all_frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=["time", "forecast_lead_days", "model"],
        keep="last",
    ).sort_values(["time", "forecast_lead_days", "model"]).reset_index(drop=True)

    path = RAW_DIR / (
        f"historical_previous_runs_{start_date:%Y%m%d}_{end_date:%Y%m%d}.parquet"
    )
    df.to_parquet(path, index=False)
    logger.info(
        "Previous Runs 训练数据已保存: %s (%s rows, leads=%s)",
        path,
        len(df),
        sorted(df["forecast_lead_days"].unique().tolist()),
    )
    return path
