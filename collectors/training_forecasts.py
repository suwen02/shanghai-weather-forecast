# -*- coding: utf-8 -*-
"""训练用历史 NWP 预报采集与持久化。"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from collectors.open_meteo import OpenMeteoCollector
from config.settings import ML_CONFIG, RAW_DIR


def collect_training_forecasts(years: Optional[int] = None) -> Optional[Path]:
    """采集历史 NWP 预报并保存为训练数据 parquet。"""
    years = years or ML_CONFIG.historical_years
    end_date = date.today() - timedelta(days=1)
    start_date = date(end_date.year - years, end_date.month, end_date.day)

    df = OpenMeteoCollector().collect_historical_forecasts(start_date, end_date)
    if df.empty:
        return None

    path = RAW_DIR / f"historical_forecasts_{years}yr.parquet"
    df.to_parquet(path, index=False)
    return path
