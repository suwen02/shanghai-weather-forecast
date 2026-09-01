# -*- coding: utf-8 -*-
"""实时刷新状态、数据指纹和原子发布工具。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from zoneinfo import ZoneInfo


def snapshot_fingerprint(snapshot: Dict[str, Any]) -> str:
    """为天气数据主体生成稳定 SHA-256 指纹，忽略响应耗时等瞬时元数据。"""
    stable = {
        key: snapshot[key]
        for key in ("current", "hourly", "daily")
        if key in snapshot
    }
    payload = json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class RefreshStateStore:
    """持久化上一次成功发布的天气数据指纹。"""

    path: Path

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def should_refresh(self, fingerprint: str) -> bool:
        """当上游天气快照发生变化时返回 True。"""
        return self._load().get("fingerprint") != fingerprint

    def mark_refreshed(self, fingerprint: str, refreshed_at: str) -> None:
        """记录一次成功发布。"""
        payload = {"fingerprint": fingerprint, "refreshed_at": refreshed_at}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def atomic_publish_json(
    payload: Dict[str, Any],
    versioned_path: Path,
    latest_path: Path,
) -> None:
    """先写版本文件，再通过原子替换更新 latest.json。"""
    versioned_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    version_tmp = versioned_path.with_suffix(versioned_path.suffix + ".tmp")
    version_tmp.write_text(serialized, encoding="utf-8")
    os.replace(version_tmp, versioned_path)

    latest_tmp = latest_path.with_suffix(latest_path.suffix + ".tmp")
    latest_tmp.write_text(serialized, encoding="utf-8")
    os.replace(latest_tmp, latest_path)


def build_short_term_forecast(
    snapshot: Dict[str, Any],
    now: Optional[datetime] = None,
    timezone: str = "Asia/Shanghai",
    horizon_hours: int = 48,
    stale_after_minutes: int = 90,
) -> Dict[str, Any]:
    """把 Open-Meteo 逐小时快照转换为面向发布的短临预报结构。"""
    tz = ZoneInfo(timezone)
    now = now.astimezone(tz) if now is not None else datetime.now(tz)
    hourly = snapshot.get("hourly") or {}
    times = hourly.get("time") or []

    parsed = []
    for i, value in enumerate(times):
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)
        parsed.append((i, dt))

    past = [dt for _, dt in parsed if dt <= now]
    data_as_of = max(past) if past else now
    age_minutes = max(0, int((now - data_as_of).total_seconds() // 60))

    selected = [(i, dt) for i, dt in parsed if dt > now][:horizon_hours]
    variables = [key for key in hourly.keys() if key != "time"]
    hours = []
    for i, dt in selected:
        row = {"time": dt.isoformat()}
        for variable in variables:
            values = hourly.get(variable) or []
            row[variable] = values[i] if i < len(values) else None
        hours.append(row)

    return {
        "data_as_of": data_as_of.isoformat(),
        "data_age_minutes": age_minutes,
        "is_stale": age_minutes > stale_after_minutes,
        "hours": hours,
    }


def fetch_latest_snapshot(
    collector: Any,
    past_hours: int = 6,
    forecast_hours: int = 48,
) -> Dict[str, Any]:
    """通过现有 OpenMeteoCollector 获取最新 best_match 逐小时快照。"""
    from config.settings import API_ENDPOINTS, SHANGHAI_LAT, SHANGHAI_LON, TIMEZONE

    hourly_variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "pressure_msl",
        "cloud_cover",
        "precipitation_probability",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]
    params = {
        "latitude": SHANGHAI_LAT,
        "longitude": SHANGHAI_LON,
        "hourly": ",".join(hourly_variables),
        "models": "best_match",
        "timezone": TIMEZONE,
        "past_hours": past_hours,
        # Open-Meteo 会把当前整点计入 forecast_hours；多取1小时以保留完整未来窗口。
        "forecast_hours": forecast_hours + 1,
    }
    return collector._get(API_ENDPOINTS["deterministic"], params)
