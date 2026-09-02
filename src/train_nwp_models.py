# -*- coding: utf-8 -*-
"""使用历史 NWP 共识特征重新训练并覆盖现有模型 artifacts。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import (
    ML_CONFIG,
    RAW_DIR,
    PROCESSED_DIR,
)
from features.nwp_aware_engineer import NwpAwareFeatureEngineer
from models.temperature import TemperaturePredictor
from models.precipitation import PrecipitationPredictor
from models.calibration import CalibrationManager


def _latest_matching(pattern: str) -> Path:
    files = sorted(RAW_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"未找到训练数据: {RAW_DIR / pattern}")
    return files[0]


def _fit_calibration(
    temp_model: TemperaturePredictor,
    precip_model: PrecipitationPredictor,
    calibration: CalibrationManager,
    X: pd.DataFrame,
    y_temp: pd.Series,
    y_precip: pd.Series,
) -> None:
    n = len(X)
    cal_start = int(n * (1 - ML_CONFIG.calibration_fraction))
    X_cal = X.iloc[cal_start:]
    y_temp_cal = y_temp.iloc[cal_start:].values
    y_precip_cal = y_precip.iloc[cal_start:].values

    X_temp = temp_model.scaler.transform(temp_model._align_features(X_cal))
    q_preds = {}
    for q in temp_model.quantiles:
        label = f"p{int(q * 100):02d}"
        q_preds[label] = temp_model.models[q].predict(X_temp)
    calibration.fit_conformal(y_temp_cal, q_preds)

    X_precip = precip_model.scaler.transform(precip_model._align_features(X_cal))
    y_binary = (y_precip_cal >= ML_CONFIG.precip_occurrence_threshold).astype(int)
    p_raw = precip_model.classifier.predict_proba(X_precip)[:, 1]
    calibration.fit_isotonic(y_binary, p_raw)


def train_nwp_models(
    daily_path: Optional[Path] = None,
    forecast_path: Optional[Path] = None,
) -> dict:
    """训练 NWP-aware 温度/降水模型并保存到现有 artifact 路径。"""
    daily_path = daily_path or _latest_matching("historical_daily_*.parquet")
    forecast_path = forecast_path or _latest_matching("historical_forecasts_*.parquet")

    historical = pd.read_parquet(daily_path)
    historical_forecasts = pd.read_parquet(forecast_path)

    engineer = NwpAwareFeatureEngineer()
    df, feature_cols, temp_target, precip_target = engineer.build_training_features(
        historical,
        historical_forecasts,
    )
    if not engineer.has_nwp_training_features(feature_cols):
        raise RuntimeError(
            "训练特征中没有 NWP 共识列；拒绝生成会导致未来多天输入近似相同的 legacy 模型"
        )

    valid = df[temp_target].notna() & df[precip_target].notna()
    df = df[valid].reset_index(drop=True)
    df = engineer.impute_missing(df, feature_cols)

    X = df[feature_cols]
    y_temp = df[temp_target]
    y_precip = df[precip_target]

    temp_model = TemperaturePredictor()
    precip_model = PrecipitationPredictor()
    calibration = CalibrationManager()

    temp_metrics = temp_model.train(X, y_temp, feature_names=feature_cols)
    precip_metrics = precip_model.train(X, y_precip, feature_names=feature_cols)
    _fit_calibration(temp_model, precip_model, calibration, X, y_temp, y_precip)

    temp_model.save()
    precip_model.save()
    calibration.save()

    processed_path = PROCESSED_DIR / "training_features_nwp.parquet"
    df.to_parquet(processed_path, index=False)

    return {
        "temperature": temp_metrics,
        "precipitation": precip_metrics,
        "n_features": len(feature_cols),
        "n_samples": len(df),
        "nwp_training_aware": True,
        "nwp_feature_count": sum("_model_" in name for name in feature_cols),
        "daily_path": str(daily_path),
        "forecast_path": str(forecast_path),
        "processed_path": str(processed_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="重新训练 NWP-aware 上海天气模型")
    parser.add_argument("--daily-path", type=Path)
    parser.add_argument("--forecast-path", type=Path)
    args = parser.parse_args()

    result = train_nwp_models(args.daily_path, args.forecast_path)
    print(result)


if __name__ == "__main__":
    main()
