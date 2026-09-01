# -*- coding: utf-8 -*-
"""
主编排管线

四种运行模式：
- init:    采集历史数据 + 训练模型
- train:   仅训练模型
- predict: 每日预测（采集当日数据→加载模型→预测→可视化）
- full:    init + predict

管线流程：
1. 采集历史数据（5年中心站+3年30站）
2. 特征工程（约150个特征）
3. 训练温度/降水模型
4. 校准（保形预测+等保序回归）
5. 每日预测7天
6. 生成可视化报告
"""

import logging
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from config.settings import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR,
    ML_CONFIG, TIMEZONE, CITY_NAME, CITY_NAME_EN,
    TEMP_MODEL_PATH, PRECIP_MODEL_PATH, CALIBRATION_PATH,
)
from collectors.open_meteo import OpenMeteoCollector, run_daily_collection, collect_training_history
from collectors.cma_stations import CMAStationCollector, run_station_collection, collect_station_history
from features.engineer import FeatureEngineer
from features.prediction_frame import build_forecast_scaffold
from models.temperature import TemperaturePredictor
from models.precipitation import PrecipitationPredictor
from models.calibration import CalibrationManager
from src.realtime import (
    RefreshStateStore, atomic_publish_json, build_short_term_forecast,
    fetch_latest_snapshot, snapshot_fingerprint,
)

logger = logging.getLogger(__name__)


class WeatherPipeline:
    """
    天气预报主管线

    编排数据采集、特征工程、模型训练、预测和报告生成。
    """

    def __init__(self):
        self.engineer = FeatureEngineer()
        self.temp_model = TemperaturePredictor()
        self.precip_model = PrecipitationPredictor()
        self.calibration = CalibrationManager()
        self.collector = OpenMeteoCollector()
        self.station_collector = CMAStationCollector()

    # =========================================================================
    # 步骤1: 采集历史数据
    # =========================================================================
    def step1_collect_history(self, years: int = 5, station_years: int = 3) -> Dict:
        """
        采集训练用历史数据

        Args:
            years: 中心站历史年数
            station_years: 多站点历史年数

        Returns:
            采集结果字典
        """
        logger.info(f"=== 步骤1: 采集历史数据 ({years}年) ===")
        results = {}

        # 中心站历史
        center_files = collect_training_history(years)
        results.update(center_files)

        # 多站点历史
        station_files = collect_station_history(station_years)
        results.update(station_files)

        logger.info(f"历史数据采集完成: {len(results)}个文件")
        return results

    # =========================================================================
    # 步骤2: 训练模型
    # =========================================================================
    def step2_train_models(self) -> Dict:
        """
        训练温度和降水模型

        从Parquet文件加载历史数据，构建特征，训练模型，
        在校准集上拟合保形预测和等保序回归。

        Returns:
            训练指标字典
        """
        logger.info("=== 步骤2: 训练模型 ===")

        # 加载历史数据
        daily_path = RAW_DIR / "historical_daily_5yr.parquet"
        if not daily_path.exists():
            # 尝试其他文件名
            parquet_files = list(RAW_DIR.glob("historical_daily_*.parquet"))
            if parquet_files:
                daily_path = parquet_files[0]
            else:
                raise FileNotFoundError(f"未找到历史逐日数据: {daily_path}")

        historical = pd.read_parquet(daily_path)
        logger.info(f"加载历史数据: {len(historical)}行")

        # 特征工程
        df, feature_cols, temp_target, precip_target = self.engineer.build_training_features(
            historical
        )

        # 去除缺失目标的行
        valid_mask = df[temp_target].notna() & df[precip_target].notna()
        df = df[valid_mask].reset_index(drop=True)
        logger.info(f"有效训练样本: {len(df)}行, {len(feature_cols)}个特征")

        # 填充缺失值
        df = self.engineer.impute_missing(df, feature_cols)

        X = df[feature_cols]
        y_temp = df[temp_target]
        y_precip = df[precip_target]

        # 训练温度模型
        logger.info("训练温度模型...")
        temp_metrics = self.temp_model.train(X, y_temp, feature_names=feature_cols)

        # 训练降水模型
        logger.info("训练降水模型...")
        precip_metrics = self.precip_model.train(X, y_precip, feature_names=feature_cols)

        # 校准
        logger.info("拟合校准模型...")
        self._fit_calibration(X, y_temp, y_precip, feature_cols)

        # 保存模型
        self.temp_model.save()
        self.precip_model.save()
        self.calibration.save()

        # 保存处理后数据
        processed_path = PROCESSED_DIR / "training_features.parquet"
        df.to_parquet(processed_path, index=False)
        logger.info(f"训练特征已保存: {processed_path}")

        return {
            "temperature": temp_metrics,
            "precipitation": precip_metrics,
            "n_features": len(feature_cols),
            "n_samples": len(df),
        }

    def _fit_calibration(
        self,
        X: pd.DataFrame,
        y_temp: pd.Series,
        y_precip: pd.Series,
        feature_cols: list,
    ):
        """在校准子集上拟合校准模型"""
        n = len(X)
        cal_start = int(n * (1 - ML_CONFIG.calibration_fraction))

        X_cal = X.iloc[cal_start:]
        y_temp_cal = y_temp.iloc[cal_start:].values
        y_precip_cal = y_precip.iloc[cal_start:].values

        # 温度保形预测
        X_cal_aligned = self.temp_model._align_features(X_cal)
        X_cal_scaled = self.temp_model.scaler.transform(X_cal_aligned)

        quantile_preds_cal = {}
        for q in self.temp_model.quantiles:
            label = f"p{int(q*100):02d}"
            quantile_preds_cal[label] = self.temp_model.models[q].predict(X_cal_scaled)

        self.calibration.fit_conformal(y_temp_cal, quantile_preds_cal)

        # 降水等保序回归
        X_precip_aligned = self.precip_model._align_features(X_cal)
        X_precip_scaled = self.precip_model.scaler.transform(X_precip_aligned)

        y_binary = (y_precip_cal >= ML_CONFIG.precip_occurrence_threshold).astype(int)
        p_raw = self.precip_model.classifier.predict_proba(X_precip_scaled)[:, 1]
        self.calibration.fit_isotonic(y_binary, p_raw)

    # =========================================================================
    # 步骤3: 每日预测
    # =========================================================================
    def step3_daily_predict(
        self,
        target_date: Optional[date] = None,
        realtime_snapshot: Optional[Dict] = None,
    ) -> Dict:
        """
        执行每日预测

        1. 采集当日预报数据
        2. 加载训练好的模型
        3. 获取近期历史观测
        4. 构建预测特征
        5. 生成7天温度+降水概率预报
        6. 保存JSON预测结果

        Args:
            target_date: 预测基准日期

        Returns:
            预测结果字典
        """
        if target_date is None:
            target_date = date.today()

        logger.info(f"=== 步骤3: 每日预测 ({target_date}) ===")

        # 加载模型
        self._load_models()

        # 采集当日数据
        logger.info("采集当日预报数据...")
        det_df = self.collector.collect_deterministic_forecasts(target_date)
        ens_df = self.collector.collect_ensemble_summary()
        station_df = self.station_collector.collect_station_forecasts()

        # 短临快照使用 Open-Meteo 当前最新可用 best_match 数据。
        if realtime_snapshot is None:
            realtime_snapshot = fetch_latest_snapshot(self.collector)

        # 最长滚动窗口已扩展到90天，预留120天确保滞后/滚动特征完整。
        logger.info("获取近期历史观测...")
        recent_start = target_date - timedelta(days=120)
        recent_end = target_date - timedelta(days=1)
        recent = self.collector.collect_historical_data(recent_start, recent_end)
        recent_daily = recent.get("daily", pd.DataFrame())

        # 先生成历史状态特征，再以未来 NWP 日期为主表构造真正的预测行。
        logger.info("构建预测特征...")
        history_features = self.engineer.build_prediction_features(
            det_df, ens_df, station_df, recent_daily
        )
        consensus = self.engineer.build_model_consensus_features(det_df)
        ens_features = self.engineer.build_ensemble_features(ens_df)
        spatial = self.engineer.build_station_spatial_features(station_df)
        pred_features = build_forecast_scaffold(
            history_features, consensus, ens_features, spatial
        )

        if pred_features.empty:
            logger.error("预测特征构建失败：未得到未来NWP日期行")
            return {}

        # 历史状态列会携带到未来行，但时间/上海季节特征必须按目标日重算。
        pred_features = self.engineer.add_temporal_features(pred_features)
        pred_features = self.engineer.add_shanghai_features(pred_features)
        pred_features = self.engineer.impute_missing(pred_features, self.engineer.feature_cols)

        n_pred = min(len(pred_features), ML_CONFIG.forecast_horizon)
        X_pred = pred_features.head(n_pred)
        forecast_dates = pd.to_datetime(X_pred["time"]).dt.strftime("%Y-%m-%d").tolist()

        # 温度预测
        logger.info("生成温度预测...")
        temp_results = self.temp_model.predict(X_pred, forecast_dates[:n_pred])

        # 应用保形校准
        for result in temp_results:
            result.quantiles = self.calibration.calibrate_temperature_prediction(
                result.quantiles
            )

        # 降水预测
        logger.info("生成降水预测...")
        precip_results = self.precip_model.predict(X_pred, forecast_dates[:n_pred])

        # 应用等保序校准
        for result in precip_results:
            p_rain = result.quantiles.get("p_rain", result.distribution_params.get("p_rain_occurrence", 0))
            calibrated_p = self.calibration.calibrate_precipitation_prediction(p_rain)
            result.quantiles["p_rain"] = round(calibrated_p, 4)
            result.distribution_params["p_rain_occurrence"] = round(calibrated_p, 4)
            result.distribution_params["p_dry"] = round(1 - calibrated_p, 4)

        # 附加基于最新数据的48小时短临预报和新鲜度信息。
        output = self._format_predictions(temp_results, precip_results, target_date)
        short_term = build_short_term_forecast(
            realtime_snapshot, timezone=TIMEZONE, horizon_hours=48
        )
        output["data_as_of"] = short_term["data_as_of"]
        output["data_age_minutes"] = short_term["data_age_minutes"]
        output["is_stale"] = short_term["is_stale"]
        output["short_term"] = short_term["hours"]
        self._save_predictions(output, target_date)

        logger.info(f"每日预测完成: {len(temp_results)}天温度 + {len(precip_results)}天降水")
        return output

    def step4_realtime_refresh(self, force: bool = False) -> Dict:
        """检查最新逐小时数据；仅在上游数据变化后重新生成完整预测。"""
        logger.info("=== 步骤4: 实时刷新检查 ===")
        snapshot = fetch_latest_snapshot(self.collector)
        fingerprint = snapshot_fingerprint(snapshot)
        state = RefreshStateStore(PREDICTIONS_DIR / ".refresh_state.json")
        freshness = build_short_term_forecast(snapshot, timezone=TIMEZONE, horizon_hours=48)

        if not force and not state.should_refresh(fingerprint):
            logger.info("上游天气数据未变化，跳过完整预测")
            return {
                "updated": False,
                "reason": "upstream_unchanged",
                "data_as_of": freshness["data_as_of"],
                "data_age_minutes": freshness["data_age_minutes"],
                "is_stale": freshness["is_stale"],
            }

        prediction = self.step3_daily_predict(realtime_snapshot=snapshot)
        if not prediction:
            logger.error("实时刷新预测失败，不更新指纹状态")
            return {"updated": False, "reason": "prediction_failed"}

        state.mark_refreshed(fingerprint, prediction.get("generated_at", ""))
        return {"updated": True, "prediction": prediction}

    def _load_models(self):
        """加载训练好的模型"""
        if TEMP_MODEL_PATH.exists():
            self.temp_model.load()
        else:
            logger.warning("温度模型文件不存在，需要先训练")

        if PRECIP_MODEL_PATH.exists():
            self.precip_model.load()
        else:
            logger.warning("降水模型文件不存在，需要先训练")

        if CALIBRATION_PATH.exists():
            self.calibration.load()

    def _format_predictions(self, temp_results, precip_results, report_date) -> Dict:
        """格式化预测结果为JSON结构"""
        from datetime import datetime

        output = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": CITY_NAME,
            "city_en": CITY_NAME_EN,
            "temperature": [],
            "precipitation": [],
        }

        for result in temp_results:
            output["temperature"].append({
                "date": result.target_date,
                "median": result.point_estimate,
                "quantiles": result.quantiles,
                "confidence": result.confidence,
            })

        for result in precip_results:
            output["precipitation"].append({
                "date": result.target_date,
                "expected_mm": result.point_estimate,
                "quantiles": result.quantiles,
                "params": result.distribution_params,
                "confidence": result.confidence,
            })

        return output

    def _save_predictions(self, output: Dict, report_date: date):
        """原子发布版本文件、当日兼容文件和latest.json。"""
        date_str = report_date.strftime("%Y%m%d")
        time_str = pd.Timestamp.now(tz=TIMEZONE).strftime("%H%M%S")
        versioned_path = PREDICTIONS_DIR / f"predictions_{date_str}_{time_str}.json"
        daily_path = PREDICTIONS_DIR / f"predictions_{date_str}.json"
        latest_path = PREDICTIONS_DIR / "latest.json"

        atomic_publish_json(output, versioned_path, latest_path)
        atomic_publish_json(output, daily_path, latest_path)
        logger.info(
            f"预测结果已发布: {versioned_path}; latest={latest_path}"
        )

    # =========================================================================
    # 运行模式
    # =========================================================================
    def run(self, mode: str = "full", **kwargs) -> Dict:
        """
        运行管线

        Args:
            mode: 运行模式 (init/train/predict/full/evaluate)
            **kwargs: 额外参数

        Returns:
            运行结果字典
        """
        results = {}

        if mode in ("init", "full"):
            results["history"] = self.step1_collect_history(
                years=kwargs.get("years", ML_CONFIG.historical_years),
                station_years=kwargs.get("station_years", ML_CONFIG.station_historical_years),
            )
            results["training"] = self.step2_train_models()

        elif mode == "train":
            results["training"] = self.step2_train_models()
        elif mode == "refresh":
            results["refresh"] = self.step4_realtime_refresh(
                force=kwargs.get("force", False),
            )

        if mode in ("predict", "full"):
            results["prediction"] = self.step3_daily_predict(
                target_date=kwargs.get("target_date"),
            )

        return results
