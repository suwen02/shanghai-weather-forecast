# -*- coding: utf-8 -*-
"""
上海天气预报主管线。

训练协议：历史观测先构建因果状态特征，再与 Open-Meteo Previous Runs
固定 day0–day6 提前量的多模型共识特征对齐。在线推理使用相同的 NWP 共识列和
forecast_lead_days，避免未来 7 天因为复制最近历史状态而得到相同输入。
"""

import logging
import json
from datetime import date, timedelta, datetime
from typing import Dict, Optional

import pandas as pd

from config.settings import (
    RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR,
    ML_CONFIG, CITY_NAME, CITY_NAME_EN,
    TEMP_MODEL_PATH, PRECIP_MODEL_PATH, CALIBRATION_PATH,
)
from collectors.open_meteo import OpenMeteoCollector, collect_training_history
from collectors.cma_stations import CMAStationCollector, collect_station_history
from collectors.training_forecasts import collect_training_forecasts
from features.history_window import required_history_days
from features.nwp_aware_engineer import NwpAwareFeatureEngineer
from features.nwp_fallback import build_nwp_consensus_fallback
from models.temperature import TemperaturePredictor
from models.precipitation import PrecipitationPredictor
from models.calibration import CalibrationManager

logger = logging.getLogger(__name__)


class WeatherPipeline:
    """编排数据采集、训练、校准和在线预测。"""

    def __init__(self):
        self.engineer = NwpAwareFeatureEngineer()
        self.temp_model = TemperaturePredictor()
        self.precip_model = PrecipitationPredictor()
        self.calibration = CalibrationManager()
        self.collector = OpenMeteoCollector()
        self.station_collector = CMAStationCollector()

    def step1_collect_history(self, years: int = 5, station_years: int = 3) -> Dict:
        """采集观测历史、多站点历史和固定 lead Previous Runs。"""
        logger.info("=== 步骤1: 采集历史训练数据 ===")
        results = {}

        center_files = collect_training_history(years)
        results.update(center_files)

        station_files = collect_station_history(station_years)
        results.update(station_files)

        previous_runs_path = collect_training_forecasts(years)
        if previous_runs_path is not None:
            results["historical_previous_runs"] = str(previous_runs_path)

        logger.info("历史训练数据采集完成: %s个文件", len(results))
        return results

    def _latest_raw_file(self, pattern: str):
        files = sorted(
            RAW_DIR.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return files[0] if files else None

    def step2_train_models(self) -> Dict:
        """使用固定 lead NWP 共识特征训练温度和降水模型。"""
        logger.info("=== 步骤2: 训练 lead-aware 模型 ===")

        daily_path = self._latest_raw_file("historical_daily_*.parquet")
        if daily_path is None:
            raise FileNotFoundError("未找到 historical_daily_*.parquet")
        previous_runs_path = self._latest_raw_file("historical_previous_runs_*.parquet")
        if previous_runs_path is None:
            raise FileNotFoundError(
                "未找到 historical_previous_runs_*.parquet；请先运行 init 或 collect_training_forecasts"
            )

        historical = pd.read_parquet(daily_path)
        previous_runs = pd.read_parquet(previous_runs_path)
        logger.info(
            "加载训练数据: observations=%s, previous_runs=%s",
            len(historical),
            len(previous_runs),
        )

        df, feature_cols, temp_target, precip_target = self.engineer.build_training_features(
            historical,
            previous_runs,
        )
        if not self.engineer.has_nwp_training_features(feature_cols):
            raise RuntimeError(
                "训练特征必须同时包含 forecast_lead_days 与 NWP 共识列；拒绝生成 legacy 模型"
            )

        valid_mask = df[temp_target].notna() & df[precip_target].notna()
        df = df[valid_mask].sort_values(["time", "forecast_lead_days"]).reset_index(drop=True)
        if df.empty:
            raise RuntimeError("观测与 Previous Runs 没有可对齐训练样本")
        df = self.engineer.impute_missing(df, feature_cols)

        X = df[feature_cols]
        y_temp = df[temp_target]
        y_precip = df[precip_target]

        logger.info("训练温度模型...")
        temp_metrics = self.temp_model.train(X, y_temp, feature_names=feature_cols)
        logger.info("训练降水模型...")
        precip_metrics = self.precip_model.train(X, y_precip, feature_names=feature_cols)

        logger.info("拟合校准模型...")
        self._fit_calibration(X, y_temp, y_precip, feature_cols)

        self.temp_model.save()
        self.precip_model.save()
        self.calibration.save()

        processed_path = PROCESSED_DIR / "training_features_nwp_lead.parquet"
        df.to_parquet(processed_path, index=False)

        return {
            "temperature": temp_metrics,
            "precipitation": precip_metrics,
            "n_features": len(feature_cols),
            "n_samples": len(df),
            "nwp_training_aware": True,
            "nwp_feature_count": sum("_model_" in name for name in feature_cols),
            "lead_counts": {
                int(lead): int(count)
                for lead, count in df["forecast_lead_days"].value_counts().sort_index().items()
            },
            "previous_runs_path": str(previous_runs_path),
        }

    def _fit_calibration(
        self,
        X: pd.DataFrame,
        y_temp: pd.Series,
        y_precip: pd.Series,
        feature_cols: list,
    ):
        """在末段时间样本上拟合保形温度区间与降水等保序校准。"""
        n = len(X)
        cal_start = int(n * (1 - ML_CONFIG.calibration_fraction))
        X_cal = X.iloc[cal_start:]
        y_temp_cal = y_temp.iloc[cal_start:].values
        y_precip_cal = y_precip.iloc[cal_start:].values

        X_cal_aligned = self.temp_model._align_features(X_cal)
        X_cal_scaled = self.temp_model.scaler.transform(X_cal_aligned)
        quantile_preds_cal = {}
        for q in self.temp_model.quantiles:
            label = f"p{int(q*100):02d}"
            quantile_preds_cal[label] = self.temp_model.models[q].predict(X_cal_scaled)
        self.calibration.fit_conformal(y_temp_cal, quantile_preds_cal)

        X_precip_aligned = self.precip_model._align_features(X_cal)
        X_precip_scaled = self.precip_model.scaler.transform(X_precip_aligned)
        y_binary = (y_precip_cal >= ML_CONFIG.precip_occurrence_threshold).astype(int)
        p_raw = self.precip_model.classifier.predict_proba(X_precip_scaled)[:, 1]
        self.calibration.fit_isotonic(y_binary, p_raw)

    def _models_nwp_aware(self) -> bool:
        return (
            self.engineer.has_nwp_training_features(self.temp_model.feature_names)
            and self.engineer.has_nwp_training_features(self.precip_model.feature_names)
        )

    def step3_daily_predict(self, target_date: Optional[date] = None) -> Dict:
        """采集最新 NWP，并生成 7 天 lead-aware 预测。"""
        if target_date is None:
            target_date = date.today()
        logger.info("=== 步骤3: 每日预测 (%s) ===", target_date)

        self._load_models()

        logger.info("采集当日预报数据...")
        det_df = self.collector.collect_deterministic_forecasts(target_date)
        ens_df = self.collector.collect_ensemble_summary()
        station_df = self.station_collector.collect_station_forecasts()

        if not self._models_nwp_aware():
            logger.warning(
                "检测到 legacy 模型 artifact：缺少 forecast_lead_days/NWP 特征，改用未校准 NWP 共识 fallback"
            )
            output = self._nwp_consensus_fallback(det_df, target_date)
            self._save_predictions(output, target_date)
            return output

        history_days = required_history_days(
            ML_CONFIG.lag_days,
            ML_CONFIG.rolling_windows,
        )
        logger.info("获取近期历史观测: %s天...", history_days)
        recent_start = target_date - timedelta(days=history_days)
        recent_end = target_date - timedelta(days=1)
        recent = self.collector.collect_historical_data(recent_start, recent_end)
        recent_daily = recent.get("daily", pd.DataFrame())

        logger.info("构建未来 NWP + 因果状态特征...")
        pred_features = self.engineer.build_prediction_features(
            det_df, ens_df, station_df, recent_daily
        )
        if pred_features.empty:
            logger.error("预测特征构建失败")
            return {}

        pred_features = self.engineer.impute_missing(
            pred_features,
            self.engineer.feature_cols,
        )
        n_pred = min(len(pred_features), ML_CONFIG.forecast_horizon)
        X_pred = pred_features.head(n_pred)
        forecast_dates = pd.to_datetime(X_pred["time"]).dt.date.astype(str).tolist()
        forecast_leads = X_pred["forecast_lead_days"].astype(int).tolist()

        logger.info("生成温度预测...")
        temp_results = self.temp_model.predict(X_pred, forecast_dates)
        for result in temp_results:
            result.quantiles = self.calibration.calibrate_temperature_prediction(
                result.quantiles
            )

        logger.info("生成降水预测...")
        precip_results = self.precip_model.predict(X_pred, forecast_dates)
        for result in precip_results:
            p_rain = result.quantiles.get(
                "p_rain",
                result.distribution_params.get("p_rain_occurrence", 0),
            )
            calibrated_p = self.calibration.calibrate_precipitation_prediction(p_rain)
            result.quantiles["p_rain"] = round(calibrated_p, 4)
            result.distribution_params["p_rain_occurrence"] = round(calibrated_p, 4)
            result.distribution_params["p_dry"] = round(1 - calibrated_p, 4)

        output = self._format_predictions(
            temp_results,
            precip_results,
            target_date,
            forecast_leads,
        )
        output["nwp_training_aware"] = True
        output["forecast_lead_days"] = forecast_leads
        self._save_predictions(output, target_date)
        logger.info("每日预测完成: %s天", n_pred)
        return output

    def _nwp_consensus_fallback(self, det_df: pd.DataFrame, report_date: date) -> Dict:
        """Legacy artifact 期间发布当前 NWP 共识，避免重复 ML 预测。"""
        consensus = self.engineer.build_model_consensus_features(det_df)
        return build_nwp_consensus_fallback(
            det_df=det_df,
            consensus=consensus,
            report_date=report_date,
            horizon=ML_CONFIG.forecast_horizon,
            precipitation_threshold=ML_CONFIG.precip_occurrence_threshold,
            city_name=CITY_NAME,
            city_name_en=CITY_NAME_EN,
        )

    def _load_models(self):
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

        names = list(dict.fromkeys([
            *getattr(self.temp_model, "feature_names", []),
            *getattr(self.precip_model, "feature_names", []),
        ]))
        self.engineer.feature_cols = names

    def _format_predictions(
        self,
        temp_results,
        precip_results,
        report_date,
        forecast_leads=None,
    ) -> Dict:
        output = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": CITY_NAME,
            "city_en": CITY_NAME_EN,
            "source": "ml_postprocessed_nwp",
            "calibrated": True,
            "temperature": [],
            "precipitation": [],
        }
        lead_values = list(forecast_leads or [])
        for index, result in enumerate(temp_results):
            lead_days = lead_values[index] if index < len(lead_values) else index
            output["temperature"].append({
                "date": result.target_date,
                "lead_days": int(lead_days),
                "median": result.point_estimate,
                "quantiles": result.quantiles,
                "confidence": result.confidence,
            })
        for index, result in enumerate(precip_results):
            lead_days = lead_values[index] if index < len(lead_values) else index
            output["precipitation"].append({
                "date": result.target_date,
                "lead_days": int(lead_days),
                "expected_mm": result.point_estimate,
                "quantiles": result.quantiles,
                "params": result.distribution_params,
                "confidence": result.confidence,
            })
        return output

    def _save_predictions(self, output: Dict, report_date: date):
        date_str = report_date.strftime("%Y%m%d")
        pred_path = PREDICTIONS_DIR / f"predictions_{date_str}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("预测结果已保存: %s", pred_path)

    def run(self, mode: str = "full", **kwargs) -> Dict:
        results = {}
        if mode in ("init", "full"):
            results["history"] = self.step1_collect_history(
                years=kwargs.get("years", ML_CONFIG.historical_years),
                station_years=kwargs.get("station_years", ML_CONFIG.station_historical_years),
            )
            results["training"] = self.step2_train_models()
        elif mode == "train":
            results["training"] = self.step2_train_models()

        if mode in ("predict", "full"):
            results["prediction"] = self.step3_daily_predict(
                target_date=kwargs.get("target_date"),
            )
        return results
