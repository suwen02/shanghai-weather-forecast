# -*- coding: utf-8 -*-
"""
完整管线运行器

端到端执行：
1. collect_5year_history()   — 采集5年逐日+逐小时历史数据
2. collect_multistation()    — 采集3年30站点历史数据
3. collect_todays_forecasts()— 采集当日确定性+集合+站点预报
4. train_with_accuracy_eval()— 训练/测试划分，全面评估
5. generate_accuracy_report()— 8面板精度可视化
6. make_today_prediction()   — 生成并保存7天预报

运行模式：
  python run_full_pipeline.py                    # 完整：历史+训练+预测+评估
  python run_full_pipeline.py --mode init        # 仅历史+训练
  python run_full_pipeline.py --mode predict     # 仅每日预测
  python run_full_pipeline.py --mode evaluate    # 仅评估
"""

import sys
import logging
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_full_pipeline")


def collect_5year_history():
    """采集5年历史观测数据"""
    logger.info("=" * 70)
    logger.info("步骤1: 采集5年历史观测数据")
    logger.info("=" * 70)

    from collectors.open_meteo import collect_training_history
    files = collect_training_history(years=5)
    logger.info(f"历史数据文件: {files}")
    return files


def collect_multistation_history():
    """采集3年多站点历史数据"""
    logger.info("=" * 70)
    logger.info("步骤2: 采集3年多站点历史数据 (30个站点)")
    logger.info("=" * 70)

    from collectors.cma_stations import collect_station_history
    files = collect_station_history(years=3)
    logger.info(f"多站点历史文件: {files}")
    return files


def collect_todays_forecasts():
    """采集当日预报数据"""
    logger.info("=" * 70)
    logger.info("步骤3: 采集当日预报数据")
    logger.info("=" * 70)

    from collectors.open_meteo import run_daily_collection
    from collectors.cma_stations import run_station_collection

    det_files = run_daily_collection()
    station_files = run_station_collection()

    files = {**det_files, **station_files}
    logger.info(f"当日预报文件: {files}")
    return files


def train_with_accuracy_eval():
    """训练模型并进行全面精度评估"""
    logger.info("=" * 70)
    logger.info("步骤4: 训练模型 + 精度评估")
    logger.info("=" * 70)

    from config.settings import RAW_DIR, ML_CONFIG, MODELS_DIR
    from features.engineer import FeatureEngineer
    from models.temperature import TemperaturePredictor
    from models.precipitation import PrecipitationPredictor
    from models.calibration import CalibrationManager

    # 加载历史数据
    daily_path = list(RAW_DIR.glob("historical_daily_*.parquet"))
    if not daily_path:
        logger.error("未找到历史逐日数据，请先运行数据采集")
        return None

    historical = pd.read_parquet(daily_path[0])
    logger.info(f"历史数据: {len(historical)}天")

    # 特征工程
    engineer = FeatureEngineer()
    df, feature_cols, temp_target, precip_target = engineer.build_training_features(historical)

    # 去除缺失目标
    valid_mask = df[temp_target].notna() & df[precip_target].notna()
    df = df[valid_mask].reset_index(drop=True)
    df = engineer.impute_missing(df, feature_cols)

    logger.info(f"有效样本: {len(df)}, 特征数: {len(feature_cols)}")

    # 训练/测试划分（最后365天为测试集）
    test_days = ML_CONFIG.validation_days
    n = len(df)
    train_end = n - test_days
    if train_end < 100:
        train_end = int(n * 0.7)
        test_days = n - train_end

    df_train = df.iloc[:train_end]
    df_test = df.iloc[train_end:]

    X_train = df_train[feature_cols]
    y_temp_train = df_train[temp_target]
    y_precip_train = df_train[precip_target]

    X_test = df_test[feature_cols]
    y_temp_test = df_test[temp_target].values
    y_precip_test = df_test[precip_target].values
    test_dates = df_test["time"]

    logger.info(f"训练集: {len(df_train)}天, 测试集: {len(df_test)}天")

    # 训练温度模型
    logger.info("训练温度模型...")
    temp_model = TemperaturePredictor()
    temp_metrics = temp_model.train(X_train, y_temp_train, feature_names=feature_cols)

    # 训练降水模型
    logger.info("训练降水模型...")
    precip_model = PrecipitationPredictor()
    precip_metrics = precip_model.train(X_train, y_precip_train, feature_names=feature_cols)

    # 校准
    logger.info("拟合校准模型...")
    calibration = CalibrationManager()
    cal_start = int(len(X_train) * 0.8)
    X_cal = X_train.iloc[cal_start:]
    y_temp_cal = y_temp_train.iloc[cal_start:].values
    y_precip_cal = y_precip_train.iloc[cal_start:].values

    X_cal_scaled = temp_model.scaler.transform(temp_model._align_features(X_cal))
    q_preds_cal = {}
    for q in temp_model.quantiles:
        label = f"p{int(q*100):02d}"
        q_preds_cal[label] = temp_model.models[q].predict(X_cal_scaled)
    calibration.fit_conformal(y_temp_cal, q_preds_cal)

    X_precip_cal_scaled = precip_model.scaler.transform(precip_model._align_features(X_cal))
    y_binary_cal = (y_precip_cal >= 0.1).astype(int)
    p_raw_cal = precip_model.classifier.predict_proba(X_precip_cal_scaled)[:, 1]
    calibration.fit_isotonic(y_binary_cal, p_raw_cal)

    # 测试集评估
    logger.info("在测试集上评估...")
    X_test_scaled = temp_model.scaler.transform(temp_model._align_features(X_test))
    y_temp_pred = temp_model.models[0.50].predict(X_test_scaled)

    # 分位数预测
    quantile_preds_test = {}
    for q in temp_model.quantiles:
        label = f"p{int(q*100):02d}"
        quantile_preds_test[label] = temp_model.models[q].predict(X_test_scaled)

    # 温度指标
    mae = np.mean(np.abs(y_temp_test - y_temp_pred))
    rmse = np.sqrt(np.mean((y_temp_test - y_temp_pred) ** 2))
    bias = np.mean(y_temp_pred - y_temp_test)
    ss_res = np.sum((y_temp_test - y_temp_pred) ** 2)
    ss_tot = np.sum((y_temp_test - y_temp_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    within_1 = np.mean(np.abs(y_temp_test - y_temp_pred) <= 1)
    within_2 = np.mean(np.abs(y_temp_test - y_temp_pred) <= 2)
    within_3 = np.mean(np.abs(y_temp_test - y_temp_pred) <= 3)

    # 覆盖率
    coverages = {}
    for name, lo_q, hi_q in [("50%", "p25", "p75"), ("80%", "p10", "p90"), ("90%", "p05", "p95")]:
        lo = quantile_preds_test[lo_q]
        hi = quantile_preds_test[hi_q]
        cov = np.mean((y_temp_test >= lo) & (y_temp_test <= hi))
        coverages[name] = round(float(cov), 4)
        width = np.mean(hi - lo)
        logger.info(f"  {name}覆盖率: {cov:.1%} (宽度: {width:.2f}°C)")

    # CRPS估算
    pinball_losses = []
    for q in temp_model.quantiles:
        label = f"p{int(q*100):02d}"
        pl = np.mean(np.where(
            y_temp_test >= quantile_preds_test[label],
            q * (y_temp_test - quantile_preds_test[label]),
            (q - 1) * (y_temp_test - quantile_preds_test[label]),
        ))
        pinball_losses.append(pl)
    crps = 2 * np.mean(pinball_losses)

    # 降水评估
    X_precip_test_scaled = precip_model.scaler.transform(precip_model._align_features(X_test))
    p_rain_test = precip_model.classifier.predict_proba(X_precip_test_scaled)[:, 1]
    y_precip_binary = (y_precip_test >= 0.1).astype(int)
    pred_binary = (p_rain_test >= 0.5).astype(int)

    tp = np.sum((pred_binary == 1) & (y_precip_binary == 1))
    tn = np.sum((pred_binary == 0) & (y_precip_binary == 0))
    fp = np.sum((pred_binary == 1) & (y_precip_binary == 0))
    fn = np.sum((pred_binary == 0) & (y_precip_binary == 1))

    precip_acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precip_prec = tp / max(tp + fp, 1)
    precip_rec = tp / max(tp + fn, 1)
    precip_f1 = 2 * precip_prec * precip_rec / max(precip_prec + precip_rec, 1e-8)
    brier = np.mean((p_rain_test - y_precip_binary) ** 2)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_precip_binary, p_rain_test)
    except Exception:
        auc = 0.0

    # 打印评估结果
    logger.info("=" * 70)
    logger.info("温度预测评估结果 (测试集)")
    logger.info("=" * 70)
    logger.info(f"  MAE:    {mae:.3f}°C")
    logger.info(f"  RMSE:   {rmse:.3f}°C")
    logger.info(f"  R²:     {r2:.4f}")
    logger.info(f"  偏差:   {bias:+.3f}°C")
    logger.info(f"  CRPS:   {crps:.4f}")
    logger.info(f"  ±1°C:   {within_1:.1%}")
    logger.info(f"  ±2°C:   {within_2:.1%}")
    logger.info(f"  ±3°C:   {within_3:.1%}")
    for name, cov in coverages.items():
        logger.info(f"  {name}覆盖率: {cov:.1%}")

    logger.info("=" * 70)
    logger.info("降水预测评估结果 (测试集)")
    logger.info("=" * 70)
    logger.info(f"  准确率:  {precip_acc:.1%}")
    logger.info(f"  精确率:  {precip_prec:.1%}")
    logger.info(f"  召回率:  {precip_rec:.1%}")
    logger.info(f"  F1:     {precip_f1:.3f}")
    logger.info(f"  AUC-ROC: {auc:.3f}")
    logger.info(f"  Brier:  {brier:.4f}")

    # 保存模型
    temp_model.save()
    precip_model.save()
    calibration.save()

    return {
        "temp_model": temp_model,
        "precip_model": precip_model,
        "calibration": calibration,
        "engineer": engineer,
        "feature_cols": feature_cols,
        "y_temp_test": y_temp_test,
        "y_temp_pred": y_temp_pred,
        "quantile_preds_test": quantile_preds_test,
        "y_precip_test": y_precip_test,
        "p_rain_test": p_rain_test,
        "test_dates": test_dates,
        "metrics": {
            "temperature": {
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "r2": round(float(r2), 4),
                "bias": round(float(bias), 4),
                "crps": round(float(crps), 4),
                "within_1c": round(float(within_1), 4),
                "within_2c": round(float(within_2), 4),
                "within_3c": round(float(within_3), 4),
                "coverages": coverages,
            },
            "precipitation": {
                "accuracy": round(float(precip_acc), 4),
                "precision": round(float(precip_prec), 4),
                "recall": round(float(precip_rec), 4),
                "f1": round(float(precip_f1), 4),
                "auc_roc": round(float(auc), 4),
                "brier_score": round(float(brier), 4),
            },
        },
    }


def generate_accuracy_report(results):
    """生成8面板精度评估报告"""
    logger.info("=" * 70)
    logger.info("步骤5: 生成精度评估报告")
    logger.info("=" * 70)

    from src.visualizer import WeatherVisualizer

    visualizer = WeatherVisualizer()
    path = visualizer.generate_accuracy_report(
        y_true=results["y_temp_test"],
        y_pred=results["y_temp_pred"],
        quantile_preds=results["quantile_preds_test"],
        precip_true=results["y_precip_test"],
        precip_prob=results["p_rain_test"],
        feature_importance=results["temp_model"].get_feature_importance(),
        dates=results["test_dates"],
    )
    logger.info(f"精度报告: {path}")
    return path


def make_today_prediction(results=None):
    """生成今日7天预报"""
    logger.info("=" * 70)
    logger.info("步骤6: 生成7天预报")
    logger.info("=" * 70)

    from src.pipeline import WeatherPipeline
    from src.visualizer import WeatherVisualizer

    pipeline = WeatherPipeline()
    prediction = pipeline.step3_daily_predict()

    if prediction:
        visualizer = WeatherVisualizer()
        temp_preds = prediction.get("temperature", [])
        precip_preds = prediction.get("precipitation", [])

        if temp_preds:
            visualizer.plot_temperature_distribution(temp_preds)
        if precip_preds:
            visualizer.plot_precipitation_distribution(precip_preds)
        if temp_preds and precip_preds:
            visualizer.generate_daily_report(temp_preds, precip_preds)

    return prediction


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="上海天气预报ML系统 — 完整管线运行器"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "init", "predict", "evaluate"],
        default="full",
        help="运行模式: full(完整), init(历史+训练), predict(仅预测), evaluate(仅评估)"
    )
    args = parser.parse_args()

    logger.info(f"上海天气预报ML系统启动 — 模式: {args.mode}")
    logger.info(f"日期: {date.today().isoformat()}")

    if args.mode == "full":
        # 完整流程
        collect_5year_history()
        collect_multistation_history()
        collect_todays_forecasts()
        results = train_with_accuracy_eval()
        if results:
            generate_accuracy_report(results)
            make_today_prediction(results)

    elif args.mode == "init":
        # 初始化：采集历史+训练
        collect_5year_history()
        collect_multistation_history()
        results = train_with_accuracy_eval()
        if results:
            generate_accuracy_report(results)

    elif args.mode == "predict":
        # 仅每日预测
        make_today_prediction()

    elif args.mode == "evaluate":
        # 仅评估
        results = train_with_accuracy_eval()
        if results:
            generate_accuracy_report(results)

    logger.info("管线运行完毕")


if __name__ == "__main__":
    main()
