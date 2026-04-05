# -*- coding: utf-8 -*-
"""
超参数优化与模型对比

对比四种方法在365天保留测试集上的表现：

1. 基线: sklearn GradientBoostingRegressor
2. LightGBM + Optuna: 贝叶斯超参数搜索（CRPS目标）
3. Stacking: LightGBM + XGBoost + CatBoost 均值融合
4. Stack + Conformal: Stacking + 保形预测校准

严格防止数据泄漏：移除同日观测变量。

运行方式：
  python run_optimization.py               # 默认200次试验
  python run_optimization.py --trials 100  # 自定义试验数
"""

import sys
import json
import logging
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("optimization")


def pinball_loss(y_true, y_pred, quantile):
    """Pinball损失函数"""
    diff = y_true - y_pred
    return np.mean(np.where(diff >= 0, quantile * diff, (quantile - 1) * diff))


def crps_from_quantiles(y_true, quantile_preds, quantiles):
    """从分位数预测近似计算CRPS"""
    scores = []
    for q in quantiles:
        label = f"p{int(q*100):02d}"
        if label in quantile_preds:
            scores.append(pinball_loss(y_true, quantile_preds[label], q))
    return 2.0 * np.mean(scores) if scores else float("inf")


def load_and_prepare_data():
    """加载数据并准备特征"""
    from config.settings import RAW_DIR, ML_CONFIG, LEAKED_FEATURES
    from features.engineer import FeatureEngineer

    daily_path = list(RAW_DIR.glob("historical_daily_*.parquet"))
    if not daily_path:
        raise FileNotFoundError("未找到历史逐日数据")

    historical = pd.read_parquet(daily_path[0])
    engineer = FeatureEngineer()
    df, feature_cols, temp_target, precip_target = engineer.build_training_features(historical)

    # 移除泄漏特征
    feature_cols = [c for c in feature_cols if c not in LEAKED_FEATURES]

    valid_mask = df[temp_target].notna() & df[precip_target].notna()
    df = df[valid_mask].reset_index(drop=True)
    df = engineer.impute_missing(df, feature_cols)

    # 划分训练/测试
    test_days = min(ML_CONFIG.validation_days, len(df) // 3)
    train_end = len(df) - test_days

    X_train = df.iloc[:train_end][feature_cols]
    y_temp_train = df.iloc[:train_end][temp_target].values
    y_precip_train = df.iloc[:train_end][precip_target].values

    X_test = df.iloc[train_end:][feature_cols]
    y_temp_test = df.iloc[train_end:][temp_target].values
    y_precip_test = df.iloc[train_end:][precip_target].values

    logger.info(f"数据准备完成: 训练={len(X_train)}, 测试={len(X_test)}, 特征={len(feature_cols)}")
    return X_train, y_temp_train, y_precip_train, X_test, y_temp_test, y_precip_test, feature_cols


def evaluate_temperature(y_true, y_pred, quantile_preds, quantiles):
    """评估温度预测"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    within_1 = np.mean(np.abs(y_true - y_pred) <= 1)
    within_2 = np.mean(np.abs(y_true - y_pred) <= 2)
    within_3 = np.mean(np.abs(y_true - y_pred) <= 3)

    coverages = {}
    widths = {}
    for name, lo_q, hi_q in [("90%", "p05", "p95"), ("80%", "p10", "p90"), ("50%", "p25", "p75")]:
        if lo_q in quantile_preds and hi_q in quantile_preds:
            lo = quantile_preds[lo_q]
            hi = quantile_preds[hi_q]
            coverages[name] = float(np.mean((y_true >= lo) & (y_true <= hi)))
            widths[name] = float(np.mean(hi - lo))

    crps = crps_from_quantiles(y_true, quantile_preds, quantiles)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "bias": round(float(bias), 4),
        "crps": round(float(crps), 4),
        "within_1c": round(float(within_1), 4),
        "within_2c": round(float(within_2), 4),
        "within_3c": round(float(within_3), 4),
        "coverages": coverages,
        "widths": widths,
    }


def evaluate_precipitation(y_true, p_rain):
    """评估降水预测"""
    y_binary = (y_true >= 0.1).astype(int)
    pred_binary = (p_rain >= 0.5).astype(int)

    tp = np.sum((pred_binary == 1) & (y_binary == 1))
    tn = np.sum((pred_binary == 0) & (y_binary == 0))
    fp = np.sum((pred_binary == 1) & (y_binary == 0))
    fn = np.sum((pred_binary == 0) & (y_binary == 1))

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    brier = np.mean((p_rain - y_binary) ** 2)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_binary, p_rain)
    except Exception:
        auc = 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
    }


# =============================================================================
# 方法1: 基线 (sklearn GBR)
# =============================================================================
def run_baseline(X_train, y_temp_train, y_precip_train, X_test, y_temp_test, y_precip_test):
    """基线方法：sklearn GradientBoosting"""
    logger.info("=" * 60)
    logger.info("方法1: 基线 (sklearn GBR)")
    logger.info("=" * 60)

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    base_params = dict(n_estimators=200, max_depth=6, learning_rate=0.08,
                       min_samples_leaf=15, subsample=0.8, random_state=42)

    # 温度分位数回归
    q_preds = {}
    for q in quantiles:
        model = GradientBoostingRegressor(loss="quantile", alpha=q, **base_params)
        model.fit(X_train, y_temp_train)
        label = f"p{int(q*100):02d}"
        q_preds[label] = model.predict(X_test)

    y_pred = q_preds["p50"]
    temp_metrics = evaluate_temperature(y_temp_test, y_pred, q_preds, quantiles)

    # 降水分类
    y_binary_train = (y_precip_train >= 0.1).astype(int)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                                      min_samples_leaf=15, subsample=0.8, random_state=42)
    clf.fit(X_train, y_binary_train)
    p_rain = clf.predict_proba(X_test)[:, 1]
    precip_metrics = evaluate_precipitation(y_precip_test, p_rain)

    logger.info(f"温度 MAE={temp_metrics['mae']}°C, R²={temp_metrics['r2']}")
    logger.info(f"降水 准确率={precip_metrics['accuracy']}, F1={precip_metrics['f1']}")

    return {"temperature": temp_metrics, "precipitation": precip_metrics}


# =============================================================================
# 方法2: LightGBM + Optuna
# =============================================================================
def run_lightgbm_optuna(X_train, y_temp_train, y_precip_train,
                        X_test, y_temp_test, y_precip_test, n_trials=200):
    """LightGBM + Optuna 超参数优化"""
    logger.info("=" * 60)
    logger.info(f"方法2: LightGBM + Optuna ({n_trials}次试验)")
    logger.info("=" * 60)

    import optuna
    import lightgbm as lgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # Optuna温度优化
    def temp_objective(trial):
        params = {
            "objective": "quantile",
            "metric": "quantile",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        cv_losses = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_temp_train[train_idx]
            y_val = y_temp_train[val_idx]

            q_preds_val = {}
            for q in quantiles:
                p = params.copy()
                p["alpha"] = q
                model = lgb.LGBMRegressor(**p)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                label = f"p{int(q*100):02d}"
                q_preds_val[label] = model.predict(X_val)

            crps = crps_from_quantiles(y_val, q_preds_val, quantiles)
            cv_losses.append(crps)

        return np.mean(cv_losses)

    study = optuna.create_study(direction="minimize")
    study.optimize(temp_objective, n_trials=n_trials, timeout=600)

    best_params = study.best_params
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳CRPS: {study.best_value:.4f}")

    # 用最佳参数训练最终模型
    q_preds = {}
    for q in quantiles:
        params = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "boosting_type": "gbdt",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        params.update(best_params)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_temp_train)
        label = f"p{int(q*100):02d}"
        q_preds[label] = model.predict(X_test)

    y_pred = q_preds["p50"]
    temp_metrics = evaluate_temperature(y_temp_test, y_pred, q_preds, quantiles)

    # 降水分类
    y_binary_train = (y_precip_train >= 0.1).astype(int)
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, verbose=-1, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_binary_train)
    p_rain = clf.predict_proba(X_test)[:, 1]
    precip_metrics = evaluate_precipitation(y_precip_test, p_rain)

    logger.info(f"温度 MAE={temp_metrics['mae']}°C, R²={temp_metrics['r2']}")
    logger.info(f"降水 准确率={precip_metrics['accuracy']}, F1={precip_metrics['f1']}")

    return {
        "temperature": temp_metrics,
        "precipitation": precip_metrics,
        "best_params": best_params,
        "best_crps": round(float(study.best_value), 4),
    }


# =============================================================================
# 方法3: Stacking (LightGBM + XGBoost + CatBoost)
# =============================================================================
def run_stacking(X_train, y_temp_train, y_precip_train,
                 X_test, y_temp_test, y_precip_test):
    """三算法Stacking融合"""
    logger.info("=" * 60)
    logger.info("方法3: Stacking (LightGBM + XGBoost + CatBoost)")
    logger.info("=" * 60)

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor, CatBoostClassifier

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # 温度分位数回归
    q_preds = {}
    for q in quantiles:
        label = f"p{int(q*100):02d}"

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=500,
            learning_rate=0.05, verbose=-1, random_state=42, n_jobs=-1,
        )
        lgb_model.fit(X_train, y_temp_train)
        lgb_pred = lgb_model.predict(X_test)

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            n_estimators=500, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_model.fit(X_train, y_temp_train)
        xgb_pred = xgb_model.predict(X_test)

        # CatBoost
        cb_model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            iterations=500, learning_rate=0.05,
            random_seed=42, verbose=0,
        )
        cb_model.fit(X_train, y_temp_train)
        cb_pred = cb_model.predict(X_test)

        # 均值融合
        q_preds[label] = (lgb_pred + xgb_pred + cb_pred) / 3.0

    y_pred = q_preds["p50"]
    temp_metrics = evaluate_temperature(y_temp_test, y_pred, q_preds, quantiles)

    # 降水分类融合
    y_binary_train = (y_precip_train >= 0.1).astype(int)

    lgb_clf = lgb.LGBMClassifier(n_estimators=300, verbose=-1, random_state=42, n_jobs=-1)
    lgb_clf.fit(X_train, y_binary_train)
    lgb_p = lgb_clf.predict_proba(X_test)[:, 1]

    xgb_clf = xgb.XGBClassifier(n_estimators=300, verbosity=0, random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train, y_binary_train)
    xgb_p = xgb_clf.predict_proba(X_test)[:, 1]

    cb_clf = CatBoostClassifier(iterations=300, verbose=0, random_seed=42)
    cb_clf.fit(X_train, y_binary_train)
    cb_p = cb_clf.predict_proba(X_test)[:, 1]

    p_rain = (lgb_p + xgb_p + cb_p) / 3.0
    precip_metrics = evaluate_precipitation(y_precip_test, p_rain)

    logger.info(f"温度 MAE={temp_metrics['mae']}°C, R²={temp_metrics['r2']}")
    logger.info(f"降水 准确率={precip_metrics['accuracy']}, F1={precip_metrics['f1']}")

    return {
        "temperature": temp_metrics,
        "precipitation": precip_metrics,
        "q_preds": q_preds,
    }


# =============================================================================
# 方法4: Stack + Conformal
# =============================================================================
def run_stacking_conformal(X_train, y_temp_train, y_precip_train,
                           X_test, y_temp_test, y_precip_test):
    """Stacking + 保形预测校准"""
    logger.info("=" * 60)
    logger.info("方法4: Stack + Conformal Prediction")
    logger.info("=" * 60)

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    from models.calibration import conformal_correct

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # 重新划分：70%训练 + 30%校准
    n = len(X_train)
    cal_start = int(n * 0.7)
    X_tr = X_train.iloc[:cal_start]
    y_tr = y_temp_train[:cal_start]
    X_cal = X_train.iloc[cal_start:]
    y_cal = y_temp_train[cal_start:]

    # Stacking训练
    q_preds_cal = {}
    q_preds_test = {}

    for q in quantiles:
        label = f"p{int(q*100):02d}"

        lgb_m = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=500,
            learning_rate=0.05, verbose=-1, random_state=42, n_jobs=-1,
        )
        lgb_m.fit(X_tr, y_tr)

        xgb_m = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            n_estimators=500, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_m.fit(X_tr, y_tr)

        cb_m = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            iterations=500, learning_rate=0.05,
            random_seed=42, verbose=0,
        )
        cb_m.fit(X_tr, y_tr)

        q_preds_cal[label] = (lgb_m.predict(X_cal) + xgb_m.predict(X_cal) + cb_m.predict(X_cal)) / 3.0
        q_preds_test[label] = (lgb_m.predict(X_test) + xgb_m.predict(X_test) + cb_m.predict(X_test)) / 3.0

    # 保形校准
    corrections = {}
    for name, lo_key, hi_key, target_cov in [
        ("90%", "p05", "p95", 0.90),
        ("80%", "p10", "p90", 0.80),
        ("50%", "p25", "p75", 0.50),
    ]:
        new_lo, new_hi, corr = conformal_correct(
            y_cal, q_preds_cal[lo_key], q_preds_cal[hi_key],
            q_preds_test[lo_key], q_preds_test[hi_key],
            target_cov,
        )
        q_preds_test[lo_key] = new_lo
        q_preds_test[hi_key] = new_hi
        corrections[name] = round(float(corr), 4)
        logger.info(f"保形校准 {name}: 校正量=±{corr:.4f}°C")

    y_pred = q_preds_test["p50"]
    temp_metrics = evaluate_temperature(y_temp_test, y_pred, q_preds_test, quantiles)

    logger.info(f"温度 MAE={temp_metrics['mae']}°C, R²={temp_metrics['r2']}")

    return {
        "temperature": temp_metrics,
        "conformal_corrections": corrections,
    }


# =============================================================================
# 对比报告
# =============================================================================
def generate_comparison_report(results, output_path):
    """生成对比报告图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    methods = list(results.keys())
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    # 1. 温度MAE对比
    ax = axes[0, 0]
    maes = [results[m]["temperature"]["mae"] for m in methods]
    bars = ax.bar(methods, maes, color=colors[:len(methods)])
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("MAE (°C)")
    ax.set_title("温度MAE对比")
    ax.grid(True, alpha=0.3, axis="y")

    # 2. R²对比
    ax = axes[0, 1]
    r2s = [results[m]["temperature"]["r2"] for m in methods]
    bars = ax.bar(methods, r2s, color=colors[:len(methods)])
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", fontsize=10)
    ax.set_ylabel("R²")
    ax.set_title("R²对比")
    ax.grid(True, alpha=0.3, axis="y")

    # 3. 覆盖率对比
    ax = axes[1, 0]
    x_pos = np.arange(3)
    width = 0.2
    cov_labels = ["50%", "80%", "90%"]
    for i, method in enumerate(methods):
        covs = results[method]["temperature"].get("coverages", {})
        vals = [covs.get(l, 0) * 100 for l in cov_labels]
        ax.bar(x_pos + i * width, vals, width, label=method, color=colors[i])

    targets = [50, 80, 90]
    for j, t in enumerate(targets):
        ax.axhline(y=t, color="red", linestyle="--", alpha=0.3)

    ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(cov_labels)
    ax.set_ylabel("覆盖率 (%)")
    ax.set_title("预测区间覆盖率对比")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. 精度等级对比
    ax = axes[1, 1]
    acc_labels = ["±1°C", "±2°C", "±3°C"]
    for i, method in enumerate(methods):
        t = results[method]["temperature"]
        vals = [t.get("within_1c", 0) * 100, t.get("within_2c", 0) * 100, t.get("within_3c", 0) * 100]
        ax.bar(x_pos + i * width, vals, width, label=method, color=colors[i])

    ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(acc_labels)
    ax.set_ylabel("百分比 (%)")
    ax.set_title("温度精度等级对比")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 5. 降水F1对比
    ax = axes[2, 0]
    precip_methods = [m for m in methods if "precipitation" in results[m]]
    if precip_methods:
        f1s = [results[m]["precipitation"]["f1"] for m in precip_methods]
        bars = ax.bar(precip_methods, f1s, color=colors[:len(precip_methods)])
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("F1分数")
    ax.set_title("降水预测F1对比")
    ax.grid(True, alpha=0.3, axis="y")

    # 6. 摘要表格
    ax = axes[2, 1]
    ax.axis("off")
    table_data = []
    for method in methods:
        t = results[method]["temperature"]
        p = results[method].get("precipitation", {})
        table_data.append([
            method,
            f"{t['mae']:.3f}",
            f"{t['rmse']:.3f}",
            f"{t['r2']:.4f}",
            f"{t['within_2c']*100:.1f}%",
            f"{t.get('coverages', {}).get('90%', 0)*100:.1f}%",
            f"{p.get('f1', '-')}" if isinstance(p.get('f1'), (int, float)) else "-",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["方法", "MAE", "RMSE", "R²", "±2°C", "90%覆盖", "降水F1"],
        cellLoc="center", loc="center",
        colColours=["#4a90d9"] * 7,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")

    plt.suptitle("上海天气预报ML系统 — 模型优化对比", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"对比报告已保存: {output_path}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="超参数优化与模型对比")
    parser.add_argument("--trials", type=int, default=200, help="Optuna试验次数")
    args = parser.parse_args()

    logger.info("上海天气预报ML系统 — 超参数优化")
    logger.info(f"Optuna试验数: {args.trials}")

    # 加载数据
    X_train, y_temp_train, y_precip_train, X_test, y_temp_test, y_precip_test, feature_cols = \
        load_and_prepare_data()

    results = {}

    # 方法1: 基线
    results["基线(GBR)"] = run_baseline(
        X_train, y_temp_train, y_precip_train,
        X_test, y_temp_test, y_precip_test,
    )

    # 方法2: LightGBM + Optuna
    results["LightGBM+Optuna"] = run_lightgbm_optuna(
        X_train, y_temp_train, y_precip_train,
        X_test, y_temp_test, y_precip_test,
        n_trials=args.trials,
    )

    # 方法3: Stacking
    results["Stacking"] = run_stacking(
        X_train, y_temp_train, y_precip_train,
        X_test, y_temp_test, y_precip_test,
    )
    # 移除临时数据
    if "q_preds" in results["Stacking"]:
        del results["Stacking"]["q_preds"]

    # 方法4: Stack + Conformal
    results["Stack+Conformal"] = run_stacking_conformal(
        X_train, y_temp_train, y_precip_train,
        X_test, y_temp_test, y_precip_test,
    )

    # 保存结果
    from config.settings import MODELS_DIR, REPORTS_DIR

    results_path = MODELS_DIR / "optimization_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"优化结果已保存: {results_path}")

    # 生成对比报告
    report_path = REPORTS_DIR / "optimization_comparison.png"
    generate_comparison_report(results, report_path)

    # 打印最终对比
    logger.info("=" * 70)
    logger.info("最终对比结果")
    logger.info("=" * 70)
    logger.info(f"{'方法':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'±2°C':>8} {'90%覆盖':>8}")
    logger.info("-" * 70)
    for method, r in results.items():
        t = r["temperature"]
        cov90 = t.get("coverages", {}).get("90%", 0)
        logger.info(
            f"{method:<20} {t['mae']:>8.3f} {t['rmse']:>8.3f} "
            f"{t['r2']:>8.4f} {t['within_2c']*100:>7.1f}% {cov90*100:>7.1f}%"
        )

    logger.info("优化完成")


if __name__ == "__main__":
    main()
