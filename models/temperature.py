# -*- coding: utf-8 -*-
"""
温度预测模型 — LightGBM分位数回归

核心架构：
- 每个分位数一个独立的LightGBM模型 (P05, P10, P25, P50, P75, P90, P95)
- Optuna超参数优化（CRPS目标函数）
- TimeSeriesSplit交叉验证（5折）
- 中位数(P50)作为点估计
- StandardScaler特征缩放
- 支持保存/加载模型
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from config.settings import ML_CONFIG, TEMP_MODEL_PATH

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """预测结果数据类"""
    target_date: str
    variable: str
    point_estimate: float
    quantiles: Dict[str, float]
    distribution_params: Dict
    model_info: Dict
    confidence: str


class TemperaturePredictor:
    """
    LightGBM分位数回归温度预测器

    为每个目标分位数训练独立的LightGBM模型，
    输出完整的概率分布（7个分位数）。
    """

    def __init__(self, params: Optional[Dict] = None):
        self.quantiles = ML_CONFIG.temp_quantiles
        self.params = params or ML_CONFIG.lgbm_temp_params
        self.models: Dict[float, object] = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
        eval_fraction: float = 0.15,
    ) -> Dict:
        """
        训练所有分位数模型

        Args:
            X: 特征矩阵
            y: 目标变量（temperature_2m_max）
            feature_names: 特征名列表
            eval_fraction: 验证集比例

        Returns:
            训练指标字典
        """
        import lightgbm as lgb

        self.feature_names = feature_names or list(X.columns)
        X_arr = X[self.feature_names].values if isinstance(X, pd.DataFrame) else X

        # 特征缩放
        X_scaled = self.scaler.fit_transform(X_arr)

        # 时间序列划分
        n = len(X_scaled)
        split_idx = int(n * (1 - eval_fraction))
        X_train, X_eval = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_eval = y.values[:split_idx], y.values[split_idx:]

        pinball_losses = {}

        for q in self.quantiles:
            params = self.params.copy()
            params["alpha"] = q

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_eval, y_eval)],
                callbacks=[lgb.early_stopping(
                    stopping_rounds=ML_CONFIG.early_stopping_rounds,
                    verbose=False,
                )],
            )
            self.models[q] = model

            # 计算Pinball损失
            y_pred = model.predict(X_eval)
            pinball = self._pinball_loss(y_eval, y_pred, q)
            pinball_losses[f"p{int(q*100):02d}"] = pinball
            logger.info(f"分位数P{int(q*100):02d}训练完成, Pinball损失: {pinball:.4f}")

        # 整体评估
        self.metrics = self._evaluate(X_eval, y_eval)
        self.metrics["pinball_losses"] = pinball_losses
        self.is_trained = True

        logger.info(
            f"温度模型训练完成: MAE={self.metrics.get('mae', 'N/A'):.3f}°C, "
            f"RMSE={self.metrics.get('rmse', 'N/A'):.3f}°C, "
            f"R²={self.metrics.get('r2', 'N/A'):.4f}"
        )
        return self.metrics

    def train_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Dict:
        """
        时间序列交叉验证训练

        Args:
            X: 特征矩阵
            y: 目标变量
            n_splits: 折数

        Returns:
            交叉验证指标
        """
        import lightgbm as lgb

        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
            y_tr = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            y_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]

            fold_models = {}
            for q in self.quantiles:
                params = self.params.copy()
                params["alpha"] = q
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)
                fold_models[q] = model

            # 评估该折
            preds_50 = fold_models[0.50].predict(
                X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            )
            mae = np.mean(np.abs(y_val - preds_50))
            cv_metrics.append({"fold": fold, "mae": mae, "n_val": len(y_val)})

        # 最终训练在全量数据上
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)

        n = len(X_scaled)
        eval_start = int(n * 0.85)

        for q in self.quantiles:
            params = self.params.copy()
            params["alpha"] = q
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_scaled[:eval_start], y.values[:eval_start],
                eval_set=[(X_scaled[eval_start:], y.values[eval_start:])],
                callbacks=[lgb.early_stopping(
                    stopping_rounds=ML_CONFIG.early_stopping_rounds,
                    verbose=False,
                )],
            )
            self.models[q] = model

        self.is_trained = True
        avg_mae = np.mean([m["mae"] for m in cv_metrics])
        logger.info(f"CV训练完成: 平均MAE={avg_mae:.3f}°C ({n_splits}折)")
        return {"cv_metrics": cv_metrics, "avg_mae": avg_mae}

    def predict(
        self,
        X: pd.DataFrame,
        dates: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """
        生成温度概率预报

        Args:
            X: 特征矩阵
            dates: 日期列表

        Returns:
            预测结果列表
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")

        X_aligned = self._align_features(X)
        X_scaled = self.scaler.transform(X_aligned)

        results = []
        for i in range(len(X_scaled)):
            row = X_scaled[i:i+1]
            quantile_preds = {}
            for q in self.quantiles:
                label = f"p{int(q*100):02d}"
                pred = self.models[q].predict(row)[0]
                quantile_preds[label] = round(float(pred), 2)

            # 强制分位数单调性 (p05 ≤ p10 ≤ ... ≤ p95)
            sorted_keys = sorted(quantile_preds.keys())
            prev_val = -np.inf
            for key in sorted_keys:
                quantile_preds[key] = max(quantile_preds[key], prev_val)
                prev_val = quantile_preds[key]

            point_estimate = quantile_preds["p50"]
            spread_90 = quantile_preds["p95"] - quantile_preds["p05"]

            # 置信度分类
            if spread_90 < 5:
                confidence = "high"
            elif spread_90 < 10:
                confidence = "medium"
            else:
                confidence = "low"

            target_date = dates[i] if dates and i < len(dates) else f"day_{i+1}"

            results.append(PredictionResult(
                target_date=str(target_date),
                variable="temperature_max",
                point_estimate=point_estimate,
                quantiles=quantile_preds,
                distribution_params={
                    "spread_90": round(spread_90, 2),
                    "spread_50": round(
                        quantile_preds["p75"] - quantile_preds["p25"], 2
                    ),
                },
                model_info={
                    "type": "LightGBM_Quantile_Regression",
                    "n_quantiles": len(self.quantiles),
                    "n_features": len(self.feature_names),
                },
                confidence=confidence,
            ))

        return results

    def _evaluate(self, X_eval: np.ndarray, y_eval: np.ndarray) -> Dict:
        """评估模型在验证集上的表现"""
        # 中位数预测（点估计）
        y_pred = self.models[0.50].predict(X_eval)

        mae = np.mean(np.abs(y_eval - y_pred))
        rmse = np.sqrt(np.mean((y_eval - y_pred) ** 2))
        bias = np.mean(y_pred - y_eval)

        ss_res = np.sum((y_eval - y_pred) ** 2)
        ss_tot = np.sum((y_eval - np.mean(y_eval)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 区间覆盖率
        coverages = {}
        for name, lo_q, hi_q in [("50%", 0.25, 0.75), ("80%", 0.10, 0.90), ("90%", 0.05, 0.95)]:
            if lo_q in self.models and hi_q in self.models:
                lo = self.models[lo_q].predict(X_eval)
                hi = self.models[hi_q].predict(X_eval)
                covered = np.mean((y_eval >= lo) & (y_eval <= hi))
                coverages[name] = round(float(covered), 4)

        # 精度等级
        within_1 = np.mean(np.abs(y_eval - y_pred) <= 1)
        within_2 = np.mean(np.abs(y_eval - y_pred) <= 2)
        within_3 = np.mean(np.abs(y_eval - y_pred) <= 3)

        return {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "bias": round(float(bias), 4),
            "coverages": coverages,
            "within_1c": round(float(within_1), 4),
            "within_2c": round(float(within_2), 4),
            "within_3c": round(float(within_3), 4),
        }

    def _align_features(self, X: pd.DataFrame) -> np.ndarray:
        """对齐预测特征列与训练特征列"""
        if isinstance(X, pd.DataFrame):
            aligned = pd.DataFrame(index=X.index)
            for col in self.feature_names:
                if col in X.columns:
                    aligned[col] = X[col].values
                else:
                    aligned[col] = 0.0
            return aligned.values
        return X

    @staticmethod
    def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """计算Pinball损失"""
        diff = y_true - y_pred
        loss = np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
        return float(np.mean(loss))

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性（基于中位数模型）"""
        if 0.50 not in self.models:
            return pd.DataFrame()

        model = self.models[0.50]
        importance = model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return df.head(top_n).reset_index(drop=True)

    def save(self, path: Optional[Path] = None):
        """保存模型到磁盘"""
        path = path or TEMP_MODEL_PATH
        data = {
            "models": self.models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "quantiles": self.quantiles,
            "metrics": self.metrics,
            "params": self.params,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"温度模型已保存: {path}")

    def load(self, path: Optional[Path] = None):
        """从磁盘加载模型"""
        path = path or TEMP_MODEL_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.quantiles = data["quantiles"]
        self.metrics = data.get("metrics", {})
        self.params = data.get("params", self.params)
        self.is_trained = True
        logger.info(f"温度模型已加载: {path}")
