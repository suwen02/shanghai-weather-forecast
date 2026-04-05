# -*- coding: utf-8 -*-
"""
降水预测模型 — 两阶段LightGBM

架构：
- 第一阶段：LightGBM二分类器（降雨发生概率）
  - 目标：降水量 > 0.1mm → 1，否则 → 0
  - 输出：P(降雨) ∈ [0,1]

- 第二阶段：条件分位数回归（降雨量分布）
  - 仅在降雨样本上训练
  - 对log1p(降水量)做分位数回归
  - 分位数：P05, P25, P50, P75, P90, P95, P99

- 最终输出：
  - P(降雨)、P(干)
  - 条件分位数、无条件分位数
  - 期望降水量
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import ML_CONFIG, PRECIP_MODEL_PATH
from models.temperature import PredictionResult

logger = logging.getLogger(__name__)


class PrecipitationPredictor:
    """
    两阶段降水预测器

    第一阶段：二分类判断是否降雨
    第二阶段：条件分位数回归预测降雨量分布
    """

    def __init__(
        self,
        classifier_params: Optional[Dict] = None,
        qr_params: Optional[Dict] = None,
    ):
        self.threshold = ML_CONFIG.precip_occurrence_threshold
        self.quantiles = ML_CONFIG.precip_quantiles
        self.classifier_params = classifier_params or ML_CONFIG.lgbm_precip_classifier_params
        self.qr_params = qr_params or ML_CONFIG.lgbm_precip_qr_params
        self.classifier = None
        self.qr_models: Dict[float, object] = {}
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
        训练两阶段降水模型

        Args:
            X: 特征矩阵
            y: 目标变量（precipitation_sum, mm）
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
        y_vals = y.values if isinstance(y, pd.Series) else y

        # 二分类标签
        y_binary = (y_vals >= self.threshold).astype(int)

        # 时间序列划分
        n = len(X_scaled)
        split_idx = int(n * (1 - eval_fraction))
        X_train, X_eval = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train_bin, y_eval_bin = y_binary[:split_idx], y_binary[split_idx:]
        y_train_cont, y_eval_cont = y_vals[:split_idx], y_vals[split_idx:]

        # ========================
        # 第一阶段：分类器
        # ========================
        logger.info("训练降水分类器（第一阶段）...")
        self.classifier = lgb.LGBMClassifier(**self.classifier_params)
        self.classifier.fit(
            X_train, y_train_bin,
            eval_set=[(X_eval, y_eval_bin)],
            callbacks=[lgb.early_stopping(
                stopping_rounds=ML_CONFIG.early_stopping_rounds,
                verbose=False,
            )],
        )

        # 分类指标
        y_prob = self.classifier.predict_proba(X_eval)[:, 1]
        y_pred_bin = (y_prob >= 0.5).astype(int)

        tp = np.sum((y_pred_bin == 1) & (y_eval_bin == 1))
        tn = np.sum((y_pred_bin == 0) & (y_eval_bin == 0))
        fp = np.sum((y_pred_bin == 1) & (y_eval_bin == 0))
        fn = np.sum((y_pred_bin == 0) & (y_eval_bin == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Brier分数
        brier = np.mean((y_prob - y_eval_bin) ** 2)

        # AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_eval_bin, y_prob)
        except Exception:
            auc = 0.0

        logger.info(
            f"分类器训练完成: 准确率={accuracy:.3f}, F1={f1:.3f}, "
            f"AUC={auc:.3f}, Brier={brier:.4f}"
        )

        # ========================
        # 第二阶段：条件分位数回归
        # ========================
        logger.info("训练条件降水量模型（第二阶段）...")

        # 仅用降雨样本训练
        rain_mask_train = y_train_cont >= self.threshold
        X_rain = X_train[rain_mask_train]
        y_rain = np.log1p(y_train_cont[rain_mask_train])

        rain_mask_eval = y_eval_cont >= self.threshold
        X_rain_eval = X_eval[rain_mask_eval] if rain_mask_eval.sum() > 0 else X_rain[-10:]
        y_rain_eval = np.log1p(y_eval_cont[rain_mask_eval]) if rain_mask_eval.sum() > 0 else y_rain[-10:]

        for q in self.quantiles:
            params = self.qr_params.copy()
            params["alpha"] = q

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_rain, y_rain,
                eval_set=[(X_rain_eval, y_rain_eval)],
                callbacks=[lgb.early_stopping(
                    stopping_rounds=ML_CONFIG.early_stopping_rounds,
                    verbose=False,
                )],
            )
            self.qr_models[q] = model

        logger.info(f"条件降水量模型训练完成: {len(self.qr_models)}个分位数")

        self.metrics = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "auc_roc": round(float(auc), 4),
            "brier_score": round(float(brier), 4),
            "confusion_matrix": {
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            },
            "rain_rate_actual": round(float(y_eval_bin.mean()), 4),
            "rain_rate_predicted": round(float(y_pred_bin.mean()), 4),
        }

        self.is_trained = True
        return self.metrics

    def predict(
        self,
        X: pd.DataFrame,
        dates: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """
        生成降水概率预报

        对每一天输出：
        - P(降雨), P(干)
        - 条件分位数 (给定降雨的降雨量分布)
        - 无条件分位数 (P(rain) * 条件量)
        - 期望降水量

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

            # 第一阶段：降雨概率
            p_rain = float(self.classifier.predict_proba(row)[0, 1])
            p_dry = 1.0 - p_rain

            # 第二阶段：条件分位数
            conditional_quantiles = {}
            for q in self.quantiles:
                label = f"cond_p{int(q*100):02d}"
                log_val = self.qr_models[q].predict(row)[0]
                val_mm = float(np.expm1(max(0.0, log_val)))
                conditional_quantiles[label] = round(val_mm, 2)

            # 无条件分位数
            unconditional_quantiles = {}
            for q in self.quantiles:
                cond_label = f"cond_p{int(q*100):02d}"
                uncond_label = f"uncond_p{int(q*100):02d}"
                unconditional_quantiles[uncond_label] = round(
                    p_rain * conditional_quantiles[cond_label], 2
                )

            # 期望降水量
            cond_median = conditional_quantiles.get("cond_p50", 0.0)
            expected = round(p_rain * cond_median, 2)

            # 合并分位数
            all_quantiles = {"p_rain": round(p_rain, 4)}
            all_quantiles.update(conditional_quantiles)
            all_quantiles.update(unconditional_quantiles)

            # 置信度
            if p_rain > 0.8 or p_rain < 0.2:
                confidence = "high"
            elif 0.6 <= p_rain <= 0.8 or 0.2 <= p_rain <= 0.4:
                confidence = "medium"
            else:
                confidence = "low"

            target_date = dates[i] if dates and i < len(dates) else f"day_{i+1}"

            results.append(PredictionResult(
                target_date=str(target_date),
                variable="precipitation",
                point_estimate=expected,
                quantiles=all_quantiles,
                distribution_params={
                    "p_rain_occurrence": round(p_rain, 4),
                    "p_dry": round(p_dry, 4),
                    "conditional_median": cond_median,
                    "conditional_p95": conditional_quantiles.get("cond_p95", 0.0),
                    "expected_amount": expected,
                },
                model_info={
                    "type": "TwoStage_LightGBM",
                    "classifier": "LGBMClassifier",
                    "qr": "LGBMRegressor_Quantile",
                    "n_quantiles": len(self.quantiles),
                },
                confidence=confidence,
            ))

        return results

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

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取分类器特征重要性"""
        if self.classifier is None:
            return pd.DataFrame()

        importance = self.classifier.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return df.head(top_n).reset_index(drop=True)

    def save(self, path: Optional[Path] = None):
        """保存模型到磁盘"""
        path = path or PRECIP_MODEL_PATH
        data = {
            "classifier": self.classifier,
            "qr_models": self.qr_models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "quantiles": self.quantiles,
            "threshold": self.threshold,
            "metrics": self.metrics,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"降水模型已保存: {path}")

    def load(self, path: Optional[Path] = None):
        """从磁盘加载模型"""
        path = path or PRECIP_MODEL_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.qr_models = data["qr_models"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.quantiles = data["quantiles"]
        self.threshold = data.get("threshold", 0.1)
        self.metrics = data.get("metrics", {})
        self.is_trained = True
        logger.info(f"降水模型已加载: {path}")
