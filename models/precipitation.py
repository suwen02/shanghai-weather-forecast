# -*- coding: utf-8 -*-
"""Two-stage LightGBM precipitation model with explicit wet-event semantics.

Stage 1 predicts ``P(precipitation >= 1 mm/day)``. Stage 2 models the
conditional precipitation amount for wet days. ``p_wet`` is the canonical
probability; ``p_rain`` and ``p_rain_occurrence`` remain compatibility aliases.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import ML_CONFIG, PRECIP_MODEL_PATH
from config.precipitation_thresholds import WET_EVENT_LABEL
from models.temperature import PredictionResult

logger = logging.getLogger(__name__)


class PrecipitationPredictor:
    """Two-stage predictor for wet-event probability and wet-day amount."""

    event_label = WET_EVENT_LABEL

    def __init__(
        self,
        classifier_params: Optional[Dict] = None,
        qr_params: Optional[Dict] = None,
    ):
        self.threshold = ML_CONFIG.precip_wet_threshold
        self.event_label = WET_EVENT_LABEL
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
        """Train wet-event classifier and conditional wet-day quantile models."""
        import lightgbm as lgb

        self.feature_names = feature_names or list(X.columns)
        X_arr = X[self.feature_names].values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.fit_transform(X_arr)
        y_vals = y.values if isinstance(y, pd.Series) else np.asarray(y)

        # Canonical event target: a meaningful wet day, >= 1 mm/day.
        y_binary = (y_vals >= self.threshold).astype(int)

        n = len(X_scaled)
        split_idx = int(n * (1 - eval_fraction))
        X_train, X_eval = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train_bin, y_eval_bin = y_binary[:split_idx], y_binary[split_idx:]
        y_train_cont, y_eval_cont = y_vals[:split_idx], y_vals[split_idx:]

        logger.info("训练降水 wet-event 分类器（>= %.1fmm/day）...", self.threshold)
        self.classifier = lgb.LGBMClassifier(**self.classifier_params)
        self.classifier.fit(
            X_train,
            y_train_bin,
            eval_set=[(X_eval, y_eval_bin)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=ML_CONFIG.early_stopping_rounds,
                    verbose=False,
                )
            ],
        )

        y_prob = self.classifier.predict_proba(X_eval)[:, 1]
        y_pred_bin = (y_prob >= 0.5).astype(int)
        tp = np.sum((y_pred_bin == 1) & (y_eval_bin == 1))
        tn = np.sum((y_pred_bin == 0) & (y_eval_bin == 0))
        fp = np.sum((y_pred_bin == 1) & (y_eval_bin == 0))
        fn = np.sum((y_pred_bin == 0) & (y_eval_bin == 1))

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        brier = np.mean((y_prob - y_eval_bin) ** 2)

        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_eval_bin, y_prob)
        except Exception:
            auc = 0.0

        logger.info(
            "wet-event 分类完成: accuracy=%.3f, F1=%.3f, AUC=%.3f, Brier=%.4f",
            accuracy,
            f1,
            auc,
            brier,
        )

        # Amount model is conditional on the same wet-event definition so its
        # probability gate and conditional distribution describe one event.
        wet_mask_train = y_train_cont >= self.threshold
        X_wet = X_train[wet_mask_train]
        y_wet = np.log1p(y_train_cont[wet_mask_train])
        if len(X_wet) == 0:
            raise RuntimeError("训练集中没有 >=1mm wet-event 样本")

        wet_mask_eval = y_eval_cont >= self.threshold
        X_wet_eval = X_eval[wet_mask_eval] if wet_mask_eval.sum() > 0 else X_wet[-10:]
        y_wet_eval = (
            np.log1p(y_eval_cont[wet_mask_eval])
            if wet_mask_eval.sum() > 0
            else y_wet[-10:]
        )

        self.qr_models = {}
        for q in self.quantiles:
            params = self.qr_params.copy()
            params["alpha"] = q
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_wet,
                y_wet,
                eval_set=[(X_wet_eval, y_wet_eval)],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=ML_CONFIG.early_stopping_rounds,
                        verbose=False,
                    )
                ],
            )
            self.qr_models[q] = model

        self.metrics = {
            "event_label": self.event_label,
            "event_threshold_mm": float(self.threshold),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "auc_roc": round(float(auc), 4),
            "brier_score": round(float(brier), 4),
            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            },
            "wet_rate_actual": round(float(y_eval_bin.mean()), 4),
            "wet_rate_predicted": round(float(y_pred_bin.mean()), 4),
            # Legacy metric names kept for old report consumers.
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
        """Return canonical p_wet plus conditional/unconditional amount quantiles."""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")

        X_aligned = self._align_features(X)
        X_scaled = self.scaler.transform(X_aligned)
        results = []

        for i in range(len(X_scaled)):
            row = X_scaled[i : i + 1]
            p_wet = float(self.classifier.predict_proba(row)[0, 1])
            p_dry = 1.0 - p_wet

            conditional_quantiles = {}
            for q in self.quantiles:
                label = f"cond_p{int(q * 100):02d}"
                log_val = self.qr_models[q].predict(row)[0]
                val_mm = float(np.expm1(max(0.0, log_val)))
                conditional_quantiles[label] = round(val_mm, 2)

            unconditional_quantiles = {}
            for q in self.quantiles:
                cond_label = f"cond_p{int(q * 100):02d}"
                uncond_label = f"uncond_p{int(q * 100):02d}"
                unconditional_quantiles[uncond_label] = round(
                    p_wet * conditional_quantiles[cond_label], 2
                )

            cond_median = conditional_quantiles.get("cond_p50", 0.0)
            expected = round(p_wet * cond_median, 2)
            probability = round(p_wet, 4)

            all_quantiles = {
                "p_wet": probability,
                "p_rain": probability,
            }
            all_quantiles.update(conditional_quantiles)
            all_quantiles.update(unconditional_quantiles)

            if p_wet > 0.8 or p_wet < 0.2:
                confidence = "high"
            elif 0.6 <= p_wet <= 0.8 or 0.2 <= p_wet <= 0.4:
                confidence = "medium"
            else:
                confidence = "low"

            target_date = dates[i] if dates and i < len(dates) else f"day_{i + 1}"
            results.append(
                PredictionResult(
                    target_date=str(target_date),
                    variable="precipitation",
                    point_estimate=expected,
                    quantiles=all_quantiles,
                    distribution_params={
                        "p_wet": probability,
                        "p_rain_occurrence": probability,
                        "p_dry": round(p_dry, 4),
                        "event_label": self.event_label,
                        "event_threshold_mm": float(self.threshold),
                        "conditional_median": cond_median,
                        "conditional_p95": conditional_quantiles.get("cond_p95", 0.0),
                        "expected_amount": expected,
                    },
                    model_info={
                        "type": "TwoStage_LightGBM",
                        "classifier": "LGBMClassifier",
                        "qr": "LGBMRegressor_Quantile",
                        "n_quantiles": len(self.quantiles),
                        "event_label": self.event_label,
                        "event_threshold_mm": float(self.threshold),
                    },
                    confidence=confidence,
                )
            )

        return results

    def _align_features(self, X: pd.DataFrame) -> np.ndarray:
        """Align inference columns to the artifact's training feature contract."""
        if isinstance(X, pd.DataFrame):
            aligned = pd.DataFrame(index=X.index)
            for col in self.feature_names:
                aligned[col] = X[col].values if col in X.columns else 0.0
            return aligned.values
        return X

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.classifier is None:
            return pd.DataFrame()
        importance = self.classifier.feature_importances_
        frame = pd.DataFrame(
            {
                "feature": self.feature_names[: len(importance)],
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
        return frame.head(top_n).reset_index(drop=True)

    def save(self, path: Optional[Path] = None):
        path = path or PRECIP_MODEL_PATH
        data = {
            "classifier": self.classifier,
            "qr_models": self.qr_models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "quantiles": self.quantiles,
            "threshold": self.threshold,
            "event_label": self.event_label,
            "metrics": self.metrics,
        }
        with open(path, "wb") as handle:
            pickle.dump(data, handle)
        logger.info("降水模型已保存: %s", path)

    def load(self, path: Optional[Path] = None):
        path = path or PRECIP_MODEL_PATH
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        self.classifier = data["classifier"]
        self.qr_models = data["qr_models"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.quantiles = data["quantiles"]
        self.threshold = float(data.get("threshold", ML_CONFIG.precip_wet_threshold))
        self.event_label = data.get("event_label", WET_EVENT_LABEL)
        self.metrics = data.get("metrics", {})
        self.is_trained = True
        logger.info(
            "降水模型已加载: %s (event=%s threshold=%.1fmm)",
            path,
            self.event_label,
            self.threshold,
        )
