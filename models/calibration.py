# -*- coding: utf-8 -*-
"""
预测校准模块

实现两种后处理校准方法：

1. 分割保形预测（Split-Conformal Prediction）：
   - 用于校准温度预测区间
   - 保证目标覆盖率的统计保证
   - 通过计算非一致性分数来调整区间宽度

2. 等保序回归（Isotonic Regression）：
   - 用于校准降水概率
   - 将原始分类概率映射到校准概率
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config.settings import CALIBRATION_PATH, ML_CONFIG

logger = logging.getLogger(__name__)


class CalibrationManager:
    """
    预测校准管理器

    管理温度保形预测校准和降水等保序回归校准。
    """

    def __init__(self):
        self.conformal_corrections: Dict[str, float] = {}
        self.isotonic_model: Optional[IsotonicRegression] = None
        self.is_fitted = False

    # =========================================================================
    # 保形预测（温度）
    # =========================================================================
    def fit_conformal(
        self,
        y_cal: np.ndarray,
        quantile_preds_cal: Dict[str, np.ndarray],
        target_coverages: Optional[List[Tuple[str, str, float]]] = None,
    ):
        """
        拟合保形预测校准

        在校准集上计算非一致性分数，确定校正量。

        Args:
            y_cal: 校准集真值
            quantile_preds_cal: 校准集的分位数预测，
                键为 "p05", "p10", "p25", "p50", "p75", "p90", "p95"
            target_coverages: (下界键, 上界键, 目标覆盖率) 列表
        """
        if target_coverages is None:
            target_coverages = [
                ("p05", "p95", 0.90),
                ("p10", "p90", 0.80),
                ("p25", "p75", 0.50),
            ]

        for lower_key, upper_key, target_cov in target_coverages:
            if lower_key not in quantile_preds_cal or upper_key not in quantile_preds_cal:
                logger.warning(f"缺少分位数预测: {lower_key}/{upper_key}")
                continue

            lower = quantile_preds_cal[lower_key]
            upper = quantile_preds_cal[upper_key]

            correction = self._compute_conformal_correction(
                y_cal, lower, upper, target_cov
            )

            key = f"{lower_key}_{upper_key}"
            self.conformal_corrections[key] = correction
            logger.info(
                f"保形校准 [{lower_key}, {upper_key}] "
                f"目标覆盖率={target_cov:.0%}: 校正量=±{correction:.4f}°C"
            )

    @staticmethod
    def _compute_conformal_correction(
        y_cal: np.ndarray,
        lower_cal: np.ndarray,
        upper_cal: np.ndarray,
        target_coverage: float,
    ) -> float:
        """
        计算保形预测校正量

        非一致性分数 = max(lower - y, y - upper)
        校正量 = quantile(scores, q_level)

        Args:
            y_cal: 校准集真值
            lower_cal: 下界预测
            upper_cal: 上界预测
            target_coverage: 目标覆盖率

        Returns:
            校正量（加/减到上/下界）
        """
        scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * target_coverage) / n, 1.0)
        correction = float(np.quantile(scores, q_level))
        return correction

    def apply_conformal(
        self,
        quantile_preds: Dict[str, float],
    ) -> Dict[str, float]:
        """
        应用保形预测校正

        加宽预测区间以达到目标覆盖率。

        Args:
            quantile_preds: 原始分位数预测

        Returns:
            校正后的分位数预测
        """
        corrected = quantile_preds.copy()

        for key, correction in self.conformal_corrections.items():
            parts = key.split("_")
            lower_key, upper_key = parts[0], parts[1]

            if lower_key in corrected:
                corrected[lower_key] = corrected[lower_key] - correction
            if upper_key in corrected:
                corrected[upper_key] = corrected[upper_key] + correction

        # 重新强制单调性
        sorted_keys = sorted(
            [k for k in corrected.keys() if k.startswith("p")],
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
        )
        if sorted_keys:
            prev_val = -np.inf
            for key in sorted_keys:
                corrected[key] = max(corrected[key], prev_val)
                prev_val = corrected[key]

        return corrected

    # =========================================================================
    # 等保序回归（降水概率）
    # =========================================================================
    def fit_isotonic(
        self,
        y_cal_binary: np.ndarray,
        p_cal_raw: np.ndarray,
    ):
        """
        拟合降水概率等保序回归校准

        将原始分类概率映射到校准后的概率。

        Args:
            y_cal_binary: 校准集二分类标签 (0/1)
            p_cal_raw: 校准集原始预测概率
        """
        self.isotonic_model = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self.isotonic_model.fit(p_cal_raw, y_cal_binary)
        self.is_fitted = True

        # 校准效果评估
        p_calibrated = self.isotonic_model.predict(p_cal_raw)
        brier_raw = np.mean((p_cal_raw - y_cal_binary) ** 2)
        brier_cal = np.mean((p_calibrated - y_cal_binary) ** 2)
        logger.info(
            f"等保序回归校准完成: Brier分数 {brier_raw:.4f} → {brier_cal:.4f}"
        )

    def apply_precipitation_prob(self, p_raw: float) -> float:
        """
        应用等保序回归校准到降水概率

        Args:
            p_raw: 原始预测概率

        Returns:
            校准后的概率
        """
        if self.isotonic_model is None:
            return p_raw

        calibrated = self.isotonic_model.predict([p_raw])[0]
        return float(np.clip(calibrated, 0.0, 1.0))

    def apply_precipitation_probs(self, p_raw: np.ndarray) -> np.ndarray:
        """
        批量应用等保序回归校准

        Args:
            p_raw: 原始概率数组

        Returns:
            校准后的概率数组
        """
        if self.isotonic_model is None:
            return p_raw
        return self.isotonic_model.predict(p_raw)

    # =========================================================================
    # 综合校准
    # =========================================================================
    def calibrate_temperature_prediction(
        self, quantiles: Dict[str, float]
    ) -> Dict[str, float]:
        """
        校准温度预测分位数

        Args:
            quantiles: 原始分位数预测字典

        Returns:
            校准后的分位数字典
        """
        if not self.conformal_corrections:
            return quantiles
        return self.apply_conformal(quantiles)

    def calibrate_precipitation_prediction(
        self, p_rain: float
    ) -> float:
        """
        校准降水概率

        Args:
            p_rain: 原始降雨概率

        Returns:
            校准后的降雨概率
        """
        return self.apply_precipitation_prob(p_rain)

    # =========================================================================
    # 持久化
    # =========================================================================
    def save(self, path: Optional[Path] = None):
        """保存校准数据到磁盘"""
        path = path or CALIBRATION_PATH
        data = {
            "conformal_corrections": self.conformal_corrections,
            "isotonic_model": self.isotonic_model,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"校准数据已保存: {path}")

    def load(self, path: Optional[Path] = None):
        """从磁盘加载校准数据"""
        path = path or CALIBRATION_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.conformal_corrections = data["conformal_corrections"]
        self.isotonic_model = data["isotonic_model"]
        self.is_fitted = data.get("is_fitted", True)
        logger.info(f"校准数据已加载: {path}")


def conformal_correct(
    y_cal: np.ndarray,
    lower_cal: np.ndarray,
    upper_cal: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    target_cov: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    独立函数：对测试集应用保形预测校正

    Args:
        y_cal: 校准集真值
        lower_cal: 校准集下界
        upper_cal: 校准集上界
        lower_test: 测试集下界
        upper_test: 测试集上界
        target_cov: 目标覆盖率

    Returns:
        (校正后下界, 校正后上界, 校正量)
    """
    scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
    n = len(scores)
    q_level = min(np.ceil((n + 1) * target_cov) / n, 1.0)
    correction = float(np.quantile(scores, q_level))

    return lower_test - correction, upper_test + correction, correction
