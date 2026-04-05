# -*- coding: utf-8 -*-
"""
可视化与报告生成模块

生成以下图表：
1. 温度扇形图（90%/80%/50%区间 + 中位线）
2. 降水概率柱状图 + 条件降水量
3. 每日综合报告（3面板仪表盘）
4. 模型精度8面板评估报告
5. 优化方法对比图

所有图表标签使用中文。
"""

import logging
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config.settings import REPORTS_DIR, PREDICTIONS_DIR, CITY_NAME

logger = logging.getLogger(__name__)

# 尝试设置中文字体
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


class WeatherVisualizer:
    """
    天气可视化生成器

    生成温度分布图、降水概率图和综合报告。
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 温度分布扇形图
    # =========================================================================
    def plot_temperature_distribution(
        self,
        predictions: List[Dict],
        report_date: Optional[date] = None,
    ) -> Path:
        """
        绘制温度预测扇形图

        显示90%、80%、50%预测区间和中位线。

        Args:
            predictions: 温度预测列表
            report_date: 报告日期

        Returns:
            图片文件路径
        """
        if not predictions:
            logger.warning("无温度预测数据可绘图")
            return Path()

        fig, ax = plt.subplots(figsize=(14, 6))

        dates = [p["date"] for p in predictions]
        medians = [p.get("median", p.get("quantiles", {}).get("p50", 0)) for p in predictions]
        x = range(len(dates))

        # 提取分位数
        def get_q(p, key, default=0):
            q = p.get("quantiles", {})
            return q.get(key, default)

        p05 = [get_q(p, "p05", m - 5) for p, m in zip(predictions, medians)]
        p10 = [get_q(p, "p10", m - 4) for p, m in zip(predictions, medians)]
        p25 = [get_q(p, "p25", m - 2) for p, m in zip(predictions, medians)]
        p75 = [get_q(p, "p75", m + 2) for p, m in zip(predictions, medians)]
        p90 = [get_q(p, "p90", m + 4) for p, m in zip(predictions, medians)]
        p95 = [get_q(p, "p95", m + 5) for p, m in zip(predictions, medians)]

        # 90%区间
        ax.fill_between(x, p05, p95, alpha=0.15, color="royalblue", label="90%区间 (P05-P95)")
        # 80%区间
        ax.fill_between(x, p10, p90, alpha=0.25, color="royalblue", label="80%区间 (P10-P90)")
        # 50%区间
        ax.fill_between(x, p25, p75, alpha=0.4, color="royalblue", label="50%区间 (P25-P75)")
        # 中位线
        ax.plot(x, medians, "o-", color="darkblue", linewidth=2.5, markersize=8, label="中位数预报 (P50)")

        # 温度标注
        for i, (xi, m) in enumerate(zip(x, medians)):
            color = "red" if m > 30 else ("blue" if m < 10 else "black")
            ax.annotate(
                f"{m:.1f}°C", (xi, m),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10, fontweight="bold", color=color,
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(dates, rotation=30, ha="right")
        ax.set_ylabel("温度 (°C)", fontsize=12)
        ax.set_title(f"{CITY_NAME}最高温度概率预报", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        date_str = (report_date or date.today()).strftime("%Y%m%d")
        path = self.output_dir / f"temp_distribution_{date_str}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"温度分布图已保存: {path}")
        return path

    # =========================================================================
    # 降水概率图
    # =========================================================================
    def plot_precipitation_distribution(
        self,
        predictions: List[Dict],
        report_date: Optional[date] = None,
    ) -> Path:
        """
        绘制降水概率和条件降水量

        上面板：降雨概率柱状图
        下面板：条件降水量堆叠柱状图

        Args:
            predictions: 降水预测列表
            report_date: 报告日期

        Returns:
            图片文件路径
        """
        if not predictions:
            logger.warning("无降水预测数据可绘图")
            return Path()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 1.2])

        dates = [p["date"] for p in predictions]
        x = range(len(dates))

        # 上面板：降雨概率
        p_rain = [
            p.get("quantiles", {}).get("p_rain",
                p.get("params", {}).get("p_rain_occurrence", 0))
            for p in predictions
        ]

        colors = ["#2ecc71" if p < 0.3 else "#f39c12" if p < 0.6 else "#e74c3c" for p in p_rain]
        bars = ax1.bar(x, [p * 100 for p in p_rain], color=colors, alpha=0.8, width=0.6)

        for bar, p in zip(bars, p_rain):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{p*100:.0f}%",
                ha="center", fontsize=10, fontweight="bold",
            )

        ax1.set_ylim(0, 105)
        ax1.set_ylabel("降雨概率 (%)", fontsize=11)
        ax1.set_title(f"{CITY_NAME}降水概率预报", fontsize=14, fontweight="bold")
        ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50%阈值")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(dates, rotation=30, ha="right")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis="y")

        # 下面板：条件降水量
        cond_p50 = []
        cond_p75 = []
        cond_p95 = []
        for p in predictions:
            q = p.get("quantiles", {})
            params = p.get("params", {})
            cond_p50.append(q.get("cond_p50", params.get("conditional_median", 0)))
            cond_p75.append(q.get("cond_p75", 0))
            cond_p95.append(q.get("cond_p95", params.get("conditional_p95", 0)))

        ax2.bar(x, cond_p50, color="steelblue", alpha=0.7, width=0.6, label="条件中位数 (P50)")
        ax2.bar(x, [max(0, p75 - p50) for p50, p75 in zip(cond_p50, cond_p75)],
                bottom=cond_p50, color="cornflowerblue", alpha=0.5, width=0.6, label="P50-P75")
        ax2.bar(x, [max(0, p95 - p75) for p75, p95 in zip(cond_p75, cond_p95)],
                bottom=cond_p75, color="lightsteelblue", alpha=0.4, width=0.6, label="P75-P95")

        ax2.set_ylabel("条件降水量 (mm)", fontsize=11)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(dates, rotation=30, ha="right")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        date_str = (report_date or date.today()).strftime("%Y%m%d")
        path = self.output_dir / f"precip_distribution_{date_str}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"降水分布图已保存: {path}")
        return path

    # =========================================================================
    # 每日综合报告
    # =========================================================================
    def generate_daily_report(
        self,
        temp_preds: List[Dict],
        precip_preds: List[Dict],
        report_date: Optional[date] = None,
    ) -> Path:
        """
        生成3面板综合日报

        面板1: 温度分布
        面板2: 降水概率
        面板3: 摘要表格

        Args:
            temp_preds: 温度预测列表
            precip_preds: 降水预测列表
            report_date: 报告日期

        Returns:
            报告图片路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        report_date = report_date or date.today()

        # 面板1: 温度
        ax1 = axes[0]
        if temp_preds:
            dates = [p["date"] for p in temp_preds]
            medians = [p.get("median", 0) for p in temp_preds]
            x = range(len(dates))

            p25 = [p.get("quantiles", {}).get("p25", m-2) for p, m in zip(temp_preds, medians)]
            p75 = [p.get("quantiles", {}).get("p75", m+2) for p, m in zip(temp_preds, medians)]
            p05 = [p.get("quantiles", {}).get("p05", m-5) for p, m in zip(temp_preds, medians)]
            p95 = [p.get("quantiles", {}).get("p95", m+5) for p, m in zip(temp_preds, medians)]

            ax1.fill_between(x, p05, p95, alpha=0.15, color="royalblue")
            ax1.fill_between(x, p25, p75, alpha=0.35, color="royalblue")
            ax1.plot(x, medians, "o-", color="darkblue", linewidth=2, markersize=7)

            for xi, m in zip(x, medians):
                ax1.annotate(f"{m:.1f}°C", (xi, m), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

            ax1.set_xticks(list(x))
            ax1.set_xticklabels(dates, rotation=30, ha="right")

        ax1.set_ylabel("最高温度 (°C)")
        ax1.set_title(f"{CITY_NAME}天气预报综合日报 — {report_date.isoformat()}", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # 面板2: 降水
        ax2 = axes[1]
        if precip_preds:
            dates = [p["date"] for p in precip_preds]
            x = range(len(dates))
            p_rain = [p.get("quantiles", {}).get("p_rain",
                        p.get("params", {}).get("p_rain_occurrence", 0))
                      for p in precip_preds]

            colors = ["#2ecc71" if p < 0.3 else "#f39c12" if p < 0.6 else "#e74c3c" for p in p_rain]
            bars = ax2.bar(x, [p * 100 for p in p_rain], color=colors, width=0.6)
            for bar, p in zip(bars, p_rain):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{p*100:.0f}%", ha="center", fontsize=9)

            ax2.set_xticks(list(x))
            ax2.set_xticklabels(dates, rotation=30, ha="right")

        ax2.set_ylabel("降雨概率 (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis="y")

        # 面板3: 摘要表格
        ax3 = axes[2]
        ax3.axis("off")

        if temp_preds and precip_preds:
            table_data = []
            for tp, pp in zip(temp_preds, precip_preds):
                p_rain_val = pp.get("quantiles", {}).get("p_rain",
                    pp.get("params", {}).get("p_rain_occurrence", 0))
                table_data.append([
                    tp["date"],
                    f'{tp.get("median", 0):.1f}°C',
                    f'{tp.get("quantiles", {}).get("p05", 0):.1f}~{tp.get("quantiles", {}).get("p95", 0):.1f}°C',
                    tp.get("confidence", "—"),
                    f'{p_rain_val*100:.0f}%',
                    f'{pp.get("expected_mm", 0):.1f}mm',
                    pp.get("confidence", "—"),
                ])

            table = ax3.table(
                cellText=table_data,
                colLabels=["日期", "预报温度", "90%区间", "置信度", "降雨概率", "期望降水", "置信度"],
                cellLoc="center", loc="center",
                colColours=["#4a90d9"] * 7,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # 设置表头颜色
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(color="white", fontweight="bold")

        plt.tight_layout()

        date_str = report_date.strftime("%Y%m%d")
        path = self.output_dir / f"daily_report_{date_str}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"综合日报已保存: {path}")
        return path

    # =========================================================================
    # 精度评估8面板报告
    # =========================================================================
    def generate_accuracy_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile_preds: Dict[str, np.ndarray],
        precip_true: Optional[np.ndarray] = None,
        precip_prob: Optional[np.ndarray] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        dates: Optional[pd.Series] = None,
        report_date: Optional[date] = None,
    ) -> Path:
        """
        生成8面板精度评估报告

        1. 预测vs实际散点图
        2. 误差分布直方图
        3. 时间序列（最近90天）
        4. 降水可靠性图
        5. 混淆矩阵
        6. 月度MAE分解
        7. 特征重要性
        8. 分位数覆盖率

        Returns:
            报告图片路径
        """
        fig, axes = plt.subplots(4, 2, figsize=(20, 24))
        report_date = report_date or date.today()

        # 1. 预测 vs 实际
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
        lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="完美预测")
        ax.set_xlabel("实际温度 (°C)")
        ax.set_ylabel("预测温度 (°C)")
        ax.set_title("预测 vs 实际温度")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 误差分布
        ax = axes[0, 1]
        errors = y_pred - y_true
        ax.hist(errors, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_xlabel("预测误差 (°C)")
        ax.set_ylabel("频次")
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        ax.set_title(f"误差分布 (MAE={mae:.2f}°C, RMSE={rmse:.2f}°C)")
        ax.grid(True, alpha=0.3)

        # 3. 时间序列
        ax = axes[1, 0]
        n_show = min(90, len(y_true))
        idx = range(n_show)
        ax.plot(idx, y_true[-n_show:], label="实际", color="black", linewidth=1)
        ax.plot(idx, y_pred[-n_show:], label="预测(P50)", color="royalblue", linewidth=1, alpha=0.8)
        if "p10" in quantile_preds and "p90" in quantile_preds:
            ax.fill_between(
                idx, quantile_preds["p10"][-n_show:], quantile_preds["p90"][-n_show:],
                alpha=0.2, color="royalblue", label="80%区间"
            )
        ax.set_xlabel("天数")
        ax.set_ylabel("温度 (°C)")
        ax.set_title("最近90天预测 vs 实际")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. 降水可靠性图
        ax = axes[1, 1]
        if precip_true is not None and precip_prob is not None:
            bins_edges = np.linspace(0, 1, 11)
            bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
            observed_freq = []
            for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
                mask = (precip_prob >= lo) & (precip_prob < hi)
                if mask.sum() > 0:
                    observed_freq.append(precip_true[mask].mean())
                else:
                    observed_freq.append(np.nan)
            ax.plot([0, 1], [0, 1], "r--", label="完美校准")
            ax.plot(bin_centers, observed_freq, "o-", color="steelblue", label="模型")
            ax.set_xlabel("预测降雨概率")
            ax.set_ylabel("实际降雨频率")
            ax.set_title("降水概率可靠性图")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "无降水数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("降水概率可靠性图")
        ax.grid(True, alpha=0.3)

        # 5. 混淆矩阵
        ax = axes[2, 0]
        if precip_true is not None and precip_prob is not None:
            pred_bin = (precip_prob >= 0.5).astype(int)
            true_bin = (precip_true >= 0.1).astype(int)
            tp = np.sum((pred_bin == 1) & (true_bin == 1))
            tn = np.sum((pred_bin == 0) & (true_bin == 0))
            fp = np.sum((pred_bin == 1) & (true_bin == 0))
            fn = np.sum((pred_bin == 0) & (true_bin == 1))
            cm = np.array([[tn, fp], [fn, tp]])
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, str(val), ha="center", va="center", fontsize=14,
                       color="white" if val > cm.max()/2 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["无雨", "有雨"])
            ax.set_yticklabels(["无雨", "有雨"])
            ax.set_xlabel("预测")
            ax.set_ylabel("实际")
            ax.set_title("降水混淆矩阵")
        else:
            ax.text(0.5, 0.5, "无降水数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("降水混淆矩阵")

        # 6. 月度MAE分解
        ax = axes[2, 1]
        if dates is not None and len(dates) == len(y_true):
            months = pd.to_datetime(dates).dt.month
            monthly_mae = pd.DataFrame({"month": months, "error": np.abs(y_true - y_pred)})
            monthly = monthly_mae.groupby("month")["error"].mean()
            ax.bar(monthly.index, monthly.values, color="steelblue", alpha=0.7)
            ax.set_xlabel("月份")
            ax.set_ylabel("MAE (°C)")
            ax.set_title("月度MAE分解")
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
        else:
            ax.text(0.5, 0.5, "无日期数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("月度MAE分解")
        ax.grid(True, alpha=0.3, axis="y")

        # 7. 特征重要性 (Top 15)
        ax = axes[3, 0]
        if feature_importance is not None and not feature_importance.empty:
            top_n = feature_importance.head(15)
            y_pos = range(len(top_n))
            ax.barh(y_pos, top_n["importance"].values, color="steelblue", alpha=0.7)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(top_n["feature"].values, fontsize=8)
            ax.set_xlabel("重要性")
            ax.set_title("特征重要性 (Top 15)")
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "无特征重要性数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("特征重要性 (Top 15)")

        # 8. 分位数覆盖率
        ax = axes[3, 1]
        coverage_data = {}
        pairs = [("p25", "p75", "50%"), ("p10", "p90", "80%"), ("p05", "p95", "90%")]
        for lo_key, hi_key, label in pairs:
            if lo_key in quantile_preds and hi_key in quantile_preds:
                lo = quantile_preds[lo_key]
                hi = quantile_preds[hi_key]
                cov = np.mean((y_true >= lo) & (y_true <= hi))
                coverage_data[label] = cov * 100

        if coverage_data:
            targets = {"50%": 50, "80%": 80, "90%": 90}
            labels = list(coverage_data.keys())
            actual = [coverage_data[l] for l in labels]
            target = [targets[l] for l in labels]
            x_pos = range(len(labels))
            w = 0.35
            ax.bar([xi - w/2 for xi in x_pos], target, w, label="目标", color="lightcoral", alpha=0.7)
            ax.bar([xi + w/2 for xi in x_pos], actual, w, label="实际", color="steelblue", alpha=0.7)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(labels)
            ax.set_ylabel("覆盖率 (%)")
            ax.set_title("预测区间覆盖率")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "无覆盖率数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("预测区间覆盖率")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        date_str = report_date.strftime("%Y%m%d")
        path = self.output_dir / f"accuracy_report_{date_str}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"精度评估报告已保存: {path}")
        return path


# =============================================================================
# 预测结果JSON保存
# =============================================================================

def save_predictions_json(
    temp_results: List,
    precip_results: List,
    report_date: date,
) -> Path:
    """
    保存预测结果为JSON

    Args:
        temp_results: 温度PredictionResult列表
        precip_results: 降水PredictionResult列表
        report_date: 报告日期

    Returns:
        JSON文件路径
    """
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY_NAME,
        "temperature": [],
        "precipitation": [],
    }

    for r in temp_results:
        output["temperature"].append({
            "date": r.target_date,
            "median": r.point_estimate,
            "quantiles": r.quantiles,
            "confidence": r.confidence,
        })

    for r in precip_results:
        output["precipitation"].append({
            "date": r.target_date,
            "expected_mm": r.point_estimate,
            "quantiles": r.quantiles,
            "params": r.distribution_params,
            "confidence": r.confidence,
        })

    date_str = report_date.strftime("%Y%m%d")
    path = PREDICTIONS_DIR / f"predictions_{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"预测JSON已保存: {path}")
    return path
