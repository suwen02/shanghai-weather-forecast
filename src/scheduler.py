# -*- coding: utf-8 -*-
"""
调度守护进程

支持两种运行模式：
- --daemon: 持续运行进程，使用schedule库调度
- --run-once: 单次执行（适合系统crontab）

调度任务：
- 每日07:00 CST: 运行完整预测管线
- 每周日03:00 CST: 重新训练模型

crontab示例（CST时区）：
  0 7 * * * cd /path/to/project && python src/scheduler.py --run-once
  0 3 * * 0 cd /path/to/project && python src/scheduler.py --run-once --retrain
"""

import sys
import os
import time
import logging
import argparse
from datetime import datetime, date

# 允许从项目根目录直接执行 ``python src/scheduler.py``。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import schedule

from config.settings import (
    DAILY_PREDICTION_HOUR, WEEKLY_RETRAIN_DAY, WEEKLY_RETRAIN_HOUR,
    LOG_FORMAT, LOG_LEVEL, LOG_FILE, LOGS_DIR, TIMEZONE,
)

logger = logging.getLogger(__name__)
REALTIME_REFRESH_MINUTES = max(1, int(os.environ.get("REALTIME_REFRESH_MINUTES", "30")))


def setup_logging():
    """配置日志系统"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                LOG_FILE, maxBytes=5*1024*1024, backupCount=5,
                encoding="utf-8",
            ),
        ],
    )


def realtime_job():
    """高频实时刷新任务；上游数据未变化时自动跳过完整预测。"""
    logger.info("开始实时天气刷新检查")
    try:
        from src.pipeline import WeatherPipeline
        result = WeatherPipeline().run(mode="refresh").get("refresh", {})
        if result.get("updated"):
            logger.info("检测到新天气数据，预测已刷新")
        else:
            logger.info(f"实时刷新跳过: {result.get('reason', 'unknown')}")
    except Exception as e:
        logger.error(f"实时刷新任务失败: {e}", exc_info=True)


def daily_job():
    """每日预测任务"""
    logger.info("=" * 60)
    logger.info("开始每日预测任务")
    logger.info("=" * 60)

    try:
        from src.pipeline import WeatherPipeline
        pipeline = WeatherPipeline()
        results = pipeline.run(mode="predict")

        # 生成可视化报告
        from src.visualizer import WeatherVisualizer
        visualizer = WeatherVisualizer()

        prediction = results.get("prediction", {})
        if prediction:
            temp_preds = prediction.get("temperature", [])
            precip_preds = prediction.get("precipitation", [])

            if temp_preds:
                visualizer.plot_temperature_distribution(temp_preds)
            if precip_preds:
                visualizer.plot_precipitation_distribution(precip_preds)
            if temp_preds and precip_preds:
                visualizer.generate_daily_report(temp_preds, precip_preds)

        logger.info("每日预测任务完成")
    except Exception as e:
        logger.error(f"每日预测任务失败: {e}", exc_info=True)


def weekly_retrain():
    """每周重训练任务"""
    logger.info("=" * 60)
    logger.info("开始每周模型重训练")
    logger.info("=" * 60)

    try:
        from src.pipeline import WeatherPipeline
        pipeline = WeatherPipeline()
        results = pipeline.run(mode="train")

        training = results.get("training", {})
        if training:
            temp_metrics = training.get("temperature", {})
            precip_metrics = training.get("precipitation", {})
            logger.info(
                f"重训练完成: 温度MAE={temp_metrics.get('mae', 'N/A')}°C, "
                f"降水F1={precip_metrics.get('f1', 'N/A')}"
            )
        logger.info("每周重训练完成")
    except Exception as e:
        logger.error(f"每周重训练失败: {e}", exc_info=True)


class WeatherScheduler:
    """
    天气预报调度器

    管理每日预测和每周重训练任务。
    """

    def __init__(self):
        self.is_running = False

    def setup_schedule(self):
        """设置调度任务"""
        # 每30分钟检查一次上游数据；数据不变则不重跑完整预测。
        schedule.every(REALTIME_REFRESH_MINUTES).minutes.do(realtime_job)
        logger.info(f"实时刷新任务已设置: 每{REALTIME_REFRESH_MINUTES}分钟")

        # 每日07:00 CST预测
        schedule.every().day.at(f"{DAILY_PREDICTION_HOUR:02d}:00").do(daily_job)
        logger.info(f"每日预测任务已设置: {DAILY_PREDICTION_HOUR:02d}:00 CST")

        # 每周日03:00 CST重训练
        getattr(schedule.every(), WEEKLY_RETRAIN_DAY).at(
            f"{WEEKLY_RETRAIN_HOUR:02d}:00"
        ).do(weekly_retrain)
        logger.info(f"每周重训练任务已设置: 周{WEEKLY_RETRAIN_DAY} {WEEKLY_RETRAIN_HOUR:02d}:00 CST")

    def run_daemon(self):
        """以守护进程模式运行"""
        self.setup_schedule()
        self.is_running = True

        logger.info("调度守护进程启动")
        logger.info(f"下次运行: {schedule.next_run()}")

        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            logger.info("收到中断信号，调度器停止")
            self.is_running = False

    def run_once(self, retrain: bool = False):
        """单次执行"""
        if retrain:
            weekly_retrain()
        else:
            daily_job()

    def stop(self):
        """停止调度器"""
        self.is_running = False
        logger.info("调度器已停止")


def main():
    """命令行入口"""
    import logging.handlers
    setup_logging()

    parser = argparse.ArgumentParser(description="上海天气预报调度器")
    parser.add_argument(
        "--daemon", action="store_true",
        help="以守护进程模式运行"
    )
    parser.add_argument(
        "--run-once", action="store_true",
        help="单次执行后退出"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="执行模型重训练（与--run-once配合）"
    )
    parser.add_argument(
        "--refresh-once", action="store_true",
        help="立即检查最新天气数据并按需刷新预测"
    )
    args = parser.parse_args()

    scheduler = WeatherScheduler()

    if args.daemon:
        scheduler.run_daemon()
    elif args.refresh_once:
        realtime_job()
    elif args.run_once:
        scheduler.run_once(retrain=args.retrain)
    else:
        # 默认守护模式
        scheduler.run_daemon()


if __name__ == "__main__":
    main()
