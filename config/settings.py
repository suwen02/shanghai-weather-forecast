# -*- coding: utf-8 -*-
"""
上海天气预报系统 - 全局配置

包含所有配置参数：地理信息、气象站、API端点、机器学习参数、路径等。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# =============================================================================
# 项目路径配置
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# 自动创建数据目录
for _d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 地理信息配置
# =============================================================================
SHANGHAI_LAT = 31.2304
SHANGHAI_LON = 121.4737
TIMEZONE = "Asia/Shanghai"
CITY_NAME = "上海"
CITY_NAME_EN = "Shanghai"

# =============================================================================
# 气象站配置 (30个站点)
# =============================================================================

@dataclass
class WeatherStation:
    """气象站信息"""
    station_id: str
    name_cn: str
    zone: str
    lat: float
    lon: float

# 上海市区 (16个站点)
CITY_STATIONS: List[WeatherStation] = [
    WeatherStation("xujiahui", "徐家汇 (ASOS)", "city_center", 31.1997, 121.4368),
    WeatherStation("pudong", "浦东", "east", 31.2214, 121.5447),
    WeatherStation("baoshan", "宝山", "north", 31.4050, 121.4490),
    WeatherStation("jiading", "嘉定", "northwest", 31.3880, 121.2390),
    WeatherStation("minhang", "闵行", "southwest", 31.1130, 121.3820),
    WeatherStation("songjiang", "松江", "southwest", 31.0000, 121.2280),
    WeatherStation("qingpu", "青浦", "west", 31.1500, 121.1240),
    WeatherStation("fengxian", "奉贤", "south", 30.9180, 121.4740),
    WeatherStation("jinshan", "金山", "southwest", 30.7410, 121.3420),
    WeatherStation("chongming", "崇明", "north_island", 31.6224, 121.3975),
    WeatherStation("hongkou", "虹口", "city_center", 31.2644, 121.5050),
    WeatherStation("yangpu", "杨浦", "east", 31.2597, 121.5260),
    WeatherStation("putuo", "普陀", "west", 31.2495, 121.3970),
    WeatherStation("changning", "长宁", "west", 31.2204, 121.4247),
    WeatherStation("huangpu", "黄浦", "city_center", 31.2315, 121.4694),
    WeatherStation("jing_an", "静安", "city_center", 31.2285, 121.4480),
]

# 周边城市 (14个站点)
SURROUNDING_STATIONS: List[WeatherStation] = [
    WeatherStation("kunshan", "昆山", "jiangsu_east", 31.3856, 120.9577),
    WeatherStation("taicang", "太仓", "jiangsu_east", 31.4576, 121.1309),
    WeatherStation("suzhou", "苏州", "jiangsu", 31.2990, 120.5853),
    WeatherStation("jiaxing", "嘉兴", "zhejiang", 30.7522, 120.7555),
    WeatherStation("hangzhou_bay", "杭州湾", "zhejiang", 30.4561, 121.1149),
    WeatherStation("nantong", "南通", "jiangsu_north", 31.9807, 120.8937),
    WeatherStation("pudong_airport", "浦东机场", "east", 31.1434, 121.8052),
    WeatherStation("hongqiao_airport", "虹桥机场", "west", 31.1979, 121.3363),
    WeatherStation("zhoushan", "舟山", "zhejiang_coast", 30.0360, 122.1070),
    WeatherStation("nanhui", "南汇", "southeast", 31.0500, 121.7500),
    WeatherStation("chongming_east", "崇明东滩", "north_island", 31.5173, 121.9608),
    WeatherStation("wuxi", "无锡", "jiangsu", 31.4906, 120.3119),
    WeatherStation("changzhou", "常州", "jiangsu", 31.8106, 119.9741),
    WeatherStation("huzhou", "湖州", "zhejiang", 30.8924, 120.0879),
]

# 全部站点
ALL_STATIONS: List[WeatherStation] = CITY_STATIONS + SURROUNDING_STATIONS

# =============================================================================
# Open-Meteo API 配置
# =============================================================================

# 确定性预报模型 (8个)
DETERMINISTIC_MODELS = [
    "cma_grapes_global",  # 中国气象局GRAPES全球模式（上海最相关）
    "ecmwf_ifs025",       # ECMWF IFS 0.25°（全球最佳）
    "gfs_seamless",       # NOAA GFS
    "icon_seamless",      # DWD ICON
    "jma_seamless",       # 日本气象厅（东亚表现好）
    "gem_seamless",       # 加拿大GEM
    "ukmo_seamless",      # 英国气象局
    "best_match",         # Open-Meteo自动选择最佳模型
]

# 集合预报模型 (5个系统，共161个成员)
ENSEMBLE_MODELS = {
    "ecmwf_ifs025": 51,                  # ECMWF 51个成员
    "icon_seamless": 40,                  # DWD ICON 40个成员
    "gfs025": 31,                         # NOAA GFS 31个成员
    "gem_global": 21,                     # 加拿大GEM 21个成员
    "bom_access_global_ensemble": 18,     # 澳大利亚BOM 18个成员
}
TOTAL_ENSEMBLE_MEMBERS = sum(ENSEMBLE_MODELS.values())  # 161

# API端点
API_ENDPOINTS = {
    "deterministic": "https://api.open-meteo.com/v1/forecast",
    "ensemble": "https://ensemble-api.open-meteo.com/v1/ensemble",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
}

# 采集变量配置
HOURLY_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "pressure_msl", "cloud_cover", "precipitation", "rain",
    "showers", "snowfall", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m", "shortwave_radiation", "precipitation_probability",
]

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "precipitation_probability_max", "precipitation_probability_mean",
    "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum",
]

ENSEMBLE_HOURLY_VARIABLES = [
    "temperature_2m", "precipitation", "rain", "wind_speed_10m",
    "cloud_cover", "pressure_msl", "relative_humidity_2m",
]

# =============================================================================
# HTTP客户端配置
# =============================================================================
HTTP_RATE_LIMIT_DELAY = 0.3       # 请求间隔（秒）
HTTP_TIMEOUT = 30                  # 超时时间（秒）
HTTP_MAX_RETRIES = 3               # 最大重试次数
HTTP_USER_AGENT = "ShanghaiWeatherML/1.0"

# =============================================================================
# 机器学习配置
# =============================================================================

@dataclass
class MLConfig:
    """机器学习配置参数"""
    # 训练数据
    historical_years: int = 5              # 历史数据年数
    station_historical_years: int = 3      # 站点历史数据年数
    validation_days: int = 365             # 验证集天数（最近一年）
    calibration_fraction: float = 0.2      # 校准集比例

    # 特征工程
    lag_days: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 30])

    # 温度模型 - 分位数回归
    temp_quantiles: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )

    # 降水模型
    precip_occurrence_threshold: float = 0.1  # 降雨阈值（mm）
    precip_quantiles: List[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )

    # LightGBM默认参数
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42

    # Optuna优化
    optuna_n_trials: int = 200
    optuna_timeout: int = 600             # 超时（秒）
    cv_n_splits: int = 5                  # TimeSeriesSplit折数
    early_stopping_rounds: int = 50

    # 预测范围
    forecast_horizon: int = 7             # 预测天数
    retrain_interval_days: int = 7        # 重新训练间隔

    @property
    def lgbm_temp_params(self) -> Dict:
        """温度模型LightGBM参数"""
        return {
            "objective": "quantile",
            "metric": "quantile",
            "boosting_type": "gbdt",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_jobs": -1,
            "random_state": self.random_state,
            "verbose": -1,
        }

    @property
    def lgbm_precip_classifier_params(self) -> Dict:
        """降水分类模型LightGBM参数"""
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_jobs": -1,
            "random_state": self.random_state,
            "verbose": -1,
        }

    @property
    def lgbm_precip_qr_params(self) -> Dict:
        """降水量分位数回归LightGBM参数"""
        return {
            "objective": "quantile",
            "metric": "quantile",
            "boosting_type": "gbdt",
            "n_estimators": 400,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": 10,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_jobs": -1,
            "random_state": self.random_state,
            "verbose": -1,
        }

# 全局ML配置实例
ML_CONFIG = MLConfig()

# =============================================================================
# 数据泄漏防护 - 需要从预测特征中移除的同日观测变量
# =============================================================================
LEAKED_FEATURES = {
    "temperature_2m_min", "temperature_2m_mean", "rain_sum",
    "precipitation_hours", "temp_range",
    "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum",
}

# =============================================================================
# 调度配置
# =============================================================================
DAILY_COLLECTION_HOUR = 6     # 每日06:00 CST采集数据
DAILY_PREDICTION_HOUR = 7     # 每日07:00 CST生成预报
WEEKLY_RETRAIN_DAY = "sunday"  # 每周日03:00重新训练
WEEKLY_RETRAIN_HOUR = 3

# =============================================================================
# 日志配置
# =============================================================================
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "shanghai_weather_ml.log"

# =============================================================================
# 模型保存路径
# =============================================================================
TEMP_MODEL_PATH = MODELS_DIR / "temperature_model.pkl"
PRECIP_MODEL_PATH = MODELS_DIR / "precipitation_model.pkl"
CALIBRATION_PATH = MODELS_DIR / "calibration_data.pkl"
OPTIMIZATION_RESULTS_PATH = MODELS_DIR / "optimization_results.json"
TUNED_PARAMS_PATH = MODELS_DIR / "tuned_params.json"

# =============================================================================
# 上海特色气象指标
# =============================================================================
# 台风季节：6月-11月
TYPHOON_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]
# 梅雨季节：通常6月中旬-7月上旬
MEIYU_START_DOY = 163   # 约6月12日
MEIYU_END_DOY = 193     # 约7月12日
# 东亚季风特征月份
MONSOON_WET_MONTHS = [6, 7, 8, 9]   # 夏季风（湿润）
MONSOON_DRY_MONTHS = [12, 1, 2, 3]  # 冬季风（干燥）
