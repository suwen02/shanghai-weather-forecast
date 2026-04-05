# 上海天气预报ML系统

基于多模型NWP集合预报的概率性天气预测系统，使用LightGBM分位数回归和保形预测校准。

## 系统架构

```
shanghai-weather-forecast/
├── config/settings.py           # 全局配置：站点、API、ML参数
├── collectors/
│   ├── open_meteo.py            # 多模型NWP + 集合 + 历史数据采集
│   └── cma_stations.py          # 30个气象站数据采集
├── features/engineer.py         # 特征工程管线（约150个特征）
├── models/
│   ├── temperature.py           # LightGBM分位数回归温度模型
│   ├── precipitation.py         # 两阶段降水模型
│   └── calibration.py           # 保形预测 + 等保序回归校准
├── src/
│   ├── pipeline.py              # 主编排管线
│   ├── visualizer.py            # 图表与报告生成
│   └── scheduler.py             # 调度守护进程
├── run_full_pipeline.py         # 完整管线运行器
└── run_optimization.py          # 超参数优化与模型对比
```

## 数据源（全部免费，无需API密钥）

### 确定性预报模型（8个）
| 模型 | 组织 | 说明 |
|------|------|------|
| CMA GRAPES Global | 中国气象局 | 中国区域最优模型 |
| ECMWF IFS 0.25° | 欧洲中期天气预报中心 | 全球最佳确定性模型 |
| GFS Seamless | NOAA | 美国全球预报系统 |
| ICON Seamless | DWD | 德国气象局 |
| JMA Seamless | 日本气象厅 | 东亚表现优异 |
| GEM Seamless | 加拿大环境部 | 加拿大全球环境模型 |
| UKMO Seamless | 英国气象局 | 英国统一模型 |
| Best Match | Open-Meteo | 自动选择最佳模型 |

### 集合预报系统（5个，共161个成员）
| 系统 | 成员数 | 组织 |
|------|--------|------|
| ECMWF IFS | 51 | ECMWF |
| ICON | 40 | DWD |
| GFS | 31 | NOAA |
| GEM Global | 21 | 加拿大 |
| BOM ACCESS | 18 | 澳大利亚 |

### 气象站网络（30个站点）
覆盖上海市16个区和周边14个城市，包括徐家汇(ASOS)、浦东、宝山、崇明等。

## ML模型

### 温度预测：LightGBM分位数回归
- 每个分位数一个独立模型：P05, P10, P25, P50, P75, P90, P95
- Optuna超参数优化（200次试验，CRPS目标函数）
- TimeSeriesSplit 5折交叉验证
- 保形预测校准保证区间覆盖率

### 降水预测：两阶段模型
- **第一阶段**：LightGBM二分类器判断是否降雨（阈值0.1mm）
- **第二阶段**：条件分位数回归预测降雨量分布（log1p变换）
- 等保序回归校准降雨概率

### 模型融合方案
- Stacking：LightGBM + XGBoost + CatBoost 均值融合
- Stack + Conformal：融合模型 + 保形预测校准

## 特征工程（约150个特征）

| 类别 | 特征数 | 说明 |
|------|--------|------|
| 时间特征 | 8 | 周期性sin/cos编码、季节、周末 |
| 滞后特征 | ~36 | 1,2,3,5,7,14天滞后 |
| 滚动窗口 | ~96 | 3,7,14,30天均值/标准差/最小/最大 |
| 物理导出 | ~7 | 温差、露点差、饱和水汽压、风寒、热指数 |
| 多模型共识 | ~18 | 8个模型的均值/标准差/极差 |
| 集合散度 | ~70 | 161个成员的统计量 |
| 空间特征 | ~9 | 30站点的跨站温度/降水/湿度差异 |
| 上海特色 | ~5 | 台风季、梅雨季、季风指标 |

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行完整管线
```bash
# 完整流程：采集5年历史 + 训练 + 预测 + 评估
python run_full_pipeline.py

# 仅初始化（采集历史 + 训练）
python run_full_pipeline.py --mode init

# 每日预测
python run_full_pipeline.py --mode predict

# 仅评估
python run_full_pipeline.py --mode evaluate
```

### 超参数优化
```bash
# 默认200次试验
python run_optimization.py

# 自定义试验数
python run_optimization.py --trials 100
```

### 调度运行
```bash
# 守护进程模式（每日07:00预测，每周日03:00重训练）
python src/scheduler.py --daemon

# 单次执行（配合系统crontab）
python src/scheduler.py --run-once

# 单次执行 + 重训练
python src/scheduler.py --run-once --retrain
```

## 输出

### 预测JSON
```json
{
  "city": "上海",
  "temperature": [
    {
      "date": "2026-04-05",
      "median": 22.5,
      "quantiles": {"p05": 18.2, "p10": 19.1, "p25": 20.8, "p50": 22.5, "p75": 24.1, "p90": 25.3, "p95": 26.0},
      "confidence": "high"
    }
  ],
  "precipitation": [
    {
      "date": "2026-04-05",
      "expected_mm": 2.3,
      "quantiles": {"p_rain": 0.65, "cond_p50": 3.5, "cond_p95": 15.2},
      "confidence": "medium"
    }
  ]
}
```

### 生成的图表
| 文件 | 说明 |
|------|------|
| `temp_distribution_YYYYMMDD.png` | 温度概率扇形图 |
| `precip_distribution_YYYYMMDD.png` | 降水概率 + 条件降水量 |
| `daily_report_YYYYMMDD.png` | 3面板综合日报 |
| `accuracy_report_YYYYMMDD.png` | 8面板精度评估 |
| `optimization_comparison.png` | 4种方法对比 |

## 评估指标

### 温度
- MAE、RMSE、R²、偏差
- 区间覆盖率（50%、80%、90%）
- ±1°C、±2°C、±3°C精度
- Pinball损失、CRPS
- 月度MAE分解

### 降水
- 准确率、精确率、召回率、F1
- AUC-ROC、Brier分数
- 混淆矩阵
- 可靠性图

## 技术栈

- **数据采集**: requests + Open-Meteo API
- **数据存储**: Apache Parquet (pyarrow)
- **特征工程**: pandas + numpy
- **ML框架**: LightGBM + XGBoost + CatBoost
- **超参数优化**: Optuna (贝叶斯TPE)
- **校准**: scikit-learn (保形预测 + 等保序回归)
- **可视化**: matplotlib
- **调度**: schedule

## 设计决策

1. **温度单位**: 摄氏度（适合上海用户）
2. **存储格式**: Parquet（高效列式存储）
3. **主模型**: LightGBM（训练快、精度高）
4. **CMA GRAPES**: 中国气象局专为中国区域优化的模型
5. **上海特色**: 台风季（6-11月）、梅雨季、东亚季风
6. **历史深度**: 5年训练数据
7. **文档语言**: 中文

## 许可证

MIT License
