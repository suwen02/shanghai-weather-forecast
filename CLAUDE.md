# 上海天气预报ML系统 — 项目规范

## 项目概述
基于Open-Meteo多模型NWP集合预报的概率性天气预测系统，使用LightGBM分位数回归、
两阶段降水模型和保形预测校准。

## 代码规范
- 所有代码注释和文档使用**中文**
- 温度单位：**摄氏度**
- 数据存储格式：**Parquet**
- Python 3.9+
- 类型注解：使用标准typing模块

## 目录结构
- `config/` — 配置（站点、API、ML参数）
- `collectors/` — 数据采集（Open-Meteo、CMA站点）
- `features/` — 特征工程管线
- `models/` — ML模型（温度、降水、校准）
- `src/` — 核心管线、可视化、调度
- `data/` — 运行时数据（gitignored）
- `logs/` — 日志文件（gitignored）

## 运行方式
```bash
python run_full_pipeline.py                    # 完整管线
python run_full_pipeline.py --mode init        # 初始化
python run_full_pipeline.py --mode predict     # 每日预测
python run_optimization.py                     # 超参数优化
python src/scheduler.py --daemon               # 调度守护
```

## 数据泄漏防护
config/settings.py中的LEAKED_FEATURES集合定义了需要从预测特征中排除的同日观测变量。

## 关键设计决策
1. CMA GRAPES模型优先（中国区域专用）
2. 保形预测保证覆盖率统计保证
3. 等保序回归校准降水概率
4. 台风季/梅雨季/季风季作为上海特色特征
