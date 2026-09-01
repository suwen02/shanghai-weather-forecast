# 上海天气实时刷新设计

## 目标

在不破坏现有 LightGBM 日预测模型的前提下，让系统根据 Open-Meteo 最新可用数据持续刷新，并提供未来 48 小时短临预报、数据新鲜度、版本化结果和稳定的 `latest.json`。

## 现状问题

1. `src/scheduler.py` 只执行每日 07:00 预测和每周重训练，没有高频刷新。
2. `WeatherPipeline.step3_daily_predict()` 以历史观测行为主表，再将未来 NWP 按日期左连接，导致未来 NWP 行没有进入最终预测矩阵；随后 `tail(7)` 实际取得的是历史行。
3. 输出按天覆盖，缺少 `data_as_of`、数据年龄和 stale 状态，也没有稳定的最新结果入口。
4. 现有训练模型并未完整使用历史多模型 NWP 特征，因此本次不把未经历史训练的实时变量直接塞入 LightGBM 特征；短期实时信息单独作为 48 小时短临输出，日预测先修正预测日期行构造。

## 架构

### 1. 实时快照

新增 `src/realtime.py`，通过现有 `OpenMeteoCollector._get()` 请求 `best_match` 的过去 6 小时 + 未来 48 小时逐小时数据。Open-Meteo Forecast API 自动选择最新可用模式，因此轮询得到的是当前服务端最新预报。

快照指纹只覆盖 `current/hourly/daily` 数据主体，忽略 `generationtime_ms` 等瞬时响应元数据。若指纹未变化，则跳过昂贵的完整日预测。

### 2. 正确的未来预测行

新增 `features/prediction_frame.py`。以多模型 consensus 的未来日期作为主表，ensemble/spatial 最新数据优先合并，再把最后一个已观测历史状态的 lag/rolling 特征携带到未来行。时间/上海季节特征随后按未来日期重新计算。

这样修复当前“未来 NWP 行被左连接丢弃、最终使用历史 tail 行预测未来日期”的错误。

### 3. 双层输出

- `temperature` / `precipitation`：保留现有 ML 7 天日预测。
- `short_term`：直接基于最新 Open-Meteo `best_match` 的未来 48 小时逐小时数据，用于实时温度、湿度、气压、降水概率/量和风场更新。

输出增加：

- `data_as_of`
- `data_age_minutes`
- `is_stale`
- `short_term`

### 4. 发布与调度

每次实际刷新时输出：

- `predictions_YYYYMMDD_HHMMSS.json`：不可变版本
- `predictions_YYYYMMDD.json`：当天兼容入口
- `latest.json`：原子替换的最新入口

调度器每 30 分钟执行一次 refresh gate；上游数据指纹未变化时不运行完整预测。每日 07:00 和周日重训继续保留。

## 错误处理

- Open-Meteo 快照请求失败：本次 refresh 失败，不更新 refresh state，也不覆盖 `latest.json`。
- state 文件损坏：视为首次刷新。
- 单次预测失败：不记录新指纹，下一次调度可自动重试。
- `latest.json` 通过临时文件 + `os.replace()` 原子发布，避免消费者读取半写文件。

## 测试

单元测试覆盖：

1. 数据指纹稳定性与瞬时元数据忽略。
2. 指纹变化 gate。
3. 版本文件和 `latest.json` 原子发布。
4. 48 小时短临结构与 freshness 计算。
5. 未来预测 scaffold 使用未来 NWP 日期，而不是历史 tail 行。
6. ensemble/spatial 最新数据覆盖历史同名 NaN。
