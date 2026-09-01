# Realtime Weather Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让上海天气项目按最新 Open-Meteo 数据高频刷新，并修复未来预测行构造错误，同时发布 48 小时短临和 `latest.json`。

**Architecture:** 实时快照与发布逻辑放在独立 `src/realtime.py`；未来预测行构造放在 `features/prediction_frame.py`。现有 `WeatherPipeline` 负责把这两个组件接入 ML 预测，`scheduler.py` 每 30 分钟执行 refresh gate。

**Tech Stack:** Python 3.9+, pandas, requests, schedule, pytest, Open-Meteo Forecast API.

**Spec:** `docs/superpowers/specs/2026-08-31-realtime-weather-design.md`

## Global Constraints

- 所有代码注释和文档使用中文。
- 温度单位保持摄氏度。
- 现有 Parquet 数据格式保持不变。
- 不把未经历史训练的实时变量直接加入已训练 LightGBM 特征 schema。
- `latest.json` 必须原子发布。

---

### Task 1: 实时快照、指纹和原子发布

**Files:**
- Create: `src/realtime.py`
- Test: `tests/test_realtime.py`

**Interfaces:**
- Produces: `snapshot_fingerprint(snapshot) -> str`
- Produces: `RefreshStateStore.should_refresh(fingerprint) -> bool`
- Produces: `fetch_latest_snapshot(collector, past_hours=6, forecast_hours=48) -> dict`
- Produces: `build_short_term_forecast(snapshot, ...) -> dict`
- Produces: `atomic_publish_json(payload, versioned_path, latest_path) -> None`

- [x] **Step 1: Write failing tests** for fingerprint stability, transient metadata, refresh gate, short-term output, and atomic publish.
- [x] **Step 2: Run tests and verify RED.**
- [x] **Step 3: Implement minimal realtime utilities.**
- [x] **Step 4: Run tests and verify GREEN.**

### Task 2: 正确构造未来预测行

**Files:**
- Create: `features/prediction_frame.py`
- Test: `tests/test_prediction_frame.py`

**Interfaces:**
- Produces: `build_forecast_scaffold(history_features, consensus, ensemble, spatial) -> DataFrame`

- [x] **Step 1: Write failing tests** proving future NWP dates are retained.
- [x] **Step 2: Run tests and verify RED.**
- [x] **Step 3: Implement future-date scaffold.**
- [x] **Step 4: Add regression test for ensemble/spatial overriding historical NaN.**
- [x] **Step 5: Fix merge precedence and run tests GREEN.**

### Task 3: 接入主管线

**Files:**
- Modify: `src/pipeline.py`

**Interfaces:**
- `step3_daily_predict(target_date=None, realtime_snapshot=None) -> dict`
- `step4_realtime_refresh(force=False) -> dict`
- `run(mode="refresh")`

- [ ] **Step 1:** Import realtime/scaffold helpers.
- [ ] **Step 2:** Fetch or reuse realtime snapshot in prediction.
- [ ] **Step 3:** Replace historical `tail()` prediction rows with future NWP scaffold and recompute future time/Shanghai features.
- [ ] **Step 4:** Derive prediction dates from scaffold time column.
- [ ] **Step 5:** Attach freshness and 48-hour short-term output.
- [ ] **Step 6:** Publish versioned, daily-compatible, and `latest.json` outputs atomically.
- [ ] **Step 7:** Add refresh gate mode.

### Task 4: 高频调度

**Files:**
- Modify: `src/scheduler.py`

- [ ] **Step 1:** Add `realtime_job()` calling `WeatherPipeline().run(mode="refresh")`.
- [ ] **Step 2:** Register it every 30 minutes.
- [ ] **Step 3:** Add `--refresh-once` for system cron/launchd validation.
- [ ] **Step 4:** Preserve daily prediction and weekly retraining schedules.

### Task 5: 集成验证

**Files:**
- Test: `tests/test_realtime.py`
- Test: `tests/test_prediction_frame.py`

- [ ] **Step 1:** Run `python -m pytest -q`.
- [ ] **Step 2:** Run `python -m py_compile src/realtime.py features/prediction_frame.py`.
- [ ] **Step 3:** On a network-enabled Mac, run `python src/scheduler.py --refresh-once`.
- [ ] **Step 4:** Verify `data/predictions/latest.json` and versioned output contain `short_term`, `data_as_of`, `data_age_minutes`, and `is_stale`.
