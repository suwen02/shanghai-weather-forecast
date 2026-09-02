# NWP Training Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trained weather models consume date-varying historical/current NWP features so seven-day forecasts no longer collapse to identical values.

**Architecture:** Add a focused historical-NWP feature adapter, merge those features into the training frame before feature selection, persist historical forecast data in the collection pipeline, and add regression tests that prove daily NWP variation reaches model inputs. Preserve causal observation-state features and the existing future scaffold.

**Tech Stack:** Python 3.12, pandas, pytest, LightGBM/scikit-learn interfaces already used by the repository.

**Spec:** `docs/superpowers/specs/2026-09-02-nwp-training-protocol-design.md`

## Global Constraints

- Keep all future-target observation features causal; never use future observations.
- Reuse `FeatureEngineer.build_model_consensus_features` naming for training and inference.
- Legacy model artifacts must remain loadable.
- Do not fabricate variation in the displayed forecast.
- UI source is not available in the public repository; data/model changes must remain independently testable.

---

### Task 1: Historical NWP consensus adapter

**Files:**
- Create: `features/nwp_training.py`
- Test: `tests/test_nwp_training.py`

**Interfaces:**
- Produces: `merge_historical_nwp_features(observations: pd.DataFrame, historical_forecasts: pd.DataFrame, consensus_builder: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from features.nwp_training import merge_historical_nwp_features


def test_merge_historical_nwp_features_preserves_daily_variation():
    observations = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01", "2026-08-02"]),
        "temperature_2m_max": [31.0, 32.0],
        "precipitation_sum": [0.0, 2.0],
    })
    forecasts = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-01", "2026-08-02"]),
        "model": ["m1", "m1"],
        "temperature_2m_max": [30.0, 34.0],
        "temperature_2m_min": [24.0, 26.0],
        "precipitation_sum": [0.0, 5.0],
    })

    def builder(df):
        return pd.DataFrame({
            "time": df["time"],
            "tmax_max_model_mean": df["temperature_2m_max"],
        })

    result = merge_historical_nwp_features(observations, forecasts, builder)
    assert result["tmax_max_model_mean"].tolist() == [30.0, 34.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nwp_training.py::test_merge_historical_nwp_features_preserves_daily_variation -v`
Expected: FAIL because `features.nwp_training` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
from collections.abc import Callable
import pandas as pd


def merge_historical_nwp_features(observations, historical_forecasts, consensus_builder):
    out = observations.copy()
    out["time"] = pd.to_datetime(out["time"])
    if historical_forecasts.empty:
        return out
    consensus = consensus_builder(historical_forecasts.copy())
    if consensus.empty:
        return out
    consensus["time"] = pd.to_datetime(consensus["time"])
    return out.merge(consensus, on="time", how="left")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nwp_training.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/nwp_training.py tests/test_nwp_training.py
git commit -m "feat: merge historical NWP consensus into training rows"
```

### Task 2: Make FeatureEngineer select NWP training features

**Files:**
- Modify: `features/engineer.py`
- Test: `tests/test_nwp_training.py`

**Interfaces:**
- Changes: `FeatureEngineer.build_training_features(historical_daily, historical_forecasts=None)`

- [ ] **Step 1: Add failing integration test**

```python
from features.engineer import FeatureEngineer


def test_training_feature_names_include_nwp_consensus():
    # Build at least 400 rows so lag/YoY operations are valid.
    dates = pd.date_range("2025-01-01", periods=400, freq="D")
    observations = pd.DataFrame({
        "time": dates,
        "temperature_2m_max": range(400),
        "temperature_2m_min": range(400),
        "temperature_2m_mean": range(400),
        "precipitation_sum": [0.0] * 400,
    })
    forecasts = pd.DataFrame({
        "time": dates,
        "model": ["m1"] * 400,
        "temperature_2m_max": range(400),
        "temperature_2m_min": range(400),
        "precipitation_sum": [0.0] * 400,
    })
    _, feature_cols, _, _ = FeatureEngineer().build_training_features(observations, forecasts)
    assert "tmax_max_model_mean" in feature_cols
```

- [ ] **Step 2: Run test and confirm failure**

Run: `pytest tests/test_nwp_training.py::test_training_feature_names_include_nwp_consensus -v`
Expected: FAIL because the method accepts one positional dataset and does not merge historical forecasts.

- [ ] **Step 3: Implement minimal integration**

Import `merge_historical_nwp_features`. Add optional `historical_forecasts` and call the helper before temporal/lag/rolling feature generation using `self.build_model_consensus_features`.

- [ ] **Step 4: Add explicit NWP-awareness helper**

Add:

```python
@staticmethod
def has_nwp_training_features(feature_cols):
    return any("_model_" in name for name in feature_cols)
```

Use it for diagnostics and tests.

- [ ] **Step 5: Run the focused tests**

Run: `pytest tests/test_nwp_training.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/engineer.py tests/test_nwp_training.py
git commit -m "fix: train models with historical NWP consensus features"
```

### Task 3: Persist historical forecast training data

**Files:**
- Modify: `collectors/open_meteo.py`
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline_nwp_training.py`

**Interfaces:**
- Produces: `collect_training_forecasts(years: int) -> str | None`
- `WeatherPipeline.step2_train_models()` loads `historical_forecasts_*.parquet` and passes it to `build_training_features`.

- [ ] **Step 1: Write failing pipeline contract test**

Use monkeypatch to provide a synthetic historical observation frame and historical forecast frame, replace model training methods with lightweight fakes, run `step2_train_models()`, and assert the feature list passed to both models includes `tmax_max_model_mean`.

- [ ] **Step 2: Run the test and verify failure**

Run: `pytest tests/test_pipeline_nwp_training.py -v`
Expected: FAIL because `step2_train_models()` does not load or pass historical forecasts.

- [ ] **Step 3: Add historical forecast persistence**

Add `collect_training_forecasts(years)` around `collect_historical_forecasts`, saving `RAW_DIR / f"historical_forecasts_{years}yr.parquet"`.

- [ ] **Step 4: Wire collection and training**

`step1_collect_history()` collects historical forecasts for the configured history window. `step2_train_models()` finds the newest `historical_forecasts_*.parquet`, loads it, and passes it to `build_training_features`. If absent, raise an actionable `FileNotFoundError` instead of silently producing a legacy model.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_pipeline_nwp_training.py tests/test_nwp_training.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add collectors/open_meteo.py src/pipeline.py tests/test_pipeline_nwp_training.py
git commit -m "fix: persist and consume historical forecasts during training"
```

### Task 4: Forecast-input divergence regression guard

**Files:**
- Create: `features/prediction_frame.py` if absent on the branch
- Test: `tests/test_prediction_frame.py`

**Interfaces:**
- Produces: `build_forecast_scaffold(history_features, consensus, ensemble, spatial)`

- [ ] **Step 1: Add regression test**

Construct seven future consensus rows with different `tmax_max_model_mean` values and one latest-history row. Assert the output contains seven rows, the NWP column remains different, and the lag-state column remains constant.

- [ ] **Step 2: Run and confirm failure if file is absent/current behavior is wrong**

Run: `pytest tests/test_prediction_frame.py -v`
Expected: FAIL on main because the future-scaffold module is not present.

- [ ] **Step 3: Add the already validated scaffold implementation from the realtime handoff**

Use future NWP dates as the primary table, merge ensemble/spatial by time, then copy only missing historical state columns from the latest observed row.

- [ ] **Step 4: Run test**

Run: `pytest tests/test_prediction_frame.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/prediction_frame.py tests/test_prediction_frame.py
git commit -m "fix: preserve date-varying NWP inputs for future rows"
```

### Task 5: Full verification and production handoff

**Files:**
- Modify: `README.md` only if usage changed.

- [ ] **Step 1: Run full test suite**

Run: `pytest -q`
Expected: 0 failures.

- [ ] **Step 2: Compile changed Python modules**

Run: `python -m py_compile features/nwp_training.py features/prediction_frame.py features/engineer.py collectors/open_meteo.py src/pipeline.py`
Expected: exit 0.

- [ ] **Step 3: Retrain with historical NWP data on the deployment host**

Run: `python -m src.pipeline` or the repository's supported training entrypoint with `mode=train` after historical forecasts are collected.
Expected: saved model feature names include `_model_` columns.

- [ ] **Step 4: Trigger one real refresh**

Run the worker refresh endpoint or `python src/scheduler.py --refresh-once`.
Expected: seven forecast dates, no artificial duplication, and current NWP source values vary where upstream forecasts vary.

- [ ] **Step 5: Apply UI solid-surface change when deployment source is available**

Move perforation to decorative outer pseudo-elements/layers. Give text-bearing components an opaque `background` with no transparent hole pattern beneath their content boxes. Verify at desktop and mobile widths.
