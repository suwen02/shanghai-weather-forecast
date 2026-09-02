# Lead-Aware NWP Training Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each of the seven forecast days consume the NWP forecast that corresponds to its target date and lead time, so the model no longer receives near-identical inputs.

**Architecture:** Use Open-Meteo Previous Runs as the historical fixed-lead source. Compute causal observation features on unique dates first, then expand each verifying date across lead1..lead7 and merge NWP consensus features. Live inference uses the same consensus feature names plus `forecast_lead_days`; legacy artifacts fall back to uncalibrated NWP consensus until retraining.

**Tech Stack:** Python 3.12, pandas, requests, pytest, existing LightGBM/scikit-learn model interfaces.

**Spec:** `docs/superpowers/specs/2026-09-02-nwp-training-protocol-design.md`

## Global Constraints

- Never compute lag/rolling features after duplicating a date across leads.
- Never use future observations in prediction features.
- Historical NWP and live NWP must use identical `*_model_*` feature names.
- `forecast_lead_days` is a required trained feature.
- Previous Runs archive is capped at 2024-01-01 for the default multi-model dataset.
- Legacy artifacts must not publish repeated ML output; use explicit NWP fallback.
- The UI source is not in the public repository; do not claim the visual fix is deployed until that source is accessible.

---

### Task 1: Normalize Previous Runs by lead

**Files:**
- Modify: `collectors/training_forecasts.py`
- Test: `tests/test_previous_runs_training.py`

**Interfaces:**
- `normalize_previous_runs_hourly(payload: dict, model: str, lead_days: Sequence[int]) -> pd.DataFrame`
- `collect_training_forecasts(...) -> Optional[Path]`

- [x] **Step 1: RED** — test hourly `_previous_day1/_previous_day2` fields aggregate into daily max/min/mean temperature and precipitation sum, keyed by `time`, `forecast_lead_days`, `model`.
- [x] **Step 2: GREEN** — implement parser and drop unavailable lead horizons.
- [x] **Step 3: Collection** — request `temperature_2m_previous_day1..7` and `precipitation_previous_day1..7` model-by-model from `https://previous-runs-api.open-meteo.com/v1/forecast`, in 90-day chunks.
- [x] **Step 4: Persistence** — save `historical_previous_runs_<start>_<end>.parquet`.

### Task 2: Build lead-specific NWP consensus

**Files:**
- Modify: `features/nwp_training.py`
- Test: `tests/test_nwp_training.py`

**Interfaces:**
- `build_lead_consensus_features(previous_runs) -> pd.DataFrame`
- `expand_observation_features_by_lead(observation_features, lead_consensus) -> pd.DataFrame`

- [x] **Step 1: RED** — require lead1 and lead2 for the same verifying date to preserve different NWP values.
- [x] **Step 2: GREEN** — group Previous Runs by lead and reuse `FeatureEngineer.build_model_consensus_features` so training columns exactly match live inference names.
- [x] **Step 3: Expansion** — inner-join already-built observation features to lead consensus by verifying date, producing one row per available lead.

### Task 3: Preserve causal lag/rolling semantics

**Files:**
- Modify: `features/nwp_aware_engineer.py`
- Test: `tests/test_nwp_training.py`

**Interfaces:**
- `NwpAwareFeatureEngineer.build_training_features(historical_daily, previous_runs)`
- `NwpAwareFeatureEngineer.has_nwp_training_features(feature_cols)`

- [x] **Step 1: RED** — prove lag values are calculated before lead expansion and remain unchanged across duplicated lead rows.
- [x] **Step 2: GREEN** — call the existing base training feature builder on unique observation dates first, then expand/merge lead consensus.
- [x] **Step 3: Feature contract** — append `forecast_lead_days` and numeric NWP consensus columns to `feature_cols`.
- [x] **Step 4: Guard** — define NWP-aware as both `forecast_lead_days` and at least one `*_model_*` feature.

### Task 4: Preserve live seven-day divergence

**Files:**
- Modify: `features/prediction_frame.py`
- Modify: `features/nwp_aware_engineer.py`
- Test: `tests/test_prediction_frame.py`

- [x] **Step 1: RED** — seven future consensus rows must retain distinct NWP values and expose lead1..lead7 while historical state remains constant.
- [x] **Step 2: GREEN** — use future NWP dates as the primary scaffold and merge ensemble/spatial data by target date.
- [x] **Step 3: State handling** — copy only missing observation-state columns from the latest historical row; never overwrite NWP values.
- [x] **Step 4: Target-date features** — recompute calendar/seasonal features after the future scaffold is created.

### Task 5: Integrate training and safe legacy behavior

**Files:**
- Modify: `src/pipeline.py`
- Modify: `src/train_nwp_models.py`
- Modify: `scripts/apply_nwp_training_fix.py`
- Test: `tests/test_apply_nwp_training_fix.py`

- [x] **Step 1: Training integration** — load the latest `historical_previous_runs_*.parquet` and refuse to save a model missing the lead-aware NWP feature contract.
- [x] **Step 2: Model metadata** — report NWP feature count and per-lead sample counts.
- [x] **Step 3: Prediction integration** — load model feature names into the engineer, use target dates from future rows, and predict the first configured forecast horizon rows.
- [x] **Step 4: Legacy fallback** — if either loaded artifact is legacy, publish current deterministic NWP consensus with `source=nwp_consensus_fallback` and `calibrated=false` instead of repeated ML output.
- [x] **Step 5: Deployment patch** — update the incremental patch script so an existing realtime worker can be migrated without replacing unrelated worker code.

### Task 6: Verification and deployment

**Files:**
- Modify: `.github/workflows/nwp-fix-tests.yml`

- [x] **Step 1: TDD evidence** — GitHub Actions captured expected RED failures for missing lead-aware APIs.
- [x] **Step 2: Regression suite** — run `tests/test_nwp_training.py`, `tests/test_prediction_frame.py`, `tests/test_previous_runs_training.py`, `tests/test_apply_nwp_training_fix.py` with repository root on `PYTHONPATH`.
- [x] **Step 3: Compile** — `py_compile` all changed modules including `src/pipeline.py`.
- [ ] **Step 4: Host retraining** — on the deployment source host, collect Previous Runs, run `python src/train_nwp_models.py`, and verify saved feature names include both `forecast_lead_days` and `*_model_*`.
- [ ] **Step 5: Production refresh** — trigger `/api/refresh` and verify seven target dates plus either `nwp_training_aware=true` or legacy-safe NWP fallback metadata.
- [ ] **Step 6: UI patch** — once the CLI-deployed UI source is accessible, move perforation to outer decoration and give every text-bearing surface an opaque solid background; verify desktop/mobile screenshots.
