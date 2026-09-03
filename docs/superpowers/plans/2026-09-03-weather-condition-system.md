# Weather Condition System V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make daily weather type, rain probability, and precipitation amount semantically separate, then build the historical evaluation path required before promoting learned models.

**Architecture:** Open-Meteo deterministic daily fields feed a pure condition-consensus module. The fallback formatter publishes dominant condition plus three event probabilities. The blog consumes backend condition directly. Forecast runs are persisted and verified later by lead day before any learned weather-condition model is promoted.

**Tech Stack:** Python, pandas, pytest, Open-Meteo, Supabase/Postgres, Next.js/React/TypeScript.

**Spec:** `docs/superpowers/specs/2026-09-03-weather-condition-system-design.md`

## Global Constraints
- Do not use trace-rain probability to choose the primary weather icon.
- `p_trace`, `p_wet`, and `p_heavy` are distinct event frequencies and are not calibrated probabilities.
- Keep lead days 0 through 6 explicit.
- Preserve existing temperature/precipitation payload compatibility during migration.
- Use TDD for every behavioral change.

---

### Task 1: Collect condition signals and derive consensus
**Files:** Modify `config/settings.py`, create `features/weather_condition.py`, create `tests/test_weather_condition.py`.
**Produces:** `summarize_daily_condition(det_df, target_time)` and `precipitation_event_probabilities(det_df, target_time)`.
- [ ] Write failing tests for brief-showers=>cloudy primary, sustained-rain=>rain, WMO storm/snow preservation, and separate 0.1/1/10 mm probabilities.
- [ ] Run focused pytest and confirm RED.
- [ ] Add `weather_code` and `cloud_cover_mean` to daily variables and implement pure condition functions.
- [ ] Run focused pytest and confirm GREEN.

### Task 2: Publish condition and corrected probability schema
**Files:** Modify `features/nwp_fallback.py`, `tests/test_nwp_fallback.py`.
**Consumes:** Task 1 pure functions.
**Produces:** top-level `conditions`; precipitation rows with `p_trace/p_wet/p_heavy`; compatibility `p_rain_occurrence=p_wet`.
- [ ] Add failing formatter tests.
- [ ] Implement minimal formatter integration.
- [ ] Run full backend regression suite.

### Task 3: Frontend consumes backend condition
**Files:** Modify `my-blog/src/lib/weather-visual.ts`, `my-blog/src/app/[locale]/projects/weather/live/page.tsx`, visual contract test.
**Consumes:** `conditions[].kind/secondary`, `p_wet`.
- [ ] Add failing contract test that forbids primary icon fallback from rain probability when backend condition exists.
- [ ] Map backend condition directly to hand-drawn icon and show optional shower-risk annotation.
- [ ] Use `p_wet` as displayed fallback rain probability.
- [ ] Build Vercel Preview and verify SSR.

### Task 4: Persist forecast runs for evaluation
**Files:** Create migration/schema for `weather_forecast_runs` and `weather_verifications`; modify publishing worker/pipeline; create tests for idempotent run keys.
**Produces:** immutable issued forecasts keyed by `(location, issued_at, valid_date, lead_days, model/source)` and later truth rows.
- [ ] Add schema tests/migration.
- [ ] Write forecast-run persistence adapter and idempotency tests.
- [ ] Backfill from future refreshes; do not fabricate historical forecasts.

### Task 5: Evaluation report and promotion gate
**Files:** Create `evaluation/weather_metrics.py`, CLI/report script, tests.
**Produces:** lead-wise condition accuracy/macro-F1, precipitation Brier/reliability, temperature MAE/pinball/coverage and baseline comparison.
- [ ] Add synthetic metric tests.
- [ ] Implement metrics and lead-wise report.
- [ ] Add promotion gate requiring improvement over consensus and best-match baselines.

### Task 6: Complete NWP-aware ML retraining path
**Files:** Existing `features/nwp_aware_engineer.py`, `src/pipeline.py`, training scripts/tests.
- [ ] Keep causal rolling/YoY contract green.
- [ ] Ensure inference loads >=400 days for retained 365-day lags.
- [ ] Retrain artifacts on fixed-lead Previous Runs data.
- [ ] Validate by lead before deployment.

### Task 7: Probability calibration and operational freshness
**Files:** Calibration/evaluation modules, worker refresh path, monitoring metadata, frontend labels.
- [ ] Calibrate `p_wet/p_heavy` by lead using held-out issued forecasts once sufficient samples exist.
- [ ] Publish reliability metadata and only label probabilities calibrated after passing reliability checks.
- [ ] Enforce freshness SLA and expose successful model count/source age in payload.

### Task 8: Production rollout
- [ ] Deploy backend to Preview/smoke endpoint and verify real Shanghai payload conditions are not all forced to rain.
- [ ] Deploy frontend Preview and inspect desktop/mobile/print.
- [ ] Promote backend then frontend production.
- [ ] Verify production Supabase snapshot and live page HTML.