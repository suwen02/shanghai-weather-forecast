# Lead-Aware NWP Training and Forecast Divergence Design

## Goal

Fix the production defect where all seven forecast days collapse to nearly identical ML outputs, while preserving calibrated probabilistic predictions and the existing real-time refresh worker.

## Root cause

Production logs prove that current deterministic and ensemble NWP forecasts are fetched successfully on each refresh. The defect occurs after collection:

1. The legacy model was trained only on observation-derived temporal, lag and rolling features, so live `*_model_*` NWP consensus columns were absent from the saved model `feature_names` and were silently discarded by `_align_features()`.
2. Future prediction rows carried the same latest observed state into every future date. Without trained NWP inputs and an explicit horizon feature, the seven model inputs were nearly identical.
3. A first proposed historical source (`Historical Forecast API`) was rejected after checking current Open-Meteo documentation: that API stitches the first hours of successive runs and therefore does not preserve the fixed 1–7 day lead semantics needed here.

## Correct historical source

Use Open-Meteo **Previous Runs API**. It exposes `_previous_day1` through `_previous_day7`, where each value was forecast exactly 24, 48, ... 168 hours before its valid time. Most models are archived from January 2024. This directly matches the operational seven-day correction problem.

For each model and valid local date:

- request hourly `temperature_2m_previous_dayN` and `precipitation_previous_dayN` for N=1..7;
- aggregate hourly temperature to daily max/min/mean and precipitation to daily sum;
- retain `forecast_lead_days=N` and `model`;
- aggregate across models using the same `FeatureEngineer.build_model_consensus_features` naming used online.

## Training order — leakage guard

Observation lag/rolling features MUST be computed **before** lead expansion.

Correct order:

`unique observation dates -> temporal/physical/lag/rolling state -> Previous Runs consensus by valid date + lead -> duplicate already-built observation rows for lead1..lead7 -> merge NWP -> train`

Incorrect order:

`duplicate observation rows by lead -> shift/rolling`

The incorrect order would make `.shift()` traverse duplicate lead rows rather than prior days and would corrupt causal state features.

Each training row therefore represents:

`(verifying_date, forecast_lead_days, causal_observation_state, fixed-lead_NWP_consensus) -> verifying_observation_target`

The target is the same verifying observation for all available lead rows of a date, while NWP inputs and `forecast_lead_days` differ.

## Live inference

1. Fetch current deterministic NWP, ensemble and spatial forecasts.
2. Compute recent observation state only through yesterday.
3. Use current future NWP dates as the primary scaffold.
4. Carry only historical state values forward; never overwrite future NWP columns.
5. Add `forecast_lead_days` from the day difference between the latest observed date and each target date.
6. Recompute target-date temporal/seasonal features after the scaffold is created.
7. Align to the saved NWP-aware model feature names, predict, then calibrate.

## Legacy artifact safety

A model is NWP-aware only when its feature names contain both:

- `forecast_lead_days`; and
- at least one `*_model_*` consensus feature.

If either temperature or precipitation artifact is legacy, the worker must not publish its repeated ML output as if it were current. Instead it publishes the current deterministic multi-model NWP consensus as an explicitly uncalibrated fallback (`source=nwp_consensus_fallback`, `calibrated=false`) until retraining is completed.

## Data availability

Previous Runs training data is capped at 2024-01-01 even if the observation archive is longer. The full observation history is still loaded first so lags/rolling values at the start of the NWP archive have valid causal context; only after feature construction are rows inner-joined to the available fixed-lead NWP dates.

## UI companion change

Keep the perforated-paper motif only outside content surfaces. Every element containing readable text—forecast cards, headings, labels, explanatory text, badges, metrics, buttons—must have an opaque solid background with no punch-hole/background pattern underneath the glyph bounding box. Punch holes may remain on decorative left/right margins or pseudo-elements outside the text surface.

The actual UI source is not present in the public repository. The current Vercel project was CLI-deployed from an unpushed source tree, and the host connector is unavailable in this session. Therefore the model/data fix is implemented and tested in GitHub; the UI change remains a deployment-source patch and must not be falsely reported as applied until that source is accessible.

## Validation

- Previous Runs parser test: hourly `_previous_dayN` fields aggregate to daily rows keyed by valid date, lead and model.
- Lead consensus test: lead1 and lead2 for the same valid date preserve different NWP values.
- Causality test: observation lag values are computed once and remain unchanged when the row is expanded across leads.
- Live scaffold test: seven future dates carry lead values 1..7 and retain date-varying NWP values while state features remain causal.
- Pipeline compile test: integrated `src/pipeline.py` compiles on Python 3.12.
- Legacy safety: old artifacts route to NWP consensus fallback instead of the repeated ML path.
- Production smoke test after deployment: seven distinct dates, distinct upstream NWP values where models differ, and either `nwp_training_aware=true` (retrained model) or `source=nwp_consensus_fallback` (legacy-safe mode).
- UI visual test after source access: no perforation is visible under text at desktop and mobile widths.
