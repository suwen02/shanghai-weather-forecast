# Lead-Aware NWP Training and Forecast Divergence Design

## Goal

Fix the production defect where all seven forecast days collapse to nearly identical ML outputs, while preserving calibrated probabilistic predictions and the existing real-time refresh worker.

## Root cause

Production logs prove that current deterministic and ensemble NWP forecasts are fetched successfully on each refresh. The defect occurs after collection:

1. The legacy model was trained only on observation-derived temporal, lag and rolling features, so live `*_model_*` NWP consensus columns were absent from the saved model `feature_names` and were silently discarded by `_align_features()`.
2. Future prediction rows carried the same latest observed state into every future date. Without trained NWP inputs and an explicit horizon feature, the seven model inputs were nearly identical.
3. A first proposed historical source (`Historical Forecast API`) was rejected after checking current Open-Meteo documentation: that API stitches the first hours of successive runs and therefore does not preserve fixed lead semantics for this seven-card forecast.

## Correct historical source

Use Open-Meteo **Previous Runs API**. The product displays **today + the next six days**, so its seven cards map to `_previous_day0` through `_previous_day6`:

- `day0` = the current model run for the valid time shown on today's card;
- `day1` = the value forecast 24 hours before that valid time;
- ...;
- `day6` = the value forecast 144 hours before that valid time.

Most supported model archives are available from January 2024. For each model and valid local date:

- request hourly `temperature_2m_previous_dayN` and `precipitation_previous_dayN` for N=0..6;
- aggregate hourly temperature to daily max/min/mean and precipitation to daily sum;
- retain `forecast_lead_days=N` and `model`;
- aggregate across models using the same `FeatureEngineer.build_model_consensus_features` naming used online.

## Training order — leakage guard

Observation lag/rolling features MUST be computed **before** lead expansion.

Correct order:

`unique observation dates -> temporal/physical/lag/rolling state -> Previous Runs consensus by valid date + lead -> duplicate already-built observation rows for lead0..lead6 -> merge NWP -> train`

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
5. Define the forecast origin as `latest_observation_date + 1 day` (today), then compute `forecast_lead_days = target_date - forecast_origin`; the seven cards therefore map to 0..6.
6. Recompute target-date temporal/seasonal features after the scaffold is created.
7. Align to the saved NWP-aware model feature names, predict, then calibrate.

## Legacy artifact safety

A model is NWP-aware only when its feature names contain both:

- `forecast_lead_days`; and
- at least one `*_model_*` consensus feature.

If either temperature or precipitation artifact is legacy, the worker must not publish its repeated ML output as if it were current. Instead it publishes the current deterministic multi-model NWP consensus as an explicitly uncalibrated fallback (`source=nwp_consensus_fallback`, `calibrated=false`) until retraining is completed. The fallback must emit finite numeric values even when some model statistics are NaN, and each card must carry the correct lead0..lead6 metadata.

## Data availability

Previous Runs training data is capped at 2024-01-01 even if the observation archive is longer. The full observation history is still loaded first so lags/rolling values at the start of the NWP archive have valid causal context; only after feature construction are rows inner-joined to the available fixed-lead NWP dates.

## UI companion change

Keep the perforated-paper motif only outside content surfaces. Every element containing readable text—forecast cards, headings, labels, explanatory text, badges, metrics, buttons—must have an opaque solid background with no punch-hole/background pattern underneath the glyph bounding box. Punch holes may remain on decorative left/right margins or pseudo-elements outside the text surface.

The actual UI source is not present in the public repository. The current Vercel project was CLI-deployed from an unpushed source tree, and the host connector is unavailable in this session. Therefore the model/data fix is implemented and tested in GitHub; the UI change remains a deployment-source patch and must not be falsely reported as applied until that source is accessible.

## Validation

- Previous Runs parser test: hourly `_previous_dayN` fields aggregate to daily rows keyed by valid date, lead and model.
- Lead consensus test: lead0 and lead1 for the same valid date preserve different NWP values.
- Causality test: observation lag values are computed once and remain unchanged when the row is expanded across leads.
- Live scaffold test: seven future dates carry lead values 0..6 and retain date-varying NWP values while state features remain causal.
- Fallback test: legacy fallback uses lead0..6 and never emits NaN/Inf numeric values.
- Pipeline compile test: integrated `src/pipeline.py` compiles on Python 3.12.
- Legacy safety: old artifacts route to NWP consensus fallback instead of the repeated ML path.
- Production smoke test after deployment: seven distinct dates, distinct upstream NWP values where models differ, and either `nwp_training_aware=true` (retrained model) or `source=nwp_consensus_fallback` (legacy-safe mode).
- UI visual test after source access: no perforation is visible under text at desktop and mobile widths.
