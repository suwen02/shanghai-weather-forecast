# NWP Training Protocol and Forecast Divergence Design

## Goal

Fix the production defect where all seven forecast days collapse to nearly identical ML outputs, while preserving calibrated probabilistic predictions and the existing real-time refresh worker.

## Root cause

Production evidence shows that current deterministic and ensemble NWP forecasts are fetched successfully for each refresh, and `prediction_frame.py` constructs future rows from those forecast dates. However, training still derives `feature_cols` only from historical observation rows. The NWP consensus columns therefore never enter `TemperaturePredictor.feature_names` or `PrecipitationPredictor.feature_names`; at inference time `_align_features()` silently drops those forecast-varying columns. The remaining lag/rolling state is copied from the latest observed day to every future row, so the model sees nearly identical inputs for all seven dates.

## Architecture

1. Build historical NWP consensus features using the exact same feature names as live inference (`FeatureEngineer.build_model_consensus_features`).
2. Persist historical forecast data and merge its consensus features into historical observation rows before training feature selection.
3. Require training to contain a minimum set of forecast-source features; fail loudly instead of silently training an observation-only model.
4. Keep state features causal: lag/rolling features may be carried forward from the latest observation, but current/future NWP features are merged by target date and must retain their daily variation.
5. Add a fallback daily forecast surface based on current NWP consensus if a legacy model without NWP feature names is loaded, rather than presenting seven identical ML values as if they were current predictions.

## Data flow

Historical training:
`historical observations + historical NWP forecasts -> daily NWP consensus -> time merge -> temporal/lag/rolling features -> feature selection -> train/calibrate/save`

Live inference:
`latest observed-state features + current deterministic consensus + ensemble + spatial -> future scaffold by forecast date -> align to trained NWP-aware feature_names -> predict/calibrate -> JSON`

## Compatibility

Existing model artifacts remain loadable. A model is considered legacy when none of its feature names match the NWP consensus feature family (`*_model_mean`, `*_model_std`, `*_model_min`, `*_model_max`, `*_model_range`, `*_model_count`). The worker must expose this state in metadata and use the NWP fallback until a newly trained artifact is deployed.

## UI companion change

The visible forecast page should keep the perforated-paper motif only as an outer decorative layer. Every text-bearing region—header labels, forecast-day cards, explanatory copy, metrics and badges—gets an opaque solid-color surface so punch holes never appear underneath glyphs. This UI source currently exists only in the CLI-deployed Vercel source tree and is not present on the public GitHub branch, so the code-level UI change must be applied when the deployment source or Yorushika host becomes available.

## Validation

- Unit test: historical NWP values that vary by date produce varying training consensus columns.
- Unit test: training feature selection contains NWP consensus feature names.
- Unit test: future scaffold keeps different NWP values for different dates while state columns remain causal.
- Regression test: a synthetic NWP-aware predictor receives at least two non-identical rows for a 7-day forecast.
- Production smoke test: refresh output contains seven distinct dates and at least one forecast-varying model input/output metric; metadata states `nwp_training_aware=true`.
- UI visual test: no perforation is visible behind any text bounding box.
