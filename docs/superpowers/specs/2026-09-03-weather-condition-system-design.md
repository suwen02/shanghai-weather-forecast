# Weather Condition System V2 Design

## Problem
The production snapshot currently omits weather condition codes. The frontend therefore infers a daily weather icon from the fraction of deterministic models with daily precipitation >= 0.1 mm. That quantity measures trace-rain model agreement, not the dominant daily weather state, and it turns many cloudy days with brief showers into rain icons.

## Product semantics
- `condition.kind` describes the dominant weather state for the target point/day.
- `condition.secondary` describes a meaningful secondary hazard such as showers.
- `p_trace` means deterministic-model fraction with daily precipitation >= 0.1 mm.
- `p_wet` means deterministic-model fraction with daily precipitation >= 1.0 mm and is the default user-facing rain probability until calibrated probabilities are available.
- `p_heavy` means deterministic-model fraction with daily precipitation >= 10.0 mm.
- `model_agreement` is model agreement, not a calibrated probability.
- Frontend icons must use backend `condition.kind`; rain probability alone must never choose the primary icon.

## Condition derivation
Each deterministic model produces daily `weather_code`, `cloud_cover_mean`, `precipitation_hours`, and `precipitation_sum`. A model-level condition is derived with these rules:
1. Snow and thunderstorm WMO classes remain snow/storm when present.
2. Rain-class WMO codes become `rain` only when precipitation is sustained (`precipitation_hours >= 6`) or material (`precipitation_sum >= 5 mm`). Brief/light rain is treated as secondary shower risk and the primary condition comes from cloud cover.
3. Fog codes remain fog.
4. Otherwise cloud-cover primary classes are: `<35% sunny`, `35-79.999% partly-cloudy`, `>=80% cloudy`.
5. Cross-model primary condition is the modal model-level condition. Tie-breaking favors the less severe non-precipitating state unless storm/snow has a strict plurality.
6. If the primary state is not rain/storm/snow and `p_trace >= 0.4`, set `secondary='showers'`.

## Output schema
Add top-level `conditions` rows with `date`, `lead_days`, `kind`, optional `secondary`, `weather_code`, `model_agreement`, `cloud_cover_mean`, and `model_count`. Add precipitation probability fields `p_trace`, `p_wet`, `p_heavy`. Preserve legacy `params.p_rain_occurrence`, but point it to `p_wet` so existing UI does not treat trace precipitation as rain probability.

## Evaluation and promotion
Persist issued forecasts separately from the current snapshot. Score condition accuracy/macro-F1, precipitation Brier score and reliability, temperature error/quantile coverage by lead day. A learned condition or precipitation model may replace the consensus baseline only after it beats simple model-consensus and best-match baselines on held-out forecasts.

## Non-goals for the first iteration
- No new learned multiclass weather-condition model.
- No claim that model vote fractions are calibrated probabilities.
- No UI inference of weather state from precipitation probability.