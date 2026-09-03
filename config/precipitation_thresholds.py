# -*- coding: utf-8 -*-
"""Canonical precipitation-event thresholds used across API, ML and evaluation."""

TRACE_THRESHOLD_MM = 0.1
WET_THRESHOLD_MM = 1.0
HEAVY_THRESHOLD_MM = 10.0
WET_EVENT_LABEL = "wet_ge_1mm"


def apply_precipitation_thresholds(ml_config):
    """Attach the canonical event contract while preserving the legacy alias.

    ``precip_occurrence_threshold`` historically meant trace rain (0.1 mm).
    From the condition-system migration onward it is a compatibility alias for
    the user-facing wet event (>= 1 mm/day). New code should use the explicit
    ``precip_*_threshold`` attributes.
    """
    ml_config.precip_trace_threshold = TRACE_THRESHOLD_MM
    ml_config.precip_wet_threshold = WET_THRESHOLD_MM
    ml_config.precip_heavy_threshold = HEAVY_THRESHOLD_MM
    ml_config.precip_occurrence_threshold = WET_THRESHOLD_MM
    return ml_config
