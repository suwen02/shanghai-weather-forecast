# -*- coding: utf-8 -*-
"""数据采集模块。"""

from config.settings import DAILY_VARIABLES

# Keep the legacy collector's broad daily request compatible while ensuring
# the condition classifier receives the signals it needs. Mutating the shared
# list before importing open_meteo means its module-level DAILY_VARIABLES
# reference sees the same augmented list without duplicating collector logic.
CONDITION_DAILY_VARIABLES = (
    "weather_code",
    "cloud_cover_mean",
    "precipitation_hours",
    "precipitation_sum",
)
for _variable in CONDITION_DAILY_VARIABLES:
    if _variable not in DAILY_VARIABLES:
        DAILY_VARIABLES.append(_variable)

from .open_meteo import OpenMeteoCollector
from .cma_stations import CMAStationCollector
