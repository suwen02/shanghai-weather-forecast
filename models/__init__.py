# -*- coding: utf-8 -*-
"""机器学习模型模块"""
from config.precipitation_thresholds import WET_EVENT_LABEL
from .temperature import TemperaturePredictor
from .precipitation import PrecipitationPredictor
from .calibration import CalibrationManager

# Semantic metadata is class-level so legacy pickles can still be loaded while
# all newly trained/runtime predictors expose the canonical event target.
PrecipitationPredictor.event_label = WET_EVENT_LABEL
