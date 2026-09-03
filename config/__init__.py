# -*- coding: utf-8 -*-
"""配置模块"""
from .settings import *
from .precipitation_thresholds import apply_precipitation_thresholds

# Migration shim: settings.py keeps the legacy field for old artifacts, while
# every runtime import receives the explicit trace/wet/heavy contract.
apply_precipitation_thresholds(ML_CONFIG)
