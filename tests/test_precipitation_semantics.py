import numpy as np
import pandas as pd

from config.settings import ML_CONFIG
from models.precipitation import PrecipitationPredictor


def test_precipitation_classifier_targets_wet_event_not_trace_event():
    assert ML_CONFIG.precip_trace_threshold == 0.1
    assert ML_CONFIG.precip_wet_threshold == 1.0
    assert ML_CONFIG.precip_heavy_threshold == 10.0

    predictor = PrecipitationPredictor()
    assert predictor.threshold == ML_CONFIG.precip_wet_threshold
    assert predictor.event_label == "wet_ge_1mm"


class IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class FixedClassifier:
    def predict_proba(self, values):
        return np.asarray([[0.3, 0.7] for _ in range(len(values))], dtype=float)


class FixedRegressor:
    def predict(self, values):
        return np.asarray([np.log1p(2.0) for _ in range(len(values))], dtype=float)


def test_precipitation_prediction_publishes_p_wet_with_legacy_aliases():
    predictor = PrecipitationPredictor()
    predictor.is_trained = True
    predictor.feature_names = ["x"]
    predictor.scaler = IdentityScaler()
    predictor.classifier = FixedClassifier()
    predictor.qr_models = {q: FixedRegressor() for q in predictor.quantiles}

    result = predictor.predict(pd.DataFrame({"x": [1.0]}), ["2026-09-03"])[0]

    assert result.quantiles["p_wet"] == 0.7
    assert result.quantiles["p_rain"] == 0.7
    assert result.distribution_params["p_wet"] == 0.7
    assert result.distribution_params["p_rain_occurrence"] == 0.7
    assert result.model_info["event_label"] == "wet_ge_1mm"
