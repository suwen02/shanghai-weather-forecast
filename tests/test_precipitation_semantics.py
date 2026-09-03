from config.settings import ML_CONFIG
from models.precipitation import PrecipitationPredictor


def test_precipitation_classifier_targets_wet_event_not_trace_event():
    assert ML_CONFIG.precip_trace_threshold == 0.1
    assert ML_CONFIG.precip_wet_threshold == 1.0
    assert ML_CONFIG.precip_heavy_threshold == 10.0

    predictor = PrecipitationPredictor()
    assert predictor.threshold == ML_CONFIG.precip_wet_threshold
    assert predictor.event_label == "wet_ge_1mm"
