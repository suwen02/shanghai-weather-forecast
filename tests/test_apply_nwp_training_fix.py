from scripts.apply_nwp_training_fix import patch_pipeline


def test_patch_pipeline_switches_training_to_nwp_aware_engineer():
    source = '''from features.engineer import FeatureEngineer\nfrom models.temperature import TemperaturePredictor\n\nclass WeatherPipeline:\n    def __init__(self):\n        self.engineer = FeatureEngineer()\n\n    def step1_collect_history(self, years=5, station_years=3):\n        results = {}\n        center_files = collect_training_history(years)\n        results.update(center_files)\n        station_files = collect_station_history(station_years)\n        results.update(station_files)\n        return results\n\n    def step2_train_models(self):\n        historical = pd.read_parquet(daily_path)\n        logger.info(f"加载历史数据: {len(historical)}行")\n        df, feature_cols, temp_target, precip_target = self.engineer.build_training_features(\n            historical\n        )\n'''

    patched = patch_pipeline(source)

    assert "from features.nwp_aware_engineer import NwpAwareFeatureEngineer" in patched
    assert "self.engineer = NwpAwareFeatureEngineer()" in patched
    assert "collect_training_forecasts(years)" in patched
    assert "historical_forecasts = pd.read_parquet(forecast_path)" in patched
    assert "historical, historical_forecasts" in patched
    assert "has_nwp_training_features" in patched
