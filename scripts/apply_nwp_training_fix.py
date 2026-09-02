#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 lead-aware Previous Runs 训练协议增量应用到现有 WeatherPipeline。"""

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: 期望命中 1 次，实际 {count} 次")
    return text.replace(old, new, 1)


def patch_pipeline(text: str) -> str:
    text = replace_once(
        text,
        "from features.engineer import FeatureEngineer\n",
        "from features.engineer import FeatureEngineer\n"
        "from features.nwp_aware_engineer import NwpAwareFeatureEngineer\n"
        "from collectors.training_forecasts import collect_training_forecasts\n",
        "pipeline NWP imports",
    )
    text = replace_once(
        text,
        "        self.engineer = FeatureEngineer()\n",
        "        self.engineer = NwpAwareFeatureEngineer()\n",
        "pipeline engineer",
    )
    text = replace_once(
        text,
        "        station_files = collect_station_history(station_years)\n"
        "        results.update(station_files)\n",
        "        station_files = collect_station_history(station_years)\n"
        "        results.update(station_files)\n\n"
        "        previous_runs_path = collect_training_forecasts(years)\n"
        "        if previous_runs_path is not None:\n"
        "            results[\"historical_previous_runs\"] = str(previous_runs_path)\n",
        "pipeline Previous Runs collection",
    )
    text = replace_once(
        text,
        "        df, feature_cols, temp_target, precip_target = self.engineer.build_training_features(\n"
        "            historical\n"
        "        )\n",
        "        previous_run_files = sorted(\n"
        "            RAW_DIR.glob(\"historical_previous_runs_*.parquet\"),\n"
        "            key=lambda p: p.stat().st_mtime,\n"
        "            reverse=True,\n"
        "        )\n"
        "        if not previous_run_files:\n"
        "            raise FileNotFoundError(\n"
        "                \"未找到固定提前量 Previous Runs 训练数据；请先运行 init 或 collect_training_forecasts\"\n"
        "            )\n"
        "        previous_runs_path = previous_run_files[0]\n"
        "        previous_runs = pd.read_parquet(previous_runs_path)\n"
        "        df, feature_cols, temp_target, precip_target = self.engineer.build_training_features(\n"
        "            historical, previous_runs\n"
        "        )\n"
        "        if not self.engineer.has_nwp_training_features(feature_cols):\n"
        "            raise RuntimeError(\n"
        "                \"训练特征必须包含 forecast_lead_days 与 NWP 共识列，拒绝生成 legacy 模型\"\n"
        "            )\n",
        "pipeline Previous Runs training features",
    )
    return text


def apply(project: Path) -> None:
    project = project.resolve()
    pipeline_path = project / "src" / "pipeline.py"
    if not pipeline_path.exists():
        raise RuntimeError(f"未找到 pipeline: {pipeline_path}")

    text = pipeline_path.read_text(encoding="utf-8")
    pipeline_path.write_text(patch_pipeline(text), encoding="utf-8")
    print(f"已应用 lead-aware Previous Runs 训练协议: {pipeline_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    apply(args.project)
