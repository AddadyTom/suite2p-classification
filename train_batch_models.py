"""
Batch model training script.
Trains several model variants for comparison in the investigate_cell UI.
Usage: .venv/bin/python train_batch_models.py
"""
import subprocess
import sys
from pathlib import Path

PYTHON = str(Path(__file__).parent / ".venv" / "bin" / "python")

MORPHOLOGY_FEATURES = [
    'area_to_radius_sq', 'aspect_ratio', 'bright_pixels_to_radius_sq',
    'compact', 'max_width', 'radius', 'solidity'
]

TRACE_FEATURES = [
    'corr_f_fneu', 'mrs',
    'peak_to_q95_ratio', 'peak_to_q99_ratio',
    'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'range_f', 'range_fcorr', 'skew_diff_fcorr',
    'skew_f', 'skew_fcorr', 'skew_fneu',
    'std_f', 'std_fcorr'
]

TOP10_SHAP_FEATURES = [
    'q99', 'q50', 'skew_fcorr', 'mrs',
    'corr_f_fneu', 'range_fcorr', 'compact',
    'range_f', 'solidity', 'std_f'
]

MODELS = [
    {
        "output": "models/lgb_morphology_only.pkl",
        "name": "LGB Morphology Only",
        "description": "LightGBM trained on 7 morphology/shape features only (no trace stats).",
        "features": MORPHOLOGY_FEATURES,
    },
    {
        "output": "models/lgb_trace_only.pkl",
        "name": "LGB Trace Only",
        "description": "LightGBM trained on 19 trace-statistics features only (no morphology).",
        "features": TRACE_FEATURES,
    },
    {
        "output": "models/lgb_top10_shap.pkl",
        "name": "LGB Top-10 SHAP",
        "description": "LightGBM trained on the 10 highest-importance features from the SHAP report.",
        "features": TOP10_SHAP_FEATURES,
    },
    {
        "output": "models/lgb_deep.pkl",
        "name": "LGB Deep (26 features)",
        "description": "LightGBM with deeper trees (max_depth=12, num_leaves=127) on all 26 active features.",
        "features": None,  # uses ACTIVE_FEATURES
        "extra_args": ["--deep"],
    },
    {
        "output": "models/xgb_baseline.pkl",
        "name": "XGBoost Baseline",
        "description": "XGBoost classifier trained on all 26 active features.",
        "features": None,  # uses ACTIVE_FEATURES
        "extra_args": ["--xgb"],
    },
]


def features_arg(feat_list):
    return ",".join(feat_list) if feat_list else None


def run_model(cfg):
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    cmd = [
        PYTHON, "train_model.py",
        "--output", cfg["output"],
        "--name", name,
        "--description", cfg["description"],
    ]
    if cfg.get("features"):
        cmd += ["--features", features_arg(cfg["features"])]
    for arg in cfg.get("extra_args", []):
        cmd.append(arg)

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"  [FAILED] {name} — exit code {result.returncode}")
    else:
        print(f"  [DONE] {name}")


if __name__ == "__main__":
    print(f"Starting batch training of {len(MODELS)} models...")
    for cfg in MODELS:
        run_model(cfg)
    print(f"\n{'='*60}")
    print("Batch training complete. Models saved to models/")
    print("Load them in investigate_cell UI to compare.")
