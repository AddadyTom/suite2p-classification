# Suite2p ROI Classification

Classifies Suite2p ROIs as cell / not-cell with a LightGBM model (27 features), plus a local web dashboard for curation and feature exploration.

## Layout

```
fe_engine/                   feature engine (single source of feature definitions)
  fe_definitions.py          feature functions, FEATURE_REGISTRY, ACTIVE_FEATURES
  fe_preprocessor.py         step 1: Suite2p sessions -> preprocessed_cache/*.npz
  fe_loop_runner.py          CV + SHAP feature-selection loop (writes fe_baseline.json, fe_report.md)
train_model.py               step 2: train on preprocessed_cache/ -> models/regular/
apply_AI.py                  inference: predict and overwrite iscell.npy for one session
eval_session.py              score a model against a session's ground-truth labels
investigate_cell.py          curation / explainer dashboard (http://localhost:5000)
playground.html              feature playground page served by the dashboard at /playground
models/regular/              trained model (.pkl) + its feature list (.json)
docs/                        feature reference PDF
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

A session is a Suite2p plane folder (e.g. `plane0`) with `stat.npy`, `F.npy`, `Fneu.npy` and labels (`iscell_final.npy`, `iscell_manual.npy` or `iscell.npy`). Data stays outside the repo; pass its path on the command line.

## Usage

All commands run from the repository root.

**Classify a session** (backs up `iscell.npy` to `iscell_backup_before_AI.npy`, then overwrites it with predictions and probabilities):
```bash
python apply_AI.py /path/to/suite2p/plane0            # uses models/regular
python apply_AI.py /path/to/suite2p/plane0 my_model.pkl
```

**Evaluate against ground truth:**
```bash
python eval_session.py /path/to/suite2p/plane0 --model models/regular/suite2p_best_lgb.pkl
```

**Dashboard:**
```bash
python investigate_cell.py --session /path/to/suite2p/plane0 --port 5000
```

**Retrain:**
```bash
python fe_engine/fe_preprocessor.py --source /path/to/all/sessions   # once, builds preprocessed_cache/
python train_model.py                                                # 5-fold GroupKFold CV, then saves models/regular/
```
Sessions whose name or path contains "yael" are excluded from training.

## Adding or removing a feature

1. In `fe_engine/fe_definitions.py`, write `def get_my_feature(cache): ...` returning one value per ROI.
2. Register it in `FEATURE_REGISTRY` and add its name to `ACTIVE_FEATURES` (remove a name to drop a feature).
3. Run `python train_model.py`.

Note: `apply_AI.py` computes features with its own `extract_features`, separately from `fe_engine/`, so a new feature also needs to be added there before inference can use it.
