# Suite2p Cell Classification Explainer & Curation Dashboard

This workspace contains a unified, self-contained suite for automated cell classification and active learning curation of Suite2p calcium imaging ROIs.

It features a shape-invariant morphology and trace-kinetics feature engine, a LightGBM classifier with 5-fold cross-validated training logic, and an interactive frontend dashboard for visual inspection, real-time threshold tuning, and SHAP decision explanations.

---

## 🚀 Key Features & UI Dashboard

1. **Interactive Curation & Feature Playground**:
   - Visual curation interface displaying ROIs alongside traces (Raw, Neuropil, Corrected).
   - Adjust classification thresholds with a real-time slider that instantly plots changes in **F1-Score**, **Precision**, and **Recall**.
   - Keyboard curation using Arrow Keys to browse cells and the Spacebar to toggle labels.
   
2. **SHAP-Ranked Feature Directory**:
   - All active features are ordered dynamically by their cross-validated SHAP impact descending.
   - Click any feature to view its exact mathematical Python formula, biological rationale, and syntax-highlighted execution code.
   - Interactive search and filter controls for feature directory discovery.
   - Real-time custom formula evaluator allowing you to type, test, and inject custom mathematical combinations of features on the fly.

3. **Machine Learning Classifier**:
   - Fast, robust LightGBM model trained using GroupKFold cross-validation on 22 imaging sessions (67,778 candidate cells).
   - Evaluates to `0.863` Precision, `0.850` Recall, and `0.856` F1 Score.

---

## 📦 Installation & Setup (Colleague's / Professor's Computer)

### 1. Clone the Repository
Clone the repository to get the code, scripts, and the pre-trained LightGBM model:
```bash
git clone git@github.com:AddadyTom/suite2p-classification.git
cd suite2p-classification
```

### 2. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or manually: `pip install numpy scipy scikit-learn lightgbm joblib`)*

---

## 📂 Model & Data Locations

### Where is the trained model located?
* **No action required**: The trained classifier and feature metadata are **pre-packaged inside the repository** under:
  - Model Binary: `models/suite2p_best_lgb.pkl`
  - Model Metadata: `models/suite2p_best_lgb.json`
* When running the code, the scripts automatically detect and load the model from this folder relative to the repository root.

### Where should I put my imaging data?
* **Anywhere on your computer**: You do not need to move your data inside the repository.
* The imaging folder must be a standard Suite2p plane output directory (e.g. `plane0`) containing:
  - `stat.npy`, `F.npy`, `Fneu.npy`, and `iscell.npy` (or a ground truth manual/final labels file).
* Simply pass the absolute path to your folder when running the scripts (see below).

---

## 💻 How to Run the Explainer Dashboard

To launch the interactive dashboard on a local Suite2p folder:

```bash
PYTHONPATH=. .venv/bin/python investigate_cell.py --port 5000 --session /path/to/your/suite2p/plane0
```

1. Open your browser and navigate to `http://localhost:5000`.
2. Inspect individual cells, view their SHAP contribution breakdown, check classification statistics, or type custom formula expressions in the playground.

---

## ⚙️ Running Automated Inference (Command Line)

If you want to run the classifier and apply predictions directly to the `iscell.npy` file without launching the web interface:

```bash
python apply_AI.py /path/to/your/suite2p/plane0
```

* **What it does**: 
  1. Backs up the original `iscell.npy` file.
  2. Extracts the active features.
  3. Applies the classification decision threshold.
  4. Runs Non-Maximum Suppression (NMS) to prune overlapping ROIs.
  5. Overwrites `iscell.npy` with the predicted classifications (0/1) and exact probability scores.

---

## 🛠 How to Add or Remove Features

All feature definitions are managed in a single file: [`fe_engine/fe_definitions.py`](fe_engine/fe_definitions.py).

### A. Removing a Feature
1. Open [`fe_engine/fe_definitions.py`](fe_engine/fe_definitions.py).
2. Scroll down to `ACTIVE_FEATURES = [...]` (around line 300).
3. Remove the target feature name string from the array.

### B. Adding a Feature
1. Open [`fe_engine/fe_definitions.py`](fe_engine/fe_definitions.py) and write an extractor function that takes `cache` and returns a 1D numpy array:
   ```python
   def get_my_feature(cache):
       F_corr = _get_fcorr(cache) # (n_cells, n_frames)
       return np.mean(F_corr, axis=1) # (n_cells,)
   ```
2. Register it in `FEATURE_REGISTRY`:
   ```python
   FEATURE_REGISTRY = {
       ...
       'my_feature': get_my_feature,
   }
   ```
3. Add the string `'my_feature'` to `ACTIVE_FEATURES`.

---

## 🎯 Retraining the Model

Once you have changed the active feature set, run:
```bash
.venv/bin/python train_model.py
```
This script will compile all cached sessions, run a 5-fold cross-validation, display average scores, and save the updated classifier to `models/suite2p_best_lgb.pkl` along with its feature schema JSON.

---

## 🧪 Leak-free evaluation & image features (`feat/image-features`)

```bash
# 1. One feature table per session (trace + intensity-normalized + ops.npy image features)
PYTHONPATH=. .venv/bin/python scripts/build_feature_tables.py --cache preprocessed_cache --out feature_tables
# 2. Repeated GroupKFold-by-session CV with nested threshold selection, feature-group ablation and SHAP
PYTHONPATH=. .venv/bin/python scripts/evaluate_cv.py --tables feature_tables --out results
```

* Labels: `iscell_final.npy` > `iscell_backup_before_AI.npy` > `iscell.npy`. Yael sessions are never used; `stav22` (held out) and `Stav3/21` (suspect labels) are excluded by default; exact duplicate sessions (Stav1 = Stav5) are dropped.
* The decision threshold and `n_estimators` are chosen with an inner session-grouped CV on the training folds only; the validation fold is never used for tuning.
* Image features (`fe_engine/image_features.py`) need `ops.npy` matching `stat.npy`; sessions failing the alignment check get NaN image features and fall back to the trace/morphology model.
* Results: `results/REPORT.md`, `results/cv_table.md`, `results/cv_results.json`.
