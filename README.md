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

## 📦 Installation & Setup (Colleague's Computer)

To set up and run this project on another computer:

### 1. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install numpy scipy scikit-learn lightgbm joblib
```

---

## 💻 How to Use the Explainer Dashboard

To launch the dashboard loading a local Suite2p session (e.g. `plane0` containing `F.npy`, `Fneu.npy`, `stat.npy`, `iscell.npy`):

```bash
PYTHONPATH=. .venv/bin/python investigate_cell.py --port 5000 --session /path/to/suite2p/plane0
```

1. Open your browser and navigate to `http://localhost:5000`.
2. Inspect individual cells, view their SHAP contribution breakdown, check classification statistics, or type custom formula expressions in the playground.

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
This script will compile all cached sessions, run a 5-fold cross-validation, display average scores, and save the updated classifier to `models/regular/suite2p_best_lgb.pkl` along with its feature schema JSON.
