# Suite2p AI Classifier & Curation Dashboard (Unified Release)

This directory contains a complete, self-contained suite for automated cell classification and active learning curation of Suite2p ROI outputs. 

---

## 🚀 Key Features & Highlights

1. **Robust Feature Engineering**: Leverages shape-invariant morphology (Border Solidity, MRS) and calcium trace kinetics (Asymmetry, peak statistics), eliminating raw pixel size dependencies.
2. **Simplified Machine Learning Models**:
   - **Regular** (Recommended Default): LightGBM classifier utilizing 24 features including shape-invariant morphology and continuous spatial ranking ranks.
   - **No Index (Option A)**: Pure biological model utilizing 26 features (including **Temporal SNR**, **Activity Ratio**, and **Peak Density**), completely free of spatial location/indexing bias.
3. **Unified & Adaptive Inference (`apply_AI.py`)**: A single script that automatically detects the model's feature size (24 or 26) and applies the corresponding F1-optimized threshold.
4. **Correlation-Based NMS**: Automatically suppresses duplicate/overlapping ROIs only if they share both high spatial overlap (IoU > 0.3) and trace correlation ($r \ge 0.85$).
5. **Fully Backward-Compatible**: Compiled and serialized using compatible LightGBM and NumPy 1.x binaries, making it 100% compatible with older environments (including **NumPy 1.24.3**).

---

## 📦 Requirements & Installation

Ensure you have the required packages installed in your python environment:
```bash
pip install numpy==1.24.3 scipy scikit-learn lightgbm joblib
```

---

## 1. Automated Inference (`apply_AI.py`)

Run this script on a finished Suite2p plane folder (containing `F.npy`, `Fneu.npy`, `stat.npy`, and `iscell.npy`):

```bash
python apply_AI.py /path/to/suite2p/plane0 [model_preset_or_path]
```

### Parameters:
* **`[model_preset_or_path]`** (Optional, default: `regular`): Choose a pre-trained model preset or provide an absolute path to a custom model `.pkl` file.

#### Presets Available:
* **`regular`** (Recommended default): LightGBM model utilizing continuous spatial rank index (24 features).
* **`noidx`**: LightGBM model utilizing pure biological features without index (Option A, 26 features).

### What it does:
1. Creates a backup of your original `iscell.npy` as `iscell_backup_before_AI.npy` (if not already present).
2. Auto-detects the model's feature length (24 or 26) and extracts the matching features.
3. Automatically applies the F1-maximizing fixed decision threshold:
   - **Regular (24 features)**: `0.66`
   - **No Index (26 features)**: `0.69`
4. Runs Non-Maximum Suppression (NMS) to prune overlapping duplicate ROIs.
5. Overwrites `iscell.npy` with the predicted classifications (0/1) and exact model probability scores.

---

## 2. Interactive Curation Dashboard (`investigate_cell.py`)

Launch the local web dashboard for visual curation, SHAP feature inspection, and real-time threshold tuning:

```bash
python investigate_cell.py
```

### Usage Steps:
1. Open your browser and navigate to `http://localhost:5000` (it will auto-open).
2. Input the absolute path to your Suite2p session directory.
3. The dashboard allows you to:
   - **Sort/Filter by Category**: Instantly filter ROIs by True Positives (TP), False Positives (FP), False Negatives (FN), or Uncertain cells.
   - **Adjust Classification Threshold**: Drag the real-time threshold slider to immediately visualize how decision boundaries affect F1-score, Precision, and Recall.
   - **Keyboard Curation**: Use arrow keys to browse cells and press **Spacebar** to toggle tags (Cell vs Artifact).
