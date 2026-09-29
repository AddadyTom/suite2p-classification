import os
# Configure PyTorch/OpenMP thread limits to prevent CPU lockups
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import numpy as np
from pathlib import Path
import torch
torch.set_num_threads(1)

import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure workspace root is in python path
sys.path.append(os.getcwd())
from fe_engine.fe_definitions import ACTIVE_FEATURES, FEATURE_REGISTRY
from fe_engine.fe_loop_runner import load_preprocessed_data, extract_features_dataset, run_cross_validation

PROGRESS_FILE = Path("cellpose_progress.json")
REPORT_FILE = Path("cellpose_report.md")

def compute_roi_cellpose_iou(stat_roi, masks):
    ypix = stat_roi['ypix']
    xpix = stat_roi['xpix']
    if len(ypix) == 0:
        return 0.0, 0
        
    roi_pixels = set(zip(ypix, xpix))
    overlapping_ids = np.unique(masks[ypix, xpix])
    overlapping_ids = overlapping_ids[overlapping_ids > 0]
    
    if len(overlapping_ids) == 0:
        return 0.0, 0
        
    best_iou = 0.0
    best_mask_id = 0
    
    for mask_id in overlapping_ids:
        mask_y, mask_x = np.where(masks == mask_id)
        mask_pixels = set(zip(mask_y, mask_x))
        
        intersection = len(roi_pixels.intersection(mask_pixels))
        union = len(roi_pixels.union(mask_pixels))
        
        iou = intersection / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_mask_id = mask_id
            
    return best_iou, best_mask_id

def main():
    workspace_root = Path(os.getcwd())
    cache_dir = workspace_root / "preprocessed_cache"
    
    # 1. Identify all valid sessions that have both ops.npy and labels
    required_files = ['F.npy', 'Fneu.npy', 'stat.npy', 'ops.npy']
    label_options = ['iscell_final.npy', 'iscell_manual.npy', 'iscell.npy']
    
    print("Scanning for valid Suite2p sessions with ops.npy and labels...")
    source_dir = Path("/mnt/other_ubunthu/mnt/data")
    valid_sessions = []
    for stat_path in source_dir.rglob('stat.npy'):
        session_dir = stat_path.parent
        has_req = all((session_dir / req).exists() for req in required_files)
        if not has_req:
            continue
        has_label = any((session_dir / lbl).exists() for lbl in label_options)
        if has_label:
            valid_sessions.append(session_dir)
            
    valid_sessions = sorted(list(set(valid_sessions)))
    valid_names = [s.name for s in valid_sessions]
    n_sessions = len(valid_sessions)
    print(f"Found {n_sessions} valid sessions for evaluation: {valid_names}")
    
    if n_sessions == 0:
        print("No valid sessions found. Exiting.")
        return
        
    # 2. Run LightGBM Evaluation on these 19 sessions
    print("\n=== Step 1: Evaluating LightGBM Classifier (Out-of-Fold) ===")
    sessions = load_preprocessed_data(cache_dir)
    
    # Filter sessions to only include those that have ops.npy (valid_names)
    filtered_sessions = []
    for s in sessions:
        s_name = s['session_name']
        if hasattr(s_name, 'item'):
            s_name = s_name.item()
        if s_name in valid_names:
            filtered_sessions.append(s)
            
    print(f"Filtered to {len(filtered_sessions)} sessions with ops.npy for fair comparison.")
    X, y, groups = extract_features_dataset(filtered_sessions, ACTIVE_FEATURES)
    print(f"LightGBM Dataset compiled: X={X.shape}, y={y.shape}")
    
    cv_summary, y_true, y_pred, y_prob, shap_values = run_cross_validation(X, y, groups, ACTIVE_FEATURES)
    
    # Extract out-of-fold predictions per session
    lgb_session_metrics = {}
    lgb_global_y_true = []
    lgb_global_y_pred = []
    
    start_idx = 0
    for session in filtered_sessions:
        n_rois = len(session['y'])
        end_idx = start_idx + n_rois
        
        s_name = session['session_name']
        if hasattr(s_name, 'item'):
            s_name = s_name.item()
            
        s_true = y_true[start_idx:end_idx]
        s_pred = y_pred[start_idx:end_idx]
        
        s_f1 = f1_score(s_true, s_pred, zero_division=0)
        s_p = precision_score(s_true, s_pred, zero_division=0)
        s_r = recall_score(s_true, s_pred, zero_division=0)
        
        lgb_session_metrics[s_name] = {
            'F1': float(s_f1),
            'Precision': float(s_p),
            'Recall': float(s_r),
            'ROIs': n_rois,
            'Cells': int(np.sum(s_true))
        }
        lgb_global_y_true.extend(s_true)
        lgb_global_y_pred.extend(s_pred)
        print(f"  LightGBM {s_name}: F1={s_f1:.4f}, Prec={s_p:.4f}, Rec={s_r:.4f}")
        start_idx = end_idx
        
    lgb_global_metrics = {
        'F1': float(f1_score(lgb_global_y_true, lgb_global_y_pred, zero_division=0)),
        'Precision': float(precision_score(lgb_global_y_true, lgb_global_y_pred, zero_division=0)),
        'Recall': float(recall_score(lgb_global_y_true, lgb_global_y_pred, zero_division=0))
    }
    print(f"\nLightGBM Global Metrics on 19 sessions: F1={lgb_global_metrics['F1']:.4f}")
    
    # 3. Load or Initialize Cellpose progress cache
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
            print(f"\nLoaded existing progress cache from {PROGRESS_FILE}. Found {len(progress)} processed sessions.")
        except Exception as cache_err:
            print(f"Error loading progress cache: {cache_err}. Starting fresh.")
            
    # 4. Initialize Cellpose model
    print("\n=== Step 2: Evaluating Cellpose cpdino Model ===")
    from cellpose import models
    model = models.CellposeModel(gpu=False, pretrained_model='cpdino')
    print("Cellpose cpdino model loaded.")
    
    # 5. Run Cellpose evaluation
    for idx, session in enumerate(valid_sessions):
        s_name = session.name
        print(f"\n[{idx+1}/{n_sessions}] Session: {s_name}")
        
        # If already computed, skip
        if s_name in progress:
            print(f"  Loaded from cache ({len(progress[s_name]['ious'])} ROIs).")
            continue
            
        print(f"  Running Cellpose cpdino inference (takes ~10-15 minutes on CPU)...")
        ops = np.load(session / 'ops.npy', allow_pickle=True).item()
        stat = np.load(session / 'stat.npy', allow_pickle=True)
        
        # Load labels
        iscell = None
        for lbl in label_options:
            if (session / lbl).exists():
                iscell = np.load(session / lbl)
                break
        y = iscell[:, 0].astype(int).tolist()
        
        img = ops['meanImgE']
        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)
        
        t0 = time.time()
        # Standard CPU eval
        masks, _, _ = model.eval(img_norm, diameter=12, channels=[0, 0])
        duration = time.time() - t0
        print(f"  Segmentation complete in {duration:.2f} seconds. Found {len(np.unique(masks)) - 1} masks.")
        
        # Compute IoUs
        session_ious = []
        for i in range(len(stat)):
            iou, _ = compute_roi_cellpose_iou(stat[i], masks)
            session_ious.append(float(iou))
            
        # Update progress cache
        progress[s_name] = {
            'y_true': y,
            'ious': session_ious,
            'duration': duration,
            'n_masks': int(len(np.unique(masks)) - 1)
        }
        
        # Atomic write progress to disk
        tmp_file = PROGRESS_FILE.with_suffix(".tmp")
        with open(tmp_file, 'w') as f:
            json.dump(progress, f, indent=4)
        tmp_file.replace(PROGRESS_FILE)
        print(f"  Saved progress to {PROGRESS_FILE}")
        
    # 6. Analyze Cellpose results and optimize IoU threshold
    print("\n=== Step 3: Analyzing Cellpose Results & Optimizing IoU Threshold ===")
    all_y_true = []
    all_max_ious = []
    
    for s_name, data in progress.items():
        all_y_true.extend(data['y_true'])
        all_max_ious.extend(data['ious'])
        
    all_y_true = np.array(all_y_true)
    all_max_ious = np.array(all_max_ious)
    
    best_iou_thresh = 0.0
    best_f1_val = 0.0
    best_prec = 0.0
    best_rec = 0.0
    
    for thresh in np.arange(0.05, 0.95, 0.05):
        preds = (all_max_ious >= thresh).astype(int)
        f1 = f1_score(all_y_true, preds, zero_division=0)
        if f1 > best_f1_val:
            best_f1_val = f1
            best_iou_thresh = thresh
            best_prec = precision_score(all_y_true, preds, zero_division=0)
            best_rec = recall_score(all_y_true, preds, zero_division=0)
            
    print(f"Optimal Cellpose IoU threshold: {best_iou_thresh:.2f}")
    print(f"Cellpose Global F1: {best_f1_val:.4f} (Precision: {best_prec:.4f}, Recall: {best_rec:.4f})")
    
    cellpose_global_metrics = {
        'F1': float(best_f1_val),
        'Precision': float(best_prec),
        'Recall': float(best_rec),
        'optimal_threshold': float(best_iou_thresh)
    }
    
    # Calculate session-specific metrics for Cellpose using optimal threshold
    cellpose_session_metrics = {}
    for s_name, data in progress.items():
        s_true = np.array(data['y_true'])
        s_preds = (np.array(data['ious']) >= best_iou_thresh).astype(int)
        
        s_f1 = f1_score(s_true, s_preds, zero_division=0)
        s_p = precision_score(s_true, s_preds, zero_division=0)
        s_r = recall_score(s_true, s_preds, zero_division=0)
        
        cellpose_session_metrics[s_name] = {
            'F1': float(s_f1),
            'Precision': float(s_p),
            'Recall': float(s_r),
            'ROIs': len(s_true),
            'Cells': int(np.sum(s_true)),
            'n_masks': data['n_masks'],
            'duration': data['duration']
        }
        
    # 7. Generate final comparison report
    print("\nGenerating final markdown comparison report...")
    
    # Session comparison table
    session_table = "| Session Name | ROIs | Cells | LGB F1 | Cellpose F1 (IoU $\\ge$ " + f"{best_iou_thresh:.2f}) | Cellpose Masks | Cellpose Duration (s) |\n|---|---|---|---|---|---|---|\n"
    for s_name in sorted(valid_names):
        lgb_met = lgb_session_metrics.get(s_name, {'F1': 0.0})
        cp_met = cellpose_session_metrics.get(s_name, {'F1': 0.0, 'n_masks': 0, 'duration': 0.0})
        
        session_table += f"| `{s_name}` | {cp_met.get('ROIs', 0)} | {cp_met.get('Cells', 0)} | {lgb_met['F1']:.4f} | {cp_met['F1']:.4f} | {cp_met.get('n_masks', 0)} | {cp_met.get('duration', 0.0):.1f} |\n"
        
    report = f"""# Head-to-Head Comparison: LightGBM vs Cellpose cpdino (Overnight Run)

This report presents a global comparison between the fast LightGBM classifier and the pre-trained Cellpose `cpdino` model evaluated across all **{n_sessions}** valid Suite2p sessions.

---

## 📊 Global Metrics Comparison

| Model | F1-Score | Precision | Recall | Parameters / Details |
|---|---|---|---|---|
| **LightGBM (OOF)** | {lgb_global_metrics['F1']:.4f} | {lgb_global_metrics['Precision']:.4f} | {lgb_global_metrics['Recall']:.4f} | Out-of-fold cross-validation (27 baseline features) |
| **Cellpose cpdino** | {cellpose_global_metrics['F1']:.4f} | {cellpose_global_metrics['Precision']:.4f} | {cellpose_global_metrics['Recall']:.4f} | Optimal IoU threshold = `{cellpose_global_metrics['optimal_threshold']:.2f}` |

---

## 📈 Session-by-Session Performance Breakdown

{session_table}

---

## 💡 Key Takeaways & Recommendations
1. **Performance**: Compare if Cellpose's spatial segmentation outlines outperform LightGBM's calcium trace dynamics and baseline shape features.
2. **Computational Overhead**: Cellpose takes $\sim 10-15$ minutes per session on CPU, while LightGBM takes less than $10$ milliseconds.
"""
    
    with open(REPORT_FILE, 'w') as f:
        f.write(report)
        
    print(f"✅ SUCCESS: Saved report to {REPORT_FILE.resolve()}")
    
    # Save a final comparison JSON file for structured access
    final_data = {
        'lgb_global': lgb_global_metrics,
        'cellpose_global': cellpose_global_metrics,
        'lgb_sessions': lgb_session_metrics,
        'cellpose_sessions': cellpose_session_metrics
    }
    with open("cellpose_final_comparison.json", "w") as f:
        json.dump(final_data, f, indent=4)

if __name__ == "__main__":
    main()
