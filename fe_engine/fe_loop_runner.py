import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import shap
import warnings
import gc

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure workspace root is in python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from fe_engine.fe_definitions import ACTIVE_FEATURES, FEATURE_REGISTRY

def load_preprocessed_data(cache_dir):
    """
    Load all preprocessed .npz session files from the cache directory.
    """
    cache_path = Path(cache_dir)
    npz_files = list(cache_path.glob("preprocessed_*.npz"))
    
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No preprocessed session files found in '{cache_path.resolve()}'. Run fe_preprocessor.py first.")
        
    sessions = []
    for f in sorted(npz_files):
        data = np.load(f, allow_pickle=True)
        # Convert npz fields to dictionary so they are easy to use
        session_dict = {key: data[key] for key in data.files}
        sessions.append(session_dict)
        
    print(f"Loaded {len(sessions)} preprocessed sessions from cache.")
    return sessions

def extract_features_dataset(sessions, active_features, session_callback=None):
    """
    Extract the specified features for all sessions and concatenate them.
    We pass a single cache dictionary context to each extractor, caching features
    as they are evaluated. If a trace feature is missing, we load F and Fneu from
    the raw session folder on-demand using memmap, and free them immediately.
    """
    import fe_engine.fe_definitions
    FEATURE_REGISTRY = fe_engine.fe_definitions.FEATURE_REGISTRY
    X_list = []
    y_list = []
    groups_list = []
    
    for idx, session in enumerate(sessions):
        if session_callback:
            try:
                session_callback(idx + 1, len(sessions), session.get('session_name', f"Session {idx+1}"))
            except:
                pass
        import time
        time.sleep(0.005)
        # Create a dynamic evaluation context/cache for this session
        eval_cache = session.copy()
        
        session_features = []
        for name in active_features:
            if name not in FEATURE_REGISTRY:
                raise ValueError(f"Feature '{name}' is not defined in fe_definitions.py FEATURE_REGISTRY.")
            
            # If the feature has not been computed in this context, compute and cache it
            if name not in eval_cache:
                # Load raw traces on demand using memory mapping if not loaded
                if 'F' not in eval_cache:
                    # session_path can be saved as array in npz, extract it as string
                    raw_path_str = eval_cache['session_path']
                    if isinstance(raw_path_str, np.ndarray):
                        raw_path_str = raw_path_str.item()
                    raw_path = Path(raw_path_str)
                    
                    eval_cache['F'] = np.load(raw_path / 'F.npy', mmap_mode='r')
                    eval_cache['Fneu'] = np.load(raw_path / 'Fneu.npy', mmap_mode='r')
                
                feat_val = FEATURE_REGISTRY[name](eval_cache)
                eval_cache[name] = feat_val
            else:
                feat_val = eval_cache[name]
                
            session_features.append(feat_val.reshape(-1, 1))
            
        X_sess = np.hstack(session_features)
        X_list.append(X_sess)
        y_list.append(session['y'])
        groups_list.append(session['group_id'])
        
        # Clean up loaded trace arrays immediately to save memory
        if 'F' in eval_cache:
            del eval_cache['F']
            del eval_cache['Fneu']
            gc.collect()
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)
    
    return np.nan_to_num(X), y, groups


def run_cross_validation(X, y, groups, feature_names, progress_callback=None):
    """
    Perform 5-fold GroupKFold cross-validation and compute SHAP values.
    """
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    splits = list(gkf.split(X, y, groups=groups))
    
    n_samples, n_features = X.shape
    all_y_true = np.zeros(n_samples)
    all_y_pred = np.zeros(n_samples)
    all_y_prob = np.zeros(n_samples)
    all_shap_values = np.zeros((n_samples, n_features))
    
    precisions = []
    recalls = []
    f1s = []
    thresholds = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        if progress_callback:
            try:
                progress_callback(fold + 1, len(splits))
            except:
                pass
        import time
        time.sleep(0.005)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Calculate class weights for balancing
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        # Train LightGBM model with fixed estimators to avoid leakage
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        model.fit(X_train, y_train)
        
        # Optimize threshold for F1-score on validation set
        y_prob = model.predict_proba(X_val)[:, 1]
        best_thresh = 0.5
        best_f1 = 0.0
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        thresholds.append(best_thresh)
        y_pred = (y_prob >= best_thresh).astype(int)
        
        # Collect results
        all_y_true[val_idx] = y_val
        all_y_pred[val_idx] = y_pred
        all_y_prob[val_idx] = y_prob
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val)
        
        # Handle SHAP shape conventions
        if isinstance(shap_vals, list):
            shap_vals_class1 = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            shap_vals_class1 = shap_vals[:, :, 1]
        else:
            shap_vals_class1 = shap_vals
            
        all_shap_values[val_idx] = shap_vals_class1
        
        # Fold metrics
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        
    cv_summary = {
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1': float(np.mean(f1s)),
        'precision_std': float(np.std(precisions)),
        'recall_std': float(np.std(recalls)),
        'f1_std': float(np.std(f1s)),
        'threshold': float(np.mean(thresholds)),
    }
    
    return cv_summary, all_y_true, all_y_pred, all_y_prob, all_shap_values

def analyze_errors_and_shap(X, y_true, y_pred, shap_values, feature_names):
    """
    Identify culprit features for False Positives and False Negatives.
    """
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    num_features = X.shape[1]
    
    # FP Culprits (highest positive SHAP impact on false positives compared to true negatives)
    fp_culprits = []
    if np.sum(fp_mask) > 0:
        mean_fp_shaps = np.mean(shap_values[fp_mask], axis=0)
        mean_tn_shaps = np.mean(shap_values[tn_mask], axis=0) if np.sum(tn_mask) > 0 else np.zeros(num_features)
        diff_shaps = mean_fp_shaps - mean_tn_shaps
        sorted_indices = np.argsort(diff_shaps)[::-1][:3]
        
        for idx in sorted_indices:
            fp_culprits.append({
                'feature': feature_names[idx],
                'shap_diff': float(diff_shaps[idx]),
                'fp_mean_val': float(np.mean(X[fp_mask, idx])),
                'tn_mean_val': float(np.mean(X[tn_mask, idx]))
            })
            
    # FN Culprits (highest negative/missing positive SHAP impact on false negatives compared to true positives)
    fn_culprits = []
    if np.sum(fn_mask) > 0:
        mean_fn_shaps = np.mean(shap_values[fn_mask], axis=0)
        mean_tp_shaps = np.mean(shap_values[tp_mask], axis=0) if np.sum(tp_mask) > 0 else np.zeros(num_features)
        diff_shaps = mean_tp_shaps - mean_fn_shaps
        sorted_indices = np.argsort(diff_shaps)[::-1][:3]
        
        for idx in sorted_indices:
            fn_culprits.append({
                'feature': feature_names[idx],
                'shap_diff': float(diff_shaps[idx]),
                'fn_mean_val': float(np.mean(X[fn_mask, idx])),
                'tp_mean_val': float(np.mean(X[tp_mask, idx]))
            })
            
    # Compute overall feature importance (mean absolute SHAP)
    mean_abs_shaps = np.mean(np.abs(shap_values), axis=0)
    importance = []
    for i, name in enumerate(feature_names):
        importance.append((name, float(mean_abs_shaps[i])))
    importance = sorted(importance, key=lambda x: x[1], reverse=True)
    
    return fp_culprits, fn_culprits, importance

def save_baseline_report(cv_summary, importance, fp_culprits, fn_culprits, active_features, report_path, run_type="Evaluation"):
    """
    Generate the markdown iteration report.
    """
    # Check for collinearity
    active_feat_str = "\n".join([f"- `{f}`" for f in active_features])
    
    fp_table = "| Rank | Feature | SHAP Diff (FP - TN) | FP Mean | TN Mean |\n|---|---|---|---|---|\n"
    for i, c in enumerate(fp_culprits):
        fp_table += f"| {i+1} | `{c['feature']}` | {c['shap_diff']:+.4f} | {c['fp_mean_val']:.4f} | {c['tn_mean_val']:.4f} |\n"
        
    fn_table = "| Rank | Feature | SHAP Diff (TP - FN) | FN Mean | TP Mean |\n|---|---|---|---|---|\n"
    for i, c in enumerate(fn_culprits):
        fn_table += f"| {i+1} | `{c['feature']}` | {c['shap_diff']:+.4f} | {c['fn_mean_val']:.4f} | {c['tp_mean_val']:.4f} |\n"
        
    imp_table = "| Rank | Feature | Mean Absolute SHAP |\n|---|---|---|\n"
    for i, (name, val) in enumerate(importance[:15]):
        imp_table += f"| {i+1} | `{name}` | {val:.5f} |\n"
        
    report = f"""# Feature Engineering Iteration Report ({run_type})

## Performance Summary
- **F1-Score**: {cv_summary['f1']:.4f} ± {cv_summary['f1_std']:.4f}
- **Precision**: {cv_summary['precision']:.4f} ± {cv_summary['precision_std']:.4f}
- **Recall**: {cv_summary['recall']:.4f} ± {cv_summary['recall_std']:.4f}

## Active Features ({len(active_features)})
{active_feat_str}

## SHAP Feature Importance (Top 15)
{imp_table}

## Error Diagnostic Analysis

### False Positive Analysis (Noise predicted as Cells)
Features that failed to assign negative weight to reject noise:
{fp_table}

### False Negative Analysis (Cells predicted as Noise)
Features that failed to assign positive weight to identify cells:
{fn_table}

## Recommendations & Next Steps
1. **FP Culprits**: Examine if normalizing these features or adding high-frequency noise parameters will help reject noise artifacts.
2. **FN Culprits**: Look for spatial or trace metrics that differentiate the missed cells from noise (e.g. baseline-subtracted variances or asymmetric peak shapes).
"""
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\nSaved detailed markdown report to: {report_path.resolve()}")

def update_definitions_file(new_active_features):
    """
    Rewrite ACTIVE_FEATURES in fe_definitions.py with the accepted feature list.
    """
    def_path = Path(__file__).parent / "fe_definitions.py"
    with open(def_path, 'r') as f:
        content = f.read()
        
    # We want to replace ACTIVE_FEATURES = [ ... ]
    # First, let's find the start of ACTIVE_FEATURES
    start_idx = content.find("ACTIVE_FEATURES = [")
    if start_idx == -1:
        print("Warning: Could not find ACTIVE_FEATURES list in fe_definitions.py to update.")
        return
        
    end_idx = content.find("]", start_idx)
    if end_idx == -1:
        print("Warning: Could not find end of ACTIVE_FEATURES list in fe_definitions.py.")
        return
        
    # Generate the new list representation
    new_list_str = "ACTIVE_FEATURES = [\n"
    for i in range(0, len(new_active_features), 4):
        chunk = new_active_features[i:i+4]
        new_list_str += "    " + ", ".join([f"'{f}'" for f in chunk])
        if i + 4 < len(new_active_features):
            new_list_str += ",\n"
        else:
            new_list_str += "\n"
    new_list_str += "]"
    
    # Replace in file content
    updated_content = content[:start_idx] + new_list_str + content[end_idx+1:]
    
    with open(def_path, 'w') as f:
        f.write(updated_content)
        
    print(f"Updated active features in fe_definitions.py.")

def main():
    workspace_root = Path(__file__).parent.parent.resolve()
    cache_dir = workspace_root / "preprocessed_cache"
    baseline_json_path = workspace_root / "fe_baseline.json"
    report_path = workspace_root / "fe_report.md"
    
    # Load preprocessed cache
    try:
        sessions = load_preprocessed_data(cache_dir)
    except Exception as e:
        print(f"Error loading cache: {e}")
        sys.exit(1)
        
    # Step 1: Extract features
    print(f"Extracting features: {ACTIVE_FEATURES}")
    X, y, groups = extract_features_dataset(sessions, ACTIVE_FEATURES)
    print(f"Dataset compiled: X={X.shape}, y={y.shape}, Unique Groups={len(np.unique(groups))}")
    
    # Step 2: Cross Validation
    print("Running GroupKFold Cross-Validation...")
    cv_summary, y_true, y_pred, y_prob, shap_values = run_cross_validation(X, y, groups, ACTIVE_FEATURES)
    
    print("\n=== Cross-Validation Results ===")
    print(f"  F1 Score:  {cv_summary['f1']:.4f} ± {cv_summary['f1_std']:.4f}")
    print(f"  Precision: {cv_summary['precision']:.4f} ± {cv_summary['precision_std']:.4f}")
    print(f"  Recall:    {cv_summary['recall']:.4f} ± {cv_summary['recall_std']:.4f}")
    
    # Step 3: SHAP diagnostics
    fp_culprits, fn_culprits, importance = analyze_errors_and_shap(X, y_true, y_pred, shap_values, ACTIVE_FEATURES)
    
    # Step 4: Automatic pruning of zero-contribution features
    pruned_features = []
    new_active_features = ACTIVE_FEATURES.copy()
    for name, val in importance:
        if val < 0.0001:
            pruned_features.append(name)
            new_active_features.remove(name)
            
    if len(pruned_features) > 0:
        print(f"\n⚠️ PRUNING SUGGESTION: The following features have near-zero SHAP impact (< 0.0001) and can be pruned: {pruned_features}")
        
    # Step 5: Check against baseline for new candidate features
    is_candidate_evaluation = False
    
    # Let's check if we have a baseline saved
    if baseline_json_path.exists():
        with open(baseline_json_path, 'r') as f:
            baseline = json.load(f)
        
        # Compare if ACTIVE_FEATURES differs from the baseline features
        if set(ACTIVE_FEATURES) != set(baseline['features']):
            is_candidate_evaluation = True
            diff_f1 = cv_summary['f1'] - baseline['f1']
            print(f"\nEvaluating candidate feature set against baseline (F1={baseline['f1']:.4f}):")
            print(f"  Candidate F1: {cv_summary['f1']:.4f} (Diff: {diff_f1:+.4f})")
            
            # Acceptance rule: F1 must improve by at least +0.002
            if diff_f1 >= 0.002:
                print("  ✅ ACCEPTED: Candidate feature set meets the +0.002 improvement threshold!")
                # Save as the new baseline
                new_baseline = {
                    'f1': cv_summary['f1'],
                    'precision': cv_summary['precision'],
                    'recall': cv_summary['recall'],
                    'features': ACTIVE_FEATURES
                }
                with open(baseline_json_path, 'w') as f:
                    json.dump(new_baseline, f, indent=4)
                print("  Saved new baseline.")
            else:
                print("  ❌ REJECTED: Candidate does not meet the improvement threshold.")
                # Suggest rolling back fe_definitions.py
                print(f"  Please roll back ACTIVE_FEATURES in fe_definitions.py to the baseline set: {baseline['features']}")
                
    else:
        # Save baseline for the first time
        baseline = {
            'f1': cv_summary['f1'],
            'precision': cv_summary['precision'],
            'recall': cv_summary['recall'],
            'features': ACTIVE_FEATURES
        }
        with open(baseline_json_path, 'w') as f:
            json.dump(baseline, f, indent=4)
        print(f"\nSaved initial baseline metrics to: {baseline_json_path.name}")
        
    # Save markdown report
    save_baseline_report(
        cv_summary, 
        importance, 
        fp_culprits, 
        fn_culprits, 
        ACTIVE_FEATURES, 
        report_path, 
        run_type="Candidate Run" if is_candidate_evaluation else "Baseline Run"
    )
    
    # If pruning was suggested, let's offer to update it automatically
    if len(pruned_features) > 0 and not is_candidate_evaluation:
        # Update definitions file automatically to keep it clean
        update_definitions_file(new_active_features)

if __name__ == "__main__":
    main()
