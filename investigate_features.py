import os
import sys
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure workspace root is in python path
sys.path.append(str(Path(__file__).parent.resolve()))

def analyze_features(data_path_str, source_path_str):
    data_path = Path(data_path_str)
    source_path = Path(source_path_str) if source_path_str else None

    # 1. Load compiled dataset
    print(f"Loading data from {data_path.resolve()}...")
    try:
        X = np.load(data_path / 'X_dataset.npy')
        y = np.load(data_path / 'y_dataset.npy')
        groups = np.load(data_path / 'groups_dataset.npy')
    except FileNotFoundError as e:
        print(f"Error: Missing compiled dataset files. Run prepare_data.py first. Details: {e}")
        return

    num_samples, num_features = X.shape
    print(f"Dataset Loaded: {num_samples} ROIs, {num_features} Features, {len(np.unique(groups))} Sessions")

    # 2. Get feature names
    if num_features == 25:
        from apply_AI import FEATURE_NAMES_25
        feature_names = FEATURE_NAMES_25
    elif num_features == 27:
        from apply_AI import FEATURE_NAMES_27
        feature_names = FEATURE_NAMES_27
    else:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    # 3. Map group IDs to session names if source path is provided
    group_to_session = {}
    if source_path and source_path.exists():
        session_folders = []
        for p in source_path.rglob('stat.npy'):
            session_folders.append(p.parent)
        session_folders.sort()
        group_to_session = {i: folder.name for i, folder in enumerate(session_folders)}
        print(f"Mapped {len(group_to_session)} group IDs to session directory names.")
    else:
        group_to_session = {i: f"Group_{i}" for i in np.unique(groups)}
        print("No source directory provided. Session names will fallback to Group IDs.")

    # 4. Perform 5-fold CV and collect predictions and SHAP values
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups=groups))

    # Containers for out-of-fold metrics and SHAP values
    all_y_true = np.zeros(num_samples)
    all_y_pred = np.zeros(num_samples)
    all_y_prob = np.zeros(num_samples)
    all_shap_values = np.zeros((num_samples, num_features))
    session_metrics = {}

    print("\nRunning Cross-Validation and SHAP explanations (this may take a moment)...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        groups_val = groups[val_idx]

        # Calculate class balancing weight
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Model definition
        model = lgb.LGBMClassifier(
            n_estimators=1000,
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

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # Optimize threshold
        y_prob = model.predict_proba(X_val)[:, 1]
        best_thresh = 0.5
        best_f1 = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        y_pred = (y_prob >= best_thresh).astype(int)

        # Store out-of-fold predictions
        all_y_true[val_idx] = y_val
        all_y_pred[val_idx] = y_pred
        all_y_prob[val_idx] = y_prob

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val)

        # Handle different SHAP output formats (list or ndarray)
        if isinstance(shap_vals, list):
            shap_vals_class1 = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            shap_vals_class1 = shap_vals[:, :, 1]
        else:
            shap_vals_class1 = shap_vals

        all_shap_values[val_idx] = shap_vals_class1

        # Track session-specific performance in this fold
        for u_group in np.unique(groups_val):
            session_name = group_to_session.get(u_group, f"Group_{u_group}")
            session_mask = (groups_val == u_group)
            
            s_true = y_val[session_mask]
            s_pred = y_pred[session_mask]
            
            s_p = precision_score(s_true, s_pred, zero_division=0)
            s_r = recall_score(s_true, s_pred, zero_division=0)
            s_f1 = f1_score(s_true, s_pred, zero_division=0)
            
            session_metrics[session_name] = {
                'F1': s_f1,
                'Precision': s_p,
                'Recall': s_r,
                'ROIs': len(s_true),
                'Cells': np.sum(s_true)
            }

        print(f"  Fold {fold+1} Completed (using Threshold={best_thresh:.2f})")

    # 5. Output Session Performance Table
    print("\n" + "="*70)
    print(" SESSION PERFORMANCE BREAKDOWN (Sorted by F1-Score Descending)")
    print("="*70)
    session_df = pd.DataFrame.from_dict(session_metrics, orient='index')
    session_df = session_df.sort_values(by='F1', ascending=False)
    print(session_df.to_string())

    # 6. Global CV Summary
    overall_p = precision_score(all_y_true, all_y_pred)
    overall_r = recall_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred)
    print("\n" + "="*70)
    print(f" GLOBAL CV METRICS: Precision={overall_p:.4f}, Recall={overall_r:.4f}, F1={overall_f1:.4f}")
    print("="*70)

    # 7. Identify Error Masks
    tp_mask = (all_y_true == 1) & (all_y_pred == 1)
    tn_mask = (all_y_true == 0) & (all_y_pred == 0)
    fp_mask = (all_y_true == 0) & (all_y_pred == 1)  # Predicted cell, is noise
    fn_mask = (all_y_true == 1) & (all_y_pred == 0)  # Predicted noise, is cell

    num_fp = np.sum(fp_mask)
    num_fn = np.sum(fn_mask)
    print(f"Total Errors Analyzed: {num_fp} False Positives, {num_fn} False Negatives")

    # Helper function to compute and format diagnostic summaries
    def analyze_error_group(error_mask, is_fp=True):
        if np.sum(error_mask) == 0:
            print("  No error cases found to analyze.")
            return

        # Mean SHAP impact of features on this error subset
        err_shaps = all_shap_values[error_mask]
        mean_shaps = np.mean(err_shaps, axis=0)

        # Mean SHAP values for correct classes
        tp_shaps = all_shap_values[tp_mask]
        tn_shaps = all_shap_values[tn_mask]
        mean_tp_shaps = np.mean(tp_shaps, axis=0) if len(tp_shaps) > 0 else np.zeros(num_features)
        mean_tn_shaps = np.mean(tn_shaps, axis=0) if len(tn_shaps) > 0 else np.zeros(num_features)

        # Sort features based on direction of interest
        if is_fp:
            # For FP, we want to know what features have the most positive impact (pushing towards 1)
            sorted_indices = np.argsort(mean_shaps)[::-1]
        else:
            # For FN, we want to know what features have the most negative impact (pushing towards 0)
            sorted_indices = np.argsort(mean_shaps)

        # Get top 5 culprit features
        culprits = sorted_indices[:5]
        
        # Calculate mean physical values for classes
        mean_vals_error = np.mean(X[error_mask], axis=0)
        mean_vals_tp = np.mean(X[tp_mask], axis=0) if np.sum(tp_mask) > 0 else np.zeros(num_features)
        mean_vals_tn = np.mean(X[tn_mask], axis=0) if np.sum(tn_mask) > 0 else np.zeros(num_features)

        print("\n  Rank | Feature Name             | SHAP (Error)| SHAP (Correct) | Error Mean | TP (Cell) Mean | TN (Noise) Mean")
        print("  " + "-"*109)
        for rank, idx in enumerate(culprits):
            name = feature_names[idx]
            shap_imp = mean_shaps[idx]
            shap_correct = mean_tn_shaps[idx] if is_fp else mean_tp_shaps[idx]
            val_err = mean_vals_error[idx]
            val_tp = mean_vals_tp[idx]
            val_tn = mean_vals_tn[idx]
            print(f"  {rank+1:4d} | {name:24s} | {shap_imp:+11.4f} | {shap_correct:+14.4f} | {val_err:10.4f} | {val_tp:14.4f} | {val_tn:15.4f}")

        # Show Missing Impact comparison
        if is_fp:
            # Missing negative rejection impact in FP: FP SHAP - TN SHAP (should be positive/large)
            diff_shaps = mean_shaps - mean_tn_shaps
            print("\n  Missing Rejection Impact (Features that failed to assign negative weight to reject noise):")
            print("  Rank | Feature Name             | SHAP Difference (FP - TN) | FP Mean    | TN Mean")
            print("  " + "-"*83)
            diff_indices = np.argsort(diff_shaps)[::-1][:5]
        else:
            # Missing positive cell identification impact in FN: TP SHAP - FN SHAP (should be positive/large)
            diff_shaps = mean_tp_shaps - mean_shaps
            print("\n  Missing Identification Impact (Features that failed to assign positive weight to identify cell):")
            print("  Rank | Feature Name             | SHAP Difference (TP - FN) | FN Mean    | TP Mean")
            print("  " + "-"*83)
            diff_indices = np.argsort(diff_shaps)[::-1][:5]

        for rank, idx in enumerate(diff_indices):
            name = feature_names[idx]
            d_shap = diff_shaps[idx]
            val_err = mean_vals_error[idx]
            val_ref = mean_vals_tn[idx] if is_fp else mean_vals_tp[idx]
            print(f"  {rank+1:4d} | {name:24s} | {d_shap:+25.4f} | {val_err:10.4f} | {val_ref:10.4f}")

    print("\n" + "="*70)
    print(" FALSE POSITIVE ANALYSIS: What features pushed noise to be classified as cells?")
    print("="*70)
    analyze_error_group(fp_mask, is_fp=True)

    print("\n" + "="*70)
    print(" FALSE NEGATIVE ANALYSIS: What features pushed cells to be classified as noise?")
    print("="*70)
    analyze_error_group(fn_mask, is_fp=False)
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative Feature Investigation Tool (SHAP-Driven)")
    parser.add_argument('--data', type=str, default=".", help="Directory containing compiled dataset .npy files")
    parser.add_argument('--source', type=str, default="/mnt/other_ubunthu/mnt/data", help="Raw dataset root directory for session folders mapping")
    args = parser.parse_args()

    analyze_features(args.data, args.source)
