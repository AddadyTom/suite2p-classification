import os
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from pathlib import Path

def train_lgb_model(data_dir, output_model_path, preset='regular'):
    data_path = Path(data_dir)
    
    # Resolve names based on preset
    if preset == 'rich':
        x_name = 'X_dataset_rich.npy'
        y_name = 'y_dataset_rich.npy'
        groups_name = 'groups_dataset_rich.npy'
        default_out = 'models/rich/suite2p_best_lgb.pkl'
    else:
        x_name = 'X_dataset.npy'
        y_name = 'y_dataset.npy'
        groups_name = 'groups_dataset.npy'
        default_out = 'models/regular/suite2p_best_lgb.pkl'

    model_path = Path(output_model_path) if output_model_path else Path(default_out)
    
    # Load compiled arrays
    print(f"Loading data ({preset} preset) from {data_path.resolve()}...")
    try:
        X = np.load(data_path / x_name)
        y = np.load(data_path / y_name)
        groups = np.load(data_path / groups_name)
    except FileNotFoundError as e:
        print(f"Error: Could not load datasets. Run prepare_data.py first. {e}")
        return
        
    X = np.nan_to_num(X)
    print(f"Dataset Loaded: X={X.shape}, y={y.shape}, Unique Sessions={len(np.unique(groups))}")
    
    # GroupKFold Cross-Validation
    unique_groups = len(np.unique(groups))
    if unique_groups >= 2:
        gkf = GroupKFold(n_splits=min(5, unique_groups))
        splits = list(gkf.split(X, y, groups))
    else:
        print("Warning: Only 1 session group found. Falling back to StratifiedKFold cross-validation...")
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
        
    precisions = []
    recalls = []
    f1s = []
    
    print("\nStarting Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Calculate class balancing weight
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        # Initialize LightGBM with the deployment hyperparameters
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
        
        # Train with early stopping on validation fold
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
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
                
        # Compute final fold metrics
        y_pred = (y_prob >= best_thresh).astype(int)
        fold_p = precision_score(y_val, y_pred)
        fold_r = recall_score(y_val, y_pred)
        fold_f1 = f1_score(y_val, y_pred)
        
        precisions.append(fold_p)
        recalls.append(fold_r)
        f1s.append(fold_f1)
        
        print(f"  Fold {fold+1}: Precision={fold_p:.4f}, Recall={fold_r:.4f}, F1={fold_f1:.4f} (Threshold={best_thresh:.2f})")
        
    print("\n=== Cross-Validation Summary ===")
    print(f"  Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  Average Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  Average F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    # Train final model on 100% of data
    print("\nTraining final model on ALL data...")
    n_pos_all = np.sum(y == 1)
    n_neg_all = np.sum(y == 0)
    scale_pos_weight_all = n_neg_all / n_pos_all if n_pos_all > 0 else 1.0
    
    final_model = lgb.LGBMClassifier(
        n_estimators=500,  # Slightly lower than cv limit since no early stopping
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        scale_pos_weight=scale_pos_weight_all,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    final_model.fit(X, y)
    
    # Save the model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"\n✅ SUCCESS: Final model saved to: {model_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Cell Classifier")
    parser.add_argument('--data', type=str, default=".", help="Directory containing compiled dataset .npy files")
    parser.add_argument('--output', type=str, default=None, help="Output path for the trained model .pkl")
    parser.add_argument('--preset', type=str, choices=['regular', 'rich'], default="regular", help="Feature set preset ('regular' or 'rich')")
    args = parser.parse_args()
    
    train_lgb_model(args.data, args.output, args.preset)
