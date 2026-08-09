import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import json
from pathlib import Path
from fe_engine.fe_loop_runner import load_preprocessed_data, extract_features_dataset
from fe_engine.fe_definitions import ACTIVE_FEATURES

def train_and_save():
    print(f"Starting training pipeline with {len(ACTIVE_FEATURES)} active features...")
    
    # 1. Load preprocessed sessions
    cache_dir = Path("preprocessed_cache")
    if not cache_dir.exists():
        raise FileNotFoundError("preprocessed_cache directory not found! Run preprocessing first.")
        
    sessions = load_preprocessed_data(cache_dir)
    
    # 2. Filter: Train on STAV models only (DO NOT TRAIN ON YAEL MODELS)
    stav_sessions = []
    for s in sessions:
        name = str(s.get('session_name', '')).lower()
        path = str(s.get('session_path', '')).lower()
        if 'yael' in name or 'yael' in path:
            print(f" Skipping Yael session: {s.get('session_name')}")
            continue
        if 'stav22' in name or 'stav22' in path:
            print(f" Skipping Stav22 session (held out for validation): {s.get('session_name')}")
            continue
        stav_sessions.append(s)
        
    print(f"\nFiltered training set: {len(stav_sessions)} Stav sessions (excluded Yael sessions).")
    
    # 3. Compile datasets
    X, y, groups = extract_features_dataset(stav_sessions, ACTIVE_FEATURES)
    print(f"Dataset compiled: X={X.shape}, y={y.shape}, Positive={np.sum(y==1)}, Negative={np.sum(y==0)}")
    
    # 4. 5-Fold GroupKFold Cross-Validation
    unique_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, unique_groups))
    splits = list(gkf.split(X, y, groups))
    
    precisions, recalls, f1s = [], [], []
    
    print('\nEvaluating 5-Fold GroupKFold Cross-Validation:')
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
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
        
        y_prob = model.predict_proba(X_val)[:, 1]
        
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (y_prob >= thresh).astype(int)
            score = f1_score(y_val, preds)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh
                
        y_pred = (y_prob >= best_thresh).astype(int)
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f = f1_score(y_val, y_pred, zero_division=0)
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        print(f'  Fold {fold+1}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f} (Optimal Thresh={best_thresh:.2f})')
        
    print('\n' + '='*60)
    print('CROSS-VALIDATION RESULTS (STAV ONLY):')
    print(f'  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}')
    print(f'  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}')
    print(f'  F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print('='*60)
    
    # 5. Train Final Model on 100% of Stav data
    print('\nTraining Final Production Model on 100% of Stav data...')
    n_pos_all = np.sum(y == 1)
    n_neg_all = np.sum(y == 0)
    scale_pos_weight_all = n_neg_all / n_pos_all if n_pos_all > 0 else 1.0
    
    final_model = lgb.LGBMClassifier(
        n_estimators=500,
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
    final_model.fit(X, y, feature_name=ACTIVE_FEATURES)
    
    # Save model and metadata config
    model_dir = Path("models/regular")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "suite2p_best_lgb.pkl"
    meta_path = model_dir / "suite2p_best_lgb.json"
    
    joblib.dump(final_model, model_path)
    with open(meta_path, 'w') as f:
        json.dump({'active_features': ACTIVE_FEATURES, 'custom_features': {}}, f, indent=4)
        
    print(f'\n Model successfully saved to {model_path.resolve()}')
    print(f' Sibling metadata saved to {meta_path.resolve()}')

if __name__ == "__main__":
    train_and_save()
