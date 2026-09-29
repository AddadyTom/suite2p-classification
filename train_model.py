import os
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import json
from pathlib import Path
from fe_engine.fe_loop_runner import load_preprocessed_data, extract_features_dataset
from fe_engine.fe_definitions import ACTIVE_FEATURES

try:
    import xgboost as xgb_lib
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def train_and_save(output_path=None, model_name=None, description=None, features_list=None,
                   include_bright_pixels=False, deep=False, use_xgb=False):
    base_dir = Path(__file__).parent.resolve()
    
    target_features = list(features_list) if features_list else list(ACTIVE_FEATURES)
    if include_bright_pixels and 'number_of_bright_pixels' not in target_features:
        target_features.append('number_of_bright_pixels')
        target_features = sorted(target_features)

    if output_path:
        out_p = Path(output_path)
        if not out_p.is_absolute():
            out_p = base_dir / out_p
    else:
        out_p = base_dir / "models" / "suite2p_best_lgb.pkl"
        
    out_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p = out_p.with_suffix(".json")

    disp_name = model_name if model_name else out_p.stem.replace("_", " ").title()
    disp_desc = description if description else f"LightGBM classifier trained on {len(target_features)} active features."

    print(f"Starting training pipeline for '{disp_name}' with {len(target_features)} active features...")
    
    # 1. Load preprocessed sessions
    cache_dir = base_dir / "preprocessed_cache"
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
    X, y, groups = extract_features_dataset(stav_sessions, target_features)
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
        
        if use_xgb:
            if not HAS_XGB:
                raise ImportError("xgboost is not installed. Run: pip install xgboost")
            model = xgb_lib.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=8,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                eval_metric='logloss',
                early_stopping_rounds=50,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            _max_depth = 12 if deep else 8
            _num_leaves = 127 if deep else 63
            model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=_max_depth,
                num_leaves=_num_leaves,
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
    
    if use_xgb:
        if not HAS_XGB:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        final_model = xgb_lib.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            scale_pos_weight=scale_pos_weight_all,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        final_model.fit(X, y)
    else:
        _max_depth = 12 if deep else 8
        _num_leaves = 127 if deep else 63
        final_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=_max_depth,
            num_leaves=_num_leaves,
            scale_pos_weight=scale_pos_weight_all,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        final_model.fit(X, y, feature_name=target_features)
    
    joblib.dump(final_model, out_p)
    meta_data = {
        'name': disp_name,
        'description': disp_desc,
        'active_features': target_features,
        'custom_features': {}
    }
    with open(meta_p, 'w') as f:
        json.dump(meta_data, f, indent=4)
        
    print(f'\n Model successfully saved to {out_p.resolve()}')
    print(f' Sibling metadata saved to {meta_p.resolve()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Cell Classifier Model")
    parser.add_argument('--output', '-o', type=str, default=None, help="Output .pkl file path (e.g. models/my_model.pkl)")
    parser.add_argument('--name', '-n', type=str, default=None, help="Human-readable model name")
    parser.add_argument('--description', '-d', type=str, default=None, help="Model description")
    parser.add_argument('--include-bright-pixels', action='store_true', help="Include number_of_bright_pixels feature (27 features total)")
    parser.add_argument('--features', type=str, default=None, help="Comma-separated list of features to use (overrides ACTIVE_FEATURES)")
    parser.add_argument('--deep', action='store_true', help="Use deeper trees: max_depth=12, num_leaves=127")
    parser.add_argument('--xgb', action='store_true', help="Use XGBoost instead of LightGBM")
    args = parser.parse_args()

    features_list = [f.strip() for f in args.features.split(",")] if args.features else None

    train_and_save(
        output_path=args.output,
        model_name=args.name,
        description=args.description,
        include_bright_pixels=args.include_bright_pixels,
        features_list=features_list,
        deep=args.deep,
        use_xgb=args.xgb,
    )
