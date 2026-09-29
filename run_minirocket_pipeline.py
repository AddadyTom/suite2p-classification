import os
import gc
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score

from apply_AI import FEATURE_NAMES_27
from fe_engine.fe_loop_runner import load_preprocessed_data, extract_features_dataset
from minirocket_embedder import MiniRocketExtractor, extract_session_minirocket

def main():
    cache_dir = Path("preprocessed_cache")
    if not cache_dir.exists():
        raise FileNotFoundError("preprocessed_cache directory not found!")
        
    sessions = load_preprocessed_data(cache_dir)
    
    # Filter STAV only (excluding Yael and Stav22)
    stav_sessions = []
    stav22_session = None
    
    for s in sessions:
        name = str(s.get('session_name', '')).lower()
        path = str(s.get('session_path', '')).lower()
        if 'yael' in name or 'yael' in path:
            continue
        if 'stav22' in name or 'stav22' in path:
            stav22_session = s
            continue
        stav_sessions.append(s)
        
    if stav22_session is None:
        stav22_raw = Path("/mnt/other_ubunthu/mnt/data/22-ordered/stav22")
        val_stat = np.load(stav22_raw / "stat.npy", allow_pickle=True)
        val_iscell = np.load(stav22_raw / "iscell.npy")
        stav22_session = {
            'session_name': 'stav22',
            'session_path': str(stav22_raw),
            'stat': val_stat,
            'iscell': val_iscell
        }
        
    print(f"Loaded {len(stav_sessions)} Stav training sessions. Loaded Stav22 validation session.")
    
    # 1. Initialize MiniRocket with 32 kernels x 6 dilations = 384 features
    num_kernels = 32
    max_dilations = 6
    print(f"\n--- Initializing MiniRocket (kernels={num_kernels}, dilations={max_dilations}) ---")
    extractor = MiniRocketExtractor(num_kernels=num_kernels, max_dilations=max_dilations, seed=42)
    
    # Fit biases on a small subset of traces from the first session
    first_path_str = stav_sessions[0]['session_path']
    if isinstance(first_path_str, np.ndarray):
        first_path_str = first_path_str.item()
    first_raw = Path(first_path_str)
    sample_F = np.load(first_raw / 'F.npy', mmap_mode='r')[:50]
    sample_Fneu = np.load(first_raw / 'Fneu.npy', mmap_mode='r')[:50]
    sample_Fc = (sample_F - 0.7 * sample_Fneu)
    sample_Fc = sample_Fc / np.maximum(np.median(np.std(sample_Fc, axis=1)), 1.0)
    sample_Fc_ds = sample_Fc[:, ::4]
    extractor.fit_biases(sample_Fc_ds)
    del sample_F, sample_Fneu, sample_Fc, sample_Fc_ds
    gc.collect()
    
    # 2. Extract MiniRocket features session-by-session for training set
    train_emb_file = Path("minirocket_train_embeddings.npy")
    val_emb_file = Path("minirocket_val_embeddings.npy")
    
    if train_emb_file.exists():
        print(f"\nLoading existing training MiniRocket embeddings from {train_emb_file}...")
        X_train_rocket = np.load(train_emb_file)
    else:
        print("\n--- Extracting MiniRocket Features for Training Sessions ---")
        train_embeddings_list = []
        for s in stav_sessions:
            raw_path_str = s['session_path']
            if isinstance(raw_path_str, np.ndarray):
                raw_path_str = raw_path_str.item()
            print(f"Extracting MiniRocket for {s['session_name']}...")
            emb = extract_session_minirocket(raw_path_str, extractor, downsample_factor=4)
            train_embeddings_list.append(emb)
            gc.collect()
            
        X_train_rocket = np.vstack(train_embeddings_list)
        np.save(train_emb_file, X_train_rocket)
        
    print(f"Total training MiniRocket shape: {X_train_rocket.shape}")
    
    # 3. Extract MiniRocket features for Stav22 validation set
    if val_emb_file.exists():
        print(f"Loading existing Stav22 MiniRocket embeddings from {val_emb_file}...")
        X_val_rocket = np.load(val_emb_file)
    else:
        print("\n--- Extracting MiniRocket Features for Stav22 Validation Session ---")
        val_raw_str = stav22_session['session_path']
        if isinstance(val_raw_str, np.ndarray):
            val_raw_str = val_raw_str.item()
        X_val_rocket = extract_session_minirocket(val_raw_str, extractor, downsample_factor=4)
        np.save(val_emb_file, X_val_rocket)
        
    print(f"Stav22 MiniRocket shape: {X_val_rocket.shape}")
    
    # 4. Extract baseline 27 features and combine
    print("\n--- Compiling baseline features ---")
    X_baseline_train, y_train, groups_train = extract_features_dataset(stav_sessions, FEATURE_NAMES_27)
    X_baseline_val, y_val, groups_val = extract_features_dataset([stav22_session], FEATURE_NAMES_27)
    
    X_comb_train = np.hstack([X_baseline_train, X_train_rocket])
    X_comb_val = np.hstack([X_baseline_val, X_val_rocket])
    print(f"Combined Train shape: {X_comb_train.shape}, Combined Val shape: {X_comb_val.shape}")
    
    # 5. Run 5-fold GroupKFold Cross-Validation
    print("\n--- 5-Fold GroupKFold Cross-Validation (Baseline + MiniRocket) ---")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_comb_train, y_train, groups_train))
    precisions, recalls, f1s = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, y_tr = X_comb_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_comb_train[va_idx], y_train[va_idx]
        
        n_pos = np.sum(y_tr == 1)
        n_neg = np.sum(y_tr == 0)
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        
        m = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8, num_leaves=63,
            scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        probs = m.predict_proba(X_va)[:, 1]
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= t).astype(int)
            f = f1_score(y_va, preds)
            if f > best_f:
                best_f = f
                best_t = t
                
        preds = (probs >= best_t).astype(int)
        p = precision_score(y_va, preds, zero_division=0)
        r = recall_score(y_va, preds, zero_division=0)
        f = f1_score(y_va, preds, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        print(f"Fold {fold+1}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f} (Optimal Thresh={best_t:.2f})")
        
    print(f"\nMean CV Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Mean CV Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"Mean CV F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # 6. Train final model on 100% of Stav training data
    print("\n--- Training Final Model on 100% Stav Data ---")
    n_pos_all = np.sum(y_train == 1)
    n_neg_all = np.sum(y_train == 0)
    final_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=63,
        scale_pos_weight=n_neg_all/n_pos_all, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1
    )
    final_model.fit(X_comb_train, y_train)
    
    probs_val = final_model.predict_proba(X_comb_val)[:, 1]
    
    cell_probs = probs_val[y_val == 1]
    art_probs = probs_val[y_val == 0]
    print("\nStav22 Probability Distributions:")
    print("  ALL:        min={:.3f}, 10th={:.3f}, 50th={:.3f}, 90th={:.3f}, max={:.3f}".format(
        np.min(probs_val), np.percentile(probs_val, 10), np.percentile(probs_val, 50), np.percentile(probs_val, 90), np.max(probs_val)
    ))
    print("  TRUE CELLS: min={:.3f}, 10th={:.3f}, 50th={:.3f}, 90th={:.3f}, max={:.3f}".format(
        np.min(cell_probs), np.percentile(cell_probs, 10), np.percentile(cell_probs, 50), np.percentile(cell_probs, 90), np.max(cell_probs)
    ))
    print("  ARTIFACTS:  min={:.3f}, 10th={:.3f}, 50th={:.3f}, 90th={:.3f}, max={:.3f}".format(
        np.min(art_probs), np.percentile(art_probs, 10), np.percentile(art_probs, 50), np.percentile(art_probs, 90), np.max(art_probs)
    ))
    
    # Calculate performance at 0.50 and optimal threshold
    best_t_val, best_f_val = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (probs_val >= t).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f_val:
            best_f_val = f
            best_t_val = t
            
    preds_50 = (probs_val >= 0.5).astype(int)
    p_50 = precision_score(y_val, preds_50, zero_division=0)
    r_50 = recall_score(y_val, preds_50, zero_division=0)
    f_50 = f1_score(y_val, preds_50, zero_division=0)
    
    preds_opt = (probs_val >= best_t_val).astype(int)
    p_opt = precision_score(y_val, preds_opt, zero_division=0)
    r_opt = recall_score(y_val, preds_opt, zero_division=0)
    
    print(f"\nStav22 Performance at 0.50 Threshold:")
    print(f"  Precision: {p_50:.4f}, Recall: {r_50:.4f}, F1: {f_50:.4f}")
    print(f"Stav22 Performance at Optimal Threshold ({best_t_val:.2f}):")
    print(f"  Precision: {p_opt:.4f}, Recall: {r_opt:.4f}, F1: {best_f_val:.4f}")

if __name__ == "__main__":
    main()
