import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
import joblib
from pathlib import Path

def main():
    data_dir = Path(".")
    X_path = data_dir / "X_vision_dataset.npy"
    y_path = data_dir / "y_vision_dataset.npy"
    groups_path = data_dir / "groups_vision_dataset.npy"
    
    if not X_path.exists():
        print(f"Dataset not found at {X_path}. Please run prepare_vision_data.py first.")
        return
        
    X = np.load(X_path)
    y = np.load(y_path)
    groups = np.load(groups_path)
    
    print(f"Loaded dataset: {X.shape[0]} ROIs, {X.shape[1]} features.")
    
    # K-Fold Cross Validation
    gkf = GroupKFold(n_splits=5)
    
    models = []
    aucs = []
    f1s = []
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'verbose': -1
    }
    
    print("\nTraining LightGBM on combined Numerical + Vision Features...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds = model.predict(X_val)
        auc = roc_auc_score(y_val, preds)
        f1 = f1_score(y_val, (preds > 0.5).astype(int))
        
        aucs.append(auc)
        f1s.append(f1)
        models.append(model)
        
        print(f"Fold {fold+1} | AUC: {auc:.4f} | F1: {f1:.4f}")
        
    print(f"\nMean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"Mean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    
    # Save the best model
    best_fold = np.argmax(aucs)
    best_model = models[best_fold]
    
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "suite2p_vision_lgb.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nSaved best LightGBM model to {model_path}")

if __name__ == '__main__':
    main()
