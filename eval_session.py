import os
import sys
import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure workspace root is in python path
sys.path.append(str(Path(__file__).parent.resolve()))
from apply_AI import extract_features, FEATURE_NAMES_25, FEATURE_NAMES_27

def evaluate_session(session_path_str, model_path_str):
    session_path = Path(session_path_str)
    model_path = Path(model_path_str)
    
    if not session_path.exists():
        print(f"Error: Session directory '{session_path}' does not exist.")
        return
        
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' does not exist.")
        return
        
    print(f"Loading data from {session_path.resolve()}...")
    try:
        F = np.load(session_path / 'F.npy')
        Fneu = np.load(session_path / 'Fneu.npy')
        stat = np.load(session_path / 'stat.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error: Missing required files (F.npy, Fneu.npy, stat.npy) in session folder. {e}")
        return
        
    # Find ground truth label file
    iscell_gt = None
    label_options = ['iscell_final.npy', 'iscell_manual.npy', 'iscell.npy']
    for lbl_name in label_options:
        lbl_path = session_path / lbl_name
        if lbl_path.exists():
            iscell_gt = np.load(lbl_path)
            print(f"Loaded ground truth labels from: {lbl_name}")
            break
            
    if iscell_gt is None:
        print("Warning: No curated ground truth file (iscell_final.npy, iscell_manual.npy, or iscell.npy) found.")
        print("Cannot compute evaluation metrics.")
        return
        
    y_true = iscell_gt[:, 0].astype(int)
    
    # Load model and auto-detect expected features count
    print(f"Loading model from {model_path.resolve()}...")
    model = joblib.load(model_path)
    
    if hasattr(model, 'n_features_in_'):
        num_features = model.n_features_in_
    elif hasattr(model, 'n_features_'):
        num_features = model.n_features_
    else:
        num_features = 25
        
    print(f"Extracting {num_features} features dynamically...")
    if num_features == 25:
        X = extract_features(F, Fneu, stat, FEATURE_NAMES_25)
    elif num_features == 27:
        X = extract_features(F, Fneu, stat, FEATURE_NAMES_27)
    else:
        raise ValueError(f"Model expects unsupported feature count: {num_features}")
        
    X = np.nan_to_num(X)
    
    # Predict probabilities and apply optimal threshold
    y_prob = model.predict_proba(X)[:, 1]
    
    # Threshold preset matching apply_AI logic
    threshold = 0.66 if num_features == 25 else 0.69
    y_pred = (y_prob >= threshold).astype(int)
    
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n==========================================")
    print(f" PERFORMANCE SUMMARY: {session_path.name}")
    print(f"==========================================")
    print(f"  Total ROIs:        {len(y_true)}")
    print(f"  Ground Truth Cells: {np.sum(y_true)}")
    print(f"  Predicted Cells:   {np.sum(y_pred)}")
    print(f"  Precision:         {p:.4f}")
    print(f"  Recall:            {r:.4f}")
    print(f"  F1-Score:          {f1:.4f} (Threshold={threshold:.2f})")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cell Classifier on Single Session")
    parser.add_argument('session', type=str, help="Path to Suite2p session folder")
    parser.add_argument('--model', type=str, default="models/regular/suite2p_best_lgb.pkl", help="Path to the model .pkl file")
    args = parser.parse_args()
    
    evaluate_session(args.session, args.model)
