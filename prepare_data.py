import os
import sys
import argparse
import numpy as np
import hashlib
from pathlib import Path
from joblib import Parallel, delayed

# Ensure workspace root is in python path
sys.path.append(str(Path(__file__).parent.resolve()))
from apply_AI import extract_features, FEATURE_NAMES_25, FEATURE_NAMES_27, FEATURE_NAMES_38

def process_single_session(session_path, group_id, label_options, cache_dir, feature_names, cache_suffix):
    try:
        # Define cache path
        path_str = str(session_path.resolve())
        path_hash = hashlib.md5(path_str.encode('utf-8')).hexdigest()
        cache_file = cache_dir / f"session_{path_hash}_{cache_suffix}.npz"
        
        # Load from cache if it exists
        if cache_file.exists():
            try:
                data = np.load(cache_file)
                return data['X'], data['y'], np.full(len(data['y']), group_id)
            except Exception as cache_err:
                print(f"Error loading cache for {session_path}: {cache_err}. Re-extracting...")
        
        # Load suite2p files
        F = np.load(session_path / 'F.npy', mmap_mode='r')
        Fneu = np.load(session_path / 'Fneu.npy', mmap_mode='r')
        stat = np.load(session_path / 'stat.npy', allow_pickle=True)
        
        # Find ground truth label file
        iscell = None
        for lbl_name in label_options:
            lbl_path = session_path / lbl_name
            if lbl_path.exists():
                iscell = np.load(lbl_path)
                break
                
        if iscell is None:
            raise FileNotFoundError(f"No cell labels found in {session_path}")
            
        y = iscell[:, 0].astype(int)
        
        # Extract features using apply_AI's pipeline (ensures perfect alignment)
        X = extract_features(F, Fneu, stat, feature_names)
        
        # Save to cache
        try:
            np.savez(cache_file, X=X, y=y)
        except Exception as save_err:
            print(f"Error saving cache for {session_path}: {save_err}")
            
        return X, y, np.full(len(y), group_id)
    except Exception as e:
        print(f"Error processing {session_path}: {e}")
        return None

def process_sessions(source_dir, output_dir, preset='regular'):
    source_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    cache_dir = Path(__file__).parent / "feature_cache"
    cache_dir.mkdir(exist_ok=True)
    
    required_files = ['F.npy', 'Fneu.npy', 'stat.npy']
    label_options = ['iscell_final.npy', 'iscell_manual.npy', 'iscell.npy']
    
    print(f"Scanning for valid Suite2p sessions under: {source_path}")
    valid_sessions = []
    
    # Recursive search for stat.npy to identify sessions
    for stat_path in source_path.rglob('stat.npy'):
        session_dir = stat_path.parent
        has_req = all((session_dir / req).exists() for req in required_files)
        if not has_req:
            continue
        has_label = any((session_dir / lbl).exists() for lbl in label_options)
        if has_label:
            valid_sessions.append(session_dir)
            
    valid_sessions = sorted(list(set(valid_sessions)))
    n_sessions = len(valid_sessions)
    print(f"Found {n_sessions} valid sessions for feature extraction.")
    
    if n_sessions == 0:
        print("No sessions found with ground truth labels. Exiting.")
        return
        
    # Select feature config based on preset
    if preset == 'noidx':
        feature_names = FEATURE_NAMES_27
        cache_suffix = '27features_v2'
        x_name = 'X_dataset_noidx.npy'
        y_name = 'y_dataset_noidx.npy'
        groups_name = 'groups_dataset_noidx.npy'
    elif preset == 'rich':
        feature_names = FEATURE_NAMES_38
        cache_suffix = '38features'
        x_name = 'X_dataset_rich.npy'
        y_name = 'y_dataset_rich.npy'
        groups_name = 'groups_dataset_rich.npy'
    else:
        feature_names = FEATURE_NAMES_25
        cache_suffix = '25features'
        x_name = 'X_dataset.npy'
        y_name = 'y_dataset.npy'
        groups_name = 'groups_dataset.npy'

    # Parallel feature extraction using joblib
    results = Parallel(n_jobs=-1)(
        delayed(process_single_session)(session_path, group_id, label_options, cache_dir, feature_names, cache_suffix)
        for group_id, session_path in enumerate(valid_sessions)
    )
    
    X_all = []
    y_all = []
    groups_all = []
    
    for r in results:
        if r is not None:
            X_s, y_s, gr_s = r
            X_all.append(X_s)
            y_all.append(y_s)
            groups_all.append(gr_s)
            
    if len(X_all) == 0:
        print("No data extracted. Exiting.")
        return
        
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    groups = np.concatenate(groups_all)
    
    print(f"\nSuccessfully compiled dataset:")
    print(f"  Total ROIs: {X.shape[0]}")
    print(f"  Features:   {X.shape[1]}")
    print(f"  Cells (1):  {np.sum(y == 1)}")
    print(f"  Noise (0):  {np.sum(y == 0)}")
    
    np.save(out_path / x_name, X)
    np.save(out_path / y_name, y)
    np.save(out_path / groups_name, groups)
    print(f"Saved dataset arrays ({preset} preset) to: {out_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Suite2p Cell Prediction Dataset")
    parser.add_argument('--source', type=str, default="/mnt/other_ubunthu/mnt/data", help="Directory containing Suite2p session folders")
    parser.add_argument('--output', type=str, default=".", help="Directory to save compiled .npy arrays")
    parser.add_argument('--preset', type=str, choices=['regular', 'noidx', 'rich'], default="regular", help="Feature set preset ('regular' = 25 features, 'noidx' = 27 features, 'rich' = 38 features)")
    args = parser.parse_args()
    
    process_sessions(args.source, args.output, args.preset)
