import os
import sys
import gc
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

# Ensure workspace root is in python path
workspace_root = Path(__file__).parent.resolve()
sys.path.append(str(workspace_root))

from fe_engine.fe_definitions import FEATURE_REGISTRY

def precompute_missing_for_session(f, missing_features):
    try:
        # Load existing npz data
        data = np.load(f, allow_pickle=True)
        session_dict = {key: data[key] for key in data.files}
        
        # Resolve raw path
        raw_path_str = session_dict['session_path']
        if isinstance(raw_path_str, np.ndarray):
            raw_path_str = raw_path_str.item()
        raw_path = Path(raw_path_str)
        
        if not raw_path.exists():
            return False, f"Raw session folder '{raw_path}' does not exist."
            
        # Load raw traces using memmap
        F_arr = np.load(raw_path / 'F.npy', mmap_mode='r')
        Fneu_arr = np.load(raw_path / 'Fneu.npy', mmap_mode='r')
        
        session_dict['F'] = F_arr
        session_dict['Fneu'] = Fneu_arr
        
        # Re-extract all missing features
        for feat in missing_features:
            extractor = FEATURE_REGISTRY[feat]
            session_dict[feat] = extractor(session_dict)
            
        # Remove raw traces from dictionary
        del session_dict['F']
        del session_dict['Fneu']
        gc.collect()
        
        # Save back to same file (compressed)
        np.savez_compressed(f, **session_dict)
        return True, raw_path.name
    except Exception as e:
        return False, f"{f.name}: {e}"

def main():
    cache_dir = workspace_root / "preprocessed_cache"
    npz_files = list(cache_dir.glob("preprocessed_*.npz"))
    
    if len(npz_files) == 0:
        print("No cache files found.")
        sys.exit(1)
        
    # Check one file to see what features are missing
    first_file = npz_files[0]
    data = np.load(first_file, allow_pickle=True)
    existing_keys = set(data.files)
    
    missing_features = [feat for feat in FEATURE_REGISTRY.keys() if feat not in existing_keys]
    
    if not missing_features:
        print("All features in FEATURE_REGISTRY are already precomputed and cached in the npz files!")
        sys.exit(0)
        
    print(f"Found {len(missing_features)} missing features in the cache: {missing_features}")
    # Using n_jobs=2 to prevent OOM crash
    print(f"Precomputing them sequentially (n_jobs=1) for all {len(npz_files)} sessions...")
    
    results = Parallel(n_jobs=1)(
        delayed(precompute_missing_for_session)(f, missing_features)
        for f in npz_files
    )
    
    success_count = sum(1 for r in results if r[0])
    print(f"Finished! Successfully updated cache for {success_count}/{len(npz_files)} sessions.")
    
    errors = [r[1] for r in results if not r[0]]
    if errors:
        print("Errors:")
        for err in errors:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
