import os
import sys
import argparse
import hashlib
import gc
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew

# Ensure workspace root is in python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

def extract_cell_features(f, fneu, group_id, session_name, path_hash, session_path_str,
                           npix_val, solidity_val, mrs_val, compact_val, aspect_ratio_val, radius_val,
                           bright_pix_val, bright_ratio_val, roi_idx_norm_val):
    """
    Extract all 27 baseline features for a single cell trace in a memory-safe, 1D manner.
    """
    f_corr = f - 0.7 * fneu
    
    # 1. Compute standard deviation of first differences
    diff_f = np.diff(f)
    std_diff_f = np.std(diff_f)
    if std_diff_f <= 0:
        std_diff_f = 1e-6
        
    diff_fcorr = np.diff(f_corr)
    std_diff_fcorr = np.std(diff_fcorr)
    if std_diff_fcorr <= 0:
        std_diff_fcorr = 1e-6
        
    # 2. Skews and standard deviations (normalized)
    skew_f_val = float(skew(f))
    std_f_val = float(np.std(f) / std_diff_f)
    
    # max to mean and cv
    mean_f_val = np.mean(f)
    mean_f_val_safe = mean_f_val if mean_f_val > 0 else 1e-10
    max_to_mean_f_val = float(np.max(f) / mean_f_val_safe)
    cv_f_val = float(np.std(f) / mean_f_val_safe)
    
    skew_fneu_val = float(skew(fneu))
    
    # trace correlation
    std_f = np.std(f)
    std_fneu = np.std(fneu)
    if std_f > 0 and std_fneu > 0:
        cov = np.mean((f - mean_f_val) * (fneu - np.mean(fneu)))
        corr_f_fneu_val = float(cov / (std_f * std_fneu))
    else:
        corr_f_fneu_val = 0.0
        
    skew_fcorr_val = float(skew(f_corr))
    std_fcorr_val = float(np.std(f_corr) / std_diff_fcorr)
    
    # 3. Normalized Quantiles
    median_val = np.median(f_corr)
    q10_val = float((np.percentile(f_corr, 10) - median_val) / std_diff_fcorr)
    q25_val = float((np.percentile(f_corr, 25) - median_val) / std_diff_fcorr)
    q50_val = float(median_val / std_diff_fcorr)
    q75_val = float((np.percentile(f_corr, 75) - median_val) / std_diff_fcorr)
    q90_val = float((np.percentile(f_corr, 90) - median_val) / std_diff_fcorr)
    q95_val = float((np.percentile(f_corr, 95) - median_val) / std_diff_fcorr)
    q99_val = float((np.percentile(f_corr, 99) - median_val) / std_diff_fcorr)
    
    # 4. Peaks and Asymmetry
    peaks, _ = find_peaks(f_corr, height=np.mean(f_corr) + 2*np.std(f_corr), distance=10)
    if len(peaks) > 0:
        widths, _, left_ips, right_ips = peak_widths(f_corr, peaks, rel_height=0.5)
        left_widths = peaks - left_ips
        right_widths = right_ips - peaks
        left_widths[left_widths < 0.1] = 0.1
        asymmetry_ratios = right_widths / left_widths
        
        avg_asym_val = float(np.mean(asymmetry_ratios))
        max_asym_val = float(np.max(asymmetry_ratios))
        max_width_val = float(np.max(widths))
    else:
        avg_asym_val = 1.0
        max_asym_val = 1.0
        max_width_val = 0.0
        
    # 5. Amplitudes and SNR
    range_fcorr_val = float((np.max(f_corr) - np.min(f_corr)) / std_diff_fcorr)
    range_f_val = float((np.max(f) - np.min(f)) / std_diff_f)
    snr_val = float(np.std(f_corr) / std_diff_fcorr)
    
    # 6. Activity Ratio and Peak Density
    above_med = f_corr[f_corr > median_val]
    below_med = f_corr[f_corr <= median_val]
    std_above = np.std(above_med) if len(above_med) > 0 else 0.0
    std_below = np.std(below_med) if len(below_med) > 0 else 0.0
    activity_ratio_val = float(std_above / std_below) if std_below > 0 else 0.0
    peak_density_val = float(len(peaks) / len(f_corr)) if len(f_corr) > 0 else 0.0
    
    # Return dictionary of feature values
    return {
        'number_of_bright_pixels': bright_pix_val,
        'bright_pixels_ratio': bright_ratio_val,
        'solidity': solidity_val,
        'mrs': mrs_val,
        'skew_f': skew_f_val,
        'std_f': std_f_val,
        'max_to_mean_f': max_to_mean_f_val,
        'cv_f': cv_f_val,
        'skew_fneu': skew_fneu_val,
        'corr_f_fneu': corr_f_fneu_val,
        'skew_fcorr': skew_fcorr_val,
        'std_fcorr': std_fcorr_val,
        'q10': q10_val,
        'q25': q25_val,
        'q50': q50_val,
        'q75': q75_val,
        'q90': q90_val,
        'q95': q95_val,
        'q99': q99_val,
        'avg_asym': avg_asym_val,
        'max_asym': max_asym_val,
        'max_width': max_width_val,
        'range_fcorr': range_fcorr_val,
        'range_f': range_f_val,
        'snr': snr_val,
        'activity_ratio': activity_ratio_val,
        'peak_density': peak_density_val
    }

def preprocess_single_session(session_path, group_id, label_options, cache_dir):
    """
    Process a single session, precompute ALL baseline features cell-by-cell.
    Uses memmap and strictly 1D calculations to keep RAM footprint <50MB.
    """
    try:
        path_str = str(session_path.resolve())
        path_hash = hashlib.md5(path_str.encode('utf-8')).hexdigest()
        cache_file = cache_dir / f"preprocessed_{path_hash}.npz"
        
        # If cache exists, skip
        if cache_file.exists():
            return f"Session {session_path.name} already preprocessed (skipped)."
            
        # Load traces using memory mapping (uses almost zero RAM)
        F = np.load(session_path / 'F.npy', mmap_mode='r')
        Fneu = np.load(session_path / 'Fneu.npy', mmap_mode='r')
        
        # Load labels
        iscell = None
        for lbl_name in label_options:
            lbl_path = session_path / lbl_name
            if lbl_path.exists():
                iscell = np.load(lbl_path)
                break
                
        if iscell is None:
            return f"Error: No labels found in {session_path.name}"
            
        y = iscell[:, 0].astype(int)
        n_cells = F.shape[0]
        
        # Load spatial properties from stat.npy
        stat = np.load(session_path / 'stat.npy', allow_pickle=True)
        
        npix = np.zeros(n_cells, dtype=np.float32)
        solidity = np.zeros(n_cells, dtype=np.float32)
        mrs = np.zeros(n_cells, dtype=np.float32)
        compact = np.zeros(n_cells, dtype=np.float32)
        aspect_ratio = np.zeros(n_cells, dtype=np.float32)
        radius = np.zeros(n_cells, dtype=np.float32)
        number_of_bright_pixels = np.zeros(n_cells, dtype=np.float32)
        bright_pixels_ratio = np.zeros(n_cells, dtype=np.float32)
        
        for i in range(n_cells):
            s = stat[i]
            npix[i] = float(s.get('npix', 0))
            solidity[i] = float(s.get('solidity', 1.0))
            mrs[i] = float(s.get('mrs', 0.0))
            compact[i] = float(s.get('compact', 0.0))
            aspect_ratio[i] = float(s.get('aspect_ratio', 1.0))
            radius[i] = float(s.get('radius', 0.0))
            
            # Bright pixels
            lam = s.get('lam', np.zeros(0))
            max_lam = np.max(lam) if len(lam) > 0 else 0.0
            bright_pix_val = float(np.sum(lam > 0.1 * max_lam)) if max_lam > 0 else 0.0
            number_of_bright_pixels[i] = bright_pix_val
            bright_pixels_ratio[i] = bright_pix_val / npix[i] if npix[i] > 0 else 0.0
            
        # Free stat.npy from memory immediately (crucial!)
        del stat
        gc.collect()
        
        # Pre-extract all features cell-by-cell
        results = []
        for i in range(n_cells):
            # Load 1D slice from disk using memmap
            f_slice = np.array(F[i])
            fneu_slice = np.array(Fneu[i])
            
            # Extract features for this cell
            cell_feats = extract_cell_features(
                f_slice, fneu_slice, group_id, session_path.name, path_hash, path_str,
                npix[i], solidity[i], mrs[i], compact[i], aspect_ratio[i], radius[i],
                number_of_bright_pixels[i], bright_pixels_ratio[i], float(i / n_cells)
            )
            results.append(cell_feats)
            
        # Compile lists of feature columns
        precomputed_features = {}
        for name in results[0].keys():
            precomputed_features[name] = np.array([r[name] for r in results], dtype=np.float32)
            
        # Save cache (without raw F or Fneu arrays)
        np.savez_compressed(
            cache_file,
            y=y.astype(np.int32),
            group_id=np.full(n_cells, group_id, dtype=np.int32),
            session_name=session_path.name,
            session_hash=path_hash,
            session_path=str(session_path.resolve()),
            # Spatial scalars not present in active features
            npix=npix,
            compact=compact,
            aspect_ratio=aspect_ratio,
            radius=radius,
            # Precomputed trace features (includes solidity, mrs, bright pixels, and peak features)
            **precomputed_features
        )
        return f"Preprocessed {session_path.name} ({n_cells} ROIs) -> cached."
    except Exception as e:
        return f"Failed {session_path.name}: {e}"

def main():
    parser = argparse.ArgumentParser(description="One-Time Suite2p Session Preprocessor")
    parser.add_argument('--source', type=str, default="/mnt/other_ubunthu/mnt/data", help="Root folder for Suite2p sessions")
    parser.add_argument('--output', type=str, default=None, help="Output preprocessed cache directory")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of sessions to preprocess")
    parser.add_argument('--n-jobs', type=int, default=2, help="Number of parallel jobs (default: 2)")
    args = parser.parse_args()
    
    workspace_root = Path(__file__).parent.parent.resolve()
    cache_dir = Path(args.output) if args.output else workspace_root / "preprocessed_cache"
    cache_dir.mkdir(exist_ok=True)
    
    required_files = ['F.npy', 'Fneu.npy', 'stat.npy']
    label_options = ['iscell_final.npy', 'iscell_manual.npy', 'iscell.npy']
    
    print(f"Scanning sessions under: {args.source}")
    valid_sessions = []
    
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist.")
        sys.exit(1)
        
    for stat_path in source_path.rglob('stat.npy'):
        session_dir = stat_path.parent
        has_req = all((session_dir / req).exists() for req in required_files)
        if not has_req:
            continue
        has_label = any((session_dir / lbl).exists() for lbl in label_options)
        if has_label:
            valid_sessions.append(session_dir)
            
    valid_sessions = sorted(list(set(valid_sessions)))
    if args.limit:
        valid_sessions = valid_sessions[:args.limit]
        
    n_sessions = len(valid_sessions)
    print(f"Found {n_sessions} valid sessions to preprocess.")
    if n_sessions == 0:
        print("Nothing to process.")
        return
        
    # Run in parallel with a memory-safe job limit (2 jobs is very fast and safe now)
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(preprocess_single_session)(session_path, group_id, label_options, cache_dir)
        for group_id, session_path in enumerate(valid_sessions)
    )
    
    for r in results:
        print(r)
        
    print("\nPreprocessing complete.")

if __name__ == "__main__":
    main()
