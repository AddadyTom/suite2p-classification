import sys
import numpy as np
import joblib
from pathlib import Path
import shutil
import warnings
import json
from scipy.stats import skew
from scipy.signal import find_peaks, peak_widths

warnings.filterwarnings("ignore")

FEATURE_NAMES_25 = [
    'number_of_bright_pixels', 'bright_pixels_ratio', 'solidity', 'mrs', 'roi_idx_norm',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fneu', 'corr_f_fneu',
    'skew_fcorr', 'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'avg_asym', 'max_asym', 'max_width', 'range_fcorr', 'range_f'
]

FEATURE_NAMES_27 = [
    'number_of_bright_pixels', 'bright_pixels_ratio', 'solidity', 'mrs',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fneu', 'corr_f_fneu',
    'skew_fcorr', 'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'avg_asym', 'max_asym', 'max_width', 'range_fcorr', 'range_f',
    'snr', 'activity_ratio', 'peak_density'
]

FEATURE_NAMES_38 = [
    'number_of_bright_pixels', 'bright_pixels_ratio', 'solidity', 'mrs',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fneu', 'corr_f_fneu',
    'skew_fcorr', 'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'avg_asym', 'max_asym', 'max_width', 'range_fcorr', 'range_f',
    'snr', 'activity_ratio', 'peak_density',
    'std_f_norm', 'std_fcorr_norm', 'range_f_norm', 'range_fcorr_norm',
    'q10_norm', 'q25_norm', 'q50_norm', 'q75_norm', 'q90_norm', 'q95_norm', 'q99_norm'
]

# ==========================================
# 1. FEATURE EXTRACTION LOGIC
# ==========================================
def extract_peak_features(F):
    peaks, _ = find_peaks(F, height=np.mean(F) + 2*np.std(F), distance=10)
    if len(peaks) == 0:
        return 1.0, 1.0, 0.0  
        
    widths, _, left_ips, right_ips = peak_widths(F, peaks, rel_height=0.5)
    left_widths = peaks - left_ips
    right_widths = right_ips - peaks
    left_widths[left_widths < 0.1] = 0.1
    
    asymmetry_ratios = right_widths / left_widths
    max_width = np.max(widths)
    
    return np.mean(asymmetry_ratios), np.max(asymmetry_ratios), max_width

def extract_features(F, Fneu, stat, num_features_or_names):
    n_cells = F.shape[0]
    features = []
    
    if isinstance(num_features_or_names, (list, tuple, np.ndarray)):
        feature_names = num_features_or_names
        print(f"Extracting {len(feature_names)} features dynamically for {n_cells} ROIs...")
        for i in range(n_cells):
            f = F[i]
            fneu = Fneu[i]
            f_corr = f - 0.7 * fneu
            s = stat[i]
            
            std_diff_f = np.std(np.diff(f))
            if std_diff_f <= 0:
                std_diff_f = 1e-6
            std_diff_fcorr = np.std(np.diff(f_corr))
            if std_diff_fcorr <= 0:
                std_diff_fcorr = 1e-6
            is_noidx = (len(feature_names) == 27)
            
            avg_asym, max_asym, max_width = None, None, None
            bright_pix = None
            
            row = []
            for name in feature_names:
                # Spatial features
                if name == 'npix':
                    row.append(float(s.get('npix', 0)))
                elif name == 'skew_spatial':
                    row.append(float(s.get('skew', 0)))
                elif name == 'compact':
                    row.append(float(s.get('compact', 0.0)))
                elif name == 'aspect_ratio':
                    row.append(float(s.get('aspect_ratio', 1.0)))
                elif name == 'radius':
                    row.append(float(s.get('radius', 0.0)))
                elif name == 'solidity':
                    row.append(float(s.get('solidity', 1.0)))
                elif name == 'mrs':
                    row.append(float(s.get('mrs', 0.0)))
                elif name == 'number_of_bright_pixels':
                    if bright_pix is None:
                        lam = s.get('lam', np.zeros(0))
                        max_lam = np.max(lam) if len(lam) > 0 else 0.0
                        bright_pix = float(np.sum(lam > 0.1 * max_lam)) if max_lam > 0 else 0.0
                    row.append(bright_pix)
                elif name == 'bright_pixels_ratio':
                    if bright_pix is None:
                        lam = s.get('lam', np.zeros(0))
                        max_lam = np.max(lam) if len(lam) > 0 else 0.0
                        bright_pix = float(np.sum(lam > 0.1 * max_lam)) if max_lam > 0 else 0.0
                    npix = float(s.get('npix', 0))
                    row.append(bright_pix / npix if npix > 0 else 0.0)
                    
                # Index features
                elif name == 'roi_idx_norm':
                    row.append(float(i / n_cells) if n_cells > 0 else 0.0)
                elif name == 'roi_idx_norm_3bin':
                    norm_val = i / n_cells if n_cells > 0 else 0.0
                    row.append(float(np.digitize(norm_val, [0.1, 0.4])))
                elif name == 'roi_idx_raw' or name == 'roi_idx':
                    row.append(float(i))
                    
                # Trace Stats features
                elif name == 'skew_f':
                    row.append(float(skew(f)))
                elif name == 'std_f':
                    val = float(np.std(f))
                    if is_noidx:
                        val /= std_diff_f
                    row.append(val)
                elif name == 'max_f':
                    row.append(float(np.max(f)))
                elif name == 'mean_f':
                    row.append(float(np.mean(f)))
                elif name == 'max_to_mean_f':
                    mean_f_val = np.mean(f)
                    row.append(float(np.max(f) / mean_f_val) if mean_f_val > 0 else 1.0)
                elif name == 'cv_f':
                    mean_f_val = np.mean(f)
                    row.append(float(np.std(f) / mean_f_val) if mean_f_val > 0 else 0.0)
                elif name == 'skew_fneu':
                    row.append(float(skew(fneu)))
                elif name == 'corr_f_fneu':
                    row.append(float(np.corrcoef(f, fneu)[0, 1]) if np.std(f)>0 and np.std(fneu)>0 else 0.0)
                elif name == 'skew_fcorr':
                    row.append(float(skew(f_corr)))
                elif name == 'std_fcorr':
                    val = float(np.std(f_corr))
                    if is_noidx:
                        val /= std_diff_fcorr
                    row.append(val)
                    
                # Quantiles
                elif name.startswith('q') and name[1:].isdigit():
                    pct = float(name[1:]) / 100.0
                    val = float(np.quantile(f_corr, pct))
                    if is_noidx:
                        median_val = np.median(f_corr)
                        if pct == 0.5:
                            val = median_val / std_diff_fcorr
                        else:
                            val = (val - median_val) / std_diff_fcorr
                    row.append(val)
                    
                # Dynamics
                elif name in ('avg_asym', 'max_asym', 'max_width'):
                    if avg_asym is None:
                        avg_asym, max_asym, max_width = extract_peak_features(f_corr)
                    if name == 'avg_asym':
                        row.append(avg_asym)
                    elif name == 'max_asym':
                        row.append(max_asym)
                    elif name == 'max_width':
                        row.append(max_width)
                        
                # Amplitudes
                elif name == 'range_fcorr':
                    val = float(np.max(f_corr) - np.min(f_corr))
                    if is_noidx:
                        val /= std_diff_fcorr
                    row.append(val)
                elif name == 'range_f':
                    val = float(np.max(f) - np.min(f))
                    if is_noidx:
                        val /= std_diff_f
                    row.append(val)
                    
                # New Biological
                elif name == 'snr':
                    diff_f = np.diff(f_corr)
                    std_diff = np.std(diff_f)
                    row.append(float(np.std(f_corr) / std_diff) if std_diff > 0 else 0.0)
                elif name == 'activity_ratio':
                    median_val = np.median(f_corr)
                    above_med = f_corr[f_corr > median_val]
                    below_med = f_corr[f_corr <= median_val]
                    std_above = np.std(above_med) if len(above_med) > 0 else 0.0
                    std_below = np.std(below_med) if len(below_med) > 0 else 0.0
                    row.append(float(std_above / std_below) if std_below > 0 else 0.0)
                elif name == 'peak_density':
                    peaks, _ = find_peaks(f_corr, height=np.mean(f_corr) + 2*np.std(f_corr), distance=10)
                    row.append(float(len(peaks) / len(f_corr)) if len(f_corr) > 0 else 0.0)
                    
                # Explicitly Normalized Features for rich sets
                elif name == 'std_f_norm':
                    row.append(float(np.std(f) / std_diff_f))
                elif name == 'std_fcorr_norm':
                    row.append(float(np.std(f_corr) / std_diff_fcorr))
                elif name == 'range_f_norm':
                    row.append(float((np.max(f) - np.min(f)) / std_diff_f))
                elif name == 'range_fcorr_norm':
                    row.append(float((np.max(f_corr) - np.min(f_corr)) / std_diff_fcorr))
                elif name.endswith('_norm') and name.startswith('q') and name[1:-5].isdigit():
                    pct = float(name[1:-5]) / 100.0
                    val = float(np.quantile(f_corr, pct))
                    median_val = np.median(f_corr)
                    if pct == 0.5:
                        row.append(median_val / std_diff_fcorr)
                    else:
                        row.append((val - median_val) / std_diff_fcorr)
                    
                else:
                    print(f"Warning: Unknown feature name '{name}'. Defaulting to 0.0.")
                    row.append(0.0)
            features.append(row)
        return np.nan_to_num(np.array(features))

    num_features = num_features_or_names
    print(f"Extracting {num_features} features for {n_cells} ROIs...")
    for i in range(n_cells):
        f = F[i]
        fneu = Fneu[i]
        f_corr = f - 0.7 * fneu
        s = stat[i]
        
        avg_asym, max_asym, max_width = extract_peak_features(f_corr)
        
        # Spatial
        lam = s.get('lam', np.zeros(0))
        max_lam = np.max(lam) if len(lam) > 0 else 0.0
        bright_pix = np.sum(lam > 0.1 * max_lam) if max_lam > 0 else 0
        
        solidity = s.get('solidity', 1.0)
        mrs = s.get('mrs', 0)
        
        npix = float(s.get('npix', 0))
        bright_ratio = bright_pix / npix if npix > 0 else 0.0
        
        if num_features == 25:
            # Spatial (5) - Continuous Index
            spatial = [bright_pix, bright_ratio, solidity, mrs, i / n_cells if n_cells > 0 else 0.0]
        elif num_features in (27, 38):
            # Spatial (4) - No Index
            spatial = [bright_pix, bright_ratio, solidity, mrs]
        else:
            raise ValueError(f"Unsupported number of features: {num_features}. Model must expect 25, 27 or 38 features.")
            
        std_diff_f = np.std(np.diff(f))
        if std_diff_f <= 0:
            std_diff_f = 1e-6
        std_diff_fcorr = np.std(np.diff(f_corr))
        if std_diff_fcorr <= 0:
            std_diff_fcorr = 1e-6

        # Trace Stats (8)
        mean_f_val = np.mean(f)
        max_to_mean_f = np.max(f) / mean_f_val if mean_f_val > 0 else 1.0
        cv_f = np.std(f) / mean_f_val if mean_f_val > 0 else 0.0
        
        if num_features == 27:
            std_f_val = np.std(f) / std_diff_f
            std_fcorr_val = np.std(f_corr) / std_diff_fcorr
        else:
            std_f_val = np.std(f)
            std_fcorr_val = np.std(f_corr)

        trace_stats = [
            skew(f), std_f_val, max_to_mean_f, cv_f,
            skew(fneu),
            np.corrcoef(f, fneu)[0, 1] if np.std(f)>0 and np.std(fneu)>0 else 0,
            skew(f_corr), std_fcorr_val
        ]
        
        # Percentiles (7)
        q_raw = np.quantile(f_corr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        if num_features == 27:
            q = []
            median_val = q_raw[2]
            for idx, val in enumerate(q_raw):
                if idx == 2:
                    q.append(median_val / std_diff_fcorr)
                else:
                    q.append((val - median_val) / std_diff_fcorr)
        else:
            q = q_raw.tolist()
        
        # Dynamics (3)
        dynamics = [avg_asym, max_asym, max_width]
        
        # Trace Amplitudes (2)
        range_fcorr = np.max(f_corr) - np.min(f_corr)
        range_f = np.max(f) - np.min(f)
        if num_features == 27:
            range_fcorr = range_fcorr / std_diff_fcorr
            range_f = range_f / std_diff_f
            
        amplitudes = [range_fcorr, range_f]
        
        row = spatial + trace_stats + q + dynamics + amplitudes
        
        if num_features in (27, 38):
            # New Biological (3) for No Index/Rich models
            diff_f = np.diff(f_corr)
            std_diff = np.std(diff_f)
            snr_val = np.std(f_corr) / std_diff if std_diff > 0 else 0.0
            
            median_val = np.median(f_corr)
            above_med = f_corr[f_corr > median_val]
            below_med = f_corr[f_corr <= median_val]
            std_above = np.std(above_med) if len(above_med) > 0 else 0.0
            std_below = np.std(below_med) if len(below_med) > 0 else 0.0
            activity_ratio = std_above / std_below if std_below > 0 else 0.0
            
            peaks, _ = find_peaks(f_corr, height=np.mean(f_corr) + 2*np.std(f_corr), distance=10)
            peak_density = len(peaks) / len(f_corr) if len(f_corr) > 0 else 0.0
            
            new_bio = [snr_val, activity_ratio, peak_density]
            row.extend(new_bio)
            
        if num_features == 38:
            # 11 Normalized Features:
            std_f_norm = np.std(f) / std_diff_f
            std_fcorr_norm = np.std(f_corr) / std_diff_fcorr
            range_f_norm = (np.max(f) - np.min(f)) / std_diff_f
            range_fcorr_norm = (np.max(f_corr) - np.min(f_corr)) / std_diff_fcorr
            
            q_norm = []
            median_val = q_raw[2]
            for idx, val in enumerate(q_raw):
                if idx == 2:
                    q_norm.append(median_val / std_diff_fcorr)
                else:
                    q_norm.append((val - median_val) / std_diff_fcorr)
            
            row.extend([std_f_norm, std_fcorr_norm, range_f_norm, range_fcorr_norm] + q_norm)
            
        features.append(row)
        
    return np.nan_to_num(np.array(features))

def suppress_duplicate_rois(stat, probs, preds, F, dist_threshold=15.0, corr_threshold=0.7):
    """
    Identifies spatially duplicate/highly overlapping ROIs.
    If two ROIs classified as cells (preds == 1) are close, overlap, AND have highly
    correlated calcium activity (Pearson correlation >= corr_threshold),
    suppresses the one with the lower probability score.
    """
    n_rois = len(stat)
    if n_rois == 0:
        return preds.copy(), np.zeros(0, dtype=bool)
        
    centroids = np.array([s['med'] for s in stat])
    pixel_sets = [set(zip(s['ypix'], s['xpix'])) for s in stat]
    
    cell_indices = np.where(preds == 1)[0]
    if len(cell_indices) == 0:
        return preds.copy(), np.zeros(n_rois, dtype=bool)
        
    # Sort active cells by probability descending
    sorted_indices = cell_indices[np.argsort(probs[cell_indices])[::-1]]
    suppressed = np.zeros(n_rois, dtype=bool)
    
    for idx_i, i in enumerate(sorted_indices):
        if suppressed[i]:
            continue
            
        for j in sorted_indices[idx_i + 1:]:
            if suppressed[j]:
                continue
                
            # Centroid check (fast filter)
            dy = abs(centroids[i][0] - centroids[j][0])
            dx = abs(centroids[i][1] - centroids[j][1])
            if dy < dist_threshold and dx < dist_threshold:
                # Calculate intersection
                set_i = pixel_sets[i]
                set_j = pixel_sets[j]
                intersection = len(set_i.intersection(set_j))
                if intersection > 0:
                    trace_i = F[i]
                    trace_j = F[j]
                    if np.std(trace_i) > 0 and np.std(trace_j) > 0:
                        corr = np.corrcoef(trace_i, trace_j)[0, 1]
                    else:
                        corr = 0.0

                    if corr > corr_threshold:
                        suppressed[j] = True
                        dist = np.sqrt(dy**2 + dx**2)
                        print(f"  [NMS Suppress] Suppressing ROI {j} (prob={probs[j]:.3f}) in favor of ROI {i} (prob={probs[i]:.3f}) - Centroid Dist: {dist:.1f}px, Trace Corr: {corr:.3f}")
                        
    new_preds = preds.copy()
    new_preds[suppressed] = 0
    return new_preds, suppressed

# ==========================================
# 2. INFERENCE LOGIC
# ==========================================
def apply_active_learning(session_path, model_spec='regular'):
    session_path = Path(session_path)
    base_dir = Path(__file__).parent
    
    print(f"Loading session data from {session_path}...")
    try:
        F = np.load(session_path / 'F.npy')
        Fneu = np.load(session_path / 'Fneu.npy')
        stat = np.load(session_path / 'stat.npy', allow_pickle=True)
        iscell_path = session_path / 'iscell.npy'
        iscell_exists = iscell_path.exists()
        if not iscell_exists:
            print("Notice: iscell.npy not found in session folder. A new one will be created.")
            
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Map preset model names to paths
    model_map = {
        'regular': base_dir / 'models' / 'regular' / 'suite2p_best_lgb.pkl',
        'noidx': base_dir / 'models' / 'no_index' / 'suite2p_best_lgb.pkl',
        'no_index': base_dir / 'models' / 'no_index' / 'suite2p_best_lgb.pkl',
        'rich': base_dir / 'models' / 'rich' / 'suite2p_best_lgb.pkl',
    }

    model_path = model_spec
    if model_spec.lower() in model_map:
        model_path = model_map[model_spec.lower()]
    else:
        model_path = Path(model_spec)

    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        print("Available presets: 'regular', 'noidx', 'rich'")
        return

    print(f"Loading model: {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Auto-detect features expected by the model
    if hasattr(model, 'n_features_in_'):
        num_features = model.n_features_in_
    elif hasattr(model, 'n_features_'):
        num_features = model.n_features_
    else:
        # Default fallback to regular model feature count
        num_features = 25
        print("Warning: Could not detect feature size from model metadata. Defaulting to 25.")

    # Try to infer feature names list
    feature_names = None
    if hasattr(model, 'feature_name_'):
        feature_names = list(model.feature_name_)
    elif hasattr(model, 'feature_names'):
        feature_names = list(model.feature_names)
        
    def is_default_names(names):
        if not names:
            return True
        first = str(names[0]).lower()
        return first.startswith('column_') or first.startswith('feature_') or first.isdigit()
        
    if feature_names and is_default_names(feature_names):
        feature_names = None
        
    if not feature_names:
        # Look in model directory for any JSON file describing features
        model_dir = model_path.parent
        json_files = list(model_dir.glob("*.json"))
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'features' in data:
                        if len(data['features']) == num_features:
                            feature_names = data['features']
                            print(f"Loaded feature names from metadata file: {jf.name}")
                            break
            except:
                pass

    # Fallback to predefined lists if we still don't have feature names
    if not feature_names:
        if num_features == 25:
            feature_names = FEATURE_NAMES_25
        elif num_features == 27:
            feature_names = FEATURE_NAMES_27
        elif num_features == 38:
            feature_names = FEATURE_NAMES_38
        else:
            feature_names = [f"feature_{i}" for i in range(num_features)]

    X = extract_features(F, Fneu, stat, feature_names)
    probs = model.predict_proba(X)[:, 1]

    # Select optimal decision threshold based on feature layout (25, 27 or 38)
    if num_features == 25:
        threshold = 0.66
    elif num_features in (27, 38):
        threshold = 0.69
    else:
        # Default fallback
        threshold = 0.66

    print(f"Using F1-optimized classification threshold: {threshold:.2f}")
    is_cell = np.zeros(len(probs))
    is_cell[probs >= threshold] = 1

    # Run duplicate ROI suppression (NMS) on predicted cells
    new_preds, suppressed = suppress_duplicate_rois(stat, probs, is_cell, F)
    num_suppressed = np.sum(suppressed)
    if num_suppressed > 0:
        is_cell = new_preds
        probs[suppressed] = 0.0
        print(f"Suppressed {num_suppressed} duplicate/overlapping ROIs.")
    
    print(f"\n=== ACTIVE LEARNING RESULTS ===")
    print(f"Total ROIs analyzed: {len(probs)}")
    print(f"Tagged as CELLS: {np.sum(is_cell == 1)}")
    print(f"Tagged as ARTIFACTS: {np.sum(is_cell == 0)}")
    
    # Backup original iscell.npy if it exists
    if iscell_exists:
        backup_path = session_path / 'iscell_backup_before_AI.npy'
        if not backup_path.exists():
            shutil.copy(iscell_path, backup_path)
            print(f"Backed up original iscell.npy to: {backup_path.name}")
    
    # Overwrite iscell.npy
    iscell_new = np.zeros((len(probs), 2))
    iscell_new[:, 0] = is_cell
    iscell_new[:, 1] = probs
    np.save(iscell_path, iscell_new)
    
    print("\n✅ SUCCESS: iscell.npy has been overwritten.")
    print("Refresh your Suite2p GUI results to see the updated classifications!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python apply_AI.py <path_to_suite2p_session_folder> [model_preset_or_path]")
        print("\nPresets:")
        print("  regular   - LightGBM with Continuous Index (25 features, Recommended Default)")
        print("  noidx     - LightGBM with No Index (Option A, 27 features)")
        print("  rich      - LightGBM with Rich Features (38 features)")
    else:
        model_choice = sys.argv[2] if len(sys.argv) > 2 else 'regular'
        apply_active_learning(sys.argv[1], model_choice)
