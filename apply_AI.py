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

FEATURE_NAMES_24 = [
    'number_of_bright_pixels', 'solidity', 'mrs', 'roi_idx_norm',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f',
    'skew_fneu', 'corr_f_fneu', 'skew_fcorr', 'std_fcorr', 'q10', 'q25',
    'q50', 'q75', 'q90', 'q95', 'q99', 'avg_asym', 'max_asym', 'max_width',
    'range_fcorr', 'range_f'
]

FEATURE_NAMES_25 = [
    'number_of_bright_pixels', 'bright_pixels_ratio', 'solidity', 'mrs', 'roi_idx_norm',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fneu', 'corr_f_fneu',
    'skew_fcorr', 'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'avg_asym', 'max_asym', 'max_width', 'range_fcorr', 'range_f'
]

FEATURE_NAMES_26 = [
    'number_of_bright_pixels', 'solidity', 'mrs',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fneu', 'corr_f_fneu',
    'skew_fcorr', 'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'avg_asym', 'max_asym', 'max_width', 'range_fcorr', 'range_f',
    'snr', 'activity_ratio', 'peak_density'
]

FEATURE_NAMES_27 = [
    'area_to_radius_sq', 'aspect_ratio', 'bright_pixels_to_radius_sq', 'compact',
    'corr_f_fneu', 'max_width', 'mrs',
    'number_of_bright_pixels', 'peak_to_q95_ratio', 'peak_to_q99_ratio',
    'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'radius', 'range_f', 'range_fcorr', 'skew_diff_fcorr',
    'skew_f', 'skew_fcorr', 'skew_fneu', 'solidity',
    'std_f', 'std_fcorr'
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

FEATURE_NAMES_30 = [
    'area_to_radius_sq', 'aspect_ratio', 'bright_pixels_to_radius_sq', 'compact',
    'corr_f_fneu', 'cv_f', 'max_width', 'mrs', 'npix',
    'number_of_bright_pixels', 'peak_to_q95_ratio', 'peak_to_q99_ratio',
    'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'radius', 'range_f', 'range_fcorr', 'skew_diff_fcorr',
    'skew_f', 'skew_fcorr', 'skew_fneu', 'snr', 'solidity',
    'std_f', 'std_fcorr'
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

def extract_features(F, Fneu, stat, num_features_or_names, custom_features=None):
    n_cells = F.shape[0]
    
    # 1. Resolve feature names
    if isinstance(num_features_or_names, (list, tuple, np.ndarray)):
        feature_names = list(num_features_or_names)
    elif isinstance(num_features_or_names, int):
        if num_features_or_names == 24:
            feature_names = FEATURE_NAMES_24
        elif num_features_or_names == 25:
            feature_names = FEATURE_NAMES_25
        elif num_features_or_names == 26:
            feature_names = FEATURE_NAMES_26
        elif num_features_or_names == 27:
            feature_names = FEATURE_NAMES_27
        elif num_features_or_names == 30:
            feature_names = FEATURE_NAMES_30
        elif num_features_or_names == 38:
            feature_names = FEATURE_NAMES_38
        else:
            raise ValueError(f"Unsupported number of features: {num_features_or_names}. Model must expect 24, 25, 26, 27, 30 or 38 features.")
    else:
        raise TypeError("num_features_or_names must be a list of names or an integer (24, 25, 26, 27, 30, 38).")
        
    print(f"Extracting {len(feature_names)} features dynamically for {n_cells} ROIs...")
    
    # 2. Precompute trace signals
    F_corr = F - 0.7 * Fneu
    
    # Calculate session-level standard deviation scale
    std_fcorr_all_temp = np.std(F_corr, axis=1)
    session_scale = np.median(std_fcorr_all_temp)
    if np.isnan(session_scale) or session_scale <= 0:
        session_scale = 1.0
    
    std_diff_f = np.std(np.diff(F, axis=1), axis=1)
    std_diff_f[std_diff_f <= 0] = 1e-6
    std_diff_fcorr = np.std(np.diff(F_corr, axis=1), axis=1)
    std_diff_fcorr[std_diff_fcorr <= 0] = 1e-6

    # 4. Precompute all quantiles at once if requested
    q_names = [name for name in feature_names if name.startswith('q') and (name[1:].isdigit() or (name.endswith('_norm') and name[1:-5].isdigit()))]
    
    q_map = {}
    if q_names:
        # Extract unique percentiles to compute
        q_pcts = []
        for name in q_names:
            if name.endswith('_norm'):
                pct = float(name[1:-5])
            else:
                pct = float(name[1:])
            if pct not in q_pcts:
                q_pcts.append(pct)
                
        # Compute all needed percentiles across axis=1
        q_vals = np.percentile(F_corr, q_pcts, axis=1) / session_scale # shape (len(q_pcts), n_cells)
        if len(q_pcts) == 1:
            q_vals = q_vals.reshape(1, -1)
            
        medians = np.median(F_corr, axis=1)
        
        # Build mapping of percentile to values
        pct_to_val = {pct: q_vals[idx] for idx, pct in enumerate(q_pcts)}
        
        for name in q_names:
            if name.endswith('_norm'):
                pct = float(name[1:-5])
                val = pct_to_val[pct]
                if pct == 50:
                    val = medians / std_diff_fcorr
                else:
                    val = (val - medians) / std_diff_fcorr
            else:
                pct = float(name[1:])
                val = pct_to_val[pct]
            q_map[name] = val

    # 5. Precompute peak-finding features if requested
    need_peaks = any(name in feature_names for name in ('avg_asym', 'max_asym', 'max_width', 'peak_density'))
    avg_asym_arr = np.ones(n_cells)
    max_asym_arr = np.ones(n_cells)
    max_width_arr = np.zeros(n_cells)
    peak_density_arr = np.zeros(n_cells)
    
    if need_peaks:
        mean_fcorr = np.mean(F_corr, axis=1)
        std_fcorr = np.std(F_corr, axis=1)
        
        for i in range(n_cells):
            f = F_corr[i]
            peaks, _ = find_peaks(f, height=mean_fcorr[i] + 2.0 * std_fcorr[i], distance=10)
            peak_density_arr[i] = len(peaks) / len(f) if len(f) > 0 else 0.0
            
            if len(peaks) == 0:
                avg_asym_arr[i] = 1.0
                max_asym_arr[i] = 1.0
                max_width_arr[i] = 0.0
            else:
                widths, _, left_ips, right_ips = peak_widths(f, peaks, rel_height=0.5)
                left_widths = peaks - left_ips
                right_widths = right_ips - peaks
                left_widths[left_widths < 0.1] = 0.1
                
                asymmetry_ratios = right_widths / left_widths
                avg_asym_arr[i] = np.mean(asymmetry_ratios)
                max_asym_arr[i] = np.max(asymmetry_ratios)
                max_width_arr[i] = np.max(widths)

    # 6. Precompute activity ratio
    activity_ratio_arr = np.zeros(n_cells)
    if 'activity_ratio' in feature_names:
        medians = np.median(F_corr, axis=1)
        for i in range(n_cells):
            f = F_corr[i]
            med = medians[i]
            above = f[f > med]
            below = f[f <= med]
            std_above = np.std(above) if len(above) > 0 else 0.0
            std_below = np.std(below) if len(below) > 0 else 0.0
            activity_ratio_arr[i] = std_above / std_below if std_below > 0 else 0.0

    # 7. Precompute SNR
    snr_arr = np.zeros(n_cells)
    if 'snr' in feature_names:
        std_fcorr_all = np.std(F_corr, axis=1)
        diff_fcorr_all = np.diff(F_corr, axis=1)
        std_diff_fcorr_all = np.std(diff_fcorr_all, axis=1)
        std_diff_fcorr_all[std_diff_fcorr_all <= 0] = 1e-6
        snr_arr = std_fcorr_all / std_diff_fcorr_all

    # 8. Precompute correlation
    corr_f_fneu_arr = np.zeros(n_cells)
    if 'corr_f_fneu' in feature_names:
        mean_F = np.mean(F, axis=1, keepdims=True)
        mean_Fneu = np.mean(Fneu, axis=1, keepdims=True)
        cov = np.mean((F - mean_F) * (Fneu - mean_Fneu), axis=1)
        std_F = np.std(F, axis=1)
        std_Fneu = np.std(Fneu, axis=1)
        valid = (std_F > 0) & (std_Fneu > 0)
        corr_f_fneu_arr[valid] = cov[valid] / (std_F[valid] * std_Fneu[valid] + 1e-10)

    # 9. Precompute skewness
    skew_f_arr = None
    if 'skew_f' in feature_names:
        skew_f_arr = skew(F, axis=1)
    skew_fneu_arr = None
    if 'skew_fneu' in feature_names:
        skew_fneu_arr = skew(Fneu, axis=1)
    skew_fcorr_arr = None
    if 'skew_fcorr' in feature_names:
        skew_fcorr_arr = skew(F_corr, axis=1)

    # 10. Precompute std values
    std_f_arr = None
    if 'std_f' in feature_names:
        std_f_arr = np.std(F, axis=1) / session_scale
    std_fcorr_arr = None
    if 'std_fcorr' in feature_names:
        std_fcorr_arr = np.std(F_corr, axis=1) / session_scale

    # 11. Precompute range values
    range_f_arr = None
    if 'range_f' in feature_names:
        range_f_arr = (np.max(F, axis=1) - np.min(F, axis=1)) / session_scale
    range_fcorr_arr = None
    if 'range_fcorr' in feature_names:
        range_fcorr_arr = (np.max(F_corr, axis=1) - np.min(F_corr, axis=1)) / session_scale

    # 12. Build output feature matrix X
    X = np.zeros((n_cells, len(feature_names)))
    
    # Pre-extract spatial statistics arrays
    npix_vals = np.array([float(s.get('npix', 0)) for s in stat])
    solidity_vals = np.array([float(s.get('solidity', 1.0)) for s in stat])
    mrs_vals = np.array([float(s.get('mrs', 0.0)) for s in stat])
    skew_spatial_vals = np.array([float(s.get('skew', 0.0)) for s in stat])
    compact_vals = np.array([float(s.get('compact', 0.0)) for s in stat])
    aspect_ratio_vals = np.array([float(s.get('aspect_ratio', 1.0)) for s in stat])
    radius_vals = np.array([float(s.get('radius', 0.0)) for s in stat])
    
    # Precalculate bright pixels for spatial feature indexing
    bright_pix = None
    if any(name in feature_names for name in ('number_of_bright_pixels', 'bright_pixels_ratio')):
        bright_pix = []
        for s in stat:
            lam = s.get('lam', np.zeros(0))
            max_lam = np.max(lam) if len(lam) > 0 else 0.0
            bright_pix.append(float(np.sum(lam > 0.1 * max_lam)) if max_lam > 0 else 0.0)
        bright_pix = np.array(bright_pix)
        
    ns = {}
    for col_idx, name in enumerate(feature_names):
        if custom_features and name in custom_features:
            continue
        # Spatial features
        if name == 'npix':
            X[:, col_idx] = npix_vals
        elif name == 'skew_spatial':
            X[:, col_idx] = skew_spatial_vals
        elif name == 'compact':
            X[:, col_idx] = compact_vals
        elif name == 'aspect_ratio':
            X[:, col_idx] = aspect_ratio_vals
        elif name == 'radius':
            X[:, col_idx] = radius_vals
        elif name == 'solidity':
            X[:, col_idx] = solidity_vals
        elif name == 'mrs':
            X[:, col_idx] = mrs_vals
        elif name == 'number_of_bright_pixels':
            X[:, col_idx] = bright_pix
        elif name == 'bright_pixels_ratio':
            X[:, col_idx] = np.where(npix_vals > 0, bright_pix / npix_vals, 0.0)
            
        # Index features
        elif name == 'roi_idx_norm':
            X[:, col_idx] = np.arange(n_cells) / n_cells if n_cells > 0 else np.zeros(0)
        elif name == 'roi_idx_norm_3bin':
            norm_val = np.arange(n_cells) / n_cells if n_cells > 0 else np.zeros(0)
            X[:, col_idx] = np.digitize(norm_val, [0.1, 0.4])
        elif name in ('roi_idx_raw', 'roi_idx'):
            X[:, col_idx] = np.arange(n_cells)
            
        # Trace Stats
        elif name == 'skew_f':
            X[:, col_idx] = skew_f_arr
        elif name == 'std_f':
            X[:, col_idx] = std_f_arr
        elif name == 'max_f':
            X[:, col_idx] = np.max(F, axis=1)
        elif name == 'mean_f':
            X[:, col_idx] = np.mean(F, axis=1)
        elif name == 'max_to_mean_f':
            mean_f = np.mean(F, axis=1)
            X[:, col_idx] = np.where(mean_f > 0, np.max(F, axis=1) / mean_f, 1.0)
        elif name == 'cv_f':
            mean_f = np.mean(F, axis=1)
            X[:, col_idx] = np.where(mean_f > 0, np.std(F, axis=1) / mean_f, 0.0)
        elif name == 'skew_fneu':
            X[:, col_idx] = skew_fneu_arr
        elif name == 'corr_f_fneu':
            X[:, col_idx] = corr_f_fneu_arr
        elif name == 'skew_fcorr':
            X[:, col_idx] = skew_fcorr_arr
        elif name == 'std_fcorr':
            X[:, col_idx] = std_fcorr_arr
            
        # Quantiles
        elif name in q_map:
            X[:, col_idx] = q_map[name]
            
        # Dynamics
        elif name == 'avg_asym':
            X[:, col_idx] = avg_asym_arr
        elif name == 'max_asym':
            X[:, col_idx] = max_asym_arr
        elif name == 'max_width':
            X[:, col_idx] = max_width_arr
            
        # Ranges
        elif name == 'range_fcorr':
            X[:, col_idx] = range_fcorr_arr
        elif name == 'range_f':
            X[:, col_idx] = range_f_arr
            
        # New Biological
        elif name == 'snr':
            X[:, col_idx] = snr_arr
        elif name == 'activity_ratio':
            X[:, col_idx] = activity_ratio_arr
        elif name == 'peak_density':
            X[:, col_idx] = peak_density_arr
        elif name == 'area_to_radius_sq':
            X[:, col_idx] = npix_vals / np.maximum(radius_vals ** 2, 1e-6)
        elif name == 'bright_pixels_to_radius_sq':
            X[:, col_idx] = bright_pix / np.maximum(radius_vals ** 2, 1e-6)
        elif name == 'peak_to_q95_ratio':
            q95 = q_map.get('q95', np.percentile(F_corr, 95, axis=1))
            X[:, col_idx] = np.max(F_corr, axis=1) / np.maximum(q95, 1e-6)
        elif name == 'peak_to_q99_ratio':
            q99 = q_map.get('q99', np.percentile(F_corr, 99, axis=1))
            X[:, col_idx] = np.max(F_corr, axis=1) / np.maximum(q99, 1e-6)
        elif name == 'skew_diff_fcorr':
            diff_fcorr = np.diff(F_corr, axis=1)
            X[:, col_idx] = skew(diff_fcorr, axis=1)
            
        # Explicitly Normalized Features for rich sets
        elif name == 'std_f_norm':
            X[:, col_idx] = np.std(F, axis=1) / std_diff_f
        elif name == 'std_fcorr_norm':
            X[:, col_idx] = np.std(F_corr, axis=1) / std_diff_fcorr
        elif name == 'range_f_norm':
            X[:, col_idx] = (np.max(F, axis=1) - np.min(F, axis=1)) / std_diff_f
        elif name == 'range_fcorr_norm':
            X[:, col_idx] = (np.max(F_corr, axis=1) - np.min(F_corr, axis=1)) / std_diff_fcorr
            
        else:
            print(f"Warning: Unknown feature name '{name}'. Defaulting to 0.0.")
            X[:, col_idx] = 0.0
            
        ns[name] = X[:, col_idx]
        
    # Pass 2: Calculate custom features
    if custom_features:
        from fe_engine.fe_definitions import safe_eval_formula
        for col_idx, name in enumerate(feature_names):
            if name in custom_features:
                expr = custom_features[name]
                try:
                    result = safe_eval_formula(expr, ns)
                    X[:, col_idx] = result
                    ns[name] = result
                except Exception as e:
                    print(f"Error evaluating custom feature '{name} = {expr}': {e}")
                    X[:, col_idx] = np.zeros(n_cells)
            
    return np.nan_to_num(X)

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
        F = np.load(session_path / 'F.npy', mmap_mode='r')
        Fneu = np.load(session_path / 'Fneu.npy', mmap_mode='r')
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
        'rich': base_dir / 'models' / 'rich' / 'suite2p_best_lgb.pkl',
    }

    model_path = model_spec
    if model_spec.lower() in model_map:
        model_path = model_map[model_spec.lower()]
    else:
        model_path = Path(model_spec)

    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        print("Available presets: 'regular', 'rich'")
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

    # Try to infer feature names list and custom formulas
    feature_names = None
    custom_features = {}
    
    # Try loading sibling JSON metadata first (precise matching)
    meta_path = model_path.with_suffix('.json')
    meta_loaded = False
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            if isinstance(meta_data, dict):
                feats = meta_data.get('active_features') or meta_data.get('features')
                if feats:
                    feature_names = feats
                    num_features = len(feature_names)
                custom_features = meta_data.get('custom_features', {})
                meta_loaded = True
                print(f"Loaded feature names and custom formulas from metadata file: {meta_path.name}")
        except Exception as e:
            print(f"Warning: Failed to load sibling metadata JSON: {e}")

    if not meta_loaded and not feature_names:
        # Look in model directory for any JSON file describing features
        model_dir = model_path.parent
        json_files = list(model_dir.glob("*.json"))
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        feats = data.get('active_features') or data.get('features')
                        if feats and len(feats) == num_features:
                            feature_names = feats
                            custom_features = data.get('custom_features', {})
                            meta_loaded = True
                            print(f"Loaded feature names from metadata file: {jf.name}")
                            break
            except:
                pass

    # Fallback to predefined lists if we still don't have feature names
    if not feature_names:
        if num_features == 24:
            feature_names = FEATURE_NAMES_24
        elif num_features == 25:
            feature_names = FEATURE_NAMES_25
        elif num_features == 26:
            feature_names = FEATURE_NAMES_26
        elif num_features == 27:
            feature_names = FEATURE_NAMES_27
        elif num_features == 30:
            feature_names = FEATURE_NAMES_30
        elif num_features == 38:
            feature_names = FEATURE_NAMES_38
        else:
            feature_names = [f"feature_{i}" for i in range(num_features)]

    X = extract_features(F, Fneu, stat, feature_names, custom_features=custom_features)
    probs = model.predict_proba(X)[:, 1]

    # Select optimal decision threshold based on feature layout (24, 25, 26, 27, 30 or 38)
    if num_features in (24, 25):
        threshold = 0.66
    elif num_features in (26, 38):
        threshold = 0.69
    elif num_features == 27:
        threshold = 0.66
    elif num_features == 30:
        threshold = 0.61
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
        print("  rich      - LightGBM with Rich Features (38 features)")
    else:
        model_choice = sys.argv[2] if len(sys.argv) > 2 else 'regular'
        apply_active_learning(sys.argv[1], model_choice)
