import numpy as np
import scipy.stats
from scipy.signal import find_peaks

# ==========================================
# 1. EXTRACTOR FUNCTIONS
# ==========================================

def _get_fcorr(cache):
    if '_fcorr' not in cache:
        cache['_fcorr'] = cache['F'] - np.float32(0.7) * cache['Fneu']
    return cache['_fcorr']

def _get_session_scale(cache):
    if '_session_scale' not in cache:
        F_corr = _get_fcorr(cache)
        std_fcorr = np.std(F_corr, axis=1)
        scale = np.median(std_fcorr)
        if np.isnan(scale) or scale <= 0:
            scale = 1.0
        cache['_session_scale'] = scale
    return cache['_session_scale']

def get_npix(cache): return cache['npix']
def get_solidity(cache): return cache['solidity']
def get_mrs(cache): return cache['mrs']
def get_compact(cache): return cache['compact']
def get_aspect_ratio(cache): return cache['aspect_ratio']
def get_radius(cache): return cache['radius']
def get_number_of_bright_pixels(cache): return cache['number_of_bright_pixels']
def get_bright_pixels_ratio(cache): return cache['bright_pixels_ratio']

def get_roi_idx_norm(cache):
    if 'npix' in cache:
        n_cells = cache['npix'].shape[0]
    elif 'F' in cache:
        n_cells = cache['F'].shape[0]
    else:
        return np.zeros(0, dtype=np.float32)
    return np.arange(n_cells, dtype=np.float32) / n_cells if n_cells > 0 else np.zeros(0, dtype=np.float32)

def get_skew_f(cache):
    return scipy.stats.skew(cache['F'], axis=1)

def get_std_f(cache):
    return np.std(cache['F'], axis=1) / _get_session_scale(cache)

def get_max_to_mean_f(cache):
    F = cache['F']
    mean = np.mean(F, axis=1)
    mean[mean <= 0] = 1e-10
    return np.max(F, axis=1) / mean

def get_cv_f(cache):
    F = cache['F']
    mean = np.mean(F, axis=1)
    mean[mean <= 0] = 1e-10
    return np.std(F, axis=1) / mean

def get_skew_fneu(cache):
    return scipy.stats.skew(cache['Fneu'], axis=1)

def get_corr_f_fneu(cache):
    F = cache['F']
    Fneu = cache['Fneu']
    mean_F = np.mean(F, axis=1, keepdims=True)
    mean_Fneu = np.mean(Fneu, axis=1, keepdims=True)
    cov = np.mean((F - mean_F) * (Fneu - mean_Fneu), axis=1)
    std_F = np.std(F, axis=1)
    std_Fneu = np.std(Fneu, axis=1)
    return cov / (std_F * std_Fneu + 1e-10)

def get_skew_fcorr(cache):
    F_corr = _get_fcorr(cache)
    return scipy.stats.skew(F_corr, axis=1)

def get_std_fcorr(cache):
    F_corr = _get_fcorr(cache)
    return np.std(F_corr, axis=1) / _get_session_scale(cache)

# Helper to get quantiles
def _get_quantile(cache, q_pct):
    if '_quantiles_map' not in cache:
        F_corr = _get_fcorr(cache)
        pcts = [10, 25, 50, 75, 90, 95, 99]
        q_vals = np.percentile(F_corr, pcts, axis=1)  # shape (7, n_cells)
        cache['_quantiles_map'] = {p: q_vals[idx] for idx, p in enumerate(pcts)}
    return cache['_quantiles_map'][q_pct] / _get_session_scale(cache)

def get_q10(cache): return _get_quantile(cache, 10)
def get_q25(cache): return _get_quantile(cache, 25)
def get_q50(cache): return _get_quantile(cache, 50)
def get_q75(cache): return _get_quantile(cache, 75)
def get_q90(cache): return _get_quantile(cache, 90)
def get_q95(cache): return _get_quantile(cache, 95)
def get_q99(cache): return _get_quantile(cache, 99)

def get_avg_asym(cache): return cache['avg_asym']
def get_max_asym(cache): return cache['max_asym']
def get_max_width(cache): return cache['max_width']

def get_range_fcorr(cache):
    F_corr = _get_fcorr(cache)
    return (np.max(F_corr, axis=1) - np.min(F_corr, axis=1)) / _get_session_scale(cache)

def get_range_f(cache):
    F = cache['F']
    return (np.max(F, axis=1) - np.min(F, axis=1)) / _get_session_scale(cache)

def get_snr(cache):
    F_corr = _get_fcorr(cache)
    std = np.std(F_corr, axis=1)
    diff = np.diff(F_corr, axis=1)
    std_diff = np.std(diff, axis=1)
    std_diff[std_diff <= 0] = 1e-6
    return std / std_diff

def get_activity_ratio(cache):
    if '_sorted_Fcorr' not in cache:
        F_corr = _get_fcorr(cache)
        cache['_sorted_Fcorr'] = np.sort(F_corr, axis=1)
    sorted_F = cache['_sorted_Fcorr']
    half_idx = sorted_F.shape[1] // 2
    std_below = np.std(sorted_F[:, :half_idx], axis=1)
    std_above = np.std(sorted_F[:, half_idx:], axis=1)
    std_below = np.maximum(std_below, 1e-6)
    return std_above / std_below

def get_peak_density(cache):
    F_corr = _get_fcorr(cache)
    densities = []
    n_cells = F_corr.shape[0]
    for i in range(n_cells):
        f = F_corr[i]
        peaks, _ = find_peaks(f, height=np.mean(f) + 2*np.std(f), distance=10)
        densities.append(len(peaks) / len(f) if len(f) > 0 else 0.0)
    return np.array(densities)

def get_skew_diff_fcorr(cache):
    F_corr = _get_fcorr(cache)
    diff = np.diff(F_corr, axis=1)
    return scipy.stats.skew(diff, axis=1)

def get_smoothness_ratio_w5(cache):
    import scipy.ndimage
    F_corr = _get_fcorr(cache)
    std_raw = np.std(F_corr, axis=1)
    std_raw = np.maximum(std_raw, 1e-6)
    F_smooth = scipy.ndimage.uniform_filter1d(F_corr, size=5, axis=1)
    std_smooth = np.std(F_smooth, axis=1)
    return std_smooth / std_raw

def get_smoothness_ratio_w11(cache):
    import scipy.ndimage
    F_corr = _get_fcorr(cache)
    std_raw = np.std(F_corr, axis=1)
    std_raw = np.maximum(std_raw, 1e-6)
    F_smooth = scipy.ndimage.uniform_filter1d(F_corr, size=11, axis=1)
    std_smooth = np.std(F_smooth, axis=1)
    return std_smooth / std_raw

def get_fraction_active_k3(cache):
    if '_sorted_Fcorr' not in cache:
        F_corr = _get_fcorr(cache)
        cache['_sorted_Fcorr'] = np.sort(F_corr, axis=1)
    sorted_F = cache['_sorted_Fcorr']
    n_frames = sorted_F.shape[1]
    
    med = sorted_F[:, n_frames // 2]
    std_below = np.std(sorted_F[:, :n_frames // 2], axis=1)
    std_below = np.maximum(std_below, 1e-6)
    thresh = med + 3.0 * std_below
    
    F_corr = _get_fcorr(cache)
    return np.mean(F_corr > thresh[:, None], axis=1)

def get_q99_smooth_ratio(cache):
    import scipy.ndimage
    F_corr = _get_fcorr(cache)
    F_smooth = scipy.ndimage.uniform_filter1d(F_corr, size=5, axis=1)
    
    if '_sorted_Fcorr' not in cache:
        cache['_sorted_Fcorr'] = np.sort(F_corr, axis=1)
    sorted_F = cache['_sorted_Fcorr']
    n_frames = sorted_F.shape[1]
    
    med_raw = sorted_F[:, n_frames // 2]
    q99_raw = sorted_F[:, int(0.99 * (n_frames - 1))]
    val_raw = q99_raw - med_raw
    
    sorted_smooth = np.sort(F_smooth, axis=1)
    med_smooth = sorted_smooth[:, n_frames // 2]
    q99_smooth = sorted_smooth[:, int(0.99 * (n_frames - 1))]
    val_smooth = q99_smooth - med_smooth
    val_smooth = np.maximum(val_smooth, 1e-6)
    
    return val_raw / val_smooth

def get_skew_fcorr_smooth_w5(cache):
    import scipy.ndimage
    import scipy.stats
    F_corr = _get_fcorr(cache)
    F_smooth = scipy.ndimage.uniform_filter1d(F_corr, size=5, axis=1)
    return scipy.stats.skew(F_smooth, axis=1)

def get_skew_fcorr_smooth_w11(cache):
    import scipy.ndimage
    import scipy.stats
    F_corr = _get_fcorr(cache)
    F_smooth = scipy.ndimage.uniform_filter1d(F_corr, size=11, axis=1)
    return scipy.stats.skew(F_smooth, axis=1)

def get_area_to_radius_sq(cache):
    npix = cache['npix']
    radius = cache['radius']
    denom = radius ** 2
    denom = np.maximum(denom, 1e-6)
    return npix / denom

def get_bright_pixels_to_radius_sq(cache):
    nbright = cache['number_of_bright_pixels']
    radius = cache['radius']
    denom = radius ** 2
    denom = np.maximum(denom, 1e-6)
    return nbright / denom

def get_peak_to_q99_ratio(cache):
    F = cache['F']
    Fneu = cache['Fneu']
    n_cells = F.shape[0]
    out = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        fcorr = F[i] - 0.7 * Fneu[i]
        sorted_fcorr = np.sort(fcorr)
        n_frames = len(sorted_fcorr)
        max_val = sorted_fcorr[-1]
        q99_val = sorted_fcorr[int(0.99 * (n_frames - 1))]
        med = sorted_fcorr[n_frames // 2]
        max_diff = max_val - med
        q99_diff = max(q99_val - med, 1e-6)
        out[i] = max_diff / q99_diff
    return out

def get_peak_to_q95_ratio(cache):
    F = cache['F']
    Fneu = cache['Fneu']
    n_cells = F.shape[0]
    out = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        fcorr = F[i] - 0.7 * Fneu[i]
        sorted_fcorr = np.sort(fcorr)
        n_frames = len(sorted_fcorr)
        max_val = sorted_fcorr[-1]
        q95_val = sorted_fcorr[int(0.95 * (n_frames - 1))]
        med = sorted_fcorr[n_frames // 2]
        max_diff = max_val - med
        q95_diff = max(q95_val - med, 1e-6)
        out[i] = max_diff / q95_diff
    return out

def get_range_ratio_f_fneu(cache):
    F = cache['F']
    Fneu = cache['Fneu']
    n_cells = F.shape[0]
    out = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        q5_f, q99_f = np.percentile(F[i], [5, 99])
        q5_n, q99_n = np.percentile(Fneu[i], [5, 99])
        range_f = q99_f - q5_f
        range_neu = max(q99_n - q5_n, 1e-6)
        out[i] = range_f / range_neu
    return out

def get_mean_diff_f_fneu(cache):
    F = cache['F']
    Fneu = cache['Fneu']
    n_cells = F.shape[0]
    out = np.zeros(n_cells, dtype=np.float32)
    session_scale = _get_session_scale(cache)
    for i in range(n_cells):
        diff = np.mean(F[i]) - np.mean(Fneu[i])
        out[i] = diff / session_scale
    return out

# ==========================================
# 2. FEATURE REGISTRY (strictly dictionary based)
# ==========================================

FEATURE_REGISTRY = {
    'npix': get_npix,
    'solidity': get_solidity,
    'mrs': get_mrs,
    'compact': get_compact,
    'aspect_ratio': get_aspect_ratio,
    'radius': get_radius,
    'number_of_bright_pixels': get_number_of_bright_pixels,
    'bright_pixels_ratio': get_bright_pixels_ratio,
    'roi_idx_norm': get_roi_idx_norm,
    'skew_f': get_skew_f,
    'std_f': get_std_f,
    'max_to_mean_f': get_max_to_mean_f,
    'cv_f': get_cv_f,
    'skew_fneu': get_skew_fneu,
    'corr_f_fneu': get_corr_f_fneu,
    'skew_fcorr': get_skew_fcorr,
    'std_fcorr': get_std_fcorr,
    'q10': get_q10,
    'q25': get_q25,
    'q50': get_q50,
    'q75': get_q75,
    'q90': get_q90,
    'q95': get_q95,
    'q99': get_q99,
    'avg_asym': get_avg_asym,
    'max_asym': get_max_asym,
    'max_width': get_max_width,
    'range_fcorr': get_range_fcorr,
    'range_f': get_range_f,
    'snr': get_snr,
    'activity_ratio': get_activity_ratio,
    'peak_density': get_peak_density,
    'skew_diff_fcorr': get_skew_diff_fcorr,
    'smoothness_ratio_w5': get_smoothness_ratio_w5,
    'smoothness_ratio_w11': get_smoothness_ratio_w11,
    'fraction_active_k3': get_fraction_active_k3,
    'q99_smooth_ratio': get_q99_smooth_ratio,
    'skew_fcorr_smooth_w5': get_skew_fcorr_smooth_w5,
    'skew_fcorr_smooth_w11': get_skew_fcorr_smooth_w11,
    'area_to_radius_sq': get_area_to_radius_sq,
    'bright_pixels_to_radius_sq': get_bright_pixels_to_radius_sq,
    'peak_to_q99_ratio': get_peak_to_q99_ratio,
    'peak_to_q95_ratio': get_peak_to_q95_ratio,
    'range_ratio_f_fneu': get_range_ratio_f_fneu,
    'mean_diff_f_fneu': get_mean_diff_f_fneu,
}

# ==========================================
# 3. ACTIVE FEATURES LIST (Modified during FE)
# ==========================================

ACTIVE_FEATURES = [
    'area_to_radius_sq', 'aspect_ratio', 'compact',
    'corr_f_fneu', 'max_width', 'mrs',
    'peak_to_q95_ratio', 'peak_to_q99_ratio',
    'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'radius', 'range_f', 'range_fcorr', 'skew_diff_fcorr',
    'skew_f', 'skew_fcorr', 'skew_fneu', 'solidity',
    'std_f', 'std_fcorr'
]


# ==========================================
# 4. SECURE AST MATH FORMULA EVALUATOR
# ==========================================

import ast
import operator

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    'log': np.log,
    'log10': np.log10,
    'exp': np.exp,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'sin': np.sin,
    'cos': np.cos,
}

def safe_eval_formula(expr_str: str, feature_ns: dict) -> np.ndarray:
    """
    Parse a mathematical formula string into an AST and evaluate it securely.
    Args:
        expr_str: Mathematical expression string, e.g. "log(solidity * mrs + 1e-5)"
        feature_ns: Dictionary containing 1D numpy arrays of existing features.
    """
    try:
        node = ast.parse(expr_str.strip(), mode='eval').body
    except SyntaxError as se:
        raise ValueError(f"Invalid mathematical syntax: '{expr_str}'. Detail: {se}")
    
    def eval_node(n):
        if isinstance(n, ast.Constant):  # Python 3.8+
            return n.value
        elif isinstance(n, ast.Num):      # Python < 3.8 compatibility
            return n.n
        elif isinstance(n, ast.Name):
            if n.id in feature_ns:
                return feature_ns[n.id]
            # Try case-insensitive lookup
            lower_ns = {k.lower(): v for k, v in feature_ns.items()}
            if n.id.lower() in lower_ns:
                return lower_ns[n.id.lower()]
            raise ValueError(f"Unknown feature name: '{n.id}'. Please check spelling.")
        elif isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)
            op_type = type(n.op)
            if op_type in SAFE_OPERATORS:
                if op_type == ast.Div:
                    denom = np.array(right)
                    denom = np.where(denom == 0, 1e-10, denom)
                    return left / denom
                return SAFE_OPERATORS[op_type](left, right)
            raise TypeError(f"Unsupported mathematical operator: {op_type.__name__}")
        elif isinstance(n, ast.UnaryOp):
            operand = eval_node(n.operand)
            op_type = type(n.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](operand)
            raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
        elif isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise TypeError("Only direct math function calls are allowed.")
            func_name = n.func.id
            if func_name in SAFE_FUNCTIONS:
                if len(n.args) != 1:
                    raise ValueError(f"Math function '{func_name}' expects 1 argument, got {len(n.args)}.")
                arg_val = eval_node(n.args[0])
                if func_name in ('log', 'log10'):
                    arg_val = np.maximum(arg_val, 1e-10)
                elif func_name == 'sqrt':
                    arg_val = np.maximum(arg_val, 0.0)
                return SAFE_FUNCTIONS[func_name](arg_val)
            raise ValueError(f"Unsupported function: '{func_name}()'")
            
        raise TypeError(f"Unsupported syntax expression structure: {type(n).__name__}")
        
    return eval_node(node)

