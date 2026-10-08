"""
Compute a model's features for one Suite2p plane folder with exactly the code
used to build the training tables (scripts/build_feature_tables.py), without
needing a preprocessed_cache entry. Used by the 'image' preset of apply_AI.py.
"""
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths

from fe_engine.fe_loop_runner import extract_features_dataset
from fe_engine.image_features import IMAGE_FEATURES, compute_image_features


def _morphology(stat):
    """Per-ROI stat.npy fields, as stored by fe_preprocessor.preprocess_single_session."""
    n = len(stat)
    out = {k: np.zeros(n, dtype=np.float32) for k in
           ['npix', 'solidity', 'mrs', 'compact', 'aspect_ratio', 'radius',
            'number_of_bright_pixels', 'bright_pixels_ratio']}
    defaults = {'npix': 0, 'solidity': 1.0, 'mrs': 0.0, 'compact': 0.0, 'aspect_ratio': 1.0, 'radius': 0.0}
    for i, s in enumerate(stat):
        for k, dv in defaults.items():
            out[k][i] = float(s.get(k, dv))
        lam = s.get('lam', np.zeros(0))
        max_lam = np.max(lam) if len(lam) > 0 else 0.0
        nb = float(np.sum(lam > 0.1 * max_lam)) if max_lam > 0 else 0.0
        out['number_of_bright_pixels'][i] = nb
        out['bright_pixels_ratio'][i] = nb / out['npix'][i] if out['npix'][i] > 0 else 0.0
    return out


def _max_width(F, Fneu):
    """Widest calcium transient (frames at half height), as in fe_preprocessor.extract_cell_features."""
    out = np.zeros(F.shape[0], dtype=np.float32)
    for i in range(F.shape[0]):
        f_corr = np.asarray(F[i], dtype=np.float32) - 0.7 * np.asarray(Fneu[i], dtype=np.float32)
        peaks, _ = find_peaks(f_corr, height=np.mean(f_corr) + 2 * np.std(f_corr), distance=10)
        if len(peaks) > 0:
            out[i] = float(np.max(peak_widths(f_corr, peaks, rel_height=0.5)[0]))
    return out


def compute_session_features(session_path, feature_names, progress=None):
    """
    Returns (X, image_ok): X is (n_rois, len(feature_names)) in the given order.
    image_ok is False when the model needs image features but ops.npy is missing
    or does not match stat.npy (the image columns are then all NaN).
    progress(fraction, status), if given, is called as the steps advance (fraction 0-1).
    """
    progress = progress or (lambda fraction, status: None)
    session_path = Path(session_path)
    stat = np.load(session_path / 'stat.npy', allow_pickle=True)
    n = len(stat)
    session = _morphology(stat)
    session.update(session_path=str(session_path.resolve()), session_name=session_path.name,
                   y=np.zeros(n, dtype=np.int32), group_id=np.zeros(n, dtype=np.int32))

    trace_names = [f for f in feature_names if f not in IMAGE_FEATURES]
    if 'max_width' in trace_names:
        progress(0.0, 'Measuring transient widths...')
        F = np.load(session_path / 'F.npy', mmap_mode='r')
        Fneu = np.load(session_path / 'Fneu.npy', mmap_mode='r')
        session['max_width'] = _max_width(F, Fneu)
    progress(0.15, 'Computing trace features...')
    X_trace, _, _ = extract_features_dataset(
        [session], trace_names,
        progress_callback=lambda f: progress(0.15 + 0.7 * f, 'Computing trace features...'))
    cols = dict(zip(trace_names, X_trace.T))

    image_ok = True
    img_names = [f for f in feature_names if f in IMAGE_FEATURES]
    if img_names:
        progress(0.85, 'Computing image features from ops.npy...')
        if (session_path / 'ops.npy').exists():
            img = compute_image_features(session_path, stat=stat)
        else:
            img = {k: np.full(n, np.nan, dtype=np.float32) for k in IMAGE_FEATURES}
        image_ok = not all(np.isnan(img[k]).all() for k in img_names)
        cols.update({k: img[k] for k in img_names})

    return np.column_stack([cols[f] for f in feature_names]).astype(np.float32), image_ok
