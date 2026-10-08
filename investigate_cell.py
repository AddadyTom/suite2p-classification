import os
import sys
import gc
import time
import json
import joblib
import shutil
import numpy as np
import urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from scipy.stats import skew
from scipy.signal import find_peaks, peak_widths
import webbrowser
import threading
from fe_engine.fe_definitions import FEATURE_REGISTRY
from apply_AI import run_ai_pipeline

# ==========================================
# 1. MASTER FEATURE DEFINITIONS & PRESETS
# ==========================================

BASE_DIR = Path(__file__).parent.resolve()

MASTER_FEATURE_DESCS = {
    'skew_spatial': 'Spatial skewness of cell pixels. Higher values indicate uneven intensity.',
    'compact': 'Spatial compactness. Measures circularity; real cells are circular.',
    'npix': 'Number of pixels. Indicates the physical size of the ROI.',
    'bright_pixels_ratio': 'Ratio of bright pixels to total pixels in the ROI (bright_pixels / npix).',
    'max_to_mean_f': 'Dynamic range ratio (Max F / Mean F). Measures peak transient height relative to baseline.',
    'cv_f': 'Coefficient of variation (Std F / Mean F). Scale-free measurement of trace dynamics.',
    'aspect_ratio': 'Aspect ratio of bounding box. Real cells tend to be close to 1.0.',
    'radius': 'Radius of the ROI computed from spatial moments.',
    'solidity': 'Solidity/density of cell border. Crisp borders have higher solidity.',
    'mrs': 'Morphological Roundness Score. Lower is rounder (like a sphere).',
    'roi_idx_raw': 'Raw ROI index. Position in suite2p output list.',
    'roi_idx_norm': 'Normalized ROI index. Position normalized by total ROIs in session.',
    'roi_idx_norm_3bin': 'Coarse binned normalized ROI index (0, 1, or 2) representing the Suite2p confidence prior.',
    'skew_f': 'Skewness of raw fluorescence. Real cells have high skew due to sparse bursts.',
    'std_f': 'Standard deviation of raw fluorescence. Indicates activity amplitude.',
    'max_f': 'Maximum raw fluorescence intensity.',
    'mean_f': 'Mean raw fluorescence intensity.',
    'skew_fneu': 'Skewness of neuropil trace. High skew indicates background contamination.',
    'corr_f_fneu': 'Correlation between cell and neuropil traces. High correlation indicates motion artifacts.',
    'skew_fcorr': 'Skewness of corrected trace. Highly active cells have large skew.',
    'std_fcorr': 'Standard deviation of corrected trace. Key proxy for calcium activity levels.',
    'q10': '10th percentile of corrected trace. Background noise level.',
    'q25': '25th percentile of corrected trace. Quiet period baseline.',
    'q50': '50th percentile (median) of corrected trace.',
    'q75': '75th percentile of corrected trace.',
    'q90': '90th percentile of corrected trace.',
    'q95': '95th percentile of corrected trace.',
    'q99': '99th percentile of corrected trace. Peak active level.',
    'avg_asym': 'Average rise-to-decay peak asymmetry. Calcium binds fast and decays slow.',
    'max_asym': 'Maximum rise-to-decay asymmetry across all peaks.',
    'max_width': 'Maximum width of peaks in frames. Long events indicate slow drift or artifacts.',
    'max_spk': 'Maximum value of deconvolved spikes. Indicates strongest firing burst.',
    'mean_spk_nz': 'Mean of non-zero deconvolved spikes. Average burst size.',
    'spk_rate': 'Fraction of frames with spike activity.',
    'skew_spk': 'Skewness of deconvolved spikes. Indicates sparseness of active periods.',
    'range_fcorr': 'Range of corrected trace (Max - Min). Overall trace fluctuation.',
    'range_f': 'Range of raw trace (Max - Min). Overall raw fluctuation.',
    'snr': 'Temporal signal-to-noise ratio. Real neural transients are smooth while noise fluctuates rapidly.',
    'activity_ratio': 'Ratio of variance above the median to variance below. Highlights positive-going calcium transients.',
    'peak_density': 'Density of 2-standard-deviation peaks per frame. Measures active firing frequency.'
}

# Features of the image model (models/image), computed by fe_engine/image_features.py and
# fe_engine/fe_definitions.py (intensity-normalized trace features)
for _img in ('meanImg', 'meanImgE', 'max_proj', 'Vcorr'):
    MASTER_FEATURE_DESCS[f'{_img}_contrast_z'] = (f'Mask vs surrounding 2-7 px ring in {_img}: (lam-weighted mean inside '
                                                  f'- ring mean) / ring std. Real somata stand out from their surroundings.')
    MASTER_FEATURE_DESCS[f'{_img}_lam_corr'] = (f'Correlation between the mask weights (lam) and {_img} pixels inside the mask. '
                                                f'High when the bright part of the mask sits on a bright blob.')
    MASTER_FEATURE_DESCS[f'{_img}_lam_corr_patch'] = (f'Correlation between the mask weights (0 in the ring) and {_img} over mask + ring: '
                                                      f'shape and contrast match together.')
MASTER_FEATURE_DESCS.update({
    'meanImg_ratio': 'Mean image brightness inside the mask divided by the surrounding ring.',
    'max_proj_ratio': 'Max-projection brightness inside the mask divided by the surrounding ring: did the ROI light up more than its surroundings?',
    'Vcorr_in_mean': 'Mean local-correlation (Vcorr) value inside the mask. High when the pixels fluctuate together, as in an active cell.',
    'Vcorr_in_minus_ring': 'Vcorr inside the mask minus Vcorr in the surrounding ring.',
    'meanImg_neuropil_contrast': "Mean image inside the mask vs Suite2p's own neuropil mask (which excludes other ROIs), in units of neuropil std.",
    'dff_q95': 'dF/F at the 95th percentile (F_corr - median) / median(F).',
    'dff_q99': 'dF/F at the 99th percentile.',
    'dff_q999': 'dF/F at the 99.9th percentile: size of the largest transients.',
    'dff_max': 'Maximum dF/F.',
    'dff_range': 'dF/F range between the 1st and 99th percentiles.',
    'q95_over_noise': '95th percentile of F_corr above its median, in units of the trace noise (MAD of first differences).',
    'q99_over_noise': '99th percentile of F_corr above its median, in noise units.',
    'q999_over_noise': '99.9th percentile of F_corr above its median, in noise units: transient SNR.',
    'range_over_noise': 'F_corr range (1st-99th percentile) in noise units.',
    'noise_over_f0': 'Trace noise relative to baseline F (noise / median F).',
    'f_over_fneu_baseline': 'Baseline ratio median(F) / median(Fneu): how much brighter the ROI is than its neuropil.',
})

# Legacy Feature Lists
FEATURE_NAMES_24_legacy = [
    'number_of_bright_pixels', 'solidity', 'mrs', 'roi_idx_norm',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f',
    'skew_fneu', 'corr_f_fneu', 'skew_fcorr', 'std_fcorr', 'q10', 'q25',
    'q50', 'q75', 'q90', 'q95', 'q99', 'avg_asym', 'max_asym', 'max_width',
    'range_fcorr', 'range_f'
]

FEATURE_NAMES_26_legacy = [
    'number_of_bright_pixels', 'solidity', 'mrs',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fneu', 'corr_f_fneu',
    'skew_fcorr', 'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'avg_asym', 'max_asym', 'max_width', 'range_fcorr', 'range_f',
    'snr', 'activity_ratio', 'peak_density'
]

FEATURE_NAMES_27_legacy = [
    'number_of_bright_pixels', 'solidity', 'mrs', 'roi_idx_norm_3bin',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fneu', 'corr_f_fneu',
    'skew_fcorr', 'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'avg_asym', 'max_asym', 'max_width', 'range_fcorr', 'range_f',
    'snr', 'activity_ratio', 'peak_density'
]

# New Feature Lists
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

FEATURE_NAMES_28_bin = [
    'number_of_bright_pixels', 'bright_pixels_ratio', 'solidity', 'mrs', 'roi_idx_norm_3bin',
    'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fneu', 'corr_f_fneu',
    'skew_fcorr', 'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99',
    'avg_asym', 'max_asym', 'max_width', 'range_fcorr', 'range_f',
    'snr', 'activity_ratio', 'peak_density'
]

FEATURE_NAMES_28 = [
    'skew_spatial', 'npix', 'aspect_ratio', 'radius', 'solidity', 'mrs',
    'roi_idx_raw', 'roi_idx_norm', 'skew_f', 'std_f', 'max_f', 'mean_f',
    'skew_fneu', 'corr_f_fneu', 'skew_fcorr', 'std_fcorr', 'q10', 'q25',
    'q50', 'q75', 'q90', 'q95', 'q99', 'avg_asym', 'max_asym', 'max_width',
    'range_fcorr', 'range_f'
]

FEATURE_NAMES_29 = [
    'skew_spatial', 'compact', 'npix', 'aspect_ratio', 'radius', 'solidity', 'mrs',
    'skew_f', 'std_f', 'max_f', 'mean_f', 'skew_fneu', 'corr_f_fneu', 'skew_fcorr',
    'std_fcorr', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99', 'avg_asym',
    'max_asym', 'max_width', 'max_spk', 'mean_spk_nz', 'spk_rate', 'skew_spk'
]

FEATURE_NAMES_31 = [
    'skew_spatial', 'npix', 'aspect_ratio', 'radius', 'solidity', 'mrs',
    'roi_idx_raw', 'roi_idx_norm', 'skew_f', 'std_f', 'max_f', 'mean_f',
    'skew_fneu', 'corr_f_fneu', 'skew_fcorr', 'std_fcorr', 'q10', 'q25',
    'q50', 'q75', 'q90', 'q95', 'q99', 'avg_asym', 'max_asym', 'max_width',
    'max_spk', 'mean_spk_nz', 'spk_rate', 'range_fcorr', 'range_f'
]

# ==========================================
# 2. FEATURE EXTRACTION PIPELINES
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

def extract_features_single(F_row, Fneu_row, spks_row, stat_entry, roi_idx, n_rois, num_features_or_names):
    f = F_row
    fneu = Fneu_row
    f_corr = f - 0.7 * fneu
    s = spks_row if spks_row is not None else np.zeros_like(f)
    
    # Check if we got a list of feature names
    if isinstance(num_features_or_names, (list, tuple, np.ndarray)):
        feature_names = num_features_or_names
        avg_asym, max_asym, max_width = None, None, None
        bright_pix = None
        
        row = []
        for name in feature_names:
            # Spatial features
            if name == 'npix':
                row.append(float(stat_entry.get('npix', 0)))
            elif name == 'skew_spatial':
                row.append(float(stat_entry.get('skew', 0)))
            elif name == 'compact':
                row.append(float(stat_entry.get('compact', 0.0)))
            elif name == 'aspect_ratio':
                row.append(float(stat_entry.get('aspect_ratio', 1.0)))
            elif name == 'radius':
                row.append(float(stat_entry.get('radius', 0.0)))
            elif name == 'solidity':
                row.append(float(stat_entry.get('solidity', 1.0)))
            elif name == 'mrs':
                row.append(float(stat_entry.get('mrs', 0.0)))
            elif name == 'number_of_bright_pixels':
                if bright_pix is None:
                    lam = stat_entry.get('lam', np.zeros(0))
                    max_lam = np.max(lam) if len(lam) > 0 else 0.0
                    bright_pix = float(np.sum(lam > 0.1 * max_lam)) if max_lam > 0 else 0.0
                row.append(bright_pix)
            elif name == 'bright_pixels_ratio':
                if bright_pix is None:
                    lam = stat_entry.get('lam', np.zeros(0))
                    max_lam = np.max(lam) if len(lam) > 0 else 0.0
                    bright_pix = float(np.sum(lam > 0.1 * max_lam)) if max_lam > 0 else 0.0
                npix = float(stat_entry.get('npix', 0))
                row.append(bright_pix / npix if npix > 0 else 0.0)
                
            # Index features
            elif name == 'roi_idx_norm':
                row.append(float(roi_idx / n_rois) if n_rois > 0 else 0.0)
            elif name == 'roi_idx_norm_3bin':
                norm_val = roi_idx / n_rois if n_rois > 0 else 0.0
                row.append(float(np.digitize(norm_val, [0.1, 0.4])))
            elif name == 'roi_idx_raw' or name == 'roi_idx':
                row.append(float(roi_idx))
                
            # Trace Stats features
            elif name == 'skew_f':
                row.append(float(skew(f)))
            elif name == 'std_f':
                row.append(float(np.std(f)))
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
                row.append(float(np.std(f_corr)))
                
            # Quantiles
            elif name.startswith('q') and name[1:].isdigit():
                pct = float(name[1:]) / 100.0
                row.append(float(np.quantile(f_corr, pct)))
                
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
                    
            # Spikes
            elif name == 'max_spk':
                row.append(float(np.max(s)))
            elif name == 'mean_spk_nz':
                row.append(float(np.mean(s[s > 0]) if np.sum(s > 0) > 0 else 0.0))
            elif name == 'spk_rate':
                max_s = np.max(s)
                row.append(float(np.count_nonzero(s > 0.01 * max_s) / len(s) if max_s > 0 else 0.0))
            elif name == 'skew_spk':
                row.append(float(skew(s)))
                
            # Amplitudes
            elif name == 'range_fcorr':
                row.append(float(np.max(f_corr) - np.min(f_corr)))
            elif name == 'range_f':
                row.append(float(np.max(f) - np.min(f)))
                
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
                
            elif name in FEATURE_REGISTRY:
                cache = {
                    'stat': stat_cell,
                    'F': f[None, :],
                    'Fneu': fneu[None, :],
                    '_fcorr': f_corr[None, :],
                    'npix': np.array([stat_cell['npix']]),
                    'solidity': np.array([stat_cell.get('solidity', 1.0)]),
                    'mrs': np.array([stat_cell.get('mrs', 0.0)]),
                    'compact': np.array([stat_cell.get('compact', 1.0)]),
                    'aspect_ratio': np.array([stat_cell.get('aspect_ratio', 1.0)]),
                    'radius': np.array([stat_cell.get('radius', 1.0)]),
                    'number_of_bright_pixels': np.array([np.sum(stat_cell['lam'] > 0.1 * np.max(stat_cell['lam'])) if 'lam' in stat_cell else 1]),
                    'bright_pixels_ratio': np.array([np.sum(stat_cell['lam'] > 0.1 * np.max(stat_cell['lam'])) / max(stat_cell['npix'], 1) if 'lam' in stat_cell else 1.0]),
                    'avg_asym': np.array([avg_asym if avg_asym is not None else 1.0]),
                    'max_asym': np.array([max_asym if max_asym is not None else 1.0]),
                    'max_width': np.array([max_width if max_width is not None else 0.0]),
                }
                val = FEATURE_REGISTRY[name](cache)
                row.append(float(val[0]))
            else:
                print(f"Warning: Unknown feature name '{name}'. Defaulting to 0.0.")
                row.append(0.0)
                
        return np.array(row)
        
    num_features = num_features_or_names
    avg_asym, max_asym, max_width = extract_peak_features(f_corr)
    
    if num_features == 24:
        # Spatial (4)
        lam = stat_entry.get('lam', np.zeros(0))
        max_lam = np.max(lam) if len(lam) > 0 else 0.0
        bright_pix = np.sum(lam > 0.1 * max_lam) if max_lam > 0 else 0
        
        spatial = [
            bright_pix,
            stat_entry.get('solidity', 1.0),
            stat_entry.get('mrs', 0),
            roi_idx / n_rois if n_rois > 0 else 0.0
        ]
        # Trace Stats (8)
        mean_f_val = np.mean(f)
        max_to_mean_f = np.max(f) / mean_f_val if mean_f_val > 0 else 1.0
        cv_f = np.std(f) / mean_f_val if mean_f_val > 0 else 0.0
        
        trace_stats = [
            skew(f), np.std(f), max_to_mean_f, cv_f,
            skew(fneu),
            np.corrcoef(f, fneu)[0, 1] if np.std(f)>0 and np.std(fneu)>0 else 0,
            skew(f_corr), np.std(f_corr)
        ]
        # Percentiles (7)
        q = np.quantile(f_corr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).tolist()
        # Dynamics (3)
        dynamics = [avg_asym, max_asym, max_width]
        # Trace Amplitudes (2)
        range_fcorr = np.max(f_corr) - np.min(f_corr)
        range_f = np.max(f) - np.min(f)
        amplitudes = [range_fcorr, range_f]
        
        return np.array(spatial + trace_stats + q + dynamics + amplitudes)
        
    elif num_features == 26:
        # Option A: No Index
        # Spatial (3)
        lam = stat_entry.get('lam', np.zeros(0))
        max_lam = np.max(lam) if len(lam) > 0 else 0.0
        bright_pix = np.sum(lam > 0.1 * max_lam) if max_lam > 0 else 0
        
        spatial = [
            bright_pix,
            stat_entry.get('solidity', 1.0),
            stat_entry.get('mrs', 0)
        ]
        # Trace Stats (8)
        mean_f_val = np.mean(f)
        max_to_mean_f = np.max(f) / mean_f_val if mean_f_val > 0 else 1.0
        cv_f = np.std(f) / mean_f_val if mean_f_val > 0 else 0.0
        
        trace_stats = [
            skew(f), np.std(f), max_to_mean_f, cv_f,
            skew(fneu),
            np.corrcoef(f, fneu)[0, 1] if np.std(f)>0 and np.std(fneu)>0 else 0,
            skew(f_corr), np.std(f_corr)
        ]
        # Percentiles (7)
        q = np.quantile(f_corr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).tolist()
        # Dynamics (3)
        dynamics = [avg_asym, max_asym, max_width]
        # Trace Amplitudes (2)
        range_fcorr = np.max(f_corr) - np.min(f_corr)
        range_f = np.max(f) - np.min(f)
        amplitudes = [range_fcorr, range_f]
        
        # New Biological (3)
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
        
        return np.array(spatial + trace_stats + q + dynamics + amplitudes + new_bio)
        
    elif num_features == 27:
        # Option B: 3-Bin Index
        # Spatial (4)
        lam = stat_entry.get('lam', np.zeros(0))
        max_lam = np.max(lam) if len(lam) > 0 else 0.0
        bright_pix = np.sum(lam > 0.1 * max_lam) if max_lam > 0 else 0
        
        norm_val = roi_idx / n_rois if n_rois > 0 else 0.0
        binned = float(np.digitize(norm_val, [0.1, 0.4]))
        
        spatial = [
            bright_pix,
            stat_entry.get('solidity', 1.0),
            stat_entry.get('mrs', 0),
            binned
        ]
        # Trace Stats (8)
        mean_f_val = np.mean(f)
        max_to_mean_f = np.max(f) / mean_f_val if mean_f_val > 0 else 1.0
        cv_f = np.std(f) / mean_f_val if mean_f_val > 0 else 0.0
        
        trace_stats = [
            skew(f), np.std(f), max_to_mean_f, cv_f,
            skew(fneu),
            np.corrcoef(f, fneu)[0, 1] if np.std(f)>0 and np.std(fneu)>0 else 0,
            skew(f_corr), np.std(f_corr)
        ]
        # Percentiles (7)
        q = np.quantile(f_corr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).tolist()
        # Dynamics (3)
        dynamics = [avg_asym, max_asym, max_width]
        # Trace Amplitudes (2)
        range_fcorr = np.max(f_corr) - np.min(f_corr)
        range_f = np.max(f) - np.min(f)
        amplitudes = [range_fcorr, range_f]
        
        # New Biological (3)
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
        
        return np.array(spatial + trace_stats + q + dynamics + amplitudes + new_bio)
        
    elif num_features == 28:
        # Spatial (8)
        spatial = [
            stat_entry.get('skew', 0),
            stat_entry.get('npix', 0),
            stat_entry.get('aspect_ratio', 1.0),
            stat_entry.get('radius', 0),
            stat_entry.get('solidity', 1.0),
            stat_entry.get('mrs', 0),
            roi_idx,
            roi_idx / n_rois if n_rois > 0 else 0.0
        ]
        # Trace Stats (8)
        trace_stats = [
            skew(f), np.std(f), np.max(f), np.mean(f),
            skew(fneu),
            np.corrcoef(f, fneu)[0, 1] if np.std(f)>0 and np.std(fneu)>0 else 0,
            skew(f_corr), np.std(f_corr)
        ]
        # Percentiles (7)
        q = np.quantile(f_corr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).tolist()
        # Dynamics (3)
        dynamics = [avg_asym, max_asym, max_width]
        # Trace Amplitudes (2)
        range_fcorr = np.max(f_corr) - np.min(f_corr)
        range_f = np.max(f) - np.min(f)
        amplitudes = [range_fcorr, range_f]
        
        return np.array(spatial + trace_stats + q + dynamics + amplitudes)
        
    elif num_features == 29:
        # Spatial (7)
        spatial = [
            stat_entry.get('skew', 0),
            stat_entry.get('compact', 0),
            stat_entry.get('npix', 0),
            stat_entry.get('aspect_ratio', 1.0),
            stat_entry.get('radius', 0),
            stat_entry.get('solidity', 1.0),
            stat_entry.get('mrs', 0)
        ]
        # Trace Stats (8)
        trace_stats = [
            skew(f), np.std(f), np.max(f), np.mean(f),
            skew(fneu),
            np.corrcoef(f, fneu)[0, 1] if np.std(f)>0 and np.std(fneu)>0 else 0,
            skew(f_corr), np.std(f_corr)
        ]
        # Percentiles (7)
        q = np.quantile(f_corr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).tolist()
        # Dynamics (3)
        dynamics = [avg_asym, max_asym, max_width]
        # Spikes (4)
        max_spk = np.max(s)
        mean_spk_nz = np.mean(s[s > 0]) if np.sum(s > 0) > 0 else 0
        spk_rate = np.count_nonzero(s > 0.01 * max_spk) / len(s) if max_spk > 0 else 0
        skew_spk = skew(s)
        spikes = [max_spk, mean_spk_nz, spk_rate, skew_spk]
        
        return np.array(spatial + trace_stats + q + dynamics + spikes)

    elif num_features == 31:
        # Spatial (8)
        spatial = [
            stat_entry.get('skew', 0),
            stat_entry.get('npix', 0),
            stat_entry.get('aspect_ratio', 1.0),
            stat_entry.get('radius', 0),
            stat_entry.get('solidity', 1.0),
            stat_entry.get('mrs', 0),
            roi_idx,
            roi_idx / n_rois if n_rois > 0 else 0.0
        ]
        # Trace Stats (8)
        trace_stats = [
            skew(f), np.std(f), np.max(f), np.mean(f),
            skew(fneu),
            np.corrcoef(f, fneu)[0, 1] if np.std(f)>0 and np.std(fneu)>0 else 0,
            skew(f_corr), np.std(f_corr)
        ]
        # Percentiles (7)
        q = np.quantile(f_corr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).tolist()
        # Dynamics (3)
        dynamics = [avg_asym, max_asym, max_width]
        # Spikes (3)
        max_spk = np.max(s)
        mean_spk_nz = np.mean(s[s > 0]) if np.sum(s > 0) > 0 else 0
        spk_rate = np.count_nonzero(s > 0.01 * max_spk) / len(s) if max_spk > 0 else 0
        spikes = [max_spk, mean_spk_nz, spk_rate]
        # Trace Amplitudes (2)
        range_fcorr = np.max(f_corr) - np.min(f_corr)
        range_f = np.max(f) - np.min(f)
        amplitudes = [range_fcorr, range_f]
        
        return np.array(spatial + trace_stats + q + dynamics + spikes + amplitudes)
    
    else:
        raise ValueError(f"Unsupported number of features: {num_features}")

def extract_features_vectorized(F, Fneu, spks, stat, feature_names, custom_features=None):
    n_cells = F.shape[0]
    print(f"Extracting {len(feature_names)} features dynamically using vectorized operations for {n_cells} ROIs...")
    chunk_size = 500
    stds = []
    for i in range(0, n_cells, chunk_size):
        f_c = F[i:i+chunk_size] - 0.7 * Fneu[i:i+chunk_size]
        stds.append(np.std(f_c, axis=1))
    std_fcorr_all_temp = np.concatenate(stds) if stds else np.array([])
    session_scale = np.median(std_fcorr_all_temp) if len(std_fcorr_all_temp) > 0 else 1.0
    if np.isnan(session_scale) or session_scale <= 0:
        session_scale = 1.0

    if n_cells <= chunk_size:
        return _extract_features_vectorized_chunk(F, Fneu, spks, stat, feature_names, custom_features, session_scale)

    X_chunks = []
    for i in range(0, n_cells, chunk_size):
        F_c = F[i:i+chunk_size]
        Fneu_c = Fneu[i:i+chunk_size]
        spks_c = spks[i:i+chunk_size] if spks is not None else None
        stat_c = stat[i:i+chunk_size]
        X_c = _extract_features_vectorized_chunk(F_c, Fneu_c, spks_c, stat_c, feature_names, custom_features, session_scale)
        X_chunks.append(X_c)
        import gc; gc.collect()
    return np.vstack(X_chunks)

def _extract_features_vectorized_chunk(F, Fneu, spks, stat, feature_names, custom_features=None, session_scale=1.0):
    n_cells = F.shape[0]
    feature_names = list(feature_names)

    # 2. Precompute trace signals
    F_corr = F - 0.7 * Fneu
    
    # Session scale is passed in as a parameter
        
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

    # 12. Precompute Spikes
    max_spk_arr = None
    mean_spk_nz_arr = None
    spk_rate_arr = None
    skew_spk_arr = None
    if spks is not None:
        if 'max_spk' in feature_names:
            max_spk_arr = np.max(spks, axis=1)
        if 'mean_spk_nz' in feature_names:
            mean_spk_nz_arr = np.zeros(n_cells)
            for i in range(n_cells):
                s = spks[i]
                mask = s > 0
                mean_spk_nz_arr[i] = np.mean(s[mask]) if np.sum(mask) > 0 else 0.0
        if 'spk_rate' in feature_names:
            spk_rate_arr = np.zeros(n_cells)
            max_spks = np.max(spks, axis=1)
            for i in range(n_cells):
                s = spks[i]
                max_s = max_spks[i]
                spk_rate_arr[i] = np.count_nonzero(s > 0.01 * max_s) / len(s) if max_s > 0 else 0.0
        if 'skew_spk' in feature_names:
            skew_spk_arr = skew(spks, axis=1)

    # 13. Build output feature matrix X
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
            
        # Spikes
        elif name == 'max_spk':
            X[:, col_idx] = max_spk_arr if max_spk_arr is not None else 0.0
        elif name == 'mean_spk_nz':
            X[:, col_idx] = mean_spk_nz_arr if mean_spk_nz_arr is not None else 0.0
        elif name == 'spk_rate':
            X[:, col_idx] = spk_rate_arr if spk_rate_arr is not None else 0.0
        elif name == 'skew_spk':
            X[:, col_idx] = skew_spk_arr if skew_spk_arr is not None else 0.0
            
        # New Biological
        elif name == 'snr':
            X[:, col_idx] = snr_arr
        elif name == 'activity_ratio':
            X[:, col_idx] = activity_ratio_arr
        elif name == 'peak_density':
            X[:, col_idx] = peak_density_arr
            
        # Explicitly Normalized Features for rich sets
        elif name == 'std_f_norm':
            X[:, col_idx] = np.std(F, axis=1) / std_diff_f
        elif name == 'std_fcorr_norm':
            X[:, col_idx] = np.std(F_corr, axis=1) / std_diff_fcorr
        elif name == 'range_f_norm':
            X[:, col_idx] = (np.max(F, axis=1) - np.min(F, axis=1)) / std_diff_f
        elif name == 'range_fcorr_norm':
            X[:, col_idx] = (np.max(F_corr, axis=1) - np.min(F_corr, axis=1)) / std_diff_fcorr
            
        elif name in FEATURE_REGISTRY:
            cache = {
                'stat': stat,
                'F': F,
                'Fneu': Fneu,
                '_fcorr': F_corr,
                'npix': npix_vals,
                'solidity': solidity_vals,
                'mrs': mrs_vals,
                'compact': compact_vals,
                'aspect_ratio': aspect_ratio_vals,
                'radius': radius_vals,
                'number_of_bright_pixels': bright_pix,
                'bright_pixels_ratio': np.where(npix_vals > 0, bright_pix / npix_vals, 0.0),
                'avg_asym': avg_asym_arr,
                'max_asym': max_asym_arr,
                'max_width': max_width_arr,
            }
            X[:, col_idx] = FEATURE_REGISTRY[name](cache)
            
        else:
            print(f"Warning: Unknown feature name '{name}'. Defaulting to 0.0.")
            print(f"DEBUG: FEATURE_REGISTRY keys: {list(FEATURE_REGISTRY.keys())}")
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

# ==========================================
# 3. INTERACTIVE SERVER BACKEND
# ==========================================

def _model_option_value(model_path):
    """The model dropdown lists paths relative to the repo; report the active model the same way."""
    if not model_path:
        return None
    try:
        return str(Path(model_path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(model_path)


def _json_safe(obj):
    """Replace NaN/Inf (invalid in JSON, they break the browser's parser) with None."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class SessionState:
    def __init__(self):
        self.session_path = None
        self.model_path = None
        self.scaler_path = None
        self.model = None
        self.scaler = None
        self.num_features = 29
        self.feature_names = FEATURE_NAMES_25
        self.feature_descs = {}
        self.lock = threading.Lock()
        
        # Extracted data cache
        self.F = None
        self.Fneu = None
        self.stat = None
        self.spks = None
        self.y_true = None
        self.X_extracted = None
        self.y_probs = None
        self.y_preds = None
        self.iscell_meta = None
        
        # Progress of the current Apply Settings (polled by the dashboard's progress bar)
        self.apply_progress = {'percent': 0, 'status': 'Idle', 'active': False, 'started': None}

        # Progress tracking for CV evaluation
        self.cv_progress = {
            'percent': 0,
            'status': 'Idle',
            'active': False
        }
        
        # Dataset-wide averages for reference
        self.ref_means = None
        self.ref_cells_means = None
        self.ref_noncells_means = None
        
    def scan_models(self):
        dir_path = BASE_DIR
        models = [str(f.relative_to(dir_path)) for f in dir_path.rglob("*.pkl") if "scaler" not in f.name and not any("venv" in p for p in f.parts)]
        scalers = [str(f.relative_to(dir_path)) for f in dir_path.rglob("*scaler*.pkl") if not any("venv" in p for p in f.parts)]
        
        model_details = []
        for m_str in sorted(models):
            p = dir_path / m_str
            json_p = p.with_suffix('.json')
            meta = {}
            if json_p.exists():
                try:
                    with open(json_p, 'r') as f_json:
                        meta = json.load(f_json)
                except Exception:
                    pass
            
            name = meta.get('name') or p.stem.replace('_', ' ').title()
            desc = meta.get('description') or 'Custom trained cell classification model.'
            active_feats = meta.get('active_features') or meta.get('features') or []
            
            model_details.append({
                'path': m_str,
                'filename': p.name,
                'name': name,
                'description': desc,
                'active_features': active_feats,
                'feature_count': len(active_feats),
                'custom_features': meta.get('custom_features', {}),
                'trained': meta.get('trained'),
                'training_data': meta.get('training_data'),
                'idea': meta.get('idea')
            })
            
        return sorted(models), sorted(scalers), model_details

    def get_active_model_info(self):
        meta = {}
        if self.model_path:
            json_p = self.model_path.with_suffix('.json')
            if json_p.exists():
                try:
                    with open(json_p, 'r') as f:
                        meta = json.load(f)
                except Exception:
                    pass
        active_feats = meta.get('active_features') or (self.feature_names if hasattr(self, 'feature_names') and self.feature_names else [])
        return {
            'path': str(self.model_path) if self.model_path else '',
            'filename': self.model_path.name if self.model_path else '',
            'name': meta.get('name') or (self.model_path.stem.replace('_', ' ').title() if self.model_path else 'No Model Loaded'),
            'description': meta.get('description') or 'Custom trained model.',
            'active_features': active_feats,
            'feature_count': len(active_feats),
            'custom_features': meta.get('custom_features', {}),
            'threshold': meta.get('threshold'),  # tuned decision threshold, if the model stores one
            'trained': meta.get('trained'),
            'training_data': meta.get('training_data'),
            'idea': meta.get('idea')
        }

    def load_model(self, model_name, scaler_name=None):
        dir_path = BASE_DIR
        
        m_path = Path(model_name)
        if m_path.is_absolute() and m_path.exists():
            self.model_path = m_path
        elif (dir_path / model_name).exists():
            self.model_path = dir_path / model_name
        elif m_path.exists():
            self.model_path = m_path.resolve()
        else:
            self.model_path = dir_path / model_name
            
        self.model = joblib.load(self.model_path)
        
        # Detect feature count
        if hasattr(self.model, 'n_features_in_'):
            self.num_features = self.model.n_features_in_
        elif hasattr(self.model, 'n_features_'):
            self.num_features = self.model.n_features_
        elif hasattr(self.model, 'num_features'):
            self.num_features = self.model.num_features
        elif "CatBoostClassifier" in str(type(self.model)):
            try:
                self.num_features = len(self.model.feature_names_)
            except:
                self.num_features = 29
        else:
            self.num_features = 29
            
        # Try to infer feature names list
        feature_names = None
        if hasattr(self.model, 'feature_name_'):
            feature_names = list(self.model.feature_name_)
        elif hasattr(self.model, 'feature_names'):
            feature_names = list(self.model.feature_names)
            
        def is_default_names(names):
            if not names:
                return True
            first = str(names[0]).lower()
            return first.startswith('column_') or first.startswith('feature_') or first.isdigit()
            
        if feature_names and is_default_names(feature_names):
            feature_names = None
            
        # Initialize custom features
        self.custom_features = {}
        
        # Try loading sibling JSON metadata first (precise matching)
        meta_path = self.model_path.with_suffix('.json')
        meta_loaded = False
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                if isinstance(meta_data, dict):
                    feats = meta_data.get('active_features') or meta_data.get('features')
                    if feats:
                        feature_names = feats
                        self.num_features = len(feature_names)
                    self.custom_features = meta_data.get('custom_features', {})
                    meta_loaded = True
                    print(f"Loaded feature names and custom formulas from metadata file: {meta_path.name}")
            except Exception as e:
                print(f"Warning: Failed to load sibling metadata JSON: {e}")

        if not meta_loaded and not feature_names:
            # Look in model directory for any JSON file describing features
            model_dir = self.model_path.parent
            json_files = list(model_dir.glob("*.json"))
            for jf in json_files:
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            feats = data.get('active_features') or data.get('features')
                            if feats and len(feats) == self.num_features:
                                feature_names = feats
                                self.custom_features = data.get('custom_features', {})
                                meta_loaded = True
                                print(f"Loaded feature names from metadata file: {jf.name}")
                                break
                except:
                    pass

        # Register custom features in FEATURE_REGISTRY so they can be evaluated
        if self.custom_features:
            from fe_engine.fe_definitions import safe_eval_formula
            import fe_engine.fe_definitions
            for name, expr in self.custom_features.items():
                fe_engine.fe_definitions.FEATURE_REGISTRY[name] = lambda cache, e=expr: safe_eval_formula(e, cache)

        if not feature_names:
            if self.num_features == 25:
                feature_names = FEATURE_NAMES_25
            elif self.num_features == 27:
                feature_names = FEATURE_NAMES_27
            elif self.num_features == 28:
                if "3bin" in str(self.model_path) or "bin" in str(self.model_path):
                    feature_names = FEATURE_NAMES_28_bin
                else:
                    feature_names = FEATURE_NAMES_28
            elif self.num_features == 24:
                feature_names = FEATURE_NAMES_24_legacy
            elif self.num_features == 26:
                feature_names = FEATURE_NAMES_26_legacy
            elif self.num_features == 31:
                feature_names = FEATURE_NAMES_31
            elif self.num_features == 29:
                feature_names = FEATURE_NAMES_29
            else:
                feature_names = [f"feature_{i}" for i in range(self.num_features)]

        self.feature_names = feature_names
        self.feature_descs = {name: MASTER_FEATURE_DESCS.get(name, 'Custom model feature.') for name in self.feature_names}
        print(f"Loaded model {self.model_path}. Expects {self.num_features} features.")
        
        # Determine scaler
        if scaler_name:
            s_path = Path(scaler_name)
            if s_path.is_absolute() and s_path.exists():
                self.scaler_path = s_path
            elif (dir_path / scaler_name).exists():
                self.scaler_path = dir_path / scaler_name
            elif s_path.exists():
                self.scaler_path = s_path.resolve()
            else:
                self.scaler_path = dir_path / scaler_name
                
            try:
                self.scaler = joblib.load(self.scaler_path)
                print(f"Loaded selected scaler: {self.scaler_path}")
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.num_features:
                    print(f"Warning: Selected scaler expects {self.scaler.n_features_in_} features, but model expects {self.num_features}. Disabling scaling.")
                    self.scaler = None
                    self.scaler_path = None
            except Exception as e:
                print(f"Error loading scaler {scaler_name}: {e}")
                self.scaler = None
                self.scaler_path = None
        else:
            # Auto-detect matching scaler
            scaler_prefix = self.model_path.name.replace("_mlp.pkl", "").replace("_lgb.pkl", "").replace("_best", "").replace("_standard", "")
            scaler_candidates = [
                self.model_path.parent / f"{scaler_prefix}_scaler.pkl",
                dir_path / f"{scaler_prefix}_scaler.pkl",
                dir_path / "suite2p_mlp_scaler.pkl",
                dir_path / "suite2p_scaler.pkl"
            ]
            self.scaler = None
            self.scaler_path = None
            
            # Scaler only needed for non-tree models (MLPClassifier)
            if "MLPClassifier" in str(type(self.model)):
                for c in scaler_candidates:
                    if c.exists():
                        try:
                            temp_scaler = joblib.load(c)
                            if hasattr(temp_scaler, 'n_features_in_') and temp_scaler.n_features_in_ == self.num_features:
                                self.scaler = temp_scaler
                                self.scaler_path = c
                                print(f"Auto-loaded matching scaler: {c.name}")
                                break
                        except Exception as e:
                            print(f"Error loading scaler candidate {c.name}: {e}")
                            
        self.load_dataset_averages()

    def load_dataset_averages(self):
        dir_path = BASE_DIR
        if self.num_features in (24, 27, 28):
            dataset_file = dir_path / "X_all_stav.npy"
            labels_file = dir_path / "y_all_stav.npy"
        elif self.num_features == 29:
            dataset_file = dir_path / "X_dataset.npy"
            labels_file = dir_path / "y_dataset.npy"
        else:
            dataset_file = dir_path / "X_all_stav.npy"
            labels_file = dir_path / "y_all_stav.npy"
            
        if dataset_file.exists() and labels_file.exists():
            try:
                X_ref = np.load(dataset_file)
                y_ref = np.load(labels_file)
                if X_ref.shape[1] == self.num_features:
                    self.ref_means = np.mean(X_ref, axis=0)
                    self.ref_cells_means = np.mean(X_ref[y_ref == 1], axis=0)
                    self.ref_noncells_means = np.mean(X_ref[y_ref == 0], axis=0)
                    print(f"Loaded dataset averages from {dataset_file.name}")
                    return
            except Exception as e:
                print(f"Error loading reference dataset averages: {e}")
                
        self.ref_means = None
        self.ref_cells_means = None
        self.ref_noncells_means = None

    def load_session(self, session_path_str):
        # --- Explicitly free old session data to avoid OOM when switching sessions ---
        self.F = None
        self.Fneu = None
        self.stat = None
        self.spks = None
        self.X_extracted = None
        self.y_probs = None
        self.y_preds = None
        self.y_true = None
        self.iscell_meta = None
        gc.collect()
        # ---------------------------------------------------------------------------

        self.session_path = Path(session_path_str)
        print(f"Loading session: {self.session_path}")
        
        self.F = np.load(self.session_path / 'F.npy', mmap_mode='r')
        self.Fneu = np.load(self.session_path / 'Fneu.npy', mmap_mode='r')
        self.stat = np.load(self.session_path / 'stat.npy', allow_pickle=True)
        
        spks_path = self.session_path / 'spks.npy'
        self.spks = np.load(spks_path, mmap_mode='r') if spks_path.exists() else None
        
        iscell_path = None
        for l in ['iscell_final.npy', 'iscell_manual.npy', 'iscell_backup_before_AI.npy', 'iscell.npy']:
            if (self.session_path / l).exists():
                iscell_path = self.session_path / l
                break
                    
        if iscell_path is not None:
            iscell = np.load(iscell_path)
            self.y_true = iscell[:, 0].astype(int)
            self.iscell_meta = iscell
            self.gt_path = str(iscell_path)
            self.gt_file = iscell_path.name
        else:
            self.y_true = np.zeros(len(self.F))
            self.iscell_meta = None
            self.gt_path = "None"
            self.gt_file = "None"
            print("Warning: No curated ground truth file (iscell_final.npy or iscell_manual.npy) found. Initializing with zeros.")
            
        self.X_extracted = None
        self.y_probs = None
        self.y_preds = None

    def process_all_cells(self):
        if self.X_extracted is not None:
            return
            
        with self.lock:
            if self.X_extracted is not None:
                return
                
            if self.session_path and self.model_path:
                try:
                    # Let run_ai_pipeline handle partial/full cache loading and fallback internally
                    results = run_ai_pipeline(
                        session_path=self.session_path,
                        model_spec=self.model_path,
                        scaler_spec=self.scaler_path,
                        save_iscell=False,
                        custom_features=self.custom_features
                    )
                    self._store_pipeline_results(results)
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    import traceback; traceback.print_exc()
                    self.X_extracted = None
                    # Surface the real reason (e.g. ops.npy not matching stat.npy) instead of a
                    # follow-on error from predicting on missing features
                    raise RuntimeError(f"Could not compute this model's features for the session: {e}") from e
                # 3. Predict using model (if not already predicted by run_ai_pipeline)
                if getattr(self, 'y_probs', None) is None:
                    X_proc = self.X_extracted
                    if self.scaler is not None:
                        X_proc = self.scaler.transform(X_proc)
                    self.y_probs = self.model.predict_proba(X_proc)[:, 1]
                    self.y_preds = (self.y_probs >= 0.5).astype(int)
            
            if (self.ref_means is None or len(self.ref_means) != self.num_features or self.ref_cells_means is None or self.ref_noncells_means is None) and self.X_extracted is not None:
                self.ref_means = np.mean(self.X_extracted, axis=0)
                has_gt = hasattr(self, 'y_true') and self.y_true is not None
                self.ref_cells_means = np.mean(self.X_extracted[self.y_true == 1], axis=0) if has_gt and np.sum(self.y_true == 1) > 0 else self.ref_means
                self.ref_noncells_means = np.mean(self.X_extracted[self.y_true == 0], axis=0) if has_gt and np.sum(self.y_true == 0) > 0 else self.ref_means
                print("Calculated reference averages from session.")

    def _store_pipeline_results(self, results):
        self.X_extracted = results["X_extracted"]
        self.y_probs = results["probs"]
        self.y_preds = results["is_cell"]
        # Image features can be NaN for a few ROIs (e.g. a saturated, constant patch).
        # Probabilities above were computed with the NaNs (LightGBM handles them); the
        # copy kept for explanations, statistics and JSON uses the session median instead.
        if self.X_extracted is not None and np.isnan(self.X_extracted).any():
            X = np.array(self.X_extracted, dtype=np.float64)
            med = np.nan_to_num(np.nanmedian(X, axis=0))
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(med, np.where(nan_mask)[1])
            print(f"Filled {int(nan_mask.sum())} NaN feature values with session medians for display.")
            self.X_extracted = X

    def save_iscell_to_data_dir(self, progress=None):
        if self.session_path is None or self.model_path is None:
            return False
            
        try:
            results = run_ai_pipeline(
                session_path=self.session_path,
                model_spec=self.model_path,
                scaler_spec=self.scaler_path,
                save_iscell=True,
                progress=progress
            )
            self._store_pipeline_results(results)
            print(f"Saved AI model predictions ({int(np.sum(self.y_preds))} cells) to iscell.npy in {self.session_path.name}")
            return True
        except Exception as e:
            print(f"Error saving AI model predictions to data directory: {e}")
            return False
        
    def get_batch_explanations(self, indices):
        if self.X_extracted is None:
            self.process_all_cells()
            
        if len(indices) == 0:
            return np.zeros((0, self.num_features))
            
        X_batch = self.X_extracted[indices]
        
        model_type_str = str(type(self.model))
        has_shap = False
        shap_values = None
        
        if "LGBMClassifier" in model_type_str:
            try:
                contribs = self.model.predict(X_batch, pred_contrib=True)
                shaps = contribs[:, :-1]
                base_value = contribs[:, -1]
                
                margin = base_value + np.sum(shaps, axis=1)
                prob_full = 1.0 / (1.0 + np.exp(-margin))
                
                margin_without = margin[:, np.newaxis] - shaps
                prob_without = 1.0 / (1.0 + np.exp(-margin_without))
                
                shap_values = prob_full[:, np.newaxis] - prob_without
                has_shap = True
            except Exception as e:
                print(f"Error computing batch LGBM TreeSHAP: {e}")
                
        elif "XGBClassifier" in model_type_str:
            try:
                import xgboost as xgb
                booster = self.model.get_booster()
                dmat = xgb.DMatrix(X_batch)
                contribs = booster.predict(dmat, pred_contribs=True)
                shaps = contribs[:, :-1]
                base_value = contribs[:, -1]
                
                margin = base_value + np.sum(shaps, axis=1)
                prob_full = 1.0 / (1.0 + np.exp(-margin))
                
                margin_without = margin[:, np.newaxis] - shaps
                prob_without = 1.0 / (1.0 + np.exp(-margin_without))
                
                shap_values = prob_full[:, np.newaxis] - prob_without
                has_shap = True
            except Exception as e:
                print(f"Error computing batch XGBoost TreeSHAP: {e}")
                
        elif "CatBoostClassifier" in model_type_str:
            try:
                import catboost
                pool = catboost.Pool(X_batch)
                contribs = self.model.get_feature_importance(data=pool, type='ShapValues')
                shaps = contribs[:, :-1]
                base_value = contribs[:, -1]
                
                margin = base_value + np.sum(shaps, axis=1)
                prob_full = 1.0 / (1.0 + np.exp(-margin))
                
                margin_without = margin[:, np.newaxis] - shaps
                prob_without = 1.0 / (1.0 + np.exp(-margin_without))
                
                shap_values = prob_full[:, np.newaxis] - prob_without
                has_shap = True
            except Exception as e:
                print(f"Error computing batch CatBoost TreeSHAP: {e}")
                
        if has_shap:
            return shap_values
        else:
            raise ValueError("Failure-mode clustering is only supported for tree-based models (LightGBM, XGBoost, CatBoost) which support TreeSHAP.")
            
    def get_cell_explanation(self, cell_idx):
        if self.X_extracted is None or self.y_probs is None or self.y_preds is None:
            self.process_all_cells()

        if self.X_extracted is None:
            raise ValueError("Failed to extract features for dataset.")

        has_gt = hasattr(self, 'y_true') and self.y_true is not None

        if (self.ref_means is None or len(self.ref_means) != self.num_features or self.ref_cells_means is None or self.ref_noncells_means is None):
            self.ref_means = np.mean(self.X_extracted, axis=0)
            self.ref_cells_means = np.mean(self.X_extracted[self.y_true == 1], axis=0) if has_gt and np.sum(self.y_true == 1) > 0 else self.ref_means
            self.ref_noncells_means = np.mean(self.X_extracted[self.y_true == 0], axis=0) if has_gt and np.sum(self.y_true == 0) > 0 else self.ref_means

        ref_m = self.ref_means if self.ref_means is not None else np.mean(self.X_extracted, axis=0)
        ref_c_m = self.ref_cells_means if self.ref_cells_means is not None else ref_m
        ref_nc_m = self.ref_noncells_means if self.ref_noncells_means is not None else ref_m

        x_raw = self.X_extracted[cell_idx]
        prob = float(self.y_probs[cell_idx]) if self.y_probs is not None else 0.0
        
        # 1. Feature ablation explanation (always compute)
        ablation_attributions = []
        
        x_proc = x_raw.reshape(1, -1)
        if self.scaler is not None:
            x_proc = self.scaler.transform(x_proc)
            
        base_prob = float(self.model.predict_proba(x_proc)[0, 1])
        
        for i in range(self.num_features):
            x_ablated = x_raw.copy()
            x_ablated[i] = ref_m[i]
            
            x_ablated_proc = x_ablated.reshape(1, -1)
            if self.scaler is not None:
                x_ablated_proc = self.scaler.transform(x_ablated_proc)
                
            prob_ablated = float(self.model.predict_proba(x_ablated_proc)[0, 1])
            ablation_attributions.append(base_prob - prob_ablated)
            
        # 2. TreeSHAP explanation (if tree model)
        shap_values = None
        has_shap = False
        
        model_type_str = str(type(self.model))
        
        if "LGBMClassifier" in model_type_str:
            try:
                contribs = self.model.predict(x_raw.reshape(1, -1), pred_contrib=True)[0]
                shaps = contribs[:-1]
                base_value = contribs[-1]
                
                margin = base_value + np.sum(shaps)
                prob_full = 1.0 / (1.0 + np.exp(-margin))
                
                shap_prob_contribs = []
                for s_val in shaps:
                    margin_without = margin - s_val
                    prob_without = 1.0 / (1.0 + np.exp(-margin_without))
                    shap_prob_contribs.append(prob_full - prob_without)
                    
                shap_values = shap_prob_contribs
                has_shap = True
            except Exception as e:
                print(f"Error computing LGBM TreeSHAP: {e}")
                
        elif "XGBClassifier" in model_type_str:
            try:
                import xgboost as xgb
                booster = self.model.get_booster()
                dmat = xgb.DMatrix(x_raw.reshape(1, -1))
                contribs = booster.predict(dmat, pred_contribs=True)[0]
                shaps = contribs[:-1]
                base_value = contribs[-1]
                
                margin = base_value + np.sum(shaps)
                prob_full = 1.0 / (1.0 + np.exp(-margin))
                
                shap_prob_contribs = []
                for s_val in shaps:
                    margin_without = margin - s_val
                    prob_without = 1.0 / (1.0 + np.exp(-margin_without))
                    shap_prob_contribs.append(prob_full - prob_without)
                    
                shap_values = shap_prob_contribs
                has_shap = True
            except Exception as e:
                print(f"Error computing XGBoost TreeSHAP: {e}")
                
        elif "CatBoostClassifier" in model_type_str:
            try:
                import catboost
                pool = catboost.Pool(x_raw.reshape(1, -1))
                contribs = self.model.get_feature_importance(data=pool, type='ShapValues')[0]
                shaps = contribs[:-1]
                base_value = contribs[-1]
                
                margin = base_value + np.sum(shaps)
                prob_full = 1.0 / (1.0 + np.exp(-margin))
                
                shap_prob_contribs = []
                for s_val in shaps:
                    margin_without = margin - s_val
                    prob_without = 1.0 / (1.0 + np.exp(-margin_without))
                    shap_prob_contribs.append(prob_full - prob_without)
                    
                shap_values = shap_prob_contribs
                has_shap = True
            except Exception as e:
                print(f"Error computing CatBoost TreeSHAP: {e}")
                
        # 3. Assemble attributions
        attributions = []
        for i in range(self.num_features):
            ablation_val = ablation_attributions[i]
            shap_val = shap_values[i] if (has_shap and shap_values is not None) else None
            
            attributions.append({
                'name': self.feature_names[i],
                'desc': self.feature_descs.get(self.feature_names[i], 'Custom model feature.'),
                'value': float(x_raw[i]),
                'mean_dataset': float(ref_m[i]),
                'mean_cells': float(ref_c_m[i]),
                'mean_noncells': float(ref_nc_m[i]),
                'ablation_attribution': float(ablation_val),
                'shap_attribution': float(shap_val) if shap_val is not None else None,
                'attribution': float(shap_val) if shap_val is not None else float(ablation_val)
            })
            
        attributions.sort(key=lambda x: abs(x['shap_attribution'] if x['shap_attribution'] is not None else x['ablation_attribution']), reverse=True)
        
        f_raw = self.F[cell_idx] if self.F is not None else np.zeros(100)
        fneu_raw = self.Fneu[cell_idx] if self.Fneu is not None else np.zeros_like(f_raw)
        fcorr_raw = f_raw - 0.7 * fneu_raw
        spks_raw = self.spks[cell_idx] if self.spks is not None else np.zeros_like(f_raw)
        
        n_frames = len(f_raw)
        step = max(1, n_frames // 2000)
        
        dec_indices = np.arange(0, n_frames, step)
        f_dec = f_raw[dec_indices].tolist()
        fneu_dec = fneu_raw[dec_indices].tolist()
        fcorr_dec = fcorr_raw[dec_indices].tolist()
        spks_dec = spks_raw[dec_indices].tolist()
        time_dec = dec_indices.tolist()
        
        s_entry = self.stat[cell_idx] if self.stat is not None else {}
        xpix = s_entry.get('xpix', [])
        ypix = s_entry.get('ypix', [])
        
        if len(xpix) > 0 and len(ypix) > 0:
            x_min, x_max = np.min(xpix), np.max(xpix)
            y_min, y_max = np.min(ypix), np.max(ypix)
            margin = 5
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            width = max(x_max - x_min, y_max - y_min) // 2 + margin
            
            roi_points = [[int(x), int(y)] for x, y in zip(xpix, ypix)]
            roi_bbox = {
                'xmin': int(cx - width), 'xmax': int(cx + width),
                'ymin': int(cy - width), 'ymax': int(cy + width)
            }
        else:
            roi_points = []
            roi_bbox = {'xmin': 0, 'xmax': 100, 'ymin': 0, 'ymax': 100}
            
        gt_label = int(self.y_true[cell_idx]) if (has_gt and cell_idx < len(self.y_true)) else -1
        pred_tag = int(self.y_preds[cell_idx]) if (self.y_preds is not None and cell_idx < len(self.y_preds)) else int(prob >= 0.5)

        return {
            'cell_idx': cell_idx,
            'probability': float(prob),
            'label': gt_label,
            'is_cell_tag': pred_tag,
            'has_shap': has_shap,
            'attributions': attributions,
            'trace': {
                'time': time_dec,
                'f': f_dec,
                'fneu': fneu_dec,
                'fcorr': fcorr_dec,
                'spks': spks_dec
            },
            'roi': {
                'points': roi_points,
                'bbox': roi_bbox
            }
        }


# ==========================================
# 4. HTTP REQUEST HANDLER
# ==========================================

def generate_cluster_label(top_features, category):
    if not top_features:
        return "Unknown Profile"
        
    primary_feature = top_features[0][0]
    secondary_feature = top_features[1][0] if len(top_features) > 1 else ""
    
    spatial_features = {'mrs', 'solidity', 'npix', 'aspect_ratio', 'radius', 'compact', 'skew_spatial', 'bright_pixels_ratio'}
    trace_features = {'skew_f', 'std_f', 'max_to_mean_f', 'cv_f', 'skew_fcorr', 'std_fcorr', 'q90', 'q95', 'q99', 'range_fcorr', 'snr', 'activity_ratio', 'peak_density'}
    neuropil_features = {'corr_f_fneu', 'skew_fneu'}
    index_features = {'roi_idx_norm', 'roi_idx_norm_3bin', 'roi_idx_raw'}
    
    label_base = ""
    if primary_feature in neuropil_features or secondary_feature in neuropil_features:
        label_base = "Neuropil / Background Contamination"
    elif primary_feature in index_features:
        label_base = "Spatial Index Prior Bias"
    elif primary_feature in spatial_features and secondary_feature in spatial_features:
        label_base = "Pure Morphological Clone"
    elif primary_feature in trace_features and secondary_feature in trace_features:
        label_base = "Transient Activity Bias"
    elif primary_feature in spatial_features and secondary_feature in trace_features:
        label_base = "Mixed Morphology & Activity Profile"
    else:
        label_base = f"Driven by {primary_feature.replace('_', ' ').title()}"
        
    return label_base

state = SessionState()

class DashHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
        
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed_url.query)
        path = parsed_url.path
        if path.endswith('/') and len(path) > 1:
            path = path[:-1]
        
        if path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            return
            
        elif path == '/api/models':
            try:
                models, scalers, model_details = state.scan_models()
                default_sessions = [
                    "/mnt/other_ubunthu/mnt/data/1-ordered/Stav1",
                    "/mnt/other_ubunthu/mnt/data/4-ordered/Stav4",
                    "/mnt/other_ubunthu/mnt/data/6-ordered/Stav6",
                    "/mnt/other_ubunthu/mnt/data/7-ordered/Stav7",
                    "/mnt/other_ubunthu/mnt/data/8-ordered/Stav8",
                    "/mnt/other_ubunthu/mnt/data/9-ordered/Stav9",
                    "/mnt/other_ubunthu/mnt/data/10-ordered/Stav10"
                ]
                existing_sessions = [s for s in default_sessions if os.path.exists(s)]
                
                res = {
                    'models': models,
                    'scalers': scalers,
                    'model_details': model_details,
                    'suggested_sessions': existing_sessions,
                    'active_model': _model_option_value(state.model_path),
                    'active_scaler': str(state.scaler_path) if state.scaler_path else None,
                    'active_session': str(state.session_path) if state.session_path else None,
                    'num_features': state.num_features,
                    'active_model_info': state.get_active_model_info()
                }
                self.send_json(res)
            except Exception as e:
                self.send_error_json(str(e))
            return
            
        elif path == '/api/apply_progress':
            # Read without state.lock: /api/change_settings holds it while it runs
            self.send_json(dict(state.apply_progress, elapsed=(time.time() - state.apply_progress['started'])
                                if state.apply_progress['started'] else 0))
            return

        elif path == '/api/change_settings':
            def progress(percent, status):
                state.apply_progress.update(percent=percent, status=status)

            state.apply_progress = {'percent': 0, 'status': 'Waiting for the current job to finish...',
                                    'active': True, 'started': time.time()}
            try:
                with state.lock:
                    session_path = params.get('session_path', [None])[0]
                    model_name = params.get('model_name', [None])[0]
                    scaler_name = params.get('scaler_name', [None])[0]
                    
                    if model_name:
                        progress(5, 'Loading model...')
                        state.load_model(model_name, scaler_name)
                        
                    iscell_saved = False
                    if session_path:
                        progress(15, f'Loading session {Path(session_path).name}...')
                        state.load_session(session_path)
                        iscell_saved = state.save_iscell_to_data_dir(
                            progress=lambda f, status: progress(20 + 75 * f, status))
                    progress(95, 'Refreshing the dashboard...')
                        
                    self.send_json({
                        'status': 'success',
                        'active_model': str(state.model_path) if state.model_path else None,
                        'active_scaler': str(state.scaler_path) if state.scaler_path else None,
                        'active_session': str(state.session_path) if state.session_path else None,
                        'num_features': state.num_features,
                        'iscell_saved': iscell_saved,
                        'active_model_info': state.get_active_model_info()
                    })
            except Exception as e:
                self.send_error_json(str(e))
            finally:
                state.apply_progress.update(percent=100, status='Done', active=False)
            return
            
        elif path == '/api/session_info':
            try:
                if not state.session_path:
                    self.send_json({'loaded': False})
                    return
                    
                state.process_all_cells()
                
                y_true = state.y_true
                y_probs = state.y_probs
                y_preds = state.y_preds
                
                tp = np.where((y_preds == 1) & (y_true == 1))[0].tolist()
                fp = np.where((y_preds == 1) & (y_true == 0))[0].tolist()
                tn = np.where((y_preds == 0) & (y_true == 0))[0].tolist()
                fn = np.where((y_preds == 0) & (y_true == 1))[0].tolist()
                
                uncertain = np.where((y_probs > 0.15) & (y_probs < 0.85))[0].tolist()
                
                res = {
                    'loaded': True,
                    'session_name': state.session_path.name,
                    'session_path': str(state.session_path),
                    'total_rois': len(y_true),
                    'num_cells_gt': int(np.sum(y_true)),
                    'num_cells_pred': int(np.sum(y_preds)),
                    'gt_path': getattr(state, 'gt_path', 'None'),
                    'gt_file': getattr(state, 'gt_file', 'None'),
                    'categories': {
                        'true_positives': tp,
                        'false_positives': fp,
                        'true_negatives': tn,
                        'false_negatives': fn,
                        'uncertain': uncertain
                    },
                    'probabilities': y_probs.tolist(),
                    'ground_truth': y_true.tolist(),
                    'num_features': state.num_features
                }
                self.send_json(res)
            except Exception as e:
                self.send_error_json(str(e))
            return
            
        elif path == '/api/explain_cell':
            try:
                cell_idx = int(params.get('cell_idx', [0])[0])
                session_path = params.get('session_path', [None])[0]
                model_name = params.get('model_name', [None])[0]
                scaler_name = params.get('scaler_name', [None])[0]
                
                settings_changed = False
                if model_name and (not state.model_path or (state.model_path.name != model_name and str(state.model_path) != model_name) or (scaler_name and (not state.scaler_path or (state.scaler_path.name != scaler_name and str(state.scaler_path) != scaler_name)))):
                    state.load_model(model_name, scaler_name)
                    settings_changed = True
                    
                if session_path and (not state.session_path or str(state.session_path) != session_path):
                    state.load_session(session_path)
                    settings_changed = True
                    
                if settings_changed:
                    state.process_all_cells()
                    
                if not state.session_path:
                    raise ValueError("No session loaded.")
                if cell_idx < 0 or cell_idx >= len(state.F):
                    raise ValueError(f"Cell index {cell_idx} out of range [0, {len(state.F)-1}].")
                    
                explanation = state.get_cell_explanation(cell_idx)
                self.send_json(explanation)
            except Exception as e:
                self.send_error_json(str(e))
            return
            
        elif path == '/api/category_analysis':
            try:
                threshold = float(params.get('threshold', [0.5])[0])
                state.process_all_cells()
                
                y_true = state.y_true
                y_probs = state.y_probs
                
                # Predict based on threshold
                y_preds = (y_probs >= threshold).astype(int)
                
                # Categories
                tp_indices = np.where((y_preds == 1) & (y_true == 1))[0]
                fp_indices = np.where((y_preds == 1) & (y_true == 0))[0]
                tn_indices = np.where((y_preds == 0) & (y_true == 0))[0]
                fn_indices = np.where((y_preds == 0) & (y_true == 1))[0]
                
                categories = {
                    'TP': tp_indices,
                    'FP': fp_indices,
                    'TN': tn_indices,
                    'FN': fn_indices
                }
                
                # Feature statistics
                X = state.X_extracted
                feature_names = state.feature_names
                
                overall_means = np.mean(X, axis=0)
                overall_stds = np.std(X, axis=0)
                # Avoid division by zero
                overall_stds[overall_stds == 0] = 1e-6
                
                analysis = {}
                for cat_name, idxs in categories.items():
                    if len(idxs) == 0:
                        analysis[cat_name] = []
                        continue
                    
                    cat_X = X[idxs]
                    cat_means = np.mean(cat_X, axis=0)
                    z_scores = (cat_means - overall_means) / overall_stds
                    
                    cat_features = []
                    for f_idx, f_name in enumerate(feature_names):
                        cat_features.append({
                            'feature': f_name,
                            'z_score': float(z_scores[f_idx]),
                            'cat_mean': float(cat_means[f_idx]),
                            'overall_mean': float(overall_means[f_idx]),
                            'overall_std': float(overall_stds[f_idx])
                        })
                    
                    # Sort by absolute z_score descending
                    cat_features.sort(key=lambda x: abs(x['z_score']), reverse=True)
                    analysis[cat_name] = cat_features
                    
                self.send_json(analysis)
            except Exception as e:
                self.send_error_json(str(e))
            return
            
        elif path == '/api/failure_modes':
            try:
                category = params.get('category', ['FP'])[0].upper()
                threshold = float(params.get('threshold', [0.5])[0])
                
                state.process_all_cells()
                y_true = state.y_true
                y_probs = state.y_probs
                y_preds = (y_probs >= threshold).astype(int)
                
                if category == 'TP':
                    indices = np.where((y_preds == 1) & (y_true == 1))[0]
                elif category == 'FP':
                    indices = np.where((y_preds == 1) & (y_true == 0))[0]
                elif category == 'FN':
                    indices = np.where((y_preds == 0) & (y_true == 1))[0]
                elif category == 'TN':
                    indices = np.where((y_preds == 0) & (y_true == 0))[0]
                else:
                    raise ValueError(f"Unknown category {category}")
                    
                if len(indices) == 0:
                    self.send_json({'category': category, 'clusters': []})
                    return
                    
                A = state.get_batch_explanations(indices)
                
                n_clusters = min(3, len(indices))
                
                clusters_data = []
                if n_clusters > 0:
                    from sklearn.cluster import KMeans
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    cluster_labels = kmeans.fit_predict(A)
                    
                    tp_indices = np.where((y_preds == 1) & (y_true == 1))[0]
                    if len(tp_indices) > 0:
                        tp_means = np.mean(state.X_extracted[tp_indices], axis=0)
                    else:
                        tp_means = np.mean(state.X_extracted, axis=0)
                        
                    for c in range(n_clusters):
                        c_mask = (cluster_labels == c)
                        c_indices = indices[c_mask]
                        c_attributions = A[c_mask]
                        c_raw_features = state.X_extracted[c_indices]
                        
                        mean_attributions = np.mean(c_attributions, axis=0)
                        mean_raw = np.mean(c_raw_features, axis=0)
                        
                        features_list = []
                        for f_idx, f_name in enumerate(state.feature_names):
                            features_list.append({
                                'name': f_name,
                                'desc': state.feature_descs.get(f_name, ''),
                                'mean_attribution': float(mean_attributions[f_idx]),
                                'mean_raw': float(mean_raw[f_idx]),
                                'mean_tp': float(tp_means[f_idx]),
                                'mean_dataset': float(state.ref_means[f_idx]) if state.ref_means is not None and len(state.ref_means) > f_idx else 0.0
                            })
                            
                        features_list.sort(key=lambda x: abs(x['mean_attribution']), reverse=True)
                        
                        top_features = [(f['name'], f['mean_attribution']) for f in features_list]
                        
                        cluster_label = generate_cluster_label(top_features, category)
                        
                        c_probs = y_probs[c_indices]
                        if category in ('FP', 'TP'):
                            worst_order = np.argsort(c_probs)[::-1][:5]
                        else:
                            worst_order = np.argsort(c_probs)[:5]
                            
                        worst_cells = []
                        for w_idx in worst_order:
                            cell_idx = int(c_indices[w_idx])
                            worst_cells.append({
                                'idx': cell_idx,
                                'prob': float(y_probs[cell_idx]),
                                'gt': int(y_true[cell_idx])
                            })
                            
                        clusters_data.append({
                            'cluster_id': c,
                            'label': cluster_label,
                            'size': int(np.sum(c_mask)),
                            'percentage': float(np.sum(c_mask) / len(indices) * 100),
                            'features': features_list,
                            'representative_cells': worst_cells
                        })
                        
                clusters_data.sort(key=lambda x: x['size'], reverse=True)
                
                self.send_json({
                    'category': category,
                    'total_count': len(indices),
                    'clusters': clusters_data
                })
            except Exception as e:
                self.send_error_json(str(e))
            return
            
        elif path == '/api/compare_cells':
            try:
                cell_a = int(params.get('cell_a', [0])[0])
                cell_b = int(params.get('cell_b', [0])[0])
                
                if not state.session_path:
                    raise ValueError("No session loaded.")
                n_cells = len(state.F)
                if cell_a < 0 or cell_a >= n_cells or cell_b < 0 or cell_b >= n_cells:
                    raise ValueError(f"Cell indices must be between 0 and {n_cells-1}")
                    
                # Centroids
                cent_a = state.stat[cell_a]['med'] # [y, x]
                cent_b = state.stat[cell_b]['med'] # [y, x]
                dist = float(np.sqrt((cent_a[0] - cent_b[0])**2 + (cent_a[1] - cent_b[1])**2))
                
                # Pearson Correlation
                trace_a = state.F[cell_a]
                trace_b = state.F[cell_b]
                if np.std(trace_a) > 0 and np.std(trace_b) > 0:
                    corr_f = float(np.corrcoef(trace_a, trace_b)[0, 1])
                else:
                    corr_f = 0.0
                    
                # Calculate corrected F correlation
                fneu_a = state.Fneu[cell_a]
                fneu_b = state.Fneu[cell_b]
                fcorr_a = trace_a - 0.7 * fneu_a
                fcorr_b = trace_b - 0.7 * fneu_b
                if np.std(fcorr_a) > 0 and np.std(fcorr_b) > 0:
                    corr_fcorr = float(np.corrcoef(fcorr_a, fcorr_b)[0, 1])
                else:
                    corr_fcorr = 0.0
                    
                # Overlap
                set_a = set(zip(state.stat[cell_a]['ypix'], state.stat[cell_a]['xpix']))
                set_b = set(zip(state.stat[cell_b]['ypix'], state.stat[cell_b]['xpix']))
                intersection = len(set_a.intersection(set_b))
                union = len(set_a.union(set_b))
                iou = float(intersection / union) if union > 0 else 0.0
                
                res = {
                    'cell_a': cell_a,
                    'cell_b': cell_b,
                    'centroid_a': [float(cent_a[0]), float(cent_a[1])],
                    'centroid_b': [float(cent_b[0]), float(cent_b[1])],
                    'distance_px': dist,
                    'corr_f': corr_f,
                    'corr_fcorr': corr_fcorr,
                    'intersection_pixels': intersection,
                    'iou': iou
                }
                self.send_json(res)
            except Exception as e:
                self.send_error_json(str(e))
            return
            
        elif path == '/playground' or path == '/playground.html':
            try:
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                pg_path = BASE_DIR / "playground.html"
                with open(pg_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                self.wfile.write(html_content.encode('utf-8'))
            except Exception as e:
                self.send_error_json(f"Error loading playground.html: {str(e)}")
            return

        elif path == '/api/playground/status':
            try:
                import fe_engine.fe_definitions
                importlib = __import__('importlib')
                importlib.reload(fe_engine.fe_definitions)
                from fe_engine.fe_definitions import ACTIVE_FEATURES, FEATURE_REGISTRY
                loaded_feats = getattr(state, 'feature_names', [])
                if not loaded_feats:
                    loaded_feats = []
                available_feats = list(FEATURE_REGISTRY.keys())
                
                # Fetch baseline/active model predictions for all cached sessions
                from fe_engine.fe_loop_runner import load_preprocessed_data
                workspace_dir = BASE_DIR
                cache_dir = workspace_dir / "preprocessed_cache"
                sessions = load_preprocessed_data(cache_dir)
                
                sessions_list = []
                y_prob_list = []
                y_true_list = []
                y_pred_list = []
                
                curr_idx = 0
                
                # If active session is loaded in state, add it first!
                if state.session_path:
                    state.process_all_cells()
                    n_rois = len(state.y_true)
                    sessions_list.append({
                        'name': f"{state.session_path.name} (Active Session)",
                        'start_idx': curr_idx,
                        'end_idx': curr_idx + n_rois,
                        'total_rois': n_rois
                    })
                    y_prob_list.extend(state.y_probs.tolist())
                    y_true_list.extend(state.y_true.tolist())
                    y_pred_list.extend(state.y_preds.tolist())
                    curr_idx += n_rois
                    
                # Predict using active model on cached sessions
                for session in sessions:
                    s_path_val = session.get('session_path')
                    if isinstance(s_path_val, np.ndarray):
                        s_path_val = s_path_val.item()
                    s_path = Path(s_path_val).resolve() if s_path_val else None
                    n_rois = len(session['y'])
                    
                    sessions_list.append({
                        'name': f"{s_path.name} (Cached)" if s_path else f"Session {len(sessions_list)+1} (Cached)",
                        'start_idx': curr_idx,
                        'end_idx': curr_idx + n_rois,
                        'total_rois': n_rois
                    })
                    
                    # Extract active model features
                    feats_to_use = loaded_feats if loaded_feats else ACTIVE_FEATURES
                    X_sess = np.column_stack([session[feat] for feat in feats_to_use])
                    
                    # Apply scaler if present
                    if state.scaler is not None:
                        X_sess = state.scaler.transform(X_sess)
                        
                    probs = state.model.predict_proba(X_sess)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    
                    y_prob_list.extend(probs.tolist())
                    y_true_list.extend(session['y'].tolist())
                    y_pred_list.extend(preds.tolist())
                    curr_idx += n_rois
                
                res = {
                    'available_features': sorted(available_feats),
                    'active_features': ACTIVE_FEATURES,
                    'loaded_model_features': loaded_feats,
                    'active_model': str(state.model_path) if state.model_path else 'None',
                    'active_session': str(state.session_path) if state.session_path else 'None',
                    'num_features': state.num_features,
                    'sessions': sessions_list,
                    'y_prob': y_prob_list,
                    'y_true': y_true_list,
                    'y_pred': y_pred_list
                }
                self.send_json(res)
            except Exception as e:
                self.send_error_json(str(e))
            return

        elif path == '/api/playground/progress':
            try:
                self.send_json(dict(state.cv_progress))
            except Exception as e:
                self.send_error_json(str(e))
            return

        elif path == '/api/playground/run':
            try:
                # Reset progress
                state.cv_progress = {
                    'percent': 5,
                    'status': 'Loading cached preprocessed sessions from disk...',
                    'active': True
                }

                feats_str = params.get('features', [''])[0]
                active_feats = [f.strip() for f in feats_str.split(',') if f.strip()]
                formulas = params.get('formulas', [])
                
                from fe_engine.fe_loop_runner import load_preprocessed_data, extract_features_dataset, run_cross_validation, analyze_errors_and_shap
                from fe_engine.fe_definitions import safe_eval_formula
                workspace_dir = Path("/home/tomer/Documents/suite2p-iscell-prediction")
                cache_dir = workspace_dir / "preprocessed_cache"
                
                sessions = load_preprocessed_data(cache_dir)
                
                # Update progress
                state.cv_progress['percent'] = 10
                state.cv_progress['status'] = 'Evaluating custom formulas on sessions...'

                import fe_engine.fe_definitions
                for formula in formulas:
                    formula = formula.strip()
                    if not formula or '=' not in formula:
                        continue
                    name, expr = formula.split('=', 1)
                    name = name.strip()
                    expr = expr.strip()
                    for session in sessions:
                        ns = {key: session[key] for key in session.keys() if isinstance(session[key], np.ndarray) and len(session[key].shape) == 1}
                        try:
                            session[name] = safe_eval_formula(expr, ns)
                        except Exception as eval_exc:
                            print(f"DEBUG: ns keys = {sorted(list(ns.keys()))}")
                            raise ValueError(f"Error evaluating custom formula '{name} = {expr}': {str(eval_exc)}")
                    
                    # Register custom feature temporarily in registry
                    fe_engine.fe_definitions.FEATURE_REGISTRY[name] = lambda cache, n=name: cache[n]
                
                # Update progress callback for session extraction
                def session_cb(curr, total, s_name):
                    state.cv_progress['percent'] = int(15 + (curr / total) * 35)
                    state.cv_progress['status'] = f"Extracting features ({curr}/{total}): {s_name}"
                
                X, y, groups = extract_features_dataset(sessions, active_feats, session_callback=session_cb)
                
                # Update progress callback for CV splits
                def cv_cb(fold, total_folds):
                    state.cv_progress['percent'] = int(55 + (fold / total_folds) * 30)
                    state.cv_progress['status'] = f"Running 5-Fold GroupKFold Cross-Validation (fold {fold}/{total_folds})..."

                cv_summary, y_true, y_pred, y_prob, shap_values = run_cross_validation(
                    X, y, groups, active_feats, progress_callback=cv_cb
                )
                
                state.cv_progress['percent'] = 90
                state.cv_progress['status'] = 'Analyzing errors and computing SHAP importances...'

                fp_culprits, fn_culprits, importance = analyze_errors_and_shap(X, y_true, y_pred, shap_values, active_feats)
                
                baseline = None
                baseline_path = workspace_dir / "fe_baseline.json"
                if baseline_path.exists():
                    try:
                        with open(baseline_path, 'r') as f:
                            baseline = json.load(f)
                    except:
                        pass
                
                # Train final model on 100% of cached dataset using active features
                n_pos = np.sum(y == 1)
                n_neg = np.sum(y == 0)
                scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
                
                import lightgbm as lgb
                final_model = lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=8,
                    num_leaves=63,
                    scale_pos_weight=scale_pos_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
                final_model.fit(X, y)
                
                y_prob_active = None
                y_pred_active = None
                y_true_active = None
                
                if state.session_path:
                    # Construct active session dict for dynamic feature extraction
                    active_sess = {
                        'y': state.y_true,
                        'session_name': state.session_path.name,
                        'session_path': str(state.session_path),
                        'F': state.F,
                        'Fneu': state.Fneu,
                        'spks': state.spks,
                        'stat': state.stat
                    }
                    
                    # Compute all standard features dynamically using the registry
                    from fe_engine.fe_definitions import FEATURE_REGISTRY
                    custom_names = [f.split('=', 1)[0].strip() for f in formulas if '=' in f]
                    for feat in list(FEATURE_REGISTRY.keys()):
                        if feat in custom_names:
                            continue
                        try:
                            active_sess[feat] = FEATURE_REGISTRY[feat](active_sess)
                        except Exception as feat_exc:
                            active_sess[feat] = np.zeros(len(state.y_true))
                                
                    # Evaluate custom formulas
                    for formula in formulas:
                        formula = formula.strip()
                        if not formula or '=' not in formula:
                            continue
                        name, expr = formula.split('=', 1)
                        name = name.strip()
                        expr = expr.strip()
                        ns = {key: active_sess[key] for key in active_sess.keys() if isinstance(active_sess[key], np.ndarray) and len(active_sess[key].shape) == 1}
                        try:
                            active_sess[name] = safe_eval_formula(expr, ns)
                        except Exception as eval_exc:
                            raise ValueError(f"Error evaluating custom formula '{name} = {expr}': {str(eval_exc)}")
                            
                    # Construct feature matrix X_active
                    X_active = np.column_stack([active_sess[feat] for feat in active_feats])
                    y_true_active = state.y_true
                    
                    y_prob_active = final_model.predict_proba(X_active)[:, 1]
                    y_pred_active = (y_prob_active >= 0.5).astype(int)
                
                combined_y_prob = []
                combined_y_true = []
                combined_y_pred = []
                sessions_list = []
                
                curr_idx = 0
                
                # Add active session first with predicted values from the new model
                if state.session_path and y_prob_active is not None:
                    n_rois = len(state.y_true)
                    sessions_list.append({
                        'name': f"{state.session_path.name} (Active Session - Predicted)",
                        'start_idx': curr_idx,
                        'end_idx': curr_idx + n_rois,
                        'total_rois': n_rois
                    })
                    combined_y_prob.extend(y_prob_active.tolist())
                    combined_y_true.extend(y_true_active.tolist())
                    combined_y_pred.extend(y_pred_active.tolist())
                    curr_idx += n_rois
                    
                # Add CV sessions
                cv_idx = 0
                for session in sessions:
                    s_path_val = session.get('session_path')
                    if isinstance(s_path_val, np.ndarray):
                        s_path_val = s_path_val.item()
                    s_path = Path(s_path_val).resolve() if s_path_val else None
                    n_rois = len(session['y'])
                    sessions_list.append({
                        'name': f"{s_path.name} (CV Held-Out)" if s_path else f"Session {len(sessions_list)+1} (CV Held-Out)",
                        'start_idx': curr_idx,
                        'end_idx': curr_idx + n_rois,
                        'total_rois': n_rois
                    })
                    combined_y_prob.extend(y_prob[cv_idx : cv_idx + n_rois].tolist())
                    combined_y_true.extend(y_true[cv_idx : cv_idx + n_rois].tolist())
                    combined_y_pred.extend(y_pred[cv_idx : cv_idx + n_rois].tolist())
                    cv_idx += n_rois
                    curr_idx += n_rois
                
                res = {
                    'status': 'success',
                    'cv_summary': cv_summary,
                    'baseline': baseline,
                    'shap_importance': importance,
                    'fp_culprits': fp_culprits,
                    'fn_culprits': fn_culprits,
                    'active_features': active_feats,
                    'sessions': sessions_list,
                    'y_prob': combined_y_prob,
                    'y_true': combined_y_true,
                    'y_pred': combined_y_pred
                }
                
                state.cv_progress['percent'] = 100
                state.cv_progress['status'] = 'Finished.'
                state.cv_progress['active'] = False
                
                self.send_json(res)
            except Exception as e:
                state.cv_progress['active'] = False
                state.cv_progress['status'] = f"Error: {str(e)}"
                self.send_error_json(str(e))
            return

        elif path == '/api/playground/set_baseline':
            try:
                f1 = float(params.get('f1', [0.0])[0])
                precision = float(params.get('precision', [0.0])[0])
                recall = float(params.get('recall', [0.0])[0])
                feats_str = params.get('features', [''])[0]
                feats = [f.strip() for f in feats_str.split(',') if f.strip()]
                
                workspace_dir = Path("/home/tomer/Documents/suite2p-iscell-prediction")
                baseline_path = workspace_dir / "fe_baseline.json"
                
                baseline = {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'features': feats
                }
                with open(baseline_path, 'w') as f:
                    json.dump(baseline, f, indent=4)
                    
                self.send_json({'status': 'success', 'baseline': baseline})
            except Exception as e:
                self.send_error_json(str(e))
            return

        elif path == '/api/playground/recompute':
            try:
                feature_name = params.get('feature_name', [''])[0].strip()
                if not feature_name:
                    raise ValueError("No feature name specified.")
                
                workspace_dir = Path("/home/tomer/Documents/suite2p-iscell-prediction")
                cache_dir = workspace_dir / "preprocessed_cache"
                
                import fe_engine.fe_definitions
                importlib = __import__('importlib')
                importlib.reload(fe_engine.fe_definitions)
                from fe_engine.fe_definitions import FEATURE_REGISTRY
                
                if feature_name not in FEATURE_REGISTRY:
                    raise ValueError(f"Feature '{feature_name}' not defined in fe_definitions.py.")
                
                extractor = FEATURE_REGISTRY[feature_name]
                npz_files = list(cache_dir.glob("preprocessed_*.npz"))
                
                if len(npz_files) == 0:
                    raise FileNotFoundError("No cached npz files found.")
                
                from joblib import Parallel, delayed
                def worker(f):
                    try:
                        data = np.load(f, allow_pickle=True)
                        sess = {k: data[k] for k in data.files}
                        raw_path_str = sess['session_path']
                        if isinstance(raw_path_str, np.ndarray):
                            raw_path_str = raw_path_str.item()
                        raw_path = Path(raw_path_str)
                        
                        F_arr = np.load(raw_path / 'F.npy', mmap_mode='r')
                        Fneu_arr = np.load(raw_path / 'Fneu.npy', mmap_mode='r')
                        sess['F'] = F_arr
                        sess['Fneu'] = Fneu_arr
                        
                        new_val = extractor(sess)
                        
                        del sess['F']
                        del sess['Fneu']
                        
                        sess[feature_name] = new_val
                        np.savez_compressed(f, **sess)
                        return True
                    except:
                        return False
                
                results = Parallel(n_jobs=2)(delayed(worker)(f) for f in npz_files)
                success_count = sum(1 for r in results if r)
                
                self.send_json({
                    'status': 'success',
                    'success_count': success_count,
                    'total_count': len(npz_files)
                })
            except Exception as e:
                self.send_error_json(str(e))
            return

        elif path == '/api/playground/retrain':
            try:
                feats_str = params.get('features', [''])[0]
                active_feats = [f.strip() for f in feats_str.split(',') if f.strip()]
                formulas = params.get('formulas', [])
                if not active_feats:
                    raise ValueError("No active features selected.")
                
                # Parse formulas
                custom_formulas = {}
                for formula in formulas:
                    formula = formula.strip()
                    if not formula or '=' not in formula:
                        continue
                    name, expr = formula.split('=', 1)
                    custom_formulas[name.strip()] = expr.strip()

                from fe_engine.fe_definitions import FEATURE_REGISTRY, safe_eval_formula
                import fe_engine.fe_definitions
                from fe_engine.fe_loop_runner import load_preprocessed_data, extract_features_dataset, update_definitions_file
                workspace_dir = BASE_DIR
                cache_dir = workspace_dir / "preprocessed_cache"
                
                sessions = load_preprocessed_data(cache_dir)
                
                # Evaluate custom formulas on sessions and register them
                for name, expr in custom_formulas.items():
                    for session in sessions:
                        ns = {key: session[key] for key in session.keys() if isinstance(session[key], np.ndarray) and len(session[key].shape) == 1}
                        try:
                            session[name] = safe_eval_formula(expr, ns)
                        except Exception as eval_exc:
                            raise ValueError(f"Error evaluating custom formula '{name} = {expr}': {str(eval_exc)}")
                    fe_engine.fe_definitions.FEATURE_REGISTRY[name] = lambda cache, n=name: cache[n]

                # Update definitions file only with standard (non-custom) active features
                reg_active = [f for f in active_feats if f in FEATURE_REGISTRY and f not in custom_formulas]
                update_definitions_file(reg_active)
                
                X, y, groups = extract_features_dataset(sessions, active_feats)
                
                n_pos = np.sum(y == 1)
                n_neg = np.sum(y == 0)
                scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
                
                import lightgbm as lgb
                final_model = lgb.LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=8,
                    num_leaves=63,
                    scale_pos_weight=scale_pos_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
                final_model.fit(X, y, feature_name=active_feats)
                
                model_out_path = state.model_path
                if not model_out_path:
                    model_out_path = workspace_dir / 'models' / 'suite2p_best_lgb.pkl'
                
                model_out_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(final_model, model_out_path)
                
                # Write sibling metadata JSON file
                meta_out_path = model_out_path.with_suffix('.json')
                with open(meta_out_path, 'w') as f:
                    json.dump({
                        'active_features': active_feats,
                        'custom_features': custom_formulas
                    }, f, indent=4)
                
                state.load_model(str(model_out_path))
                
                self.send_json({
                    'status': 'success',
                    'retrained_model': str(model_out_path),
                    'features': active_feats
                })
            except Exception as e:
                self.send_error_json(str(e))
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(_json_safe(data)).encode('utf-8'))
        
    def send_error_json(self, message):
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode('utf-8'))

# ==========================================
# 5. BEAUTIFUL FRONTEND HTML/JS/CSS
# ==========================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Suite2p Cell Classification Explainability Dashboard</title>
    <!-- Tailwind CSS (via CDN) -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts: Outfit -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <!-- FontAwesome for Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Outfit', 'sans-serif'],
                    },
                    colors: {
                        brand: {
                            50: '#f2f7ff',
                            100: '#e1eeff',
                            200: '#bcdbff',
                            500: '#3b82f6',
                            600: '#2563eb',
                            700: '#1d4ed8',
                            900: '#1e3a8a',
                            darkBg: '#0b0f19',
                            cardBg: '#151d30',
                            border: '#1f2d47'
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #0b0f19;
            color: #f3f4f6;
        }
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0b0f19;
        }
        ::-webkit-scrollbar-thumb {
            background: #1f2d47;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3b82f6;
        }
        .glassmorphism {
            background: rgba(21, 29, 48, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
</head>
<body class="font-sans antialiased overflow-x-hidden">

    <!-- HEADER -->
    <header class="border-b border-brand-border bg-brand-cardBg/50 sticky top-0 z-50 backdrop-blur-md">
        <div class="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
            <div class="flex items-center gap-3">
                <div class="bg-blue-600 text-white p-2 rounded-xl shadow-lg shadow-blue-500/20">
                    <i class="fa-solid fa-microscope text-xl"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-blue-400 via-indigo-200 to-purple-400 bg-clip-text text-transparent">
                        Suite2p AI Decision Explainer
                    </h1>
                    <p class="text-xs text-gray-400">Local Feature Explainability & Sensitivity Analysis Tool</p>
                </div>
            </div>
            
            <div class="flex items-center gap-4">
                <a href="/" class="text-xs font-semibold px-3 py-2 rounded-lg bg-blue-600/10 border border-blue-500/20 text-blue-400 hover:bg-blue-600/20 transition flex items-center gap-1.5">
                    <i class="fa-solid fa-microscope"></i> Cell Curation
                </a>
                <a href="/playground" class="text-xs font-semibold px-3 py-2 rounded-lg bg-brand-border text-gray-300 hover:bg-brand-border/80 transition flex items-center gap-1.5">
                    <i class="fa-solid fa-flask"></i> Feature Playground
                </a>
                <span class="text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center gap-1.5 flex-shrink-0">
                    <span class="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span> Server Active
                </span>
            </div>
        </div>
    </header>

    <main class="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        <!-- SETTINGS PANEL (CONFIGURABILITY BAR) -->
        <section class="glassmorphism p-5 rounded-2xl shadow-xl space-y-4">
            <div class="flex flex-wrap gap-4 items-end">
                <!-- Session Path -->
                <div class="flex-1 min-w-[280px]">
                    <label class="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1.5 flex items-center gap-1.5">
                        <i class="fa-regular fa-folder-open text-blue-400"></i> Suite2p Session Folder Path
                    </label>
                    <div class="flex gap-2">
                        <input type="text" id="session-path-input" 
                               class="w-full bg-brand-darkBg border border-brand-border rounded-xl px-4 py-2.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500 transition" 
                               placeholder="e.g. /mnt/data/1-ordered/Stav1">
                    </div>
                </div>
                
                <!-- Model Selection -->
                <div class="flex-1 min-w-[280px]">
                    <label class="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1.5 flex items-center gap-1.5">
                        <i class="fa-solid fa-brain text-purple-400"></i> AI Model File (.pkl)
                    </label>
                    <select id="model-select" onchange="onModelSelectChange()"
                            class="w-full bg-brand-darkBg border border-brand-border rounded-xl px-3 py-2.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500 transition">
                    </select>
                </div>
                
                <!-- Scaler Selection -->
                <div class="flex-1 min-w-[200px]">
                    <label class="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1.5 flex items-center gap-1.5">
                        <i class="fa-solid fa-sliders text-indigo-400"></i> Scaler (For MLP)
                    </label>
                    <input type="text" id="scaler-select" list="scaler-options" 
                           class="w-full bg-brand-darkBg border border-brand-border rounded-xl px-3 py-2.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500 transition"
                           placeholder="Select or enter absolute scaler path...">
                    <datalist id="scaler-options"></datalist>
                </div>
                
                <!-- Probability Threshold -->
                <div class="w-48">
                    <label class="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1.5 flex items-center gap-1.5">
                        <i class="fa-solid fa-circle-nodes text-yellow-400"></i> Predict Threshold
                    </label>
                    <div class="flex items-center gap-3 bg-brand-darkBg border border-brand-border rounded-xl px-3 py-2 text-sm text-gray-200 h-[42px]">
                        <input type="range" id="threshold-slider" min="0.05" max="0.95" step="0.05" value="0.80"
                               class="w-full h-1 bg-brand-border rounded-lg appearance-none cursor-pointer accent-blue-500"
                               oninput="document.getElementById('threshold-value').textContent = parseFloat(this.value).toFixed(2); updateThreshold(parseFloat(this.value));">
                        <span class="font-bold text-blue-400 text-sm whitespace-nowrap" id="threshold-value">0.80</span>
                    </div>
                </div>
                
                <!-- Apply Button -->
                <div>
                    <button onclick="applySettings()" id="apply-btn"
                            class="bg-blue-600 hover:bg-blue-500 text-white font-medium px-6 py-2.5 rounded-xl shadow-lg shadow-blue-500/10 hover:shadow-blue-500/25 active:scale-95 transition flex items-center gap-2">
                        <i class="fa-solid fa-sync"></i> Apply Settings
                    </button>
                </div>
            </div>
            
            <!-- Suggested paths -->
            <div id="suggested-sessions" class="text-xs text-gray-400 flex flex-wrap items-center gap-2">
                <span class="font-semibold text-gray-500 uppercase tracking-wide">Quick-Load Workspace Paths:</span>
            </div>

            <!-- SELECTED MODEL METADATA CARD -->
            <div id="model-details-card" class="bg-brand-darkBg/70 border border-brand-border/60 rounded-xl p-4 space-y-3 transition mt-2">
                <div class="flex flex-wrap items-center justify-between gap-3 border-b border-brand-border/40 pb-2.5">
                    <div class="flex items-center gap-3">
                        <div class="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
                            <i class="fa-solid fa-brain text-base"></i>
                        </div>
                        <div>
                            <h4 id="model-card-title" class="text-sm font-bold text-gray-100 flex items-center gap-2">
                                Loading Model Details...
                            </h4>
                            <p id="model-card-desc" class="text-xs text-gray-400">
                                Model description will appear here.
                            </p>
                        </div>
                    </div>
                    <div class="flex items-center gap-2">
                        <span id="model-card-count" class="px-2.5 py-1 rounded-full bg-purple-500/10 text-purple-300 border border-purple-500/20 text-xs font-semibold">
                            0 Features
                        </span>
                        <span id="model-card-file" class="px-2.5 py-1 rounded-full bg-blue-500/10 text-blue-300 border border-blue-500/20 text-xs font-mono">
                            Path
                        </span>
                    </div>
                </div>
                <div>
                    <div class="flex items-center justify-between mb-1.5">
                        <label class="text-[11px] font-semibold uppercase tracking-wider text-gray-400 flex items-center gap-1.5">
                            <i class="fa-solid fa-list-check text-blue-400"></i> Enabled Active Features
                        </label>
                        <span id="model-card-feature-subtitle" class="text-[11px] text-gray-500"></span>
                    </div>
                    <div id="model-card-features" class="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pr-1">
                        <span class="text-xs text-gray-500 italic">No features loaded</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- NO SESSION PLACEHOLDER -->
        <section id="no-session-card" class="glassmorphism p-12 rounded-3xl text-center space-y-4">
            <div class="mx-auto w-16 h-16 rounded-2xl bg-brand-border flex items-center justify-center text-blue-400">
                <i class="fa-solid fa-circle-info text-2xl"></i>
            </div>
            <div class="space-y-2">
                <h3 class="text-lg font-semibold">No Suite2p Session Loaded</h3>
                <p class="text-sm text-gray-400 max-w-md mx-auto">Please enter a valid Suite2p session directory path containing F.npy, Fneu.npy, and stat.npy to begin cell-level decision explanations.</p>
            </div>
        </section>

        <!-- DASHBOARD CONTAINER -->
        <div id="dashboard-content" class="hidden space-y-6">
            
            <!-- Ground Truth Missing Warning Banner -->
            <div id="gt-warning-banner" class="hidden bg-amber-500/10 border border-amber-500/30 rounded-2xl p-4 flex items-start gap-3 text-amber-400">
                <i class="fa-solid fa-triangle-exclamation text-lg mt-0.5"></i>
                <div class="space-y-1">
                    <h4 class="font-bold text-sm">No Curated Ground Truth File Loaded</h4>
                    <p class="text-xs text-gray-300 leading-relaxed font-normal">
                        Neither <code class="font-mono bg-brand-darkBg/60 px-1 py-0.5 rounded text-amber-300 text-[11px]">iscell_final.npy</code> nor <code class="font-mono bg-brand-darkBg/60 px-1 py-0.5 rounded text-amber-300 text-[11px]">iscell_manual.npy</code> was found in this session directory. 
                        Ground truth has been initialized to all zeros. True Positives and False Negatives counts will remain 0, and metrics (Precision, Recall, F1) cannot be calculated.
                        To evaluate model performance, please run curation in the Suite2p GUI and copy/rename the resulting <code class="font-mono bg-brand-darkBg/60 px-1 py-0.5 rounded text-amber-300 text-[11px]">iscell.npy</code> to <code class="font-mono bg-brand-darkBg/60 px-1 py-0.5 rounded text-amber-300 text-[11px]">iscell_final.npy</code>.
                    </p>
                </div>
            </div>
            
            <!-- ROW 1: STATS & NAVIGATION -->
            <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <!-- Overview Stats -->
                <div class="glassmorphism p-5 rounded-2xl space-y-4 lg:col-span-1 flex flex-col justify-between">
                    <div>
                        <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">Session Overview</h3>
                        <div class="space-y-3">
                            <div>
                                <div class="text-3xl font-extrabold text-white" id="session-name-display">-</div>
                                <div class="text-xs text-gray-400">Active Session</div>
                            </div>
                            <div class="grid grid-cols-2 gap-4 pt-2">
                                <div>
                                    <div class="text-xl font-bold text-gray-200" id="total-rois-display">0</div>
                                    <div class="text-[10px] text-gray-400 uppercase font-semibold">Total ROIs</div>
                                </div>
                                <div>
                                    <div class="text-xl font-bold text-blue-400" id="detected-features-display">0</div>
                                    <div class="text-[10px] text-gray-400 uppercase font-semibold">Model Features</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="pt-4 border-t border-brand-border space-y-1">
                        <div class="flex justify-between text-xs">
                            <span class="text-gray-400">Ground Truth Cells:</span>
                            <span class="font-bold text-emerald-400" id="gt-cells-display">0</span>
                        </div>
                        <div class="flex justify-between text-xs">
                            <span class="text-gray-400">AI Predicted Cells:</span>
                            <span class="font-bold text-blue-400" id="pred-cells-display">0</span>
                        </div>
                        <div class="pt-2 border-t border-brand-border/40 mt-2 space-y-1">
                            <div class="flex justify-between text-xs">
                                <span class="text-gray-400 font-medium">Precision:</span>
                                <span class="font-bold text-blue-300" id="precision-display">0.0%</span>
                            </div>
                            <div class="flex justify-between text-xs">
                                <span class="text-gray-400 font-medium">Recall:</span>
                                <span class="font-bold text-purple-300" id="recall-display">0.0%</span>
                            </div>
                            <div class="flex justify-between text-xs">
                                <span class="text-gray-400 font-medium">F1-Score:</span>
                                <span class="font-bold text-amber-300" id="f1-display">0.0%</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Navigation panel -->
                <div class="glassmorphism p-5 rounded-2xl lg:col-span-3 flex flex-col justify-between">
                    <div>
                        <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3 flex justify-between items-center">
                            <span>ROI Navigation Index</span>
                            <div class="flex items-center gap-2">
                                <span class="text-amber-400 text-[10px] hidden font-semibold px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 animate-pulse" id="navigation-active-category"></span>
                                <span class="text-blue-400 text-[10px]" id="navigation-active-model">-</span>
                            </div>
                        </h3>
                        
                        <div class="flex items-center gap-3">
                            <!-- Cell index input -->
                            <div class="flex items-center bg-brand-darkBg border border-brand-border rounded-xl px-3 py-1 flex-1 max-w-[200px]">
                                <span class="text-xs font-semibold text-gray-500 mr-2 uppercase">ROI</span>
                                <input type="number" id="cell-idx-input" 
                                       class="w-full bg-transparent text-lg font-bold text-white focus:outline-none" 
                                       min="0" value="0">
                            </div>
                            
                            <button onclick="loadCell(parseInt(document.getElementById('cell-idx-input').value))"
                                    class="bg-brand-border hover:bg-brand-border/80 text-gray-200 font-medium px-4 py-2.5 rounded-xl text-sm transition flex items-center gap-2">
                                Go <i class="fa-solid fa-arrow-right"></i>
                            </button>
                            
                            <!-- Arrow navigation -->
                            <div class="flex items-center gap-1.5 ml-auto">
                                <button onclick="navigateCell(-1)" class="p-2.5 rounded-xl bg-brand-border hover:bg-brand-border/80 text-gray-200 transition">
                                    <i class="fa-solid fa-chevron-left"></i> Prev
                                </button>
                                <button onclick="navigateCell(1)" class="p-2.5 rounded-xl bg-brand-border hover:bg-brand-border/80 text-gray-200 transition">
                                    Next <i class="fa-solid fa-chevron-right"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Quick Jump filters -->
                    <div class="pt-4 border-t border-brand-border mt-4">
                        <div class="text-[10px] uppercase font-semibold text-gray-400 mb-2">Jump to Specific Category:</div>
                        <div class="flex flex-wrap gap-2">
                            <button onclick="jumpCategory('true_positives')" id="btn-true_positives"
                                    class="text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/10 flex items-center gap-1.5 transition">
                                <i class="fa-solid fa-check-double"></i> True Positives (<span id="count-tp">0</span>)
                            </button>
                            <button onclick="jumpCategory('false_positives')" id="btn-false_positives"
                                    class="text-xs px-3 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/10 flex items-center gap-1.5 transition">
                                <i class="fa-solid fa-circle-xmark"></i> False Positives (<span id="count-fp">0</span>)
                            </button>
                            <button onclick="jumpCategory('false_negatives')" id="btn-false_negatives"
                                    class="text-xs px-3 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/10 flex items-center gap-1.5 transition">
                                <i class="fa-solid fa-triangle-exclamation"></i> False Negatives (<span id="count-fn">0</span>)
                            </button>
                            <button onclick="jumpCategory('uncertain')" id="btn-uncertain"
                                    class="text-xs px-3 py-1.5 rounded-lg bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 border border-blue-500/10 flex items-center gap-1.5 transition">
                                <i class="fa-solid fa-question-circle"></i> Uncertain 15-85% (<span id="count-unc">0</span>)
                            </button>
                            <button onclick="clearCategoryFilter()" id="btn-clear-filter"
                                    class="text-xs px-3 py-1.5 rounded-lg bg-gray-500/10 hover:bg-gray-500/20 text-gray-400 border border-gray-500/10 flex items-center gap-1.5 transition hidden">
                                <i class="fa-solid fa-circle-minus"></i> Clear Filter (All ROIs)
                            </button>
                        </div>
                    </div>
                    
                    <!-- Category Feature Analysis -->
                    <div class="pt-4 border-t border-brand-border/60 mt-4">
                        <div class="text-[10px] uppercase font-semibold text-gray-400 mb-2">Category Feature Profile & Impact:</div>
                        <div id="category-feature-impact" class="p-3.5 bg-brand-darkBg/60 border border-brand-border/40 rounded-xl text-xs space-y-2">
                            <span class="text-gray-500 italic">Select a category above (e.g. False Positives) to analyze its distinct feature profile...</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- MAIN DASHBOARD CONTENT GRID -->
            <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
                <!-- LEFT COLUMN: ROI Details (1/4 width on desktop) -->
                <div class="glassmorphism p-5 rounded-2xl lg:col-span-1 space-y-6 flex flex-col justify-between">
                    <div>
                        <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">ROI Details</h3>
                        
                        <div class="space-y-4">
                            <!-- Circular Gauge for Confidence -->
                            <div class="flex flex-col items-center justify-center py-2 relative">
                                <div class="text-center">
                                    <div class="text-4xl font-extrabold text-white" id="cell-prob-value">0%</div>
                                    <div class="text-xs text-gray-400 mt-1 uppercase font-semibold tracking-wider">AI Confidence</div>
                                </div>
                            </div>
                            
                            <!-- Badges -->
                            <div class="grid grid-cols-2 gap-3 pt-2">
                                <div class="bg-brand-darkBg/50 p-3 rounded-xl border border-brand-border text-center">
                                    <div class="text-xs text-gray-400 font-semibold mb-1">GT Label</div>
                                    <div id="cell-gt-badge" class="text-sm font-bold">-</div>
                                </div>
                                <div class="bg-brand-darkBg/50 p-3 rounded-xl border border-brand-border text-center">
                                    <div class="text-xs text-gray-400 font-semibold mb-1">AI Decision</div>
                                    <div id="cell-pred-badge" class="text-sm font-bold">-</div>
                                </div>
                            </div>
                            
                            <!-- Classification Outcome Status -->
                            <div class="p-3.5 rounded-xl text-center border font-semibold text-sm" id="cell-outcome-badge">
                                -
                            </div>
                            
                            <!-- Set Comparison ROI Button -->
                            <button onclick="setComparisonROIA(activeCellIdx)" class="w-full bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 border border-blue-500/20 font-medium py-2 rounded-xl text-xs transition flex items-center justify-center gap-1.5">
                                <i class="fa-solid fa-code-compare"></i> Set as ROI A for Comparison
                            </button>

                            <!-- Ground Truth File Source -->
                            <div class="bg-brand-darkBg/30 p-3 rounded-xl border border-brand-border/60 text-left space-y-1">
                                <div class="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">GT Source Path</div>
                                <div id="gt-path-display" class="text-[11px] text-gray-300 font-mono break-all leading-normal" title="No ground truth file loaded.">-</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Spatial Outline Mini-Plot -->
                    <div class="space-y-2 pt-4 border-t border-brand-border">
                        <div class="text-xs font-semibold uppercase tracking-wider text-gray-400">Spatial ROI Footprint</div>
                        <div id="roi-spatial-plot" class="w-full h-48 bg-brand-darkBg/50 rounded-xl overflow-hidden flex items-center justify-center border border-brand-border">
                        </div>
                    </div>
                </div>
                
                <!-- RIGHT COLUMN: Local Feature Contribution (3/4 width on desktop) -->
                <div class="glassmorphism p-5 rounded-2xl lg:col-span-3">
                    <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4 flex justify-between items-center">
                        <span>Local Feature Contribution (Why the model decided this)</span>
                        <div class="flex items-center gap-2">
                            <span id="shap-not-avail-alert" class="text-xs text-rose-400 font-semibold hidden"><i class="fa-solid fa-triangle-exclamation"></i> TreeSHAP N/A for MLP</span>
                            <select id="attr-method-select" onchange="updateAttributionChart()" class="bg-brand-darkBg border border-brand-border rounded-lg text-xs text-gray-300 px-2.5 py-1.5 focus:outline-none focus:border-blue-500 transition">
                                <option value="shap">TreeSHAP Contributions</option>
                                <option value="ablation">Ablation Contributions</option>
                                <option value="comparison">Side-by-Side Comparison</option>
                            </select>
                        </div>
                    </h3>
                    <div id="attribution-chart" class="w-full h-[600px]">
                    </div>
                </div>
            </div>

            <!-- FULL WIDTH SECTIONS BELOW -->
            <div class="space-y-6 mt-6">
                <!-- Session Probability Distribution & Clusters -->
                <section class="glassmorphism p-5 rounded-2xl">
                    <div class="flex justify-between items-center mb-4 flex-wrap gap-4">
                        <div>
                            <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400">Session Probability Distribution & ROI Clusters</h3>
                            <p class="text-xs text-gray-500 mt-1">Interactive overview of all predictions in the session. Click any point on the scatter plot to jump directly to that ROI.</p>
                        </div>
                        <div class="flex items-center gap-2">
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" id="hide-threshold-toggle" class="sr-only peer" onchange="plotSessionProbabilities()">
                                <div class="w-9 h-5 bg-brand-darkBg border border-brand-border rounded-full peer peer-focus:outline-none peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-gray-400 after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600 peer-checked:after:bg-white"></div>
                                <span class="ml-2 text-xs font-semibold text-gray-400">Hide Threshold & Color by GT</span>
                            </label>
                        </div>
                    </div>
                    <div class="bg-brand-darkBg/30 border border-brand-border/40 rounded-xl p-3">
                        <!-- Scatter Plot -->
                        <div id="probability-scatter-plot" class="w-full h-[450px]"></div>
                    </div>
                </section>

                <!-- ROI Suppression & Comparison Tool -->
                <section class="glassmorphism p-5 rounded-2xl">
                    <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">ROI Suppression & Comparison Tool</h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div class="space-y-4">
                            <p class="text-xs text-gray-400 leading-relaxed">
                                Enter two ROI indices to compute their centroid distance, calcium trace Pearson correlation, and pixel-wise overlap. This helps evaluate whether they are candidate duplicates for suppression.
                            </p>
                            <div class="flex items-center gap-3">
                                <div class="flex items-center bg-brand-darkBg border border-brand-border rounded-xl px-3 py-1.5 flex-1 min-w-[80px]">
                                    <span class="text-xs font-semibold text-gray-500 mr-2 uppercase">ROI A</span>
                                    <input type="number" id="compare-roi-a" class="w-full min-w-0 bg-transparent font-bold text-white focus:outline-none" min="0" value="0">
                                </div>
                                <div class="flex items-center bg-brand-darkBg border border-brand-border rounded-xl px-3 py-1.5 flex-1 min-w-[80px]">
                                    <span class="text-xs font-semibold text-gray-500 mr-2 uppercase">ROI B</span>
                                    <input type="number" id="compare-roi-b" class="w-full min-w-0 bg-transparent font-bold text-white focus:outline-none" min="0" value="1">
                                </div>
                            </div>
                            <button onclick="compareROIs()" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium px-4 py-3 rounded-xl text-sm transition flex items-center justify-center gap-2">
                                <i class="fa-solid fa-magnifying-glass-chart"></i> Compare ROIs
                            </button>
                        </div>
                        
                        <div class="md:col-span-2 bg-brand-darkBg/50 border border-brand-border/60 rounded-xl p-4 flex flex-col justify-center min-h-[140px]" id="comparison-results-container">
                            <div class="text-center text-gray-500 italic text-sm">Enter ROI indices and click Compare ROIs to view spatial and temporal similarities.</div>
                        </div>
                    </div>
                </section>

                <!-- Detailed Feature Attribution & Context Comparison -->
                <section class="glassmorphism p-5 rounded-2xl">
                    <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">Detailed Feature Attribution & Context Comparison</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-sm text-gray-300">
                            <thead class="text-xs uppercase tracking-wider text-gray-400 bg-brand-darkBg/50 border-b border-brand-border">
                                <tr>
                                    <th scope="col" class="px-4 py-3">Feature Name</th>
                                    <th scope="col" class="px-4 py-3 text-right">TreeSHAP Impact</th>
                                    <th scope="col" class="px-4 py-3 text-right">Ablation Impact</th>
                                    <th scope="col" class="px-4 py-3 text-right">Cell Raw Value</th>
                                    <th scope="col" class="px-4 py-3 text-right">Dataset Avg</th>
                                    <th scope="col" class="px-4 py-3 text-right">Cells Avg (GT=1)</th>
                                    <th scope="col" class="px-4 py-3 text-right">Non-Cells Avg (GT=0)</th>
                                    <th scope="col" class="px-4 py-3 max-w-[280px]">Explanation</th>
                                </tr>
                            </thead>
                            <tbody id="features-table-body" class="divide-y divide-brand-border/50">
                            </tbody>
                        </table>
                    </div>
                </section>

                <!-- Fluorescence Calcium Traces -->
                <section class="glassmorphism p-5 rounded-2xl">
                    <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">Fluorescence Calcium Traces</h3>
                    <div id="trace-plot" class="w-full h-80">
                    </div>
                </section>
            </div>
        </div>
    </main>

    <!-- LOADING SPINNER -->
    <div id="loader" class="hidden fixed inset-0 z-50 bg-brand-darkBg/70 flex flex-col items-center justify-center gap-3 backdrop-blur-sm">
        <div class="h-10 w-10 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
        <p class="text-sm font-semibold text-gray-300">Calculating explainability attributions...</p>
    </div>

    <!-- APPLY SETTINGS PROGRESS -->
    <div id="apply-progress" class="hidden fixed inset-0 z-50 bg-brand-darkBg/70 flex items-center justify-center backdrop-blur-sm">
        <div class="w-[420px] max-w-[90vw] bg-brand-card border border-brand-border rounded-2xl p-5 shadow-xl">
            <div class="flex items-center justify-between mb-3">
                <p class="text-sm font-semibold text-gray-200">Applying settings</p>
                <span id="apply-progress-pct" class="text-xs font-mono text-blue-300">0%</span>
            </div>
            <div class="h-2.5 w-full bg-brand-border/60 rounded-full overflow-hidden">
                <div id="apply-progress-bar" class="h-full bg-blue-500 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
            <div class="flex items-center justify-between mt-3 gap-3">
                <p id="apply-progress-status" class="text-xs text-gray-400 truncate">Starting...</p>
                <span id="apply-progress-time" class="text-xs font-mono text-gray-500 shrink-0">0s</span>
            </div>
        </div>
    </div>

    <!-- MAIN JAVASCRIPT LOGIC -->
    <script>
        let sessionInfo = null;
        let activeCellIdx = 0;
        let selectedCategoryList = null;
        let currentCellExplanation = null;
        let activeThreshold = 0.80;

        function updateThreshold(val) {
            activeThreshold = val;
            recalculateMetricsAndCategories();
            
            // If we are browsing a specific category, adjust the active cell if needed
            if (activeCategory) {
                let cells = sessionInfo.categories[activeCategory];
                if (!cells || cells.length === 0) {
                    alert(`No ROIs remaining in category: ${activeCategory.replace('_', ' ')}. Clearing filter.`);
                    clearCategoryFilter();
                } else if (cells.indexOf(activeCellIdx) === -1) {
                    // Current cell is no longer in this category. Load the closest cell in the new list.
                    let closestCell = cells[0];
                    let minDiff = Math.abs(cells[0] - activeCellIdx);
                    for (let i = 1; i < cells.length; i++) {
                        let diff = Math.abs(cells[i] - activeCellIdx);
                        if (diff < minDiff) {
                            minDiff = diff;
                            closestCell = cells[i];
                        }
                    }
                    loadCell(closestCell);
                } else {
                    updateCellOutcomeUI();
                }
            } else {
                updateCellOutcomeUI();
            }
        }

        function updateCellOutcomeUI() {
            if (!currentCellExplanation) return;
            let prob = currentCellExplanation.probability;
            let label = currentCellExplanation.label;
            let isCellPred = (prob >= activeThreshold) ? 1 : 0;
            
            let predBadge = document.getElementById('cell-pred-badge');
            if (isCellPred === 1) {
                predBadge.className = "text-sm font-bold text-emerald-400";
                predBadge.textContent = "Cell (1)";
            } else {
                predBadge.className = "text-sm font-bold text-rose-400";
                predBadge.textContent = "Non-Cell (0)";
            }
            
            let outcomeBadge = document.getElementById('cell-outcome-badge');
            if (label === 1 && isCellPred === 1) {
                outcomeBadge.className = "p-3.5 rounded-xl text-center border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 font-semibold text-sm";
                outcomeBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> True Positive (Correct Cell)';
            } else if (label === 0 && isCellPred === 0) {
                outcomeBadge.className = "p-3.5 rounded-xl text-center border border-gray-700 bg-gray-800/40 text-gray-400 font-semibold text-sm";
                outcomeBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> True Negative (Correct Noise)';
            } else if (label === 0 && isCellPred === 1) {
                outcomeBadge.className = "p-3.5 rounded-xl text-center border border-rose-500/30 bg-rose-500/10 text-rose-400 font-semibold text-sm animate-pulse";
                outcomeBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> False Positive (AI Mistake!)';
            } else {
                outcomeBadge.className = "p-3.5 rounded-xl text-center border border-amber-500/30 bg-amber-500/10 text-amber-400 font-semibold text-sm animate-pulse";
                outcomeBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> False Negative (AI Missed!)';
            }
        }

        function recalculateMetricsAndCategories() {
            if (!sessionInfo || !sessionInfo.loaded) return;
            
            let tp = [];
            let fp = [];
            let tn = [];
            let fn = [];
            let uncertain = [];
            
            let probs = sessionInfo.probabilities;
            let gt = sessionInfo.ground_truth;
            
            for (let i = 0; i < probs.length; i++) {
                let p = probs[i];
                let g = gt[i];
                let pred = (p >= activeThreshold) ? 1 : 0;
                
                if (pred === 1 && g === 1) tp.push(i);
                else if (pred === 1 && g === 0) fp.push(i);
                else if (pred === 0 && g === 0) tn.push(i);
                else if (pred === 0 && g === 1) fn.push(i);
                
                if (p > 0.15 && p < 0.85) {
                    uncertain.push(i);
                }
            }
            
            sessionInfo.categories = {
                true_positives: tp,
                false_positives: fp,
                true_negatives: tn,
                false_negatives: fn,
                uncertain: uncertain
            };
            
            let tp_count = tp.length;
            let fp_count = fp.length;
            let fn_count = fn.length;
            
            const hasGT = sessionInfo.gt_path !== "None" && sessionInfo.gt_path !== "";
            
            if (hasGT) {
                let precision = tp_count + fp_count > 0 ? tp_count / (tp_count + fp_count) : 0;
                let recall = tp_count + fn_count > 0 ? tp_count / (tp_count + fn_count) : 0;
                let f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
                
                document.getElementById('precision-display').textContent = (precision * 100).toFixed(1) + '%';
                document.getElementById('recall-display').textContent = (recall * 100).toFixed(1) + '%';
                document.getElementById('f1-display').textContent = (f1 * 100).toFixed(1) + '%';
                document.getElementById('gt-cells-display').textContent = sessionInfo.ground_truth.filter(x => x === 1).length;
                
                document.getElementById('btn-true_positives').classList.remove('hidden');
                document.getElementById('btn-false_negatives').classList.remove('hidden');
                
                document.getElementById('btn-true_positives').innerHTML = `<i class="fa-solid fa-check-double"></i> True Positives (<span id="count-tp">${tp_count}</span>)`;
                document.getElementById('btn-false_positives').innerHTML = `<i class="fa-solid fa-circle-xmark"></i> False Positives (<span id="count-fp">${fp_count}</span>)`;
                document.getElementById('btn-false_negatives').innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> False Negatives (<span id="count-fn">${fn_count}</span>)`;
            } else {
                document.getElementById('precision-display').textContent = 'N/A';
                document.getElementById('recall-display').textContent = 'N/A';
                document.getElementById('f1-display').textContent = 'N/A';
                document.getElementById('gt-cells-display').textContent = 'N/A';
                
                document.getElementById('btn-true_positives').classList.add('hidden');
                document.getElementById('btn-false_negatives').classList.add('hidden');
                
                document.getElementById('btn-false_positives').innerHTML = `<i class="fa-solid fa-brain"></i> Predicted Cells (<span id="count-fp">${fp_count}</span>)`;
            }
            
            document.getElementById('btn-uncertain').innerHTML = `<i class="fa-solid fa-question-circle"></i> Uncertain 15-85% (<span id="count-unc">${uncertain.length}</span>)`;
            
            document.getElementById('pred-cells-display').textContent = tp_count + fp_count;
            document.getElementById('count-tp').textContent = tp_count;
            document.getElementById('count-fp').textContent = fp_count;
            document.getElementById('count-fn').textContent = fn_count;
            document.getElementById('count-unc').textContent = uncertain.length;
            
            // Fetch category analysis after updating categories
            fetchCategoryAnalysis();

            // Plot/update session-wide probabilities
            if (typeof plotSessionProbabilities === 'function') {
                plotSessionProbabilities();
            }
        }
        
        window.addEventListener('DOMContentLoaded', () => {
            loadModelsList();
            
            document.getElementById('cell-idx-input').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    let val = parseInt(e.target.value);
                    if (!isNaN(val)) loadCell(val);
                }
            });

            // Handle Plotly resizing for responsiveness
            window.addEventListener('resize', () => {
                const charts = ['attribution-chart', 'trace-plot', 'probability-scatter-plot'];
                charts.forEach(id => {
                    const el = document.getElementById(id);
                    if (el && el.innerHTML) {
                        Plotly.Plots.resize(el);
                    }
                });
            });
        });

        function showLoader(show) {
            document.getElementById('loader').classList.toggle('hidden', !show);
        }

        window.allModelDetails = [];

        function renderModelCard(modelInfo) {
            if (!modelInfo) return;
            // Models that store their tuned threshold (e.g. the image model) start the slider there
            if (typeof modelInfo.threshold === 'number') {
                let t = Math.round(modelInfo.threshold * 100) / 100;
                document.getElementById('threshold-slider').value = t;
                document.getElementById('threshold-value').textContent = t.toFixed(2);
                activeThreshold = t;
                if (sessionInfo && sessionInfo.loaded) updateThreshold(t);
            }
            let titleEl = document.getElementById('model-card-title');
            let descEl = document.getElementById('model-card-desc');
            let countEl = document.getElementById('model-card-count');
            let fileEl = document.getElementById('model-card-file');
            let featsContainer = document.getElementById('model-card-features');

            if (titleEl) titleEl.textContent = modelInfo.name || 'Custom Model';
            if (descEl) descEl.textContent = modelInfo.description || 'No description provided for this model.';
            
            let count = modelInfo.feature_count || (modelInfo.active_features ? modelInfo.active_features.length : 0);
            if (countEl) countEl.textContent = `${count} Active Features`;
            if (fileEl) fileEl.textContent = modelInfo.path || modelInfo.filename || '';
            
            if (featsContainer) {
                featsContainer.innerHTML = '';
                let feats = modelInfo.active_features || [];
                if (feats.length === 0) {
                    featsContainer.innerHTML = '<span class="text-xs text-gray-500 italic">No feature list recorded</span>';
                } else {
                    feats.forEach(f => {
                        let span = document.createElement('span');
                        span.className = "px-2 py-0.5 rounded bg-brand-border/60 border border-brand-border/80 text-[11px] font-mono text-gray-300 hover:text-blue-300 hover:border-blue-500/40 transition cursor-default";
                        span.textContent = f;
                        featsContainer.appendChild(span);
                    });
                }
            }
        }

        // Hover text for a model: when it was trained, on what data, and the idea behind it
        function modelTooltip(m) {
            let lines = [m.name || m.filename || ''];
            if (m.trained) lines.push(`Trained: ${m.trained}`);
            if (m.training_data) lines.push(`Data: ${m.training_data}`);
            lines.push(`Idea: ${m.idea || m.description || 'No description recorded.'}`);
            return lines.join('\\n');
        }

        function onModelSelectChange() {
            let modelSelect = document.getElementById('model-select');
            if (!modelSelect) return;
            let val = modelSelect.value;
            let found = (window.allModelDetails || []).find(m => m.path === val || m.filename === val);
            modelSelect.title = found ? modelTooltip(found) : '';
            if (found) {
                renderModelCard(found);
            }
        }

        async function loadModelsList() {
            try {
                let res = await fetch('/api/models');
                let data = await res.json();
                
                window.allModelDetails = data.model_details || [];

                let modelSelect = document.getElementById('model-select');
                if (modelSelect) {
                    modelSelect.innerHTML = '';
                    if (data.model_details && data.model_details.length > 0) {
                        data.model_details.forEach(m => {
                            let opt = document.createElement('option');
                            opt.value = m.path;
                            opt.textContent = `[${m.feature_count} Features] ${m.name} (${m.filename})`;
                            opt.title = modelTooltip(m);
                            modelSelect.appendChild(opt);
                        });
                    } else if (data.models) {
                        data.models.forEach(m => {
                            let opt = document.createElement('option');
                            opt.value = m;
                            opt.textContent = m;
                            modelSelect.appendChild(opt);
                        });
                    }
                    if (data.active_model) {
                        modelSelect.value = data.active_model;
                    }
                    let sel = modelSelect.options[modelSelect.selectedIndex];
                    modelSelect.title = sel ? sel.title : '';
                }
                
                let scalerDatalist = document.getElementById('scaler-options');
                if (scalerDatalist) {
                    scalerDatalist.innerHTML = '<option value="">[Auto-Detect Scaler]</option>';
                    data.scalers.forEach(s => {
                        let opt = document.createElement('option');
                        opt.value = s;
                        scalerDatalist.appendChild(opt);
                    });
                }
                let scalerSelect = document.getElementById('scaler-select');
                if (scalerSelect) {
                    scalerSelect.value = data.active_scaler || '';
                }
                
                let sugDiv = document.getElementById('suggested-sessions');
                sugDiv.innerHTML = '<span class="font-semibold text-gray-500 uppercase tracking-wide">Quick-Load Workspace Paths:</span>';
                if (data.suggested_sessions && data.suggested_sessions.length > 0) {
                    data.suggested_sessions.forEach(p => {
                        let btn = document.createElement('button');
                        btn.className = "px-2 py-1 rounded bg-brand-border/40 hover:bg-brand-border text-gray-300 font-semibold cursor-pointer transition ml-2";
                        let parts = p.split('/');
                        btn.textContent = parts[parts.length - 1] || p;
                        btn.onclick = () => {
                            document.getElementById('session-path-input').value = p;
                            applySettings();
                        };
                        sugDiv.appendChild(btn);
                    });
                } else {
                    sugDiv.innerHTML += '<span class="text-gray-500 italic ml-2">None found, enter manually</span>';
                }
                
                if (data.active_model_info) {
                    renderModelCard(data.active_model_info);
                } else {
                    onModelSelectChange();
                }

                if (data.active_session) {
                    document.getElementById('session-path-input').value = data.active_session;
                    loadSessionInfo();
                }
            } catch(e) {
                console.error("Error loading models:", e);
            }
        }

        let isApplying = false;

        // Progress overlay for Apply Settings. The server reports its progress through the pipeline;
        // between updates the bar creeps a little ahead so it never looks frozen.
        let applyProgressTimer = null;
        function setApplyProgress(pct, status, elapsed) {
            document.getElementById('apply-progress-bar').style.width = `${pct}%`;
            document.getElementById('apply-progress-pct').textContent = `${Math.round(pct)}%`;
            if (status) document.getElementById('apply-progress-status').textContent = status;
            if (elapsed !== undefined) document.getElementById('apply-progress-time').textContent = `${Math.round(elapsed)}s`;
        }
        function startApplyProgress() {
            let shown = 0;
            setApplyProgress(0, 'Starting...', 0);
            document.getElementById('apply-progress').classList.remove('hidden');
            applyProgressTimer = setInterval(async () => {
                try {
                    let p = await (await fetch('/api/apply_progress')).json();
                    if (!p.active) return;
                    // Ease toward the next stage boundary while the server is busy in one stage
                    let ceiling = Math.min(p.percent + 5, 99);
                    shown = Math.max(shown, p.percent);
                    shown += (ceiling - shown) * 0.05;
                    setApplyProgress(shown, p.status, p.elapsed);
                } catch (e) { /* keep polling */ }
            }, 400);
        }
        function stopApplyProgress() {
            clearInterval(applyProgressTimer);
            applyProgressTimer = null;
            document.getElementById('apply-progress').classList.add('hidden');
        }

        async function applySettings() {
            if (isApplying) return;
            
            let sessionPath = document.getElementById('session-path-input').value.trim();
            let modelName = document.getElementById('model-select').value;
            let scalerName = document.getElementById('scaler-select').value;
            
            if (!sessionPath) {
                alert("Please enter a Suite2p session folder path.");
                return;
            }
            isApplying = true;
            
            startApplyProgress();
            try {
                let url = `/api/change_settings?session_path=${encodeURIComponent(sessionPath)}&model_name=${encodeURIComponent(modelName)}&scaler_name=${encodeURIComponent(scalerName)}`;
                let res = await fetch(url);
                let data = await res.json();
                if (data.error) {
                    alert("Error: " + data.error);
                } else {
                    if (data.active_model_info) {
                        renderModelCard(data.active_model_info);
                    }
                    setApplyProgress(100, 'Done');
                    stopApplyProgress();
                    await loadSessionInfo();
                    if (data.iscell_saved) {
                        console.log("Updated iscell.npy saved to session directory for Suite2p GUI.");
                    }
                }
            } catch(e) {
                alert("Error applying settings: " + e);
            } finally {
                stopApplyProgress();
                isApplying = false;
            }
        }

        async function loadSessionInfo() {
            showLoader(true);
            try {
                let res = await fetch('/api/session_info');
                let data = await res.json();
                
                if (data.loaded) {
                    sessionInfo = data;
                    
                    document.getElementById('session-name-display').textContent = data.session_name;
                    document.getElementById('total-rois-display').textContent = data.total_rois;
                    document.getElementById('gt-cells-display').textContent = data.num_cells_gt;
                    
                    let gtPathEl = document.getElementById('gt-path-display');
                    let gtWarningBanner = document.getElementById('gt-warning-banner');
                    if (gtPathEl) {
                        gtPathEl.textContent = data.gt_path || "None";
                        gtPathEl.title = data.gt_path || "No curated GT file loaded.";
                    }
                    if (gtWarningBanner) {
                        if (data.gt_path === "None" || !data.gt_path) {
                            gtWarningBanner.classList.remove('hidden');
                        } else {
                            gtWarningBanner.classList.add('hidden');
                        }
                    }
                    
                    let modelName = document.getElementById('model-select').value;
                    document.getElementById('navigation-active-model').textContent = `Using Model: ${modelName}`;
                    document.getElementById('detected-features-display').textContent = data.num_features;
                    
                    activeCategory = null;
                    if (typeof updateCategoryButtonStyles === 'function') {
                        updateCategoryButtonStyles();
                    }
                    recalculateMetricsAndCategories();
                    
                    document.getElementById('no-session-card').classList.add('hidden');
                    document.getElementById('dashboard-content').classList.remove('hidden');
                    
                    let firstCell = 0;
                    if (sessionInfo.categories.false_positives.length > 0) {
                        firstCell = sessionInfo.categories.false_positives[0];
                    } else if (sessionInfo.categories.uncertain.length > 0) {
                        firstCell = sessionInfo.categories.uncertain[0];
                    }
                    loadCell(firstCell);
                }
            } catch(e) {
                console.error("Error loading session info:", e);
                alert("Error loading session info: " + e);
            } finally {
                showLoader(false);
            }
        }

        async function loadCell(cellIdx) {
            if (cellIdx < 0 || cellIdx >= sessionInfo.total_rois) {
                alert(`Cell index ${cellIdx} is out of bounds.`);
                return;
            }
            
            showLoader(true);
            activeCellIdx = cellIdx;
            document.getElementById('cell-idx-input').value = cellIdx;
            
            try {
                let sessionPath = document.getElementById('session-path-input').value.trim();
                let modelName = document.getElementById('model-select').value;
                let scalerName = document.getElementById('scaler-select').value;
                
                let url = `/api/explain_cell?cell_idx=${cellIdx}`;
                if (sessionPath) url += `&session_path=${encodeURIComponent(sessionPath)}`;
                if (modelName) url += `&model_name=${encodeURIComponent(modelName)}`;
                if (scalerName) url += `&scaler_name=${encodeURIComponent(scalerName)}`;
                
                let res = await fetch(url);
                let data = await res.json();
                
                if (data.error) {
                    alert("Error explaining cell: " + data.error);
                    return;
                }
                
                currentCellExplanation = data;
                
                let probPerc = Math.round(data.probability * 100);
                document.getElementById('cell-prob-value').textContent = `${probPerc}%`;
                
                let gtBadge = document.getElementById('cell-gt-badge');
                if (data.label === 1) {
                    gtBadge.className = "text-sm font-bold text-emerald-400";
                    gtBadge.textContent = "Cell (1)";
                } else {
                    gtBadge.className = "text-sm font-bold text-rose-400";
                    gtBadge.textContent = "Non-Cell (0)";
                }
                
                updateCellOutcomeUI();
                
                let selectEl = document.getElementById('attr-method-select');
                let alertEl = document.getElementById('shap-not-avail-alert');
                
                if (data.has_shap) {
                    selectEl.options[0].disabled = false;
                    selectEl.options[2].disabled = false;
                    alertEl.classList.add('hidden');
                    if (selectEl.value !== 'ablation' && selectEl.value !== 'comparison') {
                        selectEl.value = 'shap';
                    }
                } else {
                    selectEl.options[0].disabled = true;
                    selectEl.options[2].disabled = true;
                    selectEl.value = 'ablation';
                    alertEl.classList.remove('hidden');
                }
                
                updateAttributionChart();
                plotCalciumTrace(data.trace);
                drawSpatialROI(data.roi);
                renderFeaturesTable(data.attributions);
                if (typeof plotSessionProbabilities === 'function') {
                    plotSessionProbabilities();
                }
            } catch(e) {
                console.error("Error loading cell data:", e);
                alert("Error loading cell data: " + e);
            } finally {
                showLoader(false);
            }
        }

        let activeCategory = null;

        function updateCategoryButtonStyles() {
            const categories = ['true_positives', 'false_positives', 'false_negatives', 'uncertain'];
            const hasGT = sessionInfo && sessionInfo.gt_path !== "None" && sessionInfo.gt_path !== "";
            
            const normalClasses = hasGT ? {
                'true_positives': 'bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border-emerald-500/10',
                'false_positives': 'bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border-rose-500/10',
                'false_negatives': 'bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border-amber-500/10',
                'uncertain': 'bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 border-blue-500/10'
            } : {
                'true_positives': 'bg-gray-500/10 hover:bg-gray-500/20 text-gray-400 border-gray-500/10',
                'false_positives': 'bg-gray-500/10 hover:bg-gray-500/20 text-gray-300 border-gray-500/10',
                'false_negatives': 'bg-gray-500/10 hover:bg-gray-500/20 text-gray-400 border-gray-500/10',
                'uncertain': 'bg-gray-500/10 hover:bg-gray-500/20 text-gray-300 border-gray-500/10'
            };
            const activeClasses = hasGT ? {
                'true_positives': 'bg-emerald-500 text-white border-emerald-400 shadow-md shadow-emerald-500/20',
                'false_positives': 'bg-rose-500 text-white border-rose-400 shadow-md shadow-rose-500/20',
                'false_negatives': 'bg-amber-500 text-white border-amber-400 shadow-md shadow-amber-500/20',
                'uncertain': 'bg-blue-500 text-white border-blue-400 shadow-md shadow-blue-500/20'
            } : {
                'true_positives': 'bg-gray-600 text-white border-gray-500 shadow-md shadow-gray-600/20',
                'false_positives': 'bg-gray-600 text-white border-gray-500 shadow-md shadow-gray-600/20',
                'false_negatives': 'bg-gray-600 text-white border-gray-500 shadow-md shadow-gray-600/20',
                'uncertain': 'bg-gray-600 text-white border-gray-500 shadow-md shadow-gray-600/20'
            };

            categories.forEach(cat => {
                const btn = document.getElementById(`btn-${cat}`);
                if (btn) {
                    btn.className = "text-xs px-3 py-1.5 rounded-lg border flex items-center gap-1.5 transition " + 
                                    (activeCategory === cat ? activeClasses[cat] : normalClasses[cat]);
                }
            });

            const clearBtn = document.getElementById('btn-clear-filter');
            if (clearBtn) {
                if (activeCategory) {
                    clearBtn.classList.remove('hidden');
                } else {
                    clearBtn.classList.add('hidden');
                }
            }

            const indicator = document.getElementById('navigation-active-category');
            if (indicator) {
                if (activeCategory) {
                    let displayName = activeCategory.replace('_', ' ');
                    if (!hasGT && activeCategory === 'false_positives') {
                        displayName = 'Predicted Cells';
                    }
                    displayName = displayName.charAt(0).toUpperCase() + displayName.slice(1);
                    indicator.textContent = `Browsing: ${displayName}`;
                    indicator.classList.remove('hidden');
                } else {
                    indicator.classList.add('hidden');
                }
            }
        }

        function clearCategoryFilter() {
            activeCategory = null;
            updateCategoryButtonStyles();
            updateCategoryAnalysisUI();
        }

        function navigateCell(step) {
            if (activeCategory) {
                let cells = sessionInfo.categories[activeCategory];
                if (cells && cells.length > 0) {
                    let currentPos = cells.indexOf(activeCellIdx);
                    if (currentPos !== -1) {
                        let nextPos = currentPos + step;
                        if (nextPos >= 0 && nextPos < cells.length) {
                            loadCell(cells[nextPos]);
                        } else {
                            alert(`Reached the end of the ${activeCategory.replace('_', ' ')} list.`);
                        }
                    } else {
                        loadCell(cells[0]);
                    }
                }
            } else {
                let nextCell = activeCellIdx + step;
                if (nextCell >= 0 && nextCell < sessionInfo.total_rois) {
                    loadCell(nextCell);
                }
            }
        }

        function jumpCategory(cat) {
            let cells = sessionInfo.categories[cat];
            if (!cells || cells.length === 0) {
                alert(`No ROIs in category: ${cat}`);
                return;
            }
            
            if (activeCategory === cat) {
                activeCategory = null;
                updateCategoryButtonStyles();
                updateCategoryAnalysisUI();
                return;
            }
            
            activeCategory = cat;
            updateCategoryButtonStyles();
            updateCategoryAnalysisUI();
            loadCell(cells[0]);
            
            let fmKey = '';
            if (cat === 'true_positives') fmKey = 'TP';
            else if (cat === 'false_positives') fmKey = 'FP';
            else if (cat === 'false_negatives') fmKey = 'FN';
            else if (cat === 'true_negatives') fmKey = 'TN';
            
            if (fmKey) {
                openFailureModeModal(fmKey);
            }
        }

        function updateAttributionChart() {
            if (!currentCellExplanation) return;
            
            let method = document.getElementById('attr-method-select').value;
            let attrs = currentCellExplanation.attributions;
            
            let traces = [];
            
            if (method === 'shap' && currentCellExplanation.has_shap) {
                let sorted = [...attrs].sort((a, b) => a.shap_attribution - b.shap_attribution);
                let yNames = sorted.map(a => a.name);
                let xVals = sorted.map(a => a.shap_attribution);
                let colors = sorted.map(a => a.shap_attribution >= 0 ? '#10b981' : '#f43f5e');
                
                traces.push({
                    type: 'bar',
                    x: xVals,
                    y: yNames,
                    orientation: 'h',
                    name: 'TreeSHAP Impact',
                    marker: { color: colors },
                    hovertemplate: '<b>%{y}</b><br>SHAP Impact: %{x:+.4f} probability<extra></extra>'
                });
            } else if (method === 'ablation') {
                let sorted = [...attrs].sort((a, b) => a.ablation_attribution - b.ablation_attribution);
                let yNames = sorted.map(a => a.name);
                let xVals = sorted.map(a => a.ablation_attribution);
                let colors = sorted.map(a => a.ablation_attribution >= 0 ? '#3b82f6' : '#a855f7');
                
                traces.push({
                    type: 'bar',
                    x: xVals,
                    y: yNames,
                    orientation: 'h',
                    name: 'Ablation Impact',
                    marker: { color: colors },
                    hovertemplate: '<b>%{y}</b><br>Ablation Impact: %{x:+.4f} probability<extra></extra>'
                });
            } else if (method === 'comparison' && currentCellExplanation.has_shap) {
                let sorted = [...attrs].sort((a, b) => {
                    let maxA = Math.max(Math.abs(a.shap_attribution), Math.abs(a.ablation_attribution));
                    let maxB = Math.max(Math.abs(b.shap_attribution), Math.abs(b.ablation_attribution));
                    return a.shap_attribution - b.shap_attribution;
                });
                
                let yNames = sorted.map(a => a.name);
                let shapVals = sorted.map(a => a.shap_attribution);
                let ablatVals = sorted.map(a => a.ablation_attribution);
                
                traces.push({
                    type: 'bar',
                    x: shapVals,
                    y: yNames,
                    orientation: 'h',
                    name: 'TreeSHAP Value',
                    marker: { color: '#10b981' },
                    hovertemplate: '<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>'
                });
                
                traces.push({
                    type: 'bar',
                    x: ablatVals,
                    y: yNames,
                    orientation: 'h',
                    name: 'Ablation Value',
                    marker: { color: '#3b82f6' },
                    hovertemplate: '<b>%{y}</b><br>Ablation: %{x:+.4f}<extra></extra>'
                });
            }
            
            let layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                height: 580,
                margin: { l: 120, r: 20, t: 10, b: 40 },
                barmode: 'group',
                xaxis: {
                    title: 'Probability Contribution Impact',
                    gridcolor: '#1f2d47',
                    zerolinecolor: '#3b82f6',
                    zerolinewidth: 2,
                    tickfont: { color: '#9ca3af' },
                    titlefont: { color: '#9ca3af', size: 12 }
                },
                yaxis: {
                    tickmode: 'linear',
                    dtick: 1,
                    tickfont: { color: '#f3f4f6', size: 10 },
                    automargin: true
                },
                legend: {
                    font: { color: '#9ca3af', size: 10 },
                    orientation: 'h',
                    y: 1.1,
                    x: 0.5,
                    xanchor: 'center'
                },
                hoverlabel: { bgcolor: '#151d30', font: { color: '#f3f4f6' } }
            };
            
            Plotly.newPlot('attribution-chart', traces, layout, { responsive: true, displayModeBar: false });
        }

        function plotCalciumTrace(trace) {
            let fTrace = {
                x: trace.time,
                y: trace.f,
                name: 'Raw F',
                type: 'scatter',
                line: { color: '#f43f5e', width: 1 },
                opacity: 0.6
            };
            
            let fneuTrace = {
                x: trace.time,
                y: trace.fneu,
                name: 'Neuropil Fneu',
                type: 'scatter',
                line: { color: '#a855f7', width: 1 },
                opacity: 0.5
            };
            
            let fcorrTrace = {
                x: trace.time,
                y: trace.fcorr,
                name: 'Corrected F_corr',
                type: 'scatter',
                line: { color: '#3b82f6', width: 1.8 }
            };
            
            let spksTrace = {
                x: trace.time,
                y: trace.spks,
                name: 'Deconvolved Spikes',
                type: 'scatter',
                yaxis: 'y2',
                fill: 'tozeroy',
                line: { color: '#10b981', width: 1 },
                opacity: 0.7
            };
            
            let layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 50, r: 50, t: 20, b: 40 },
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: 1.15,
                    x: 0.5,
                    xanchor: 'center',
                    font: { color: '#9ca3af', size: 10 }
                },
                xaxis: {
                    title: 'Time (Frames)',
                    gridcolor: '#1f2d47',
                    tickfont: { color: '#9ca3af' },
                    titlefont: { color: '#9ca3af', size: 12 }
                },
                yaxis: {
                    title: 'Fluorescence Intensity',
                    gridcolor: '#1f2d47',
                    tickfont: { color: '#9ca3af' },
                    titlefont: { color: '#9ca3af', size: 12 }
                },
                yaxis2: {
                    title: 'Deconvolved Activity (spks)',
                    overlaying: 'y',
                    side: 'right',
                    showgrid: false,
                    tickfont: { color: '#10b981' },
                    titlefont: { color: '#10b981', size: 12 }
                },
                hovermode: 'x unified',
                hoverlabel: { bgcolor: '#151d30', font: { color: '#f3f4f6' } }
            };
            
            Plotly.newPlot('trace-plot', [fTrace, fneuTrace, fcorrTrace, spksTrace], layout, { responsive: true });
        }

        function plotSessionProbabilities() {
            if (!sessionInfo || !sessionInfo.loaded || !sessionInfo.probabilities) return;

            const probs = sessionInfo.probabilities;
            const gt = sessionInfo.ground_truth;
            const hasGT = sessionInfo.gt_path !== "None" && sessionInfo.gt_path !== "";
            const threshold = activeThreshold;
            const N = sessionInfo.total_rois;

            const hideThresholdEl = document.getElementById('hide-threshold-toggle');
            const hideThreshold = hideThresholdEl ? hideThresholdEl.checked : false;

            // Arrays to hold data for traces
            let tp_x = [], tp_y = [], tp_text = [];
            let fp_x = [], fp_y = [], fp_text = [];
            let tn_x = [], tn_y = [], tn_text = [];
            let fn_x = [], fn_y = [], fn_text = [];

            let gt_cell_x = [], gt_cell_y = [], gt_cell_text = [];
            let gt_nocell_x = [], gt_nocell_y = [], gt_nocell_text = [];

            let pred_cell_x = [], pred_cell_y = [], pred_cell_text = [];
            let pred_noise_x = [], pred_noise_y = [], pred_noise_text = [];

            for (let i = 0; i < N; i++) {
                const p = probs[i];
                const g = gt[i];
                const pred = (p >= threshold) ? 1 : 0;

                const textLabel = `ROI ${i}<br>Prob: ${(p * 100).toFixed(1)}%<br>GT: ${hasGT ? (g === 1 ? 'Cell' : 'Non-Cell') : 'N/A'}`;

                if (hasGT) {
                    if (hideThreshold) {
                        if (g === 1) {
                            gt_cell_x.push(i);
                            gt_cell_y.push(p);
                            gt_cell_text.push(textLabel);
                        } else {
                            gt_nocell_x.push(i);
                            gt_nocell_y.push(p);
                            gt_nocell_text.push(textLabel);
                        }
                    } else {
                        if (pred === 1 && g === 1) {
                            tp_x.push(i);
                            tp_y.push(p);
                            tp_text.push(textLabel);
                        } else if (pred === 1 && g === 0) {
                            fp_x.push(i);
                            fp_y.push(p);
                            fp_text.push(textLabel);
                        } else if (pred === 0 && g === 0) {
                            tn_x.push(i);
                            tn_y.push(p);
                            tn_text.push(textLabel);
                        } else if (pred === 0 && g === 1) {
                            fn_x.push(i);
                            fn_y.push(p);
                            fn_text.push(textLabel);
                        }
                    }
                } else {
                    if (pred === 1) {
                        pred_cell_x.push(i);
                        pred_cell_y.push(p);
                        pred_cell_text.push(textLabel);
                    } else {
                        pred_noise_x.push(i);
                        pred_noise_y.push(p);
                        pred_noise_text.push(textLabel);
                    }
                }
            }

            let scatterTraces = [];

            if (hasGT) {
                if (hideThreshold) {
                    scatterTraces.push({
                        x: gt_cell_x, y: gt_cell_y, text: gt_cell_text,
                        name: 'Ground Truth: Cell', type: 'scatter', mode: 'markers',
                        marker: { color: '#10b981', size: 6, opacity: 0.8 },
                        hoverinfo: 'text'
                    });
                    scatterTraces.push({
                        x: gt_nocell_x, y: gt_nocell_y, text: gt_nocell_text,
                        name: 'Ground Truth: Non-Cell', type: 'scatter', mode: 'markers',
                        marker: { color: '#f43f5e', size: 6, opacity: 0.8 },
                        hoverinfo: 'text'
                    });
                } else {
                    scatterTraces.push({
                        x: tp_x, y: tp_y, text: tp_text,
                        name: 'True Positives', type: 'scatter', mode: 'markers',
                        marker: { color: '#10b981', size: 6, opacity: 0.8 },
                        hoverinfo: 'text'
                    });
                    scatterTraces.push({
                        x: fp_x, y: fp_y, text: fp_text,
                        name: 'False Positives', type: 'scatter', mode: 'markers',
                        marker: { color: '#f43f5e', size: 6, opacity: 0.8 },
                        hoverinfo: 'text'
                    });
                    scatterTraces.push({
                        x: fn_x, y: fn_y, text: fn_text,
                        name: 'False Negatives', type: 'scatter', mode: 'markers',
                        marker: { color: '#f59e0b', size: 6, opacity: 0.8 },
                        hoverinfo: 'text'
                    });
                    scatterTraces.push({
                        x: tn_x, y: tn_y, text: tn_text,
                        name: 'True Negatives', type: 'scatter', mode: 'markers',
                        marker: { color: '#64748b', size: 5, opacity: 0.4 },
                        hoverinfo: 'text'
                    });
                }
            } else {
                scatterTraces.push({
                    x: pred_cell_x, y: pred_cell_y, text: pred_cell_text,
                    name: 'ROIs', type: 'scatter', mode: 'markers',
                    marker: { color: '#9ca3af', size: 6, opacity: 0.8 },
                    hoverinfo: 'text'
                });
                scatterTraces.push({
                    x: pred_noise_x, y: pred_noise_y, text: pred_noise_text,
                    name: 'ROIs (Suppressed/Noise)', type: 'scatter', mode: 'markers',
                    marker: { color: '#4b5563', size: 5, opacity: 0.4 },
                    hoverinfo: 'text'
                });
            }

            // Highlight the active ROI
            if (activeCellIdx >= 0 && activeCellIdx < N) {
                const activeProb = probs[activeCellIdx];
                scatterTraces.push({
                    x: [activeCellIdx],
                    y: [activeProb],
                    text: [`ACTIVE ROI ${activeCellIdx}<br>Prob: ${(activeProb * 100).toFixed(1)}%`],
                    name: 'Active ROI',
                    type: 'scatter',
                    mode: 'markers',
                    marker: {
                        color: '#fbbf24',
                        size: 12,
                        line: { color: '#ffffff', width: 2 },
                        symbol: 'star'
                    },
                    hoverinfo: 'text',
                    showlegend: true
                });
            }

            const shapes = [];
            if (!hideThreshold) {
                shapes.push({
                    type: 'line',
                    xref: 'paper',
                    yref: 'y',
                    x0: 0,
                    y0: threshold,
                    x1: 1,
                    y1: threshold,
                    line: {
                        color: '#fbbf24',
                        width: 1.5,
                        dash: 'dash'
                    }
                });
            }

            const scatterLayout = {
                autosize: true,
                height: 450,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 50, r: 20, t: 15, b: 40 },
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: 1.1,
                    x: 0.5,
                    xanchor: 'center',
                    font: { color: '#9ca3af', size: 10 }
                },
                xaxis: {
                    title: 'ROI Index',
                    gridcolor: '#1f2d47',
                    tickfont: { color: '#9ca3af' },
                    titlefont: { color: '#9ca3af', size: 11 }
                },
                yaxis: {
                    title: 'AI Probability',
                    range: [-0.02, 1.02],
                    gridcolor: '#1f2d47',
                    tickfont: { color: '#9ca3af' },
                    titlefont: { color: '#9ca3af', size: 11 }
                },
                hovermode: 'closest',
                clickmode: 'event+select',
                shapes: shapes
            };

            Plotly.react('probability-scatter-plot', scatterTraces, scatterLayout, { responsive: true, displayModeBar: false });

            // Attach click event for selecting ROI
            const scatterDiv = document.getElementById('probability-scatter-plot');
            if (scatterDiv && !scatterDiv.dataset.clickBound) {
                scatterDiv.on('plotly_click', function(data) {
                    if (data.points && data.points.length > 0) {
                        let point = data.points[0];
                        let roiIdx = point.x;
                        if (typeof roiIdx === 'number' && roiIdx >= 0 && roiIdx < N) {
                            loadCell(roiIdx);
                        }
                    }
                });
                scatterDiv.dataset.clickBound = 'true';
            }
        }

        function drawSpatialROI(roi) {
            let container = document.getElementById('roi-spatial-plot');
            container.innerHTML = '';
            
            let points = roi.points;
            let bbox = roi.bbox;
            
            if (points.length === 0) {
                container.innerHTML = '<span class="text-xs text-gray-500">No pixel mask data</span>';
                return;
            }
            
            let canvas = document.createElement('canvas');
            canvas.className = "w-full h-full max-w-xs max-h-xs object-contain";
            canvas.width = bbox.xmax - bbox.xmin;
            canvas.height = bbox.ymax - bbox.ymin;
            
            let ctx = canvas.getContext('2d');
            
            ctx.fillStyle = '#0b0f19';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.strokeStyle = '#151d30';
            ctx.lineWidth = 1;
            for (let i = 0; i < canvas.width; i += 10) {
                ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke();
            }
            for (let i = 0; i < canvas.height; i += 10) {
                ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke();
            }
            
            ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
            points.forEach(pt => {
                let rx = pt[0] - bbox.xmin;
                let ry = pt[1] - bbox.ymin;
                ctx.fillRect(rx, ry, 1.2, 1.2);
            });
            
            ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
            
            container.appendChild(canvas);
        }

        function renderFeaturesTable(attrs) {
            let body = document.getElementById('features-table-body');
            body.innerHTML = '';
            
            let sorted = [...attrs].sort((a, b) => {
                let maxA = Math.max(
                    Math.abs(a.ablation_attribution),
                    a.shap_attribution !== null ? Math.abs(a.shap_attribution) : 0
                );
                let maxB = Math.max(
                    Math.abs(b.ablation_attribution),
                    b.shap_attribution !== null ? Math.abs(b.shap_attribution) : 0
                );
                return maxB - maxA;
            });
            
            sorted.forEach(a => {
                let row = document.createElement('tr');
                row.className = "hover:bg-brand-cardBg/30 transition duration-150";
                
                let shapText = 'N/A';
                let shapColorClass = 'text-gray-600 italic';
                if (a.shap_attribution !== null) {
                    let sign = a.shap_attribution > 0 ? '+' : '';
                    shapText = `${sign}${a.shap_attribution.toFixed(4)}`;
                    if (Math.abs(a.shap_attribution) < 0.0001) shapText = '0.0000';
                    shapColorClass = a.shap_attribution > 0.005 ? 'text-emerald-400 font-bold' : (a.shap_attribution < -0.005 ? 'text-rose-400 font-bold' : 'text-gray-500');
                }
                
                let ablatSign = a.ablation_attribution > 0 ? '+' : '';
                let ablatText = `${ablatSign}${a.ablation_attribution.toFixed(4)}`;
                if (Math.abs(a.ablation_attribution) < 0.0001) ablatText = '0.0000';
                let ablatColorClass = a.ablation_attribution > 0.005 ? 'text-blue-400 font-bold' : (a.ablation_attribution < -0.005 ? 'text-purple-400 font-bold' : 'text-gray-500');
                
                row.innerHTML = `
                    <td class="px-4 py-3 font-semibold text-gray-200">${a.name}</td>
                    <td class="px-4 py-3 text-right ${shapColorClass}">${shapText}</td>
                    <td class="px-4 py-3 text-right ${ablatColorClass}">${ablatText}</td>
                    <td class="px-4 py-3 text-right text-gray-300 font-mono">${formatNumber(a.value)}</td>
                    <td class="px-4 py-3 text-right text-gray-400 font-mono">${formatNumber(a.mean_dataset)}</td>
                    <td class="px-4 py-3 text-right text-emerald-500/80 font-mono">${formatNumber(a.mean_cells)}</td>
                    <td class="px-4 py-3 text-right text-rose-500/80 font-mono">${formatNumber(a.mean_noncells)}</td>
                    <td class="px-4 py-3 text-xs text-gray-400 max-w-[280px]">${a.desc}</td>
                `;
                
                body.appendChild(row);
            });
        }

        function formatNumber(val) {
            if (val === undefined || val === null || isNaN(val)) return '-';
            if (Math.abs(val) < 0.001 && val !== 0) return val.toExponential(2);
            if (val % 1 !== 0) return val.toFixed(3);
            return val.toLocaleString();
        }

        let categoryAnalysisData = null;

        function fetchCategoryAnalysis() {
            if (!sessionInfo || !sessionInfo.loaded) return;
            
            fetch(`/api/category_analysis?threshold=${activeThreshold}`)
                .then(response => {
                    if (!response.ok) throw new Error("Server error fetching category analysis");
                    return response.json();
                })
                .then(data => {
                    categoryAnalysisData = data;
                    updateCategoryAnalysisUI();
                })
                .catch(err => {
                    console.error("Error fetching category analysis:", err);
                });
        }

        function updateCategoryAnalysisUI() {
            let container = document.getElementById('category-feature-impact');
            if (!container) return;
            
            if (!sessionInfo || !sessionInfo.loaded) {
                container.innerHTML = '<span class="text-gray-500 italic">No session loaded.</span>';
                return;
            }
            
            if (!activeCategory) {
                container.innerHTML = '<span class="text-gray-500 italic">Select a category above (e.g. False Positives) to analyze its distinct feature profile...</span>';
                return;
            }
            
            let catKey = '';
            if (activeCategory === 'true_positives') catKey = 'TP';
            else if (activeCategory === 'false_positives') catKey = 'FP';
            else if (activeCategory === 'false_negatives') catKey = 'FN';
            else if (activeCategory === 'true_negatives') catKey = 'TN';
            else if (activeCategory === 'uncertain') {
                container.innerHTML = '<span class="text-gray-400 font-semibold">Uncertain category:</span> <span class="text-gray-400">ROIs with AI Confidence between 15% and 85%. No single feature profile determines uncertainty.</span>';
                return;
            }
            
            if (!categoryAnalysisData || !categoryAnalysisData[catKey]) {
                container.innerHTML = '<span class="text-gray-500 italic">No analysis data available.</span>';
                return;
            }
            
            let features = categoryAnalysisData[catKey];
            if (features.length === 0) {
                container.innerHTML = '<span class="text-gray-500 italic">No cells in this category for analysis.</span>';
                return;
            }
            
            // Separate into positive z_score (high) and negative z_score (low)
            let highFeatures = features.filter(f => f.z_score > 0.15).slice(0, 3);
            let lowFeatures = features.filter(f => f.z_score < -0.15).slice(0, 3);
            
            // Format category name for display
            let catNameFriendly = catKey === 'TP' ? 'True Positives' :
                                  catKey === 'FP' ? 'False Positives' :
                                  catKey === 'FN' ? 'False Negatives' : 'True Negatives';
                                  
            let badgeClass = catKey === 'TP' ? 'text-emerald-400' :
                             catKey === 'FP' ? 'text-rose-400' :
                             catKey === 'FN' ? 'text-amber-400' : 'text-gray-400';
                             
            let html = `<div>
                <div class="font-bold mb-1.5 flex items-center justify-between">
                    <span class="${badgeClass}">${catNameFriendly} Profile</span>
                    <button onclick="openFailureModeModal('${catKey}')" class="text-[9px] bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 border border-blue-500/30 px-1.5 py-0.5 rounded transition font-medium flex items-center gap-1">
                        <i class="fa-solid fa-circle-nodes"></i> Analyze Failure Modes
                    </button>
                </div>`;
                
            if (highFeatures.length === 0 && lowFeatures.length === 0) {
                html += `<div class="text-gray-400 italic">Features are close to the dataset average.</div>`;
            } else {
                if (highFeatures.length > 0) {
                    html += `<div class="mb-1.5">
                        <div class="text-emerald-400/90 font-semibold text-[10px] uppercase tracking-wider mb-0.5">Unusually High / Contributing:</div>
                        <div class="space-y-1 pl-1">`;
                    highFeatures.forEach(f => {
                        html += `<div class="flex justify-between items-center text-gray-300">
                            <span>• <code class="text-blue-300 font-mono text-[11px]">${f.feature}</code></span>
                            <span class="font-semibold text-emerald-400 font-mono">+${f.z_score.toFixed(2)} σ</span>
                        </div>`;
                    });
                    html += `</div></div>`;
                }
                
                if (lowFeatures.length > 0) {
                    html += `<div>
                        <div class="text-rose-400/90 font-semibold text-[10px] uppercase tracking-wider mb-0.5">Unusually Low / Absent:</div>
                        <div class="space-y-1 pl-1">`;
                    lowFeatures.forEach(f => {
                        html += `<div class="flex justify-between items-center text-gray-300">
                            <span>• <code class="text-blue-300 font-mono text-[11px]">${f.feature}</code></span>
                            <span class="font-semibold text-rose-400 font-mono">${f.z_score.toFixed(2)} σ</span>
                        </div>`;
                    });
                    html += `</div></div>`;
                }
            }
            
            // Add a smart diagnostic description
            let summaryDesc = "";
            if (catKey === 'FP') {
                summaryDesc = "AI classified these as cells due to high morphology/activity scores, but manual curation marked them as artifacts. Check spatial outlines and peak widths.";
            } else if (catKey === 'FN') {
                summaryDesc = "AI missed these cells (marked as noise) likely due to lower activity levels or low SNR. Check if threshold tuning or model retraining is required.";
            } else if (catKey === 'TP') {
                summaryDesc = "Clear, correct classifications showing high activity, temporal SNR, and typical biological morphology.";
            }
            
            if (summaryDesc) {
                html += `<div class="pt-2 border-t border-brand-border/30 mt-2 text-[10px] text-gray-400 leading-relaxed">${summaryDesc}</div>`;
            }
            
            html += `</div>`;
            container.innerHTML = html;
        }

        function setComparisonROIA(idx) {
            let inputA = document.getElementById('compare-roi-a');
            if (inputA) {
                inputA.value = idx;
                // Highlight to show success
                inputA.classList.add('border-blue-500');
                setTimeout(() => inputA.classList.remove('border-blue-500'), 800);
            }
        }

        function compareROIs() {
            let roiA = parseInt(document.getElementById('compare-roi-a').value);
            let roiB = parseInt(document.getElementById('compare-roi-b').value);
            
            let container = document.getElementById('comparison-results-container');
            if (!container) return;
            
            if (isNaN(roiA) || isNaN(roiB)) {
                container.innerHTML = '<span class="text-rose-400 font-semibold">Please enter valid integer ROI indices.</span>';
                return;
            }
            
            container.innerHTML = `
                <div class="flex items-center justify-center gap-3 py-4">
                    <div class="h-6 w-6 border-2 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
                    <span class="text-xs text-gray-400 font-medium">Comparing ROI ${roiA} and ROI ${roiB}...</span>
                </div>`;
                
            fetch(`/api/compare_cells?cell_a=${roiA}&cell_b=${roiB}`)
                .then(response => {
                    if (!response.ok) throw new Error("Server error comparing cells");
                    return response.json();
                })
                .then(data => {
                    let dist = data.distance_px;
                    let corrF = data.corr_f;
                    let corrFcorr = data.corr_fcorr;
                    let overlapPix = data.intersection_pixels;
                    let iou = data.iou;
                    
                    // Style alerts based on similarity thresholds
                    let isDuplicate = dist <= 15.0 && (corrF >= 0.7 || corrFcorr >= 0.7) && overlapPix > 0;
                    
                    let statusHtml = '';
                    if (isDuplicate) {
                        statusHtml = `<span class="px-2 py-0.5 text-[10px] font-bold uppercase rounded bg-rose-500/10 border border-rose-500/30 text-rose-400 animate-pulse">⚠️ Duplicate Candidate</span>`;
                    } else {
                        statusHtml = `<span class="px-2 py-0.5 text-[10px] font-bold uppercase rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">✅ Distinct ROIs</span>`;
                    }
                    
                    container.innerHTML = `
                        <div class="space-y-4">
                            <div class="flex items-center justify-between border-b border-brand-border/40 pb-2">
                                <span class="font-bold text-sm text-white flex items-center gap-2">
                                    <i class="fa-solid fa-code-compare text-blue-400"></i> ROI ${roiA} vs ROI ${roiB} Comparison
                                </span>
                                ${statusHtml}
                            </div>
                            
                            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                                <div class="bg-brand-darkBg/60 p-3 rounded-xl border border-brand-border/30 text-center">
                                    <div class="text-[10px] text-gray-400 uppercase font-semibold mb-1">Centroid Distance</div>
                                    <div class="text-lg font-bold ${dist <= 15.0 ? 'text-rose-400' : 'text-gray-200'}">${dist.toFixed(1)} px</div>
                                    <div class="text-[9px] text-gray-500 mt-0.5">${dist <= 15.0 ? 'Close (< 15px)' : 'Far'}</div>
                                </div>
                                <div class="bg-brand-darkBg/60 p-3 rounded-xl border border-brand-border/30 text-center">
                                    <div class="text-[10px] text-gray-400 uppercase font-semibold mb-1">Raw Trace Corr</div>
                                    <div class="text-lg font-bold ${corrF >= 0.7 ? 'text-rose-400' : 'text-emerald-400'}">${corrF.toFixed(3)}</div>
                                    <div class="text-[9px] text-gray-500 mt-0.5">${corrF >= 0.7 ? 'High Correlation' : 'Low Correlation'}</div>
                                </div>
                                <div class="bg-brand-darkBg/60 p-3 rounded-xl border border-brand-border/30 text-center">
                                    <div class="text-[10px] text-gray-400 uppercase font-semibold mb-1">Corrected Trace Corr</div>
                                    <div class="text-lg font-bold ${corrFcorr >= 0.7 ? 'text-rose-400' : 'text-emerald-400'}">${corrFcorr.toFixed(3)}</div>
                                    <div class="text-[9px] text-gray-500 mt-0.5">${corrFcorr >= 0.7 ? 'High Correlation' : 'Low Correlation'}</div>
                                </div>
                                <div class="bg-brand-darkBg/60 p-3 rounded-xl border border-brand-border/30 text-center">
                                    <div class="text-[10px] text-gray-400 uppercase font-semibold mb-1">Pixel Overlap</div>
                                    <div class="text-lg font-bold ${overlapPix > 0 ? 'text-rose-400' : 'text-gray-200'}">${overlapPix} px</div>
                                    <div class="text-[9px] text-gray-500 mt-0.5">IoU: ${iou.toFixed(3)}</div>
                                </div>
                            </div>
                            
                            <p class="text-[11px] text-gray-400 leading-normal bg-brand-darkBg/30 p-2.5 rounded-lg border border-brand-border/20">
                                <strong>Diagnostic:</strong> ${isDuplicate ? 
                                    `These ROIs are within 15 pixels, share overlapping pixels, and have a trace correlation &ge; 0.70. Under active duplicate suppression, only the one with the higher probability is kept.` : 
                                    `These ROIs do not meet the duplication criteria (either far apart, non-overlapping, or trace correlation &lt; 0.70) and will both be classified independently.`}
                            </p>
                        </div>`;
                })
                .catch(err => {
                    container.innerHTML = `<span class="text-rose-400 font-semibold">Error comparing cells: ${err.message}</span>`;
                });
        }

        // ==========================================
        // FAILURE MODE CLUSTERING INTERACTION
        // ==========================================
        let failureModeData = null;
        let activeClusterId = 0;

        function openFailureModeModal(category) {
            let modal = document.getElementById('failure-mode-modal');
            let title = document.getElementById('fm-modal-title');
            
            let catName = category === 'TP' ? 'True Positives' :
                          category === 'FP' ? 'False Positives' :
                          category === 'FN' ? 'False Negatives' : 'True Negatives';
            
            title.textContent = `${catName} Failure Mode Analysis`;
            modal.classList.remove('hidden');
            
            document.getElementById('fm-cluster-tabs').innerHTML = '<div class="text-gray-500 italic text-xs py-4 text-center"><i class="fa-solid fa-spinner fa-spin text-blue-500 text-lg mb-2 block"></i>clustering data...</div>';
            document.getElementById('fm-cluster-name').textContent = 'Analyzing cohort...';
            document.getElementById('fm-cluster-desc').textContent = 'Running K-Means clustering on SHAP vectors...';
            document.getElementById('fm-cluster-chart').innerHTML = '<span class="text-gray-500 italic text-xs">Computing feature attributions...</span>';
            document.getElementById('fm-cluster-cells').innerHTML = '';
            document.getElementById('fm-comparison-table-body').innerHTML = '';
            
            fetch(`/api/failure_modes?category=${category}&threshold=${activeThreshold}`)
                .then(response => {
                    if (!response.ok) throw new Error("Backend error or non-tree model loaded");
                    return response.json();
                })
                .then(data => {
                    failureModeData = data;
                    if (!data.clusters || data.clusters.length === 0) {
                        document.getElementById('fm-cluster-tabs').innerHTML = '<span class="text-gray-500 italic text-xs">No cells in this category.</span>';
                        document.getElementById('fm-cluster-name').textContent = 'No failure modes detected';
                        document.getElementById('fm-cluster-desc').textContent = 'This cohort has no samples under the current threshold.';
                        document.getElementById('fm-cluster-chart').innerHTML = '';
                        return;
                    }
                    
                    renderFailureModeSidebar();
                    selectCluster(0);
                })
                .catch(err => {
                    console.error("Error fetching failure modes:", err);
                    document.getElementById('fm-cluster-tabs').innerHTML = `<span class="text-rose-400 italic text-xs">Error: ${err.message}</span>`;
                    document.getElementById('fm-cluster-name').textContent = 'Analysis Unavailable';
                    document.getElementById('fm-cluster-desc').textContent = 'This analysis is only supported for tree-based models (like LightGBM) that natively output TreeSHAP values. MLP models do not support fast batch SHAP.';
                    document.getElementById('fm-cluster-chart').innerHTML = '';
                });
        }

        function closeFailureModeModal() {
            document.getElementById('failure-mode-modal').classList.add('hidden');
        }

        function renderFailureModeSidebar() {
            let container = document.getElementById('fm-cluster-tabs');
            container.innerHTML = '';
            
            failureModeData.clusters.forEach((c, idx) => {
                let tab = document.createElement('button');
                tab.onclick = () => selectCluster(idx);
                tab.id = `fm-tab-${idx}`;
                tab.className = `w-full text-left p-3.5 rounded-xl border flex flex-col gap-1 transition duration-200`;
                
                if (idx === 0) {
                    tab.classList.add('bg-blue-600/10', 'border-blue-500/30', 'text-blue-400');
                } else {
                    tab.className += ' bg-brand-darkBg/50 border-brand-border/40 hover:border-brand-border text-gray-300';
                }
                
                let categoryLabel = failureModeData.category;
                let outcomeWord = categoryLabel === 'FP' ? 'False Alarms' : categoryLabel === 'FN' ? 'Missed Cells' : 'Cells';
                
                tab.innerHTML = `
                    <div class="font-semibold text-xs flex justify-between items-center w-full">
                        <span class="truncate max-w-[150px]">${c.label}</span>
                        <span class="px-2 py-0.5 rounded bg-brand-border/60 text-[9px] text-gray-400 font-mono">${c.percentage.toFixed(0)}%</span>
                    </div>
                    <div class="text-[10px] text-gray-400">${c.size} ${outcomeWord}</div>
                `;
                container.appendChild(tab);
            });
        }

        function selectCluster(clusterIdx) {
            activeClusterId = clusterIdx;
            
            let tabsContainer = document.getElementById('fm-cluster-tabs');
            let tabs = tabsContainer.getElementsByTagName('button');
            for (let i = 0; i < tabs.length; i++) {
                if (i === clusterIdx) {
                    tabs[i].className = "w-full text-left p-3.5 rounded-xl border flex flex-col gap-1 transition duration-200 bg-blue-600/10 border-blue-500/30 text-blue-400";
                } else {
                    tabs[i].className = "w-full text-left p-3.5 rounded-xl border flex flex-col gap-1 transition duration-200 bg-brand-darkBg/50 border-brand-border/40 hover:border-brand-border text-gray-300";
                }
            }
            
            let c = failureModeData.clusters[clusterIdx];
            let categoryLabel = failureModeData.category;
            
            document.getElementById('fm-cluster-badge').textContent = `FAILURE MODE ${clusterIdx + 1} • ${c.percentage.toFixed(0)}% OF COHORT`;
            document.getElementById('fm-cluster-name').textContent = c.label;
            
            let descText = "";
            let outcome = categoryLabel === 'FP' ? 'False Positive (False Alarm)' :
                          categoryLabel === 'FN' ? 'False Negative (Missed Cell)' : 'Correct';
            
            if (c.label.includes("Neuropil")) {
                descText = `This group of ${c.size} cells represents a ${outcome} mode characterized by high neuropil background correlation. The model was misled by background fluorescence transients, mistaking them for genuine calcium spikes in the cell body. **Actionable fix:** Consider adjusting the neuropil subtraction coefficient or trace filters.`;
            } else if (c.label.includes("Index")) {
                descText = `This group of ${c.size} cells is dominated by their ROI index prior. The model learned to over-weight cells at the start or end of the Suite2p list due to session-level sorting biases. **Actionable fix:** Consider dropping the index prior or training on random index orders to remove index bias.`;
            } else if (c.label.includes("Morphological")) {
                descText = `This group of ${c.size} cells are "morphological clones". They have perfect roundness, solidity, and size, but almost zero actual calcium activity. The model was tricked purely by static image properties. **Actionable fix:** Consider reducing the weights of shape features or restricting tree depth to avoid pure-morphology decisions.`;
            } else if (c.label.includes("Transient")) {
                descText = `This group of ${c.size} cells is driven by transient activity/skewness metrics. The traces show high variance or peak density, but the shape or morphology is atypical. **Actionable fix:** Check if these are active dendrites or scan artifacts rather than healthy cell bodies.`;
            } else {
                descText = `This group of ${c.size} cells shows a mixed error profile driven primarily by feature '${c.features[0].name.replace('_', ' ')}'. It represents a distinct decision boundary subspace where the model's feature weights align in this specific configuration.`;
            }
            document.getElementById('fm-cluster-desc').textContent = descText;
            
            let cellsContainer = document.getElementById('fm-cluster-cells');
            cellsContainer.innerHTML = '';
            
            c.representative_cells.forEach(cell => {
                let cellBtn = document.createElement('button');
                cellBtn.onclick = () => {
                    closeFailureModeModal();
                    loadCell(cell.idx);
                };
                cellBtn.className = "w-full flex justify-between items-center px-3 py-2 rounded-lg bg-brand-darkBg/60 border border-brand-border/40 hover:bg-brand-border/20 text-gray-300 hover:text-white transition text-xs";
                cellBtn.innerHTML = `
                    <span class="font-mono">ROI ${cell.idx}</span>
                    <span class="font-semibold text-blue-400 font-mono">${(cell.prob * 100).toFixed(0)}% AI Conf</span>
                `;
                cellsContainer.appendChild(cellBtn);
            });
            
            let tableBody = document.getElementById('fm-comparison-table-body');
            tableBody.innerHTML = '';
            c.features.slice(0, 5).forEach(f => {
                let row = document.createElement('tr');
                row.className = "border-b border-brand-border/10 hover:bg-brand-darkBg/20 text-xs";
                row.innerHTML = `
                    <td class="py-2 font-mono text-gray-200 font-medium">${f.name}</td>
                    <td class="py-2 text-gray-400 max-w-[250px] truncate" title="${f.desc}">${f.desc}</td>
                    <td class="py-2 text-right font-mono font-semibold text-emerald-400">${formatNumber(f.mean_raw)}</td>
                    <td class="py-2 text-right font-mono text-gray-300">${formatNumber(f.mean_tp)}</td>
                    <td class="py-2 text-right font-mono text-gray-400">${formatNumber(f.mean_dataset)}</td>
                `;
                tableBody.appendChild(row);
            });
            
            renderPlotlyClusterChart(c.features.slice(0, 8));
        }

        function renderPlotlyClusterChart(features) {
            let chartDiv = document.getElementById('fm-cluster-chart');
            chartDiv.innerHTML = '';
            
            let names = features.map(f => f.name).reverse();
            let values = features.map(f => f.mean_attribution).reverse();
            
            let colors = values.map(v => v >= 0 ? '#10b981' : '#f43f5e');
            
            let data = [{
                type: 'bar',
                x: values,
                y: names,
                orientation: 'h',
                marker: {
                    color: colors,
                    line: { width: 0 }
                },
                text: values.map(v => (v >= 0 ? '+' : '') + v.toFixed(3)),
                textposition: 'inside',
                insidetextanchor: 'middle',
                insidetextfont: {
                    family: 'Outfit, sans-serif',
                    size: 10,
                    color: '#ffffff'
                }
            }];
            
            let layout = {
                autosize: true,
                margin: { l: 150, r: 20, t: 15, b: 35 },
                xaxis: {
                    gridcolor: '#1f2d47',
                    zerolinecolor: '#3b82f6',
                    tickfont: { family: 'Outfit, sans-serif', size: 10, color: '#9ca3af' },
                    title: { text: 'AI Confidence Probability Shift', font: { family: 'Outfit, sans-serif', size: 10, color: '#9ca3af' } }
                },
                yaxis: {
                    tickfont: { family: 'Outfit, sans-serif', size: 10, color: '#d1d5db' },
                    gridcolor: 'transparent'
                },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                showlegend: false,
                hovermode: false
            };
            
            let config = { responsive: true, displayModeBar: false };
            Plotly.newPlot(chartDiv, data, layout, config);
        }
    </script>

    <!-- FAILURE MODE MODAL -->
    <div id="failure-mode-modal" class="fixed inset-0 bg-black/60 backdrop-blur-md z-50 flex items-center justify-center p-4 sm:p-6 md:p-8 hidden">
        <div class="bg-brand-cardBg border border-brand-border/80 w-full max-w-5xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
            <!-- Modal Header -->
            <div class="px-6 py-4 border-b border-brand-border/60 flex justify-between items-center bg-brand-darkBg/50">
                <div>
                    <h2 class="text-lg font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-circle-nodes text-blue-500"></i>
                        <span id="fm-modal-title">Failure Mode Analysis</span>
                    </h2>
                    <p class="text-xs text-gray-400 mt-0.5">SHAP-based clustering breakdown of model decisions</p>
                </div>
                <button onclick="closeFailureModeModal()" class="text-gray-400 hover:text-white text-xl p-1.5 hover:bg-brand-border/40 rounded-lg transition duration-200">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>
            
            <!-- Modal Body -->
            <div class="flex-1 overflow-y-auto p-6 space-y-6 flex flex-col md:flex-row gap-6">
                <!-- Left Sidebar: Cluster Selectors & Worst Offenders -->
                <div class="w-full md:w-1/3 space-y-4 flex flex-col">
                    <div class="space-y-2">
                        <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Failure Modes Found:</div>
                        <div id="fm-cluster-tabs" class="space-y-2 max-h-[30vh] overflow-y-auto">
                            <!-- Dynamically filled -->
                        </div>
                    </div>
                    <div class="space-y-2 border-t border-brand-border/40 pt-4">
                        <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Worst Offending Cells:</div>
                        <div id="fm-cluster-cells" class="space-y-2">
                            <!-- Dynamically filled -->
                        </div>
                    </div>
                </div>
                
                <!-- Right Content Panel -->
                <div class="flex-1 space-y-6 flex flex-col">
                    <!-- Cluster Description / Diagnostic -->
                    <div class="bg-brand-darkBg/40 border border-brand-border/30 p-4 rounded-xl space-y-2">
                        <div class="text-[10px] uppercase font-semibold tracking-wider text-blue-400" id="fm-cluster-badge">Failure Mode Profile</div>
                        <h3 class="text-sm font-bold text-gray-100" id="fm-cluster-name">-</h3>
                        <p class="text-xs text-gray-300 leading-relaxed" id="fm-cluster-desc">-</p>
                    </div>
                    
                    <!-- Plotly Chart Container -->
                    <div class="space-y-2">
                        <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Feature Impact Fingerprint (SHAP values):</div>
                        <div id="fm-cluster-chart" class="w-full h-[280px] bg-brand-darkBg/50 rounded-xl border border-brand-border/30 overflow-hidden">
                            <!-- Let Plotly draw directly without centering constraints -->
                        </div>
                    </div>
                    
                    <!-- Physical Feature Comparison Table (Full Width) -->
                    <div class="bg-brand-darkBg/30 p-4 rounded-xl border border-brand-border/20 space-y-3">
                        <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Feature Physical Comparison:</div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-xs text-left text-gray-300">
                                <thead>
                                    <tr class="border-b border-brand-border/30 text-gray-400 font-semibold">
                                        <th class="py-2">Feature Name</th>
                                        <th class="py-2">Description</th>
                                        <th class="py-2 text-right">Cluster Mean</th>
                                        <th class="py-2 text-right">True Positive Mean</th>
                                        <th class="py-2 text-right">Dataset Mean</th>
                                    </tr>
                                </thead>
                                <tbody id="fm-comparison-table-body">
                                    <!-- Dynamically filled -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Modal Footer -->
            <div class="px-6 py-4 border-t border-brand-border/40 bg-brand-darkBg/30 flex justify-end">
                <button onclick="closeFailureModeModal()" class="px-4 py-2 bg-brand-border hover:bg-brand-border/80 border border-brand-border/60 rounded-xl text-xs text-gray-200 transition duration-200 font-medium">
                    Close Analysis
                </button>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# 6. APPLICATION STARTPOINT
# ==========================================

def run_server(port=5000):
    # Determine default model to load
    dir_path = BASE_DIR
    
    # Pre-load best LGB if available, else MLP
    models = [str(f.relative_to(dir_path)) for f in dir_path.rglob("*.pkl") if "scaler" not in f.name and not any("venv" in p for p in f.parts)]
    
    default_model = None
    best_model_path = "models/suite2p_best_lgb.pkl"
    if (dir_path / best_model_path).exists():
        default_model = best_model_path
    elif "suite2p_best_lgb.pkl" in [Path(m).name for m in models]:
        for m in models:
            if Path(m).name == "suite2p_best_lgb.pkl":
                default_model = m
                break
    elif len(models) > 0:
        default_model = models[0]
        
    if default_model and not state.model_path:
        try:
            state.load_model(default_model)
        except Exception as e:
            print(f"Error pre-loading default model: {e}")
            
    # Auto-load first suggested session if available if none specified
    # if not state.session_path:
    #     suggested_sessions = [
    #         "/mnt/other_ubunthu/mnt/data/1-ordered/Stav1",
    #         "/mnt/other_ubunthu/mnt/data/4-ordered/Stav4",
    #         "/mnt/other_ubunthu/mnt/data/6-ordered/Stav6"
    #     ]
    #     for s in suggested_sessions:
    #         if os.path.exists(s):
    #             try:
    #                 # state.load_session(s)
    #                 break
    #             except Exception as e:
    #                 print(f"Error loading initial session {s}: {e}")
                
    # Threaded: browsers (Chrome) open idle pre-connections that would block a single-threaded
    # server; heavy session/model work is still serialized by state.lock
    server = ThreadingHTTPServer(('localhost', port), DashHandler)
    server.daemon_threads = True
    print(f"\n==================================================================")
    print(f"  Suite2p AI Decision Explainer Server is running!")
    print(f"  --> Local Address: http://localhost:{port}")
    print(f"==================================================================\n")
    
    # Automatically open in default web browser
    webbrowser.open(f"http://localhost:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Suite2p Cell Classification Decision Investigation Tool")
    parser.add_argument('--session', type=str, help="Initial Suite2p session path to load")
    parser.add_argument('--model', type=str, help="Initial model .pkl path to load")
    parser.add_argument('--scaler', type=str, help="Initial scaler .pkl path to load")
    parser.add_argument('--port', type=int, default=5000, help="Port to run server on")
    args = parser.parse_args()
    
    if args.model:
        state.load_model(args.model, args.scaler)
        
    if args.session:
        state.load_session(args.session)
        
    run_server(args.port)
