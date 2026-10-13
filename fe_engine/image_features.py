"""
Per-ROI image features computed from the Suite2p summary images in ops.npy.

For every ROI we compare the mask against a thin surrounding ring and measure
how well the mask weights (stat['lam']) match the local image structure:

  * <img>_contrast_z     (lam-weighted mean inside mask - ring mean) / ring std
  * <img>_lam_corr       Pearson r(lam, image) over the mask pixels
  * <img>_lam_corr_patch Pearson r(lam image, image) over mask + ring
                         (lam = 0 in the ring), i.e. shape *and* contrast match
  * <img>_ratio          mask mean / ring mean (for non-negative images)

Images: meanImg, meanImgE (full frame), max_proj and Vcorr. The latter two are
cropped by Suite2p to ops['yrange'] x ops['xrange'], so their pixel indices are
shifted and anything outside the crop is treated as missing.
"""
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter

RING_INNER = 2.0   # px gap between mask edge and ring (avoids blurred soma edge)
RING_OUTER = 7.0   # px outer radius of the ring, measured from the mask edge

IMAGES = ['meanImg', 'meanImgE', 'max_proj', 'Vcorr']
RATIO_IMAGES = ['meanImg', 'max_proj']

# Below this label-free alignment score the summary images in ops.npy evidently
# do not belong to the ROIs in stat.npy (e.g. Stav1, whose ops.npy comes from a
# different run), and all image features are set to NaN for the session.
MIN_ALIGNMENT = 0.15

IMAGE_FEATURES = (
    [f'{im}_contrast_z' for im in IMAGES]
    + [f'{im}_lam_corr' for im in IMAGES]
    + [f'{im}_lam_corr_patch' for im in IMAGES]
    + [f'{im}_ratio' for im in RATIO_IMAGES]
    + ['Vcorr_in_mean', 'Vcorr_in_minus_ring', 'meanImg_neuropil_contrast']
)


def _full_frame(img, ops, cropped):
    """Embed a (possibly cropped) image into a full Ly x Lx frame, NaN outside the crop."""
    img = np.asarray(img, dtype=np.float32)
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    if img.shape == (Ly, Lx):
        return img
    if not cropped:
        raise ValueError(f"unexpected image shape {img.shape} for Ly,Lx=({Ly},{Lx})")
    out = np.full((Ly, Lx), np.nan, dtype=np.float32)
    y0, y1 = ops['yrange']
    x0, x1 = ops['xrange']
    out[y0:y1, x0:x1] = img[:y1 - y0, :x1 - x0]
    return out


def _pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def ops_alignment_score(ops, stat):
    """
    Label-free check that ops.npy images match stat.npy: mean high-passed,
    z-scored Vcorr (or meanImg) inside the union of all ROI masks minus outside.
    Matching sessions here score 0.27-0.6+; Stav1 (mismatched) scores 0.05.
    """
    im = 'Vcorr' if 'Vcorr' in ops and np.size(ops['Vcorr']) > 1 else 'meanImg'
    frame = _full_frame(ops[im], ops, cropped=im == 'Vcorr')
    valid = np.isfinite(frame)
    f = np.where(valid, frame, np.nanmedian(frame)).astype(np.float64)
    f = f - gaussian_filter(f, 8)
    f = (f - f[valid].mean()) / (f[valid].std() + 1e-9)
    roi = np.zeros(frame.shape, dtype=bool)
    for s in stat:
        roi[s['ypix'], s['xpix']] = True
    return float(f[roi & valid].mean() - f[~roi & valid].mean())


def compute_image_features(session_path, stat=None, ops=None):
    """Return {feature_name: (n_rois,) float32 array} for one Suite2p plane folder."""
    session_path = Path(session_path)
    if ops is None:
        ops = np.load(session_path / 'ops.npy', allow_pickle=True).item()
    if stat is None:
        stat = np.load(session_path / 'stat.npy', allow_pickle=True)

    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    frames = {}
    for im in IMAGES:
        if im in ops and np.size(ops[im]) > 1:
            frames[im] = _full_frame(ops[im], ops, cropped=im in ('max_proj', 'Vcorr'))

    n = len(stat)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in IMAGE_FEATURES}
    score = ops_alignment_score(ops, stat)
    if score < MIN_ALIGNMENT:
        print(f"WARNING: ops.npy images do not match stat.npy in {session_path} "
              f"(alignment {score:.2f} < {MIN_ALIGNMENT}); image features set to NaN.")
        return out
    pad = int(np.ceil(RING_OUTER)) + 1

    for i, s in enumerate(stat):
        ypix, xpix = np.asarray(s['ypix']), np.asarray(s['xpix'])
        lam = np.asarray(s['lam'], dtype=np.float32)
        if len(ypix) == 0:
            continue
        y0, y1 = max(ypix.min() - pad, 0), min(ypix.max() + pad + 1, Ly)
        x0, x1 = max(xpix.min() - pad, 0), min(xpix.max() + pad + 1, Lx)

        lam_patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        lam_patch[ypix - y0, xpix - x0] = lam
        mask = lam_patch > 0
        dist = distance_transform_edt(~mask)
        ring = (dist > RING_INNER) & (dist <= RING_OUTER)
        support = mask | ring
        w = lam / lam.sum() if lam.sum() > 0 else np.full(len(lam), 1.0 / len(lam), np.float32)

        for im, frame in frames.items():
            patch = frame[y0:y1, x0:x1]
            vals_in = patch[ypix - y0, xpix - x0]
            vals_ring = patch[ring]
            vals_ring = vals_ring[np.isfinite(vals_ring)]
            ok_in = np.isfinite(vals_in)
            if ok_in.sum() == 0 or len(vals_ring) < 3:
                continue
            in_mean = float(np.sum(vals_in[ok_in] * w[ok_in]) / np.sum(w[ok_in]))
            ring_mean = float(vals_ring.mean())
            ring_std = float(vals_ring.std())

            out[f'{im}_contrast_z'][i] = (in_mean - ring_mean) / max(ring_std, 1e-6)
            out[f'{im}_lam_corr'][i] = _pearson(lam, vals_in)
            out[f'{im}_lam_corr_patch'][i] = _pearson(lam_patch[support], patch[support])
            if im in RATIO_IMAGES:
                out[f'{im}_ratio'][i] = in_mean / ring_mean if ring_mean > 0 else np.nan
            if im == 'Vcorr':
                out['Vcorr_in_mean'][i] = in_mean
                out['Vcorr_in_minus_ring'][i] = in_mean - ring_mean

        # Contrast against Suite2p's own neuropil mask (excludes other ROIs' pixels)
        if 'meanImg' in frames and 'neuropil_mask' in s and len(s['neuropil_mask']) > 0:
            mi = frames['meanImg']
            neu = mi.ravel()[np.asarray(s['neuropil_mask'])]
            in_vals = mi[ypix, xpix]
            in_mean = float(np.sum(in_vals * w))
            out['meanImg_neuropil_contrast'][i] = (in_mean - neu.mean()) / max(float(neu.std()), 1e-6)

    return out
