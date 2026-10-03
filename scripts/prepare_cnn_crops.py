"""
Extract aligned per-ROI image crops for the CNN, one .npz per session.

Channels (C=4): meanImg, max_proj, Vcorr, lam mask. max_proj and Vcorr are
cropped by Suite2p to ops['yrange'] x ops['xrange']; they are placed back into
the full frame at that offset (not resized), so all channels line up with
stat['ypix'/'xpix']. Each image channel is scaled per session by its 1st/99th
percentile; the mask channel is lam / max(lam). Pixels outside the frame or
crop are 0.

Sessions and labels come from the feature tables (same label priority,
de-duplication and exclusions as evaluate_cv.py), so ROI order matches the CV.

    PYTHONPATH=. python scripts/prepare_cnn_crops.py --tables feature_tables \
        --out /mnt/other_ubunthu/suite2p_work/crops
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.image_features import _full_frame, ops_alignment_score, MIN_ALIGNMENT
from scripts.evaluate_cv import load_tables

CROP = 32


def session_crops(session_path):
    ops = np.load(Path(session_path) / 'ops.npy', allow_pickle=True).item()
    stat = np.load(Path(session_path) / 'stat.npy', allow_pickle=True)
    if ops_alignment_score(ops, stat) < MIN_ALIGNMENT:
        raise ValueError(f"ops.npy does not match stat.npy in {session_path}")
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    chans = []
    for im in ['meanImg', 'max_proj', 'Vcorr']:
        f = _full_frame(ops[im], ops, cropped=im != 'meanImg')
        lo, hi = np.nanpercentile(f, [1, 99])
        chans.append(np.nan_to_num((f - lo) / max(hi - lo, 1e-6), nan=0.0))
    h = CROP // 2
    pad = np.pad(np.stack(chans), ((0, 0), (h, h), (h, h)))
    out = np.zeros((len(stat), 4, CROP, CROP), dtype=np.float16)
    for i, s in enumerate(stat):
        cy, cx = int(s['med'][0]), int(s['med'][1])
        out[i, :3] = pad[:, cy:cy + CROP, cx:cx + CROP]
        ys, xs = np.asarray(s['ypix']) - cy + h, np.asarray(s['xpix']) - cx + h
        ok = (ys >= 0) & (ys < CROP) & (xs >= 0) & (xs < CROP)
        lam = np.asarray(s['lam'], dtype=np.float32)
        out[i, 3, ys[ok], xs[ok]] = lam[ok] / max(lam.max(), 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21'])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, y, g, _ = load_tables(args.tables, args.exclude)
    for sess in dict.fromkeys(g):
        f = out / (Path(sess).parent.name + '__' + Path(sess).name + '.npz')
        if f.exists():
            continue
        crops = session_crops(sess)
        ys = y[g == sess]
        assert len(crops) == len(ys), sess
        np.savez(f, crops=crops, y=ys.astype(np.int8), session=sess)
        print(f"{Path(sess).name}: {crops.shape}", flush=True)


if __name__ == '__main__':
    main()
