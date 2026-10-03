"""
Fixed-length trace input for the fusion network, one .npz per session.

Per ROI: F_corr = F - 0.7 * Fneu, expressed in noise units
(F_corr - median) / sigma, with sigma = 1.4826 * MAD(diff) / sqrt(2) (the
q*_over_noise normalization). The trace is split into N_BINS equal bins and
two channels are kept per bin: the bin mean (slow structure) and the bin max
(short transients that a mean would wash out). Values are clipped to
[-10, 100] and divided by 10.

Sessions and ROI order follow the feature tables (same as the CNN crops).

    PYTHONPATH=. python scripts/prepare_trace_arrays.py --out /mnt/other_ubunthu/suite2p_work/traces
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_cv import load_tables

N_BINS = 1000


def session_traces(session_path, chunk=500):
    F = np.load(Path(session_path) / 'F.npy', mmap_mode='r')
    Fneu = np.load(Path(session_path) / 'Fneu.npy', mmap_mode='r')
    n, T = F.shape
    edges = np.linspace(0, T, N_BINS + 1).astype(int)
    out = np.zeros((n, 2, N_BINS), dtype=np.float16)
    for i in range(0, n, chunk):
        fc = np.asarray(F[i:i + chunk], dtype=np.float32) - 0.7 * np.asarray(Fneu[i:i + chunk], dtype=np.float32)
        med = np.median(fc, axis=1, keepdims=True)
        d = np.diff(fc, axis=1)
        mad = np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1, keepdims=True)
        z = (fc - med) / np.maximum(1.4826 * mad / np.sqrt(2), 1e-6)
        mean = np.add.reduceat(z, edges[:-1], axis=1) / np.diff(edges)
        mx = np.maximum.reduceat(z, edges[:-1], axis=1)
        out[i:i + chunk, 0] = np.clip(mean, -10, 100) / 10
        out[i:i + chunk, 1] = np.clip(mx, -10, 100) / 10
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
        tr = session_traces(sess)
        assert len(tr) == (g == sess).sum(), sess
        np.savez(f, traces=tr, y=y[g == sess].astype(np.int8), session=sess)
        print(f"{Path(sess).name}: {tr.shape}", flush=True)


if __name__ == '__main__':
    main()
