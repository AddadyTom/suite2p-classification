"""
MiniRocket embeddings of the 1000-bin noise-scaled traces (prepare_trace_arrays.py),
per outer fold of evaluate_cv.py repeat 0, reduced to --pca dims.

MiniRocket is unsupervised, but its biases (and the PCA) are still fit on the
outer-training sessions only. Kernels come from minirocket_embedder.MiniRocketExtractor;
the convolutions don't depend on the fold, so each batch is convolved once and
the PPV features are computed against every fold's biases (the max features are
bias-free and shared). Both trace channels (bin mean, bin max) are used.
Output layout (folds, n_rois, pca) matches the CNN/fusion embeddings.

    PYTHONPATH=. python scripts/minirocket_features.py --traces .../traces --out .../minirocket/minirocket_pca32.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minirocket_embedder import MiniRocketExtractor
from scripts.evaluate_cv import load_tables
from scripts.train_fusion_oof import load_per_session


@torch.no_grad()
def transform_all_folds(ext, bias_sets, traces, batch=512):
    """Returns (max_feats (n, D*K), [ppv_feats (n, D*K) per fold]) as float16."""
    n = len(traces)
    D, K = len(ext.dilations), ext.num_kernels
    w = torch.tensor(ext.base_patterns[:, None, :], dtype=torch.float32)
    biases = [torch.tensor(b, dtype=torch.float32) for b in bias_sets]  # each (D, K)
    mx = np.zeros((n, D * K), dtype=np.float16)
    ppv = [np.zeros((n, D * K), dtype=np.float16) for _ in bias_sets]
    for i in range(0, n, batch):
        x = torch.tensor(traces[i:i + batch, None, :], dtype=torch.float32)
        for d_idx, d in enumerate(ext.dilations):
            conv = F.conv1d(x, w, dilation=d, padding=d * 4)  # (B, K, T)
            sl = slice(d_idx * K, (d_idx + 1) * K)
            mx[i:i + batch, sl] = conv.amax(-1).numpy()
            for f, b in enumerate(biases):
                ppv[f][i:i + batch, sl] = (conv > b[d_idx, :, None]).float().mean(-1).numpy()
    return mx, ppv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--traces', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--pca', type=int, default=32)
    ap.add_argument('--kernels', type=int, default=84)
    ap.add_argument('--dilations', type=int, default=8)
    ap.add_argument('--bias-sample', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    X_tab, y, g, _ = load_tables(args.tables, ['stav22', 'stav3/21'])
    traces = load_per_session(args.traces, 'traces', g, y)  # (n, 2, 1000) float16
    splits = list(GroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(X_tab, y, g))

    feats = [[] for _ in range(args.folds)]  # per fold: list of channel blocks
    for ch in range(traces.shape[1]):
        t0 = time.time()
        ext = MiniRocketExtractor(num_kernels=args.kernels, max_dilations=args.dilations, seed=42 + ch)
        bias_sets = []
        for k, (tr, _) in enumerate(splits):
            np.random.seed(1000 * ch + k)  # fit_biases draws quantiles from the global RNG
            sample = np.random.default_rng(k).choice(tr, size=min(args.bias_sample, len(tr)), replace=False)
            ext.fit_biases(traces[sample, ch].astype(np.float32))
            bias_sets.append(ext.biases.copy())
        mx, ppv = transform_all_folds(ext, bias_sets, traces[:, ch].astype(np.float32))
        for k in range(args.folds):
            feats[k].append(np.hstack([ppv[k], mx]))
        print(f"channel {ch}: {mx.shape[1] * 2} features x {args.folds} folds ({time.time() - t0:.0f}s)", flush=True)
        del mx, ppv

    out = np.full((args.folds, len(y), args.pca), np.nan, dtype=np.float32)
    for k, (tr, _) in enumerate(splits):
        Z = np.hstack(feats[k]).astype(np.float32)
        feats[k] = None
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
        Z = (Z - mu) / sd
        pca = PCA(n_components=args.pca, random_state=0).fit(Z[tr])
        out[k] = pca.transform(Z)
        print(f"fold {k}: PCA {args.pca} explains {pca.explained_variance_ratio_.sum():.2f} of variance", flush=True)
        del Z
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, emb=out, y=y, groups=g, folds_done=args.folds)


if __name__ == '__main__':
    main()
