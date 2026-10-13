"""
Pilot: does adding ROICaT ROInet embeddings help, on the few sessions embedded so far?

Leave-one-session-out over the sessions that have embeddings. Nested protocol as
in evaluate_cv.py (inner GroupKFold picks n_estimators and threshold). The
embeddings are reduced with PCA fitted on the training sessions only.

    PYTHONPATH=. python scripts/roicat_pilot_cv.py --emb /mnt/other_ubunthu/suite2p_work/roicat/emb
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES
from scripts.evaluate_cv import fit_nested, load_tables, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--emb', required=True)
    ap.add_argument('--pca', type=int, default=16)
    args = ap.parse_args()

    X_all, y, g, names = load_tables(args.tables, ['stav22', 'stav3/21'])
    col = {n: i for i, n in enumerate(names)}
    emb = {}
    for f in Path(args.emb).glob('*.npz'):
        d = np.load(f)
        emb[str(d['session'])] = d['latents']
    keep = np.isin(g, list(emb))
    X_all, y, g = X_all[keep], y[keep], g[keep]
    E = np.vstack([emb[s] for s in dict.fromkeys(g)])
    assert len(E) == len(y)
    sessions = list(dict.fromkeys(g))
    print(f"{len(sessions)} sessions: {[Path(s).name for s in sessions]}, {len(y)} ROIs")

    # Row-order check: if embedding rows line up with stat.npy ROIs, some embedding
    # direction must track ROI size (npix) within every session; misaligned rows give ~0.
    from scipy.stats import spearmanr
    npix = X_all[:, col['area_to_radius_sq']] * X_all[:, col['radius']] ** 2
    for s in sessions:
        k = g == s
        pcs = PCA(n_components=8, random_state=0).fit_transform(E[k])
        rho = max(abs(spearmanr(pcs[:, j], npix[k])[0]) for j in range(8))
        print(f"  alignment {Path(s).name}: max |spearman(PC, npix)| = {rho:.2f}")

    sets = {
        'baseline (27)': (ACTIVE_FEATURES, False),
        'baseline + ROICaT': (ACTIVE_FEATURES, True),
        'baseline + image': (ACTIVE_FEATURES + IMAGE_FEATURES, False),
        'baseline + image + ROICaT': (ACTIVE_FEATURES + IMAGE_FEATURES, True),
        'CONTROL: image + ROICaT, shuffled train labels': (ACTIVE_FEATURES + IMAGE_FEATURES, True),
    }
    rng = np.random.default_rng(0)
    res = {k: [] for k in sets}
    for s in sessions:
        tr, va = np.where(g != s)[0], np.where(g == s)[0]
        pca = PCA(n_components=args.pca, random_state=0).fit(E[tr])
        Ep = pca.transform(E)
        for name, (feats, use_emb) in sets.items():
            X = X_all[:, [col[f] for f in feats]]
            if use_emb:
                X = np.hstack([X, Ep])
            ytr = rng.permutation(y[tr]) if name.startswith('CONTROL') else y[tr]
            m, thr, _ = fit_nested(X[tr], ytr, g[tr])
            res[name].append(scores(y[va], m.predict_proba(X[va])[:, 1] >= thr))
        print(Path(s).name, {k: round(v[-1][2], 4) for k, v in res.items()}, flush=True)

    print("\n| Feature set | Precision | Recall | F1 (mean ± std over held-out sessions) |\n|---|---|---|---|")
    base = np.array(res['baseline (27)'])[:, 2]
    for name, rows in res.items():
        a = np.array(rows)
        d = a[:, 2] - base
        print(f"| {name} | {a[:, 0].mean():.4f} | {a[:, 1].mean():.4f} | "
              f"{a[:, 2].mean():.4f} ± {a[:, 2].std():.4f} (Δ {d.mean():+.4f}, {(d > 0).sum()}/{len(d)} up) |")


if __name__ == '__main__':
    main()
