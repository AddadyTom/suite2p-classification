"""
Does the CNN score add to LightGBM? Stack the out-of-fold CNN probability
(train_cnn_oof.py) as one extra feature, on the same repeat-0 session folds and
with the same nested threshold selection as evaluate_cv.py.

For outer fold k the CNN column is cnn[k]: inner-OOF scores on training rows,
outer-model scores on validation rows, so no CNN score was fit on its own label.

    PYTHONPATH=. python scripts/cnn_stack_cv.py --cnn /mnt/other_ubunthu/suite2p_work/cnn/cnn_oof.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES
from scripts.evaluate_cv import best_threshold, fit_nested, load_tables, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--cnn', required=True)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    X_all, y, g, names = load_tables(args.tables, ['stav22', 'stav3/21'])
    col = {n: i for i, n in enumerate(names)}
    d = np.load(args.cnn, allow_pickle=True)
    cnn, done = d['cnn'], int(d['folds_done'])
    assert (d['y'] == y).all() and (d['groups'] == g).all(), "CNN rows do not match the CV rows"

    sets = {
        'baseline (27)': (ACTIVE_FEATURES, False),
        'baseline + CNN': (ACTIVE_FEATURES, True),
        '+ image (all)': (ACTIVE_FEATURES + IMAGE_FEATURES, False),
        '+ image (all) + CNN': (ACTIVE_FEATURES + IMAGE_FEATURES, True),
    }
    res = {k: [] for k in sets}
    cnn_alone = []
    splits = list(GroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(X_all, y, g))
    for k, (tr, va) in enumerate(splits[:done]):
        c = cnn[k][:, None]
        assert not np.isnan(c).any()
        # CNN alone: threshold from its inner-OOF scores on the training rows
        t = best_threshold(y[tr], c[tr, 0])
        cnn_alone.append(scores(y[va], c[va, 0] >= t))
        for name, (feats, use_cnn) in sets.items():
            X = X_all[:, [col[f] for f in feats]]
            if use_cnn:
                X = np.hstack([X, c])
            m, thr, _ = fit_nested(X[tr], y[tr], g[tr])
            p = m.predict_proba(X[va])[:, 1]
            res[name].append(scores(y[va], p >= thr) + (roc_auc_score(y[va], p),))
        print(f"fold {k}: CNN alone F1={cnn_alone[-1][2]:.4f} AUC={roc_auc_score(y[va], c[va, 0]):.4f} | "
              + " | ".join(f"{n}: F1={v[-1][2]:.4f} AUC={v[-1][3]:.4f}" for n, v in res.items()), flush=True)

    print(f"\n{done} folds (repeat 0)\n| Model | Precision | Recall | F1 | AUC | ΔF1 vs + image (all) |\n|---|---|---|---|---|---|")
    a = np.array(cnn_alone)
    print(f"| CNN alone | {a[:, 0].mean():.4f} | {a[:, 1].mean():.4f} | {a[:, 2].mean():.4f} ± {a[:, 2].std():.4f} | | |")
    ref = np.array(res['+ image (all)'])[:, 2]
    for name, rows in res.items():
        a = np.array(rows)
        dd = a[:, 2] - ref
        print(f"| {name} | {a[:, 0].mean():.4f} | {a[:, 1].mean():.4f} | {a[:, 2].mean():.4f} ± {a[:, 2].std():.4f} | "
              f"{a[:, 3].mean():.4f} | {dd.mean():+.4f} ({(dd > 0).sum()}/{len(dd)} up) |")


if __name__ == '__main__':
    main()
