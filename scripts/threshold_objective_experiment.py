"""
Experiment: pick the decision threshold by F0.5 (a false positive costs about
twice a missed cell) instead of F1, for the image model (baseline 27 + 17 image).

Same protocol as evaluate_cv.py: repeated GroupKFold by session; inside each
outer training fold one inner session-grouped CV gives out-of-fold scores, from
which BOTH thresholds are picked (so the comparison is paired: same model, only
the cut-off differs); the held-out sessions are never used. Also prints a
precision / recall table over thresholds, for the dashboard slider.

    PYTHONPATH=. python scripts/threshold_objective_experiment.py --out results/threshold_objective_f05.md
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES
from scripts.evaluate_cv import best_threshold, load_tables, model, spw_of

BETAS = {'F1 threshold (current)': 1.0, 'F0.5 threshold (FP = 2x FN)': 0.5}


def metrics(y, pred):
    return dict(P=precision_score(y, pred, zero_division=0), R=recall_score(y, pred, zero_division=0),
                F1=fbeta_score(y, pred, beta=1, zero_division=0), F05=fbeta_score(y, pred, beta=0.5, zero_division=0),
                FP=int((pred & (y == 0)).sum()), FN=int((~pred & (y == 1)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21', 'Inbar5'])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', default='results/threshold_objective_f05.md')
    args = ap.parse_args()

    X_all, y, g, names = load_tables(args.tables, args.exclude)
    col = {n: i for i, n in enumerate(names)}
    X = X_all[:, [col[f] for f in ACTIVE_FEATURES + IMAGE_FEATURES]]
    print(f"{len(set(g))} sessions, {len(y)} ROIs, excluded {args.exclude}", flush=True)

    res = {k: [] for k in BETAS}
    thrs = {k: [] for k in BETAS}
    pooled_p, pooled_y = [], []
    for r in range(args.repeats):
        for tr, va in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X, y, g):
            oof, iters = np.zeros(len(tr)), []
            for itr, iva in GroupKFold(n_splits=3).split(tr, y[tr], g[tr]):
                a, b = tr[itr], tr[iva]
                m = model(1000, spw_of(y[a]))
                m.fit(X[a], y[a], eval_set=[(X[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
                iters.append(m.best_iteration_ or 1000)
                oof[iva] = m.predict_proba(X[b])[:, 1]
            final = model(int(np.mean(iters)), spw_of(y[tr]))
            final.fit(X[tr], y[tr])
            p = final.predict_proba(X[va])[:, 1]
            if r == 0:
                pooled_p.append(p); pooled_y.append(y[va])
            for k, beta in BETAS.items():
                t = best_threshold(y[tr], oof, beta)
                thrs[k].append(t)
                res[k].append(metrics(y[va], p >= t))
            print(f"repeat {r}: " + " | ".join(f"{k}: thr={thrs[k][-1]:.2f} P={res[k][-1]['P']:.3f} R={res[k][-1]['R']:.3f}"
                                               for k in BETAS), flush=True)

    keys = list(BETAS)
    lines = [f"Image model (44 features), {len(set(g))} sessions (excluded {', '.join(args.exclude)}), "
             f"{args.repeats}x5 session folds. Same model per fold; only the threshold objective differs, both chosen "
             "on training sessions only.\n",
             "| Threshold chosen by | threshold | Precision | Recall | F1 | F0.5 | false positives / fold | missed cells / fold |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        a = {m: np.array([row[m] for row in res[k]]) for m in res[k][0]}
        lines.append(f"| {k} | {np.mean(thrs[k]):.2f} | {a['P'].mean():.3f} ± {a['P'].std():.3f} | "
                     f"{a['R'].mean():.3f} ± {a['R'].std():.3f} | {a['F1'].mean():.3f} | {a['F05'].mean():.3f} | "
                     f"{a['FP'].mean():.0f} | {a['FN'].mean():.0f} |")
    d = {m: np.array([b[m] - a_[m] for a_, b in zip(res[keys[0]], res[keys[1]])]) for m in ['P', 'R', 'F1', 'F05', 'FP', 'FN']}
    lines.append(f"\nPaired change F0.5 vs F1 threshold: precision {d['P'].mean():+.3f} ({(d['P'] > 0).sum()}/{len(d['P'])} folds up), "
                 f"recall {d['R'].mean():+.3f}, F0.5 {d['F05'].mean():+.3f} ({(d['F05'] > 0).sum()}/{len(d['F05'])} up), "
                 f"F1 {d['F1'].mean():+.3f}; false positives {d['FP'].mean():+.0f}, missed cells {d['FN'].mean():+.0f} per fold.\n")
    p, yy = np.concatenate(pooled_p), np.concatenate(pooled_y)
    lines += ["Precision / recall by threshold (held-out sessions, repeat 0 pooled): the dashboard slider moves along this.\n",
              "| threshold | Precision | Recall | false positives per 100 predicted cells |", "|---|---|---|---|"]
    for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        pr = p >= t
        prec = precision_score(yy, pr, zero_division=0)
        lines.append(f"| {t:.2f} | {prec:.3f} | {recall_score(yy, pr):.3f} | {100 * (1 - prec):.1f} |")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print("\n" + text)


if __name__ == '__main__':
    main()
