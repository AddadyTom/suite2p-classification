"""
Fixed hold-out "labeller" experiment: which training pool generalizes to which
test sessions? Same 44 features (baseline + image) as the production model.

Pools are built from the remaining (non-test) sessions:
  A  Stav only
  B  Inbar only
  C  Stav + Inbar
  D  Stav + Inbar, Inbar rows up-weighted so both labs carry equal total weight
Threshold and n_estimators come from a session-grouped inner CV on the training
pool only. Per test session we report P/R/F1 at that threshold, F1 at the
session's own best threshold (oracle; a large gap = calibration / labelling-
style shift rather than ranking errors), and predicted vs labelled positive rate.

    PYTHONPATH=. python scripts/holdout_eval.py --test Inbar3 Inbar12 inbar1 Inbar9
    PYTHONPATH=. python scripts/holdout_eval.py --test Stav14 --pools A B
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES
from scripts.evaluate_cv import LGB_PARAMS, THRESH_GRID, best_threshold, load_tables

FEATURES = ACTIVE_FEATURES + IMAGE_FEATURES


def fit_nested_weighted(X, y, g, w, n_inner):
    """fit_nested with sample weights (class balance via scale_pos_weight on weighted counts)."""
    def model(n, yy, ww):
        spw = ww[yy == 0].sum() / max(ww[yy == 1].sum(), 1e-9)
        return lgb.LGBMClassifier(n_estimators=n, scale_pos_weight=spw, **LGB_PARAMS)
    oof, iters = np.zeros(len(y)), []
    for tr, va in GroupKFold(n_splits=n_inner).split(X, y, g):
        m = model(1000, y[tr], w[tr])
        m.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[va], y[va])], eval_sample_weight=[w[va]],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        iters.append(m.best_iteration_ or 1000)
        oof[va] = m.predict_proba(X[va])[:, 1]
    thr = best_threshold(y, oof)
    final = model(int(np.mean(iters)), y, w)
    final.fit(X, y, sample_weight=w)
    return final, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--test', nargs='+', required=True, help="session names held out for testing")
    ap.add_argument('--pools', nargs='+', default=['A', 'B', 'C', 'D'])
    ap.add_argument('--inner', type=int, default=4)
    args = ap.parse_args()

    X_all, y, g, names = load_tables(args.tables, ['stav22', 'stav3/21'])
    col = {n: i for i, n in enumerate(names)}
    X = X_all[:, [col[f] for f in FEATURES]]
    sname = np.array([Path(s).name for s in g])
    assert set(args.test) <= set(sname), f"unknown sessions: {set(args.test) - set(sname)}"
    test = np.isin(sname, args.test)
    inbar = np.char.startswith(np.char.lower(sname.astype(str)), 'inbar')
    stav = np.char.startswith(np.char.lower(sname.astype(str)), 'stav')

    pools = {}
    for p in args.pools:
        if p == 'A':
            tr, w = stav & ~test, None
        elif p == 'B':
            tr, w = inbar & ~test, None
        elif p == 'C':
            tr, w = (stav | inbar) & ~test, None
        elif p == 'D':
            tr = (stav | inbar) & ~test
            w = np.ones(len(y))
            w[inbar] = (stav & tr).sum() / max((inbar & tr).sum(), 1)
        n_s = len(set(sname[tr]))
        label = {'A': 'Stav only', 'B': 'Inbar only', 'C': 'Stav + Inbar', 'D': 'Stav + Inbar, labs equal weight'}[p]
        pools[f"{p}: {label} ({n_s} sess)"] = (np.where(tr)[0], w)
    print(f"Test sessions: {args.test}; training pools: {list(pools)}", flush=True)

    rows = []
    for pname, (tr, w) in pools.items():
        ww = np.ones(len(tr)) if w is None else w[tr]
        m, thr = fit_nested_weighted(X[tr], y[tr], g[tr], ww, args.inner)
        prob = m.predict_proba(X[test])[:, 1]
        yt, st = y[test], sname[test]
        for s in args.test:
            k = st == s
            pr = prob[k] >= thr
            oracle = max(f1_score(yt[k], prob[k] >= t, zero_division=0) for t in THRESH_GRID)
            rows.append((pname, s, thr, precision_score(yt[k], pr, zero_division=0),
                         recall_score(yt[k], pr, zero_division=0), f1_score(yt[k], pr, zero_division=0),
                         oracle, pr.mean(), yt[k].mean()))
        pr = prob >= thr
        rows.append((pname, 'ALL (pooled)', thr, precision_score(yt, pr), recall_score(yt, pr), f1_score(yt, pr),
                     max(f1_score(yt, prob >= t) for t in THRESH_GRID), pr.mean(), yt.mean()))
        print(f"[{pname}] thr={thr:.2f} pooled F1={rows[-1][5]:.3f}", flush=True)

    print("\n| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | "
          "predicted pos rate | labelled pos rate |\n|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | "
              f"{r[7]:.3f} | {r[8]:.3f} |")


if __name__ == '__main__':
    main()
