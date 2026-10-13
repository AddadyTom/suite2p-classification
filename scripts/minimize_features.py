"""
Backward elimination of morphology + image features (trace features are always kept).

Start from the 44-feature image model. At each step, run the nested session-grouped
CV of evaluate_cv.py (repeat 0, 5 folds), record F1, and drop the removable feature
with the lowest mean |SHAP| on the outer validation folds. The result is an F1 curve
from 44 features down to the trace-only set.

    PYTHONPATH=. python scripts/minimize_features.py --exclude stav22 stav3/21 Inbar5 --out results/feature_minimization.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES
from scripts.evaluate_cv import fit_nested, load_tables, scores

MORPHOLOGY = ['area_to_radius_sq', 'aspect_ratio', 'compact', 'mrs', 'radius', 'solidity']
TRACE = [f for f in ACTIVE_FEATURES if f not in MORPHOLOGY]


def cv_eval(X, y, g, splits):
    f1s, imp = [], np.zeros(X.shape[1])
    for tr, va in splits:
        m, thr, _ = fit_nested(X[tr], y[tr], g[tr])
        f1s.append(scores(y[va], m.predict_proba(X[va])[:, 1] >= thr)[2])
        imp += np.abs(m.booster_.predict(X[va], pred_contrib=True)[:, :-1]).mean(axis=0)
    return np.array(f1s), imp / len(splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21', 'Inbar5'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--out', default='results/feature_minimization.md')
    args = ap.parse_args()

    X_all, y, g, names = load_tables(args.tables, args.exclude)
    col = {n: i for i, n in enumerate(names)}
    splits = list(GroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(X_all, y, g))
    print(f"{len(set(g))} sessions, {len(y)} ROIs (excluded: {args.exclude})", flush=True)

    removable = MORPHOLOGY + IMAGE_FEATURES
    feats = TRACE + removable
    steps, ref = [], None
    while True:
        t0 = time.time()
        f1s, imp = cv_eval(X_all[:, [col[f] for f in feats]], y, g, splits)
        ref = f1s if ref is None else ref
        cand = [(imp[feats.index(f)], f) for f in feats if f in removable]
        drop = min(cand)[1] if cand else None
        steps.append({'n_features': len(feats), 'f1_mean': float(f1s.mean()), 'f1_std': float(f1s.std()),
                      'delta_vs_full': float((f1s - ref).mean()), 'folds_down': int((f1s < ref).sum()),
                      'kept_removable': [f for f in feats if f in removable], 'next_drop': drop,
                      'f1_folds': f1s.tolist()})
        print(f"{len(feats)} features: F1={f1s.mean():.4f}±{f1s.std():.4f} Δ={steps[-1]['delta_vs_full']:+.4f} "
              f"next drop: {drop} ({time.time() - t0:.0f}s)", flush=True)
        if drop is None:
            break
        feats.remove(drop)

    lines = [f"Backward elimination (trace features always kept), {len(set(g))} sessions, {args.folds} session folds, "
             f"nested threshold. Excluded: {', '.join(args.exclude)}.\n",
             "| # features | F1 | ΔF1 vs 44 (paired) | folds worse | dropped next |", "|---|---|---|---|---|"]
    for s in steps:
        lines.append(f"| {s['n_features']} | {s['f1_mean']:.4f} ± {s['f1_std']:.4f} | {s['delta_vs_full']:+.4f} | "
                     f"{s['folds_down']}/{args.folds} | {s['next_drop'] or '—'} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix('.json').write_text(json.dumps({'trace_features': TRACE, 'steps': steps}, indent=1))
    print("\n".join(lines))


if __name__ == '__main__':
    main()
