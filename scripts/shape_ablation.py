"""
Experiment: what happens if the ROI shape (morphology) features are removed?

Motivation: stav22 ROI 219, a small (44 px) ROI with a clean cell trace, is
rejected mainly because of its size/shape: in the training labels only 1% of
ROIs with radius <= 3.2 px are cells.

Compares, with the same nested session CV as evaluate_cv.py (3x5 folds, Inbar5
held out): 44 vs 38 (44 minus shape) and 32 vs 28 (32 minus shape). Then trains
each set on all those sessions and scores stav22 (never used): F1 at the trained
threshold, ROI 219's probability, and recall on small vs larger labelled cells.

    PYTHONPATH=. python scripts/shape_ablation.py --out results/shape_ablation.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES
from scripts.evaluate_cv import fit_nested, load_tables, scores

SHAPE = ['area_to_radius_sq', 'aspect_ratio', 'compact', 'mrs', 'radius', 'solidity']

# Modified features, derived from table columns:
#  radius_clip35      radius floored at 3.5 px: tiny ROIs are not penalised for being tiny,
#                     large-vs-medium size information is kept
#  skew_fneu_scaled   neuropil skew weighted by neuropil amplitude relative to the ROI
#                     (skew_fneu / range_ratio_f_fneu): a flat neuropil with a few tiny bumps
#                     no longer looks like contamination
DERIVED = {
    'radius_clip35': lambda c: np.maximum(c('radius'), 3.5),
    'skew_fneu_scaled': lambda c: c('skew_fneu') / np.maximum(c('range_ratio_f_fneu'), 1e-6),
}


def build(X, cols, feats):
    """Columns for feats from a table matrix X with column names cols, computing DERIVED ones."""
    c = lambda n: X[:, cols.index(n)]
    return np.column_stack([DERIVED[f](c) if f in DERIVED else c(f) for f in feats])


def table(path, feats):
    d = np.load(path, allow_pickle=True)
    cols = list(d['trace_names']) + list(d['img_names'])
    X = np.hstack([d['X_trace'], d['X_img']])
    return build(X, cols, feats), d['y'].astype(int), X[:, cols.index('radius')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--small-set', default='../suite2p-image-features/results/feature_sets_confirmed.json')
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21', 'Inbar5'])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', default='results/shape_ablation.md')
    ap.add_argument('--variants', action='store_true', help="run the modified-feature variants on the 32-feature set")
    args = ap.parse_args()

    f44 = ACTIVE_FEATURES + IMAGE_FEATURES
    f32 = json.loads(Path(args.small_set).read_text())['32']
    sets = {'44 (current)': f44, '38 = 44 without shape': [f for f in f44 if f not in SHAPE],
            '32 (small model)': f32, '28 = 32 without shape': [f for f in f32 if f not in SHAPE]}
    if args.variants:
        sub = lambda fs, a, b: [b if f == a else f for f in fs]
        no_r = [f for f in f32 if f != 'radius']
        sets = {'32 (small model)': f32,
                '31 = 32 without radius (size)': no_r,
                '32, radius clipped at 3.5 px': sub(f32, 'radius', 'radius_clip35'),
                '32, neuropil skew scaled by amplitude': sub(f32, 'skew_fneu', 'skew_fneu_scaled'),
                '32, radius clipped + scaled neuropil skew': sub(sub(f32, 'radius', 'radius_clip35'), 'skew_fneu', 'skew_fneu_scaled'),
                '31, no radius + scaled neuropil skew': sub(no_r, 'skew_fneu', 'skew_fneu_scaled')}

    X_all, y, g, names = load_tables(args.tables, args.exclude)
    names = list(names)
    splits = [sp for r in range(args.repeats)
              for sp in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X_all, y, g)]
    s22 = Path(args.tables) / '22-ordered__stav22.npz'

    cv, st = {}, {}
    for name, feats in sets.items():
        X = build(X_all, names, feats)
        rows = []
        for tr, va in splits:
            m, thr, _ = fit_nested(X[tr], y[tr], g[tr])
            rows.append(scores(y[va], m.predict_proba(X[va])[:, 1] >= thr))
        cv[name] = np.array(rows)
        m, thr, _ = fit_nested(X, y, g, n_inner=5)
        X22, y22, r22 = table(s22, feats)
        p = m.predict_proba(X22)[:, 1]
        pr = p >= thr
        small, big = (y22 == 1) & (r22 <= 3.5), (y22 == 1) & (r22 > 3.5)
        st[name] = dict(thr=thr, P=precision_score(y22, pr), R=recall_score(y22, pr), F1=f1_score(y22, pr),
                        roi219=float(p[219]), small_recall=float(pr[small].mean()), n_small=int(small.sum()),
                        big_recall=float(pr[big].mean()))
        print(f"[{name}] CV F1={cv[name][:, 2].mean():.4f} | stav22 F1={st[name]['F1']:.3f} ROI219 p={p[219]:.3f} "
              f"small-cell recall={st[name]['small_recall']:.2f}", flush=True)

    ref = {n: ('44 (current)' if n.startswith('38') else '32 (small model)') for n in cv if n not in ('44 (current)', '32 (small model)')}
    lines = [f"CV: {len(set(g))} sessions (excluded {', '.join(args.exclude)}), {args.repeats}x5 session folds, nested "
             "threshold. stav22: model trained on all those sessions, stav22 never used.\n",
             "| Feature set | CV precision | CV recall | CV F1 | ΔF1 vs with-shape (paired) |", "|---|---|---|---|---|"]
    for name, a in cv.items():
        d = ""
        if name in ref:
            dd = a[:, 2] - cv[ref[name]][:, 2]
            d = f"{dd.mean():+.4f} ± {dd.std():.4f} ({(dd >= 0).sum()}/{len(dd)} folds not worse)"
        lines.append(f"| {name} | {a[:, 0].mean():.3f} | {a[:, 1].mean():.3f} | {a[:, 2].mean():.4f} ± {a[:, 2].std():.4f} | {d} |")
    n_small = next(iter(st.values()))['n_small']
    lines += ["", "| Feature set | stav22 threshold | precision | recall | F1 | ROI 219 probability | "
              f"recall, small labelled cells (radius ≤ 3.5, n={n_small}) | recall, larger cells |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in st.items():
        lines.append(f"| {name} | {s['thr']:.2f} | {s['P']:.3f} | {s['R']:.3f} | {s['F1']:.3f} | {s['roi219']:.2f} | "
                     f"{s['small_recall']:.2f} | {s['big_recall']:.2f} |")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print("\n" + text)


if __name__ == '__main__':
    main()
