"""
Confirm candidate small feature sets from minimize_features.py against the full
44-feature model with the more robust 3x5 repeated session-grouped nested CV
(paired folds, same exclusions).

    PYTHONPATH=. python scripts/confirm_feature_sets.py --sizes 32 30 28 --out results/feature_sets_confirmed.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_cv import fit_nested, load_tables, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--minimization', default='results/feature_minimization.json')
    ap.add_argument('--sizes', nargs='+', type=int, required=True)
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21', 'Inbar5'])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', default='results/feature_sets_confirmed.md')
    args = ap.parse_args()

    mz = json.loads(Path(args.minimization).read_text())
    trace = mz['trace_features']
    sets = {}
    for s in mz['steps']:
        if s['n_features'] == 44 or s['n_features'] in args.sizes:
            sets[s['n_features']] = trace + s['kept_removable']
    X_all, y, g, names = load_tables(args.tables, args.exclude)
    col = {n: i for i, n in enumerate(names)}
    splits = [sp for r in range(args.repeats)
              for sp in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X_all, y, g)]

    res = {}
    for n, feats in sorted(sets.items(), reverse=True):
        X = X_all[:, [col[f] for f in feats]]
        rows = []
        for tr, va in splits:
            m, thr, _ = fit_nested(X[tr], y[tr], g[tr])
            rows.append(scores(y[va], m.predict_proba(X[va])[:, 1] >= thr))
        res[n] = np.array(rows)
        print(f"{n} features: F1={res[n][:, 2].mean():.4f}±{res[n][:, 2].std():.4f}", flush=True)

    ref = res[44][:, 2]
    lines = [f"{len(set(g))} sessions (excluded {', '.join(args.exclude)}), {args.repeats}x5 session folds, nested threshold.\n",
             "| # features | Precision | Recall | F1 | ΔF1 vs 44 (paired) | folds not worse |", "|---|---|---|---|---|---|"]
    for n in sorted(res, reverse=True):
        a, d = res[n], res[n][:, 2] - ref
        delta = "—" if n == 44 else f"{d.mean():+.4f} ± {d.std():.4f}"
        lines.append(f"| {n} | {a[:, 0].mean():.4f} | {a[:, 1].mean():.4f} | {a[:, 2].mean():.4f} ± {a[:, 2].std():.4f} | "
                     f"{delta} | {(d >= 0).sum()}/{len(d)} |")
    lines.append("")
    for n in sorted(sets, reverse=True):
        if n != 44:
            lines.append(f"**{n} features:** {', '.join(sets[n])}\n")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    Path(args.out).with_suffix('.json').write_text(json.dumps({str(n): f for n, f in sets.items()}, indent=1))
    print(text)


if __name__ == '__main__':
    main()
