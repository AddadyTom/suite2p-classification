"""
Export, per session, the ROIs where two very different models both confidently
disagree with the label: the trace/morphology LightGBM (27 baseline features)
and the image CNN. Both scores are out-of-fold (session-grouped, the ROI's own
session never seen in training), so they are honest second opinions.

Such ROIs are candidates for label review in the investigate_cell dashboard
(type the ROI index into "ROI Navigation Index").

    PYTHONPATH=. python scripts/export_review_list.py --cnn .../cnn/cnn_oof.npz --out results/review_lists
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from scripts.evaluate_cv import best_threshold, fit_nested, load_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--cnn', required=True)
    ap.add_argument('--out', default='results/review_lists')
    ap.add_argument('--confident', type=float, default=0.85, help="both probabilities beyond this (or below 1 - it)")
    args = ap.parse_args()

    X_all, y, g, names = load_tables(args.tables, ['stav22', 'stav3/21'])
    col = {n: i for i, n in enumerate(names)}
    Xb = X_all[:, [col[f] for f in ACTIVE_FEATURES]]
    d = np.load(args.cnn, allow_pickle=True)
    assert (d['y'] == y).all() and (d['groups'] == g).all()
    cnn = d['cnn']

    p_trace, p_cnn = np.zeros(len(y)), np.zeros(len(y))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(Xb, y, g)):
        m, _, _ = fit_nested(Xb[tr], y[tr], g[tr])
        p_trace[va] = m.predict_proba(Xb[va])[:, 1]
        p_cnn[va] = cnn[k, va]

    hi, lo = args.confident, 1 - args.confident
    say_cell = (p_trace >= hi) & (p_cnn >= hi) & (y == 0)
    say_not = (p_trace <= lo) & (p_cnn <= lo) & (y == 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roi_idx = np.concatenate([np.arange((g == s).sum()) for s in dict.fromkeys(g)])
    summary = ["| Session | ROIs | labelled not-cell, both models say cell | labelled cell, both say not-cell | % of ROIs |",
               "|---|---|---|---|---|"]
    for s in dict.fromkeys(g):
        k = (g == s) & (say_cell | say_not)
        rows = sorted(zip(roi_idx[k], y[k], p_trace[k], p_cnn[k]),
                      key=lambda r: -abs((r[2] + r[3]) / 2 - r[1]))
        name = Path(s).name
        with open(out / f"{name}.csv", 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['roi', 'label', 'p_trace_model', 'p_image_cnn', 'models_say', 'session_path'])
            for roi, lab, pt, pc in rows:
                w.writerow([int(roi), int(lab), f"{pt:.3f}", f"{pc:.3f}", 'cell' if lab == 0 else 'not cell', s])
        n = (g == s).sum()
        summary.append(f"| {name} | {n} | {int((k & say_cell).sum())} | {int((k & say_not).sum())} | {100 * k.sum() / n:.1f}% |")
    total = int((say_cell | say_not).sum())
    summary.append(f"| **All** | {len(y)} | {int(say_cell.sum())} | {int(say_not.sum())} | {100 * total / len(y):.1f}% |")
    text = ("ROIs where the trace model and the image CNN (both out-of-fold) confidently disagree with the label "
            f"(both probabilities ≥ {hi} or ≤ {lo:.2f}). One CSV per session, most confident first.\n\n" + "\n".join(summary) + "\n")
    (out / 'README.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
