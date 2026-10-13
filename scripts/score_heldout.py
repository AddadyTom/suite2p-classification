"""
Score a saved model on held-out feature tables at the threshold stored in its JSON.

    PYTHONPATH=. python scripts/score_heldout.py --model models/image/X.pkl --test stav22 Inbar7 Inbar8
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_cv import THRESH_GRID, best_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--model', required=True)
    ap.add_argument('--test', nargs='+', required=True, help="session folder names to score")
    args = ap.parse_args()
    model = joblib.load(args.model)
    meta = json.loads(Path(args.model).with_suffix('.json').read_text())
    thr, feats = meta['threshold'], meta['active_features']
    print(f"| session | ROIs | labelled cells | precision | recall | F1 @ {thr:.2f} | best F1 (thr) | AUC |")
    print("|---|---|---|---|---|---|---|---|")
    ys, ps = [], []
    for name in args.test:
        f = next(f for f in Path(args.tables).glob('*.npz')
                 if Path(str(np.load(f, allow_pickle=True)['session'])).name.lower() == name.lower())
        d = np.load(f, allow_pickle=True)
        cols = list(d['trace_names']) + list(d['img_names'])
        X = np.hstack([d['X_trace'], d['X_img']])[:, [cols.index(c) for c in feats]]
        y = d['y'].astype(int)
        p = model.predict_proba(X)[:, 1]
        yh = p >= thr
        bt = best_threshold(y, p)
        print(f"| {name} | {len(y)} | {y.sum()} | {precision_score(y, yh):.3f} | {recall_score(y, yh):.3f} | "
              f"{f1_score(y, yh):.3f} | {f1_score(y, p >= bt):.3f} ({bt:.2f}) | {roc_auc_score(y, p):.3f} |")
        ys.append(y); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    print(f"| pooled | {len(y)} | {y.sum()} | {precision_score(y, p >= thr):.3f} | {recall_score(y, p >= thr):.3f} | "
          f"{f1_score(y, p >= thr):.3f} | | {roc_auc_score(y, p):.3f} |")


if __name__ == '__main__':
    main()
