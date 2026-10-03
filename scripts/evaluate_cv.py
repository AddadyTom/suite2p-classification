"""
Leak-free cross-validation of the ROI classifier on per-session feature tables
(see build_feature_tables.py).

Protocol
  * Outer: GroupKFold by session (5 folds), repeated with different session
    shuffles. Every feature set is evaluated on the same splits (paired).
  * Inner: GroupKFold by session (3 folds) on the *training* sessions only.
    Inner models early-stop on their inner validation sessions; their
    out-of-fold probabilities pick the F1-optimal threshold and the mean best
    iteration fixes n_estimators for the outer model. The outer validation
    sessions are never used for early stopping or threshold selection.
  * Metrics: P/R/F1 per outer fold (ROIs pooled within the fold), mean ± std.
  * Sessions whose ops.npy fails the alignment check (all image features NaN,
    e.g. Stav1) are dropped from training of image-feature models and, at
    validation, fall back to the baseline model's prediction, as they would
    in deployment.

The old protocol from train_model.py (early stopping and threshold both tuned on
the validation fold) is also run for the baseline, for reference.

    PYTHONPATH=. python scripts/evaluate_cv.py --tables feature_tables --out results
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES, TRACE_NORM_FEATURES
from fe_engine.image_features import IMAGE_FEATURES

RAW_UNIT_FEATURES = ['q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99', 'range_f', 'range_fcorr']
IMG_CONTRAST = [f for f in IMAGE_FEATURES if 'contrast' in f or 'ratio' in f or f.startswith('Vcorr_in')]
IMG_LAMCORR = [f for f in IMAGE_FEATURES if 'lam_corr' in f]
BASE_NO_RAW = [f for f in ACTIVE_FEATURES if f not in RAW_UNIT_FEATURES]

FEATURE_SETS = {
    'baseline (27)': ACTIVE_FEATURES,
    '+ trace-norm': ACTIVE_FEATURES + TRACE_NORM_FEATURES,
    'trace-norm replaces raw q/range': BASE_NO_RAW + TRACE_NORM_FEATURES,
    '+ image: ring contrast': ACTIVE_FEATURES + IMG_CONTRAST,
    '+ image: lam correlation': ACTIVE_FEATURES + IMG_LAMCORR,
    '+ image (all)': ACTIVE_FEATURES + IMAGE_FEATURES,
    'new: replace + image': BASE_NO_RAW + TRACE_NORM_FEATURES + IMAGE_FEATURES,
}

LGB_PARAMS = dict(learning_rate=0.05, max_depth=8, num_leaves=63, subsample=0.8,
                  colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)
THRESH_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def load_tables(table_dir, exclude):
    tables = {}
    for f in sorted(Path(table_dir).glob('*.npz')):
        d = np.load(f, allow_pickle=True)
        sess = str(d['session'])
        if 'yael' in sess.lower() or any(e.lower() in sess.lower() for e in exclude):
            continue
        tables[sess] = {k: d[k] for k in d.files}

    # Drop exact duplicate sessions (identical traces + labels, e.g. Stav1 is a
    # copy of Stav5): a copy on both sides of a split leaks. Keep the copy whose
    # ops.npy passed the alignment check.
    by_content = {}
    for sess, t in tables.items():
        key = hash(t['X_trace'].tobytes()) ^ hash(t['y'].tobytes())
        by_content.setdefault(key, []).append(sess)
    for dups in by_content.values():
        if len(dups) > 1:
            dups.sort(key=lambda s: np.isnan(tables[s]['X_img']).all())
            for s in dups[1:]:
                print(f"Dropping {s}: duplicate of {dups[0]}")
                del tables[s]

    Xs, ys, groups, names = [], [], [], None
    for sess, t in tables.items():
        cols = list(t['trace_names']) + list(t['img_names'])
        if names is None:
            names = cols
        assert cols == names, f"column mismatch in {sess}"
        Xs.append(np.hstack([t['X_trace'], t['X_img']]))
        ys.append(t['y'].astype(int))
        groups += [sess] * len(t['y'])
    return np.vstack(Xs), np.concatenate(ys), np.array(groups), names


def model(n_estimators, spw):
    return lgb.LGBMClassifier(n_estimators=n_estimators, scale_pos_weight=spw, **LGB_PARAMS)


def spw_of(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)


def best_threshold(y, p):
    f1s = [f1_score(y, p >= t, zero_division=0) for t in THRESH_GRID]
    return float(THRESH_GRID[int(np.argmax(f1s))])


def fit_nested(X, y, g, n_inner=3):
    """Pick n_estimators and threshold using only (X, y, g); return fitted model + threshold."""
    oof = np.zeros(len(y))
    iters = []
    for tr, va in GroupKFold(n_splits=n_inner).split(X, y, g):
        m = model(1000, spw_of(y[tr]))
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        iters.append(m.best_iteration_ or 1000)
        oof[va] = m.predict_proba(X[va])[:, 1]
    thr = best_threshold(y, oof)
    final = model(int(np.mean(iters)), spw_of(y))
    final.fit(X, y)
    return final, thr, int(np.mean(iters))


def fit_leaky(Xtr, ytr, Xva, yva):
    """Old train_model.py protocol: early stopping + threshold tuned on the validation fold."""
    m = model(1000, spw_of(ytr))
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict_proba(Xva)[:, 1]
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        f = f1_score(yva, (p >= t).astype(int))
        if f > best_f:
            best_t, best_f = t, f
    return p, best_t


def scores(y, pred):
    return (precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0))


def summarize(rows):
    a = np.array(rows)
    return {k: (float(a[:, i].mean()), float(a[:, i].std())) for i, k in enumerate(['P', 'R', 'F1'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--out', default='results')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21'],
                    help="session-path substrings to leave out: stav22 is the held-out session in train_model.py; stav3/21 has suspect labels (70%% positive)")
    ap.add_argument('--sets', nargs='*', default=None, help="subset of FEATURE_SETS keys")
    ap.add_argument('--shap-set', default='new: replace + image')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    X_all, y, g, names = load_tables(args.tables, args.exclude)
    col = {n: i for i, n in enumerate(names)}
    sessions = np.unique(g)
    print(f"{len(sessions)} sessions, {len(y)} ROIs, {y.mean():.3f} positive")

    splits = []
    for r in range(args.repeats):
        gkf = GroupKFold(n_splits=args.folds, shuffle=True, random_state=r)
        splits += [(r, tr, va) for tr, va in gkf.split(X_all, y, g)]

    sets = {k: v for k, v in FEATURE_SETS.items()
            if args.sets is None or k in args.sets or k == 'baseline (27)'}
    no_img = np.all(np.isnan(X_all[:, [col[f] for f in IMAGE_FEATURES]]), axis=1)
    print(f"{no_img.sum()} ROIs without usable image features "
          f"({', '.join(Path(s).name for s in np.unique(g[no_img]))}) use the baseline fallback")
    base_pred = {}
    results = {}

    # Reference: old leaky protocol, baseline features
    t0 = time.time()
    Xb = X_all[:, [col[f] for f in ACTIVE_FEATURES]]
    rows = []
    for r, tr, va in splits:
        p, t = fit_leaky(Xb[tr], y[tr], Xb[va], y[va])
        rows.append(scores(y[va], p >= t))
    results['baseline (27), old leaky protocol'] = {'summary': summarize(rows), 'folds': rows}
    print(f"[leaky baseline] {summarize(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    shap_imp = None
    for name, feats in sets.items():
        t0 = time.time()
        Xs = X_all[:, [col[f] for f in feats]]
        rows, per_session, thrs = [], [], []
        shap_acc = np.zeros(len(feats)) if name == args.shap_set else None
        shap_n = 0
        uses_img = any(f in IMAGE_FEATURES for f in feats)
        for i, (r, tr, va) in enumerate(splits):
            if uses_img:
                tr = tr[~no_img[tr]]
            m, thr, n_it = fit_nested(Xs[tr], y[tr], g[tr])
            p = m.predict_proba(Xs[va])[:, 1]
            pred = p >= thr
            if name == 'baseline (27)':
                base_pred[i] = pred
            elif uses_img:
                pred = np.where(no_img[va], base_pred[i], pred)
            rows.append(scores(y[va], pred))
            thrs.append(thr)
            for s in np.unique(g[va]):
                k = g[va] == s
                per_session.append((s, float(f1_score(y[va][k], pred[k], zero_division=0))))
            if shap_acc is not None:
                va_s = va[~no_img[va]] if uses_img else va
                sv = m.booster_.predict(Xs[va_s], pred_contrib=True)[:, :-1]
                shap_acc += np.abs(sv).sum(axis=0)
                shap_n += len(va_s)
        results[name] = {
            'summary': summarize(rows), 'folds': rows, 'thresholds': thrs,
            'per_session_f1': per_session, 'features': feats,
        }
        if shap_acc is not None:
            shap_imp = sorted(zip(feats, (shap_acc / shap_n).tolist()),
                              key=lambda t: -t[1])
            results[name]['mean_abs_shap'] = shap_imp
        s = summarize(rows)
        print(f"[{name}] n_feat={len(feats)} P={s['P'][0]:.4f}±{s['P'][1]:.4f} "
              f"R={s['R'][0]:.4f}±{s['R'][1]:.4f} F1={s['F1'][0]:.4f}±{s['F1'][1]:.4f} "
              f"thr={np.mean(thrs):.2f} ({time.time() - t0:.0f}s)", flush=True)

    base_f1 = np.array(results['baseline (27)']['folds'])[:, 2] if 'baseline (27)' in results else None
    lines = ["| Feature set | # feat | Precision | Recall | F1 | ΔF1 vs baseline (paired) |",
             "|---|---|---|---|---|---|"]
    for name, res in results.items():
        s = res['summary']
        nf = len(res.get('features', ACTIVE_FEATURES))
        delta = ""
        if base_f1 is not None and 'features' in res and name != 'baseline (27)':
            d = np.array(res['folds'])[:, 2] - base_f1
            delta = f"{d.mean():+.4f} ± {d.std():.4f} ({(d > 0).sum()}/{len(d)} folds up)"
        lines.append(f"| {name} | {nf} | {s['P'][0]:.4f} ± {s['P'][1]:.4f} | "
                     f"{s['R'][0]:.4f} ± {s['R'][1]:.4f} | {s['F1'][0]:.4f} ± {s['F1'][1]:.4f} | {delta} |")
    table = "\n".join(lines)
    print("\n" + table)
    if shap_imp:
        print(f"\nTop 20 mean |SHAP| ({args.shap_set}):")
        for f, v in shap_imp[:20]:
            print(f"  {f:30s} {v:.4f}")

    meta = {'n_sessions': int(len(sessions)), 'n_rois': int(len(y)), 'pos_rate': float(y.mean()),
            'sessions': sessions.tolist(), 'repeats': args.repeats, 'folds': args.folds,
            'excluded': args.exclude}
    (out / 'cv_results.json').write_text(json.dumps({'meta': meta, 'results': results}, indent=1))
    (out / 'cv_table.md').write_text(table + "\n")


if __name__ == '__main__':
    main()
