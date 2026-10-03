"""
Compare classifier families x input sets with the protocol of evaluate_cv.py:
GroupKFold by session (repeat 0, so fold-specific out-of-fold columns from the
CNN / fusion / MiniRocket runs can be used); in each outer training fold an
inner session-grouped CV gives out-of-fold probabilities (threshold) and, for
boosters, the early-stopped number of trees; the final model is refit on the
whole outer-training fold. Paired ΔF1 vs LightGBM on the 44 manual features.

Input sets (extra columns are nested out-of-fold for outer fold k):
  manual44            27 baseline + 17 image features
  manual44+cnn_prob   + CNN out-of-fold probability
  manual44+cnn_emb    + CNN 64-d penultimate embedding
  manual44+cnn_pca16  + PCA-16 of that embedding (PCA fit on outer-training rows)
  shape+image+rocket  morphology + image features + MiniRocket trace embedding
                      (no hand-made trace features)
  shape+image+tracenn morphology + image + fusion-network trace-branch embedding
  manual44+rocket     all manual + MiniRocket
  manual44+tracenn    all manual + fusion trace-branch embedding

    PYTHONPATH=. python scripts/compare_models.py --threads 8 --cnn .../cnn_oof_emb.npz \
        --fusion .../fusion_image_tab_trace_oof.npz --rocket .../minirocket_pca32.npz
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES
from scripts.evaluate_cv import LGB_PARAMS, best_threshold, load_tables, scores, spw_of

FEATURES = ACTIVE_FEATURES + IMAGE_FEATURES
SHAPE_FEATURES = ['area_to_radius_sq', 'aspect_ratio', 'compact', 'mrs', 'radius', 'solidity']
MAX_ITER = 2000


def make_model(name, n_iter, spw, threads):
    """n_iter=None -> booster sized for early stopping."""
    n = n_iter or MAX_ITER
    if name == 'LightGBM (current)':
        return lgb.LGBMClassifier(n_estimators=n, scale_pos_weight=spw, **{**LGB_PARAMS, 'n_jobs': threads})
    if name == 'LightGBM (tuned)':
        # actual row bagging (bagging_freq), smaller trees, more regularization, lower learning rate
        return lgb.LGBMClassifier(n_estimators=n, scale_pos_weight=spw, learning_rate=0.03, num_leaves=31,
                                  min_child_samples=50, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=1.0, random_state=42,
                                  n_jobs=threads, verbosity=-1)
    if name == 'XGBoost':
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=n, learning_rate=0.05, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=spw, tree_method='hist',
                                 random_state=42, n_jobs=threads, verbosity=0,
                                 early_stopping_rounds=50 if n_iter is None else None)
    if name == 'CatBoost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=n, learning_rate=0.08, depth=6, scale_pos_weight=spw,
                                  random_seed=42, thread_count=threads, verbose=False,
                                  early_stopping_rounds=50 if n_iter is None else None)
    if name == 'RandomForest':
        return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features='sqrt', class_weight='balanced_subsample',
            n_jobs=threads, random_state=42))
    if name == 'ExtraTrees':
        return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, max_features='sqrt', class_weight='balanced_subsample',
            n_jobs=threads, random_state=42))
    if name == 'MLP':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
            max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
            random_state=42))
    if name == 'LogisticRegression':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
    raise ValueError(name)


BOOSTERS = {'LightGBM (current)', 'LightGBM (tuned)', 'XGBoost', 'CatBoost'}


def fit(name, Xtr, ytr, threads, Xes=None, yes=None, n_iter=None):
    """Fit; boosters early-stop on (Xes, yes) when given. Returns (model, iterations used)."""
    m = make_model(name, n_iter if Xes is None else None, spw_of(ytr), threads)
    if name not in BOOSTERS:
        m.fit(Xtr, ytr)
        return m, None
    if Xes is None:
        m.fit(Xtr, ytr)
        return m, n_iter
    if name.startswith('LightGBM'):
        m.fit(Xtr, ytr, eval_set=[(Xes, yes)], callbacks=[lgb.early_stopping(50, verbose=False)])
        return m, m.best_iteration_ or MAX_ITER
    if name == 'XGBoost':
        m.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
        return m, int(m.best_iteration) + 1
    m.fit(Xtr, ytr, eval_set=(Xes, yes))
    return m, int(m.get_best_iteration()) + 1


ALL_MODELS = ['LightGBM (current)', 'LightGBM (tuned)', 'XGBoost', 'CatBoost',
              'RandomForest', 'ExtraTrees', 'LogisticRegression', 'MLP']
SET_MODELS = {
    'manual44': ALL_MODELS,
    'manual44+cnn_prob': ['LightGBM (current)', 'XGBoost', 'CatBoost', 'LogisticRegression', 'MLP'],
    'manual44+cnn_emb': ['LightGBM (current)', 'XGBoost', 'CatBoost', 'LogisticRegression', 'MLP'],
    'manual44+cnn_pca16': ['LightGBM (current)', 'XGBoost', 'CatBoost', 'LogisticRegression', 'MLP'],
    'shape+image+rocket': ['LightGBM (current)', 'LogisticRegression', 'MLP'],
    'shape+image+tracenn': ['LightGBM (current)', 'LogisticRegression', 'MLP'],
    'manual44+rocket': ['LightGBM (current)', 'LogisticRegression', 'MLP'],
    'manual44+tracenn': ['LightGBM (current)', 'LogisticRegression', 'MLP'],
}
ENSEMBLE = ['LightGBM (current)', 'XGBoost', 'CatBoost']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--inner', type=int, default=3)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--cnn', default=None, help="cnn_oof npz with 'cnn' (and 'emb' for embedding sets)")
    ap.add_argument('--fusion', default=None, help="fusion npz with 'emb_trace'")
    ap.add_argument('--rocket', default=None, help="minirocket npz with 'emb'")
    ap.add_argument('--sets', nargs='*', default=None)
    ap.add_argument('--out', default='results/model_comparison.md')
    args = ap.parse_args()

    X_all, y, g, names = load_tables(args.tables, ['stav22', 'stav3/21'])
    col = {n: i for i, n in enumerate(names)}
    manual = X_all[:, [col[f] for f in FEATURES]]
    shape_img = X_all[:, [col[f] for f in SHAPE_FEATURES + IMAGE_FEATURES]]

    extra = {}
    for key, path, field in [('cnn_prob', args.cnn, 'cnn'), ('cnn_emb', args.cnn, 'emb'),
                             ('tracenn', args.fusion, 'emb_trace'), ('rocket', args.rocket, 'emb')]:
        if path:
            d = np.load(path, allow_pickle=True)
            assert (d['y'] == y).all() and (d['groups'] == g).all(), f"{path}: rows do not match the CV rows"
            if field in d.files and int(d['folds_done']) == args.folds:
                extra[key] = d[field]
    available = {'manual44': True, 'manual44+cnn_prob': 'cnn_prob' in extra,
                 'manual44+cnn_emb': 'cnn_emb' in extra, 'manual44+cnn_pca16': 'cnn_emb' in extra,
                 'shape+image+rocket': 'rocket' in extra, 'shape+image+tracenn': 'tracenn' in extra,
                 'manual44+rocket': 'rocket' in extra, 'manual44+tracenn': 'tracenn' in extra}
    sets = [s for s, ok in available.items() if ok and (args.sets is None or s in args.sets)]
    print(f"{len(set(g))} sessions, {len(y)} ROIs; input sets: {sets}", flush=True)

    def build(set_name, k, tr):
        if set_name == 'manual44':
            return manual
        e = lambda key: np.asarray(extra[key][k], dtype=np.float32).reshape(len(y), -1)
        if set_name == 'manual44+cnn_pca16':
            emb = e('cnn_emb')
            return np.hstack([manual, PCA(16, random_state=0).fit(emb[tr]).transform(emb)])
        base, key = set_name.split('+', 1) if set_name.startswith('manual44') else ('shape+image', set_name.rsplit('+', 1)[1])
        left = manual if base == 'manual44' else shape_img
        return np.hstack([left, e(key)])

    res = {(s, m): [] for s in sets for m in SET_MODELS[s]}
    for s in sets:
        if all(m in SET_MODELS[s] for m in ENSEMBLE):
            res[(s, 'Average LGB+XGB+Cat')] = []
    secs = {k: 0.0 for k in res}
    splits = list(GroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(manual, y, g))
    for k, (tr, va) in enumerate(splits):
        for s in sets:
            X = build(s, k, tr)
            oof, pva = {}, {}
            for name in SET_MODELS[s]:
                t0 = time.time()
                o = np.zeros(len(tr))
                iters = []
                for itr, iva in GroupKFold(n_splits=args.inner).split(tr, y[tr], g[tr]):
                    a, b = tr[itr], tr[iva]
                    m, it = fit(name, X[a], y[a], args.threads, X[b], y[b])
                    iters.append(it)
                    o[iva] = m.predict_proba(X[b])[:, 1]
                n_it = int(np.mean(iters)) if name in BOOSTERS else None
                m, _ = fit(name, X[tr], y[tr], args.threads, n_iter=n_it)
                p = m.predict_proba(X[va])[:, 1]
                secs[(s, name)] += time.time() - t0
                oof[name], pva[name] = o, p
                res[(s, name)].append(scores(y[va], p >= best_threshold(y[tr], o)))
            if (s, 'Average LGB+XGB+Cat') in res:
                o = np.mean([oof[m] for m in ENSEMBLE], axis=0)
                p = np.mean([pva[m] for m in ENSEMBLE], axis=0)
                res[(s, 'Average LGB+XGB+Cat')].append(scores(y[va], p >= best_threshold(y[tr], o)))
                secs[(s, 'Average LGB+XGB+Cat')] = sum(secs[(s, m)] for m in ENSEMBLE)
            print(f"fold {k} [{s}] " + " | ".join(f"{m}={res[(s, m)][-1][2]:.4f}"
                                                    for (ss, m) in res if ss == s), flush=True)

    ref = np.array(res[('manual44', 'LightGBM (current)')])[:, 2] if ('manual44', 'LightGBM (current)') in res else None
    lines = [f"{len(set(g))} sessions, {args.folds} session-grouped folds (evaluate_cv repeat 0), nested "
             f"threshold / early stopping. ΔF1 is paired vs LightGBM (current) on manual44.\n",
             "| Input set | Model | Precision | Recall | F1 | ΔF1 vs LightGBM manual44 | fit time / fold |",
             "|---|---|---|---|---|---|---|"]
    for (s, name), rows in sorted(res.items(), key=lambda t: (sets.index(t[0][0]), -np.mean(np.array(t[1])[:, 2]))):
        a = np.array(rows)
        delta = ""
        if ref is not None and (s, name) != ('manual44', 'LightGBM (current)'):
            d = a[:, 2] - ref
            delta = f"{d.mean():+.4f} ± {d.std():.4f} ({(d > 0).sum()}/{len(d)} up)"
        lines.append(f"| {s} | {name} | {a[:, 0].mean():.4f} | {a[:, 1].mean():.4f} | "
                     f"{a[:, 2].mean():.4f} ± {a[:, 2].std():.4f} | {delta} | {secs[(s, name)] / len(a):.0f}s |")
    table = "\n".join(lines)
    print("\n" + table)
    Path(args.out).write_text(table + "\n")


if __name__ == '__main__':
    main()
