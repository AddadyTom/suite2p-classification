"""
Train the production LightGBM with baseline + image features on all clean
sessions (same tables, exclusions and de-duplication as evaluate_cv.py).

n_estimators and the decision threshold come from a session-grouped inner CV
over all training sessions (fit_nested), and the threshold is stored in the
JSON next to the model, which apply_AI.py's 'image' preset reads.

    PYTHONPATH=. python scripts/train_image_model.py --out models/image/suite2p_image_lgb.pkl
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES, MIN_ALIGNMENT
from scripts.evaluate_cv import fit_nested, load_tables

FEATURES = ACTIVE_FEATURES + IMAGE_FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--out', default='models/image/suite2p_image_lgb.pkl')
    ap.add_argument('--cv-results', default='results/cv_results.json')
    ap.add_argument('--inner', type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite a model file")
    out.parent.mkdir(parents=True, exist_ok=True)

    X_all, y, g, names = load_tables(args.tables, ['stav22', 'stav3/21'])
    col = {n: i for i, n in enumerate(names)}
    X = X_all[:, [col[f] for f in FEATURES]]
    ok = ~np.all(np.isnan(X_all[:, [col[f] for f in IMAGE_FEATURES]]), axis=1)
    X, y, g = X[ok], y[ok], g[ok]
    sessions = sorted(set(g))
    print(f"Training on {len(sessions)} sessions, {len(y)} ROIs, {len(FEATURES)} features")

    model, thr, n_iter = fit_nested(X, y, g, n_inner=args.inner)
    joblib.dump(model, out)

    cv = {}
    if Path(args.cv_results).exists():
        r = json.loads(Path(args.cv_results).read_text())['results']
        for k in ['baseline (27)', '+ image (all)']:
            cv[k] = {m: {'mean': v[0], 'std': v[1]} for m, v in r[k]['summary'].items()}
    meta = {
        'name': 'LightGBM + image features',
        'description': (f"Baseline 27 features + {len(IMAGE_FEATURES)} ops.npy image features, trained on "
                        f"{len(sessions)} sessions. Leak-free CV F1 "
                        f"{cv.get('+ image (all)', {}).get('F1', {}).get('mean', float('nan')):.3f} vs baseline "
                        f"{cv.get('baseline (27)', {}).get('F1', {}).get('mean', float('nan')):.3f}."),
        'active_features': FEATURES,
        'custom_features': {},
        'threshold': thr,
        'threshold_source': f'F1-optimal on {args.inner}-fold session-grouped out-of-fold predictions',
        'n_estimators': n_iter,
        'feature_code': 'fe_engine.session_features.compute_session_features',
        'requires_ops': True,
        'min_ops_alignment': MIN_ALIGNMENT,
        'fallback': 'regular preset (models/suite2p_best_lgb.pkl) when ops.npy is missing or does not match stat.npy',
        'training_sessions': sessions,
        'cv_results': cv,
    }
    out.with_suffix('.json').write_text(json.dumps(meta, indent=4))
    print(f"Saved {out} (threshold {thr:.2f}, {n_iter} trees) and {out.with_suffix('.json')}")


if __name__ == '__main__':
    main()
