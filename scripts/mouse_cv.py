"""
Mouse-grouped evaluation: are the session-grouped scores optimistic for a new animal?

Several sessions come from the same mouse, often the same field of view. Here the
outer AND inner CV hold out whole mice (mouse ID from ops['data_path']), so a test
mouse never has a sibling session in training.

1. Feature sets (27 / 44 / 32 / 31) with 3x5 mouse-grouped folds: F1 at the nested
   (F1-optimal) threshold and at a fixed threshold (--fixed-threshold, default 0.70).
2. Labeller test, leave-one-mouse-out: every mouse is scored by a Stav-only and an
   Inbar-only model, both trained without that mouse (31-feature set).

    PYTHONPATH=. python scripts/mouse_cv.py --out results/mouse_cv.md
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from fe_engine.image_features import IMAGE_FEATURES
from scripts.evaluate_cv import fit_nested, load_tables


def mouse_of(session_path):
    ops = np.load(Path(session_path) / 'ops.npy', allow_pickle=True).item()
    dp = ops.get('data_path')
    dp = dp[0] if isinstance(dp, (list, tuple)) and dp else str(dp)
    m = re.search(r'([A-Za-z]{2}\d+)_run', str(dp))
    return m.group(1) if m else f"unknown:{Path(session_path).name}"


def prf(y, pred):
    return (precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21'])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--fixed-threshold', type=float, default=0.70)
    ap.add_argument('--out', default='results/mouse_cv.md')
    args = ap.parse_args()

    X_all, y, g, names = load_tables(args.tables, args.exclude)
    col = {n: i for i, n in enumerate(names)}
    sessions = list(dict.fromkeys(g))
    smap = {s: mouse_of(s) for s in sessions}
    mouse = np.array([smap[s] for s in g])
    lab = np.array(['stav' if Path(s).name.lower().startswith('stav') else 'inbar' for s in g])
    by_mouse = {}
    for s in sessions:
        by_mouse.setdefault(smap[s], []).append(Path(s).name)
    print(f"{len(sessions)} sessions, {len(by_mouse)} mice: " +
          "; ".join(f"{m}: {', '.join(v)}" for m, v in sorted(by_mouse.items())), flush=True)

    f44 = ACTIVE_FEATURES + IMAGE_FEATURES
    f32 = json.loads(Path('results/feature_sets_confirmed.json').read_text())['32']
    f31 = json.loads(Path('results/small_model_features.json').read_text())['features']
    sets = {'27 (previous model)': ACTIVE_FEATURES, '44 (image model)': f44, '32': f32, '31 (small model)': f31}

    # 1. feature sets, mouse-grouped
    splits = [sp for r in range(args.repeats)
              for sp in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(X_all, y, mouse)]
    res = {}
    for name, feats in sets.items():
        X = X_all[:, [col[f] for f in feats]]
        rows = []
        for tr, va in splits:
            m, thr, _ = fit_nested(X[tr], y[tr], mouse[tr])
            p = m.predict_proba(X[va])[:, 1]
            rows.append(prf(y[va], p >= thr) + prf(y[va], p >= args.fixed_threshold))
        res[name] = np.array(rows)
        print(f"[{name}] mouse-grouped F1={res[name][:, 2].mean():.4f} (fixed {args.fixed_threshold}: "
              f"{res[name][:, 5].mean():.4f})", flush=True)

    ref = res['27 (previous model)'][:, 2]
    t = args.fixed_threshold
    lines = [f"Mouse-grouped CV: {len(sessions)} sessions from {len(by_mouse)} mice (excluded {', '.join(args.exclude)}), "
             f"{args.repeats}x5 folds holding out whole mice; inner threshold/tree selection also mouse-grouped.\n",
             f"| Feature set | P | R | F1 (nested threshold) | ΔF1 vs 27 (paired) | P @ {t} | R @ {t} | F1 @ {t} |",
             "|---|---|---|---|---|---|---|---|"]
    for name, a in res.items():
        d = a[:, 2] - ref
        dd = "—" if name.startswith('27') else f"{d.mean():+.4f} ± {d.std():.4f} ({(d > 0).sum()}/{len(d)} up)"
        lines.append(f"| {name} | {a[:, 0].mean():.3f} | {a[:, 1].mean():.3f} | {a[:, 2].mean():.4f} ± {a[:, 2].std():.4f} | "
                     f"{dd} | {a[:, 3].mean():.3f} | {a[:, 4].mean():.3f} | {a[:, 5].mean():.4f} |")

    # 2. labeller test, leave-one-mouse-out (31 features)
    X = X_all[:, [col[f] for f in f31]]
    lines += ["", "Labeller test, leave-one-mouse-out (31 features): each mouse scored by a Stav-only and an Inbar-only "
              "model, both trained without that mouse; F1 at each model's nested threshold.\n",
              "| Mouse | lab | sessions | Stav-only F1 | Inbar-only F1 | own-lab advantage |", "|---|---|---|---|---|---|"]
    adv = {'stav': [], 'inbar': []}
    for mname in sorted(by_mouse):
        te = mouse == mname
        mlab = lab[te][0]
        f = {}
        for pool in ['stav', 'inbar']:
            tr = (lab == pool) & ~te
            m, thr, _ = fit_nested(X[tr], y[tr], mouse[tr])
            f[pool] = f1_score(y[te], m.predict_proba(X[te])[:, 1] >= thr, zero_division=0)
        a = f[mlab] - f['inbar' if mlab == 'stav' else 'stav']
        adv[mlab].append(a)
        lines.append(f"| {mname} | {mlab} | {', '.join(by_mouse[mname])} | {f['stav']:.3f} | {f['inbar']:.3f} | {a:+.3f} |")
        print(f"[labeller] {mname} ({mlab}): Stav-only {f['stav']:.3f}, Inbar-only {f['inbar']:.3f}", flush=True)
    for k in ['stav', 'inbar']:
        v = np.array(adv[k])
        lines.append(f"\n{k.capitalize()} mice: own-lab model better for {(v > 0).sum()}/{len(v)}, mean advantage {v.mean():+.3f}.")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print("\n" + text)


if __name__ == '__main__':
    main()
