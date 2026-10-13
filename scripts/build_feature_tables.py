"""
Build one feature table per session (baseline trace/morphology features, the
intensity-normalized trace features and the ops.npy image features) so CV
experiments can run without touching F.npy again.

Labels follow LABEL_PRIORITY: iscell_final.npy (curated) > iscell_backup_before_AI.npy
(the human labels apply_AI.py saved before overwriting) > iscell.npy. A plain
iscell.npy is only used when apply_AI.py never ran on the session (no backup).
Yael sessions are always skipped. --relabel rewrites 'y' in existing tables
without recomputing features.

    PYTHONPATH=. python scripts/build_feature_tables.py \
        --cache ../suite2p-iscell-prediction/preprocessed_cache --out feature_tables
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES, TRACE_NORM_FEATURES
from fe_engine.fe_loop_runner import extract_features_dataset
from fe_engine.image_features import IMAGE_FEATURES, compute_image_features

TRACE_FEATURES = list(dict.fromkeys(ACTIVE_FEATURES + TRACE_NORM_FEATURES))
LABEL_PRIORITY = ['iscell_final.npy', 'iscell_backup_before_AI.npy', 'iscell.npy']


def load_labels(session_path):
    for name in LABEL_PRIORITY:
        f = Path(session_path) / name
        if f.exists():
            return np.load(f)[:, 0].astype(np.int8), name
    raise FileNotFoundError(f"no label file in {session_path}")


def build_one(npz_path, out_dir):
    d = np.load(npz_path, allow_pickle=True)
    session = {k: d[k] for k in d.files}
    path = Path(str(session['session_path']))
    out_file = out_dir / f"{path.parent.name}__{path.name}.npz"
    if out_file.exists():
        return f"{path.name}: cached"
    y, label_file = load_labels(path)
    assert len(y) == len(session['y']), f"label length mismatch in {path}"
    session['y'] = y
    session['group_id'] = np.zeros(len(y), dtype=np.int32)

    X_trace, _, _ = extract_features_dataset([session], TRACE_FEATURES)

    if (path / 'ops.npy').exists():
        img = compute_image_features(path)
        X_img = np.column_stack([img[k] for k in IMAGE_FEATURES])
    else:
        X_img = np.full((len(y), len(IMAGE_FEATURES)), np.nan, dtype=np.float32)

    np.savez_compressed(
        out_file, X_trace=X_trace.astype(np.float32), trace_names=np.array(TRACE_FEATURES),
        X_img=X_img.astype(np.float32), img_names=np.array(IMAGE_FEATURES),
        y=y.astype(np.int8), label_file=label_file, session=str(path),
    )
    return f"{path.name}: {len(y)} ROIs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help="preprocessed_cache dir (labels + morphology)")
    ap.add_argument('--out', default='feature_tables')
    ap.add_argument('--n-jobs', type=int, default=2)
    ap.add_argument('--relabel', action='store_true', help="only refresh labels in existing tables")
    args = ap.parse_args()

    if args.relabel:
        for f in sorted(Path(args.out).glob('*.npz')):
            d = dict(np.load(f, allow_pickle=True))
            y, label_file = load_labels(str(d['session']))
            changed = int((y != d['y']).sum())
            d['y'], d['label_file'] = y, label_file
            np.savez_compressed(f, **d)
            print(f"{f.name}: {label_file}, {changed} labels changed")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    files = []
    for f in sorted(Path(args.cache).glob('preprocessed_*.npz')):
        p = str(np.load(f, allow_pickle=True)['session_path'])
        if 'yael' in p.lower():
            print(f"Skipping Yael session: {p}")
            continue
        files.append(f)
    print(f"Building {len(files)} sessions -> {out_dir}")
    for r in Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(build_one)(f, out_dir) for f in files):
        print(r)


if __name__ == '__main__':
    main()
