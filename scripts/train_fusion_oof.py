"""
Joint (fusion) network trained end to end, with nested out-of-fold scores on the
same session folds as train_cnn_oof.py / evaluate_cv.py (repeat 0).

Branches (--branches, comma-separated):
  image  4-channel 32x32 ROI crops (prepare_cnn_crops.py) -> small 2D CNN -> 64
  tab    the 27 baseline features, robust-standardized with training-fold
         median/IQR, clipped to +-10, NaN -> 0 -> MLP -> 32
  trace  noise-normalized trace, 1000 bins x (mean, max) (prepare_trace_arrays.py)
         -> 1D CNN with global max+avg pooling -> 64
The branch embeddings are concatenated before a shared head.

The output file has the same layout as cnn_oof.npz, so cnn_stack_cv.py evaluates
it unchanged (threshold from inner-OOF scores, stacking into LightGBM).

    PYTHONPATH=. OMP_NUM_THREADS=6 python scripts/train_fusion_oof.py --branches image,tab \
        --crops .../crops --traces .../traces --out .../fusion
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fe_engine.fe_definitions import ACTIVE_FEATURES
from scripts.evaluate_cv import load_tables
from scripts.train_cnn_oof import DEVICE, augment


class Fusion(nn.Module):
    def __init__(self, branches, n_tab):
        super().__init__()
        self.branches = branches
        dim = 0
        if 'image' in branches:
            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            self.img = nn.Sequential(block(4, 16), block(16, 32), block(32, 64), nn.Flatten(),
                                     nn.Dropout(0.3), nn.Linear(64 * 4 * 4, 64), nn.ReLU())
            dim += 64
        if 'tab' in branches:
            self.tab = nn.Sequential(nn.Linear(n_tab, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU())
            dim += 32
        if 'trace' in branches:
            def block1d(i, o, k, s):
                return nn.Sequential(nn.Conv1d(i, o, k, stride=s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())
            self.trace = nn.Sequential(block1d(2, 16, 7, 2), block1d(16, 32, 7, 2), nn.MaxPool1d(2),
                                       block1d(32, 64, 5, 1), nn.MaxPool1d(2), block1d(64, 64, 5, 1))
            self.trace_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
            dim += 64
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def branch_embeddings(self, img=None, tab=None, trace=None):
        z = {}
        if 'image' in self.branches:
            z['image'] = self.img(img)
        if 'tab' in self.branches:
            z['tab'] = self.tab(tab)
        if 'trace' in self.branches:
            h = self.trace(trace)
            z['trace'] = self.trace_fc(torch.cat([h.amax(2), h.mean(2)], 1))
        return z

    def forward(self, img=None, tab=None, trace=None):
        z = self.branch_embeddings(img, tab, trace)
        return self.head(torch.cat([z[b] for b in self.branches], 1)).squeeze(1)


class TabScaler:
    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
        self.scale = np.where(iqr > 0, iqr, 1.0)
        return self

    def transform(self, X):
        return np.nan_to_num(np.clip((X - self.med) / self.scale, -10, 10)).astype(np.float32)


def batch_inputs(data, idx, branches, scaler, train):
    kw = {}
    if 'image' in branches:
        x = torch.from_numpy(data['img'][idx].astype(np.float32)).to(DEVICE)
        kw['img'] = augment(x) if train else x
    if 'tab' in branches:
        kw['tab'] = torch.from_numpy(scaler.transform(data['tab'][idx])).to(DEVICE)
    if 'trace' in branches:
        kw['trace'] = torch.from_numpy(data['trace'][idx].astype(np.float32)).to(DEVICE)
    return kw


def train(data, rows, y, branches, epochs, seed, batch=256):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = TabScaler().fit(data['tab'][rows]) if 'tab' in branches else None
    net = Fusion(branches, data['tab'].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(rows) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    yr = y[rows]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((yr == 0).sum() / max((yr == 1).sum(), 1),
                                                           dtype=torch.float32, device=DEVICE))
    net.train()
    for _ in range(epochs):
        perm = rng.permutation(len(rows))
        for i in range(0, len(rows), batch):
            b = rows[perm[i:i + batch]]
            loss = loss_fn(net(**batch_inputs(data, b, branches, scaler, True)),
                           torch.from_numpy(y[b].astype(np.float32)).to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net, scaler


@torch.no_grad()
def trace_embedding(net, scaler, data, rows, batch=1024):
    """64-dim trace-branch embedding (no augmentation on the trace branch)."""
    net.eval()
    out = []
    for i in range(0, len(rows), batch):
        b = rows[i:i + batch]
        out.append(net.branch_embeddings(**batch_inputs(data, b, net.branches, scaler, False))['trace'].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict(net, scaler, data, rows, branches, batch=1024):
    net.eval()
    out = []
    for i in range(0, len(rows), batch):
        b = rows[i:i + batch]
        kw = batch_inputs(data, b, branches, scaler, False)
        logits = net(**kw)
        if 'image' in branches:  # light TTA: horizontal flip of the image branch only
            kw['img'] = kw['img'].flip(3).contiguous()
            logits = (logits + net(**kw)) / 2
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def load_per_session(dir_, key, g, y):
    arrs = []
    for sess in dict.fromkeys(g):
        d = np.load(Path(dir_) / (Path(sess).parent.name + '__' + Path(sess).name + '.npz'))
        assert (d['y'] == y[g == sess]).all(), sess
        arrs.append(d[key])
    return np.concatenate(arrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--crops', required=True)
    ap.add_argument('--traces', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--branches', default='image,tab')
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--inner', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--max-folds', type=int, default=None, help="stop after this many outer folds (timing runs)")
    ap.add_argument('--save-emb', action='store_true', help="also save the trace-branch embedding (nested OOF layout)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print(f"device: {DEVICE}", flush=True)
    branches = args.branches.split(',')

    X_tab, y, g, names = load_tables(args.tables, args.exclude)
    col = {n: i for i, n in enumerate(names)}
    data = {'tab': X_tab[:, [col[f] for f in ACTIVE_FEATURES]].astype(np.float32)}
    if 'image' in branches:
        data['img'] = load_per_session(args.crops, 'crops', g, y)
    if 'trace' in branches:
        data['trace'] = load_per_session(args.traces, 'traces', g, y)
    print(f"{len(y)} ROIs, branches={branches}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"fusion_{'_'.join(branches)}_oof.npz"
    gkf = GroupKFold(n_splits=args.folds, shuffle=True, random_state=0)  # == evaluate_cv repeat 0
    oof = np.full((args.folds, len(y)), np.nan, dtype=np.float32)
    emb = (np.full((args.folds, len(y), 64), np.nan, dtype=np.float16)
           if args.save_emb and 'trace' in branches else None)
    for k, (tr, va) in enumerate(gkf.split(X_tab, y, g)):
        if args.max_folds is not None and k >= args.max_folds:
            break
        t0 = time.time()
        for j, (itr, iva) in enumerate(GroupKFold(n_splits=args.inner).split(tr, y[tr], g[tr])):
            net, sc = train(data, tr[itr], y, branches, args.epochs, seed=100 * k + j)
            oof[k, tr[iva]] = predict(net, sc, data, tr[iva], branches)
            if emb is not None:
                emb[k, tr[iva]] = trace_embedding(net, sc, data, tr[iva])
        net, sc = train(data, tr, y, branches, args.epochs, seed=100 * k + 99)
        oof[k, va] = predict(net, sc, data, va, branches)
        if emb is not None:
            emb[k, va] = trace_embedding(net, sc, data, va)
        np.savez(out_file, cnn=oof, y=y, groups=g, folds_done=k + 1, branches=args.branches,
                 **({'emb_trace': emb} if emb is not None else {}))
        print(f"fold {k}: val AUC={roc_auc_score(y[va], oof[k, va]):.4f} "
              f"inner-OOF AUC={roc_auc_score(y[tr], oof[k, tr]):.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
