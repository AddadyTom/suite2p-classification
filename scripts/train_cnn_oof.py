"""
Train a small CNN on the aligned ROI crops (prepare_cnn_crops.py) with the same
session-grouped outer folds as evaluate_cv.py (repeat 0), saving only
out-of-fold probabilities.

For each outer fold k, the saved column cnn[k] holds
  * validation rows: prediction of a CNN trained on all outer-training sessions
  * training rows:   inner out-of-fold predictions (GroupKFold over the
                     outer-training sessions), so a model stacked on cnn[k]
                     never sees a CNN score that was fit on its own label.
No early stopping or model selection uses held-out labels: fixed epochs.

    PYTHONPATH=. OMP_NUM_THREADS=4 python scripts/train_cnn_oof.py \
        --crops /mnt/other_ubunthu/suite2p_work/crops --out /mnt/other_ubunthu/suite2p_work/cnn
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
from scripts.evaluate_cv import load_tables

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ROICNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(in_ch, 16), block(16, 32), block(32, 64))  # 32 -> 4
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 4 * 4, 64),
                                  nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def augment(x):
    # Random element of the dihedral group (flips + 90° rotations): ROI labels are orientation-free
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3)).contiguous()


def train(X, y, epochs, seed, batch=256):
    torch.manual_seed(seed)
    net = ROICNN(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(y) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    yt = torch.from_numpy(y.astype(np.float32))
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            idx = perm[i:i + batch]
            xb = augment(torch.from_numpy(X[idx.numpy()].astype(np.float32)).to(DEVICE))
            loss = loss_fn(net(xb), yt[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net


@torch.no_grad()
def embed(net, X, batch=1024):
    """64-dim penultimate activations (input to the final linear layer), flip-averaged."""
    net.eval()
    trunk = lambda x: net.head[:-1](net.features(x))
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(DEVICE)
        out.append(((trunk(xb) + trunk(xb.flip(3).contiguous())) / 2).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict(net, X, batch=1024):
    net.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(DEVICE)
        # light test-time augmentation: identity + horizontal flip
        logits = (net(xb) + net(xb.flip(3).contiguous())) / 2
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='feature_tables')
    ap.add_argument('--crops', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', nargs='*', default=['stav22', 'stav3/21'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--inner', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--save-emb', action='store_true',
                    help="also save the 64-dim embedding per ROI, with the same nested OOF layout as the scores")
    ap.add_argument('--tag', default='', help="suffix for the output file name")
    ap.add_argument('--outer-emb', action='store_true',
                    help="embedding-feature mode: one CNN per outer fold trained on the training sessions; its "
                         "embedding is saved for ALL rows (train and validation) so downstream models see one "
                         "consistent embedding space. No inner models; 'cnn' holds validation-row scores only.")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print(f"device: {DEVICE}", flush=True)

    X_tab, y, g, _ = load_tables(args.tables, args.exclude)
    crops = []
    for sess in dict.fromkeys(g):
        d = np.load(Path(args.crops) / (Path(sess).parent.name + '__' + Path(sess).name + '.npz'))
        assert (d['y'] == y[g == sess]).all(), sess
        crops.append(d['crops'])
    X = np.concatenate(crops)
    print(f"{len(y)} ROIs, crops {X.shape}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gkf = GroupKFold(n_splits=args.folds, shuffle=True, random_state=0)  # == evaluate_cv repeat 0
    cnn = np.full((args.folds, len(y)), np.nan, dtype=np.float32)
    emb = (np.full((args.folds, len(y), 64), np.nan, dtype=np.float16)
           if args.save_emb or args.outer_emb else None)
    out_file = out / f"cnn_oof{args.tag}.npz"
    for k, (tr, va) in enumerate(gkf.split(X_tab, y, g)):
        t0 = time.time()
        if args.outer_emb:
            net = train(X[tr], y[tr], args.epochs, seed=100 * k + 99)
            cnn[k, va] = predict(net, X[va])
            emb[k] = embed(net, X)
            np.savez(out_file, cnn=cnn, y=y, groups=g, folds_done=k + 1, emb=emb, mode='outer_emb')
            print(f"fold {k}: val AUC={roc_auc_score(y[va], cnn[k, va]):.4f} ({time.time() - t0:.0f}s)", flush=True)
            continue
        for j, (itr, iva) in enumerate(GroupKFold(n_splits=args.inner).split(tr, y[tr], g[tr])):
            net = train(X[tr[itr]], y[tr[itr]], args.epochs, seed=100 * k + j)
            cnn[k, tr[iva]] = predict(net, X[tr[iva]])
            if emb is not None:
                emb[k, tr[iva]] = embed(net, X[tr[iva]])
        net = train(X[tr], y[tr], args.epochs, seed=100 * k + 99)
        cnn[k, va] = predict(net, X[va])
        if emb is not None:
            emb[k, va] = embed(net, X[va])
        np.savez(out_file, cnn=cnn, y=y, groups=g, folds_done=k + 1,
                 **({'emb': emb} if emb is not None else {}))
        print(f"fold {k}: val AUC={roc_auc_score(y[va], cnn[k, va]):.4f} "
              f"inner-OOF AUC={roc_auc_score(y[tr], cnn[k, tr]):.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
