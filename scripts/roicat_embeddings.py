"""
Pretrained ROICaT ROInet embeddings (latent vectors) per ROI, one .npz per session.

Runs in a separate environment with roicat installed (it pins its own torch):
    MPLBACKEND=Agg OMP_NUM_THREADS=3 <roicat-venv>/bin/python scripts/roicat_embeddings.py \
        --sessions-from /mnt/other_ubunthu/suite2p_work/crops --out /mnt/other_ubunthu/suite2p_work/roicat/emb

Sessions are taken from the CNN crop folder, i.e. the same de-duplicated,
non-Yael set used by the CV. um_per_pixel is not stored in ops.npy here; it is
a guess (--um-per-pixel) that only sets how ROInet rescales the ROI images.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions-from', required=True, help="dir of <parent>__<session>.npz crop files")
    ap.add_argument('--out', required=True)
    ap.add_argument('--net-dir', default=None, help="where ROInet weights are cached")
    ap.add_argument('--um-per-pixel', type=float, default=1.5)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--only', nargs='*', default=None, help="session names to embed (pilot subset)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import roicat
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net_dir = Path(args.net_dir) if args.net_dir else out.parent / 'roinet'
    net_dir.mkdir(parents=True, exist_ok=True)
    roinet = roicat.ROInet.ROInet_embedder(dir_networkFiles=str(net_dir), device='cpu',
                                           download_method='check_local_first',
                                           forward_pass_version='latent', verbose=False)

    for f in sorted(Path(args.sessions_from).glob('*__*.npz')):
        sess = Path(str(np.load(f)['session']))
        target = out / f.name
        if target.exists() or (args.only and sess.name not in args.only):
            continue
        t0 = time.time()
        data = roicat.data_importing.Data_suite2p(
            paths_statFiles=[str(sess / 'stat.npy')], paths_opsFiles=[str(sess / 'ops.npy')],
            um_per_pixel=args.um_per_pixel, new_or_old_suite2p='new', verbose=False)
        roinet.generate_dataloader(ROI_images=data.ROI_images, um_per_pixel=data.um_per_pixel,
                                   pref_plot=False, batchSize_dataloader=args.batch,
                                   numWorkers_dataloader=0, persistentWorkers_dataloader=False,
                                   pinMemory_dataloader=False)
        latents = roinet.generate_latents()
        latents = latents.numpy() if hasattr(latents, 'numpy') else np.asarray(latents)
        n_rois = len(np.load(sess / 'stat.npy', allow_pickle=True))
        assert latents.shape[0] == n_rois, (sess, latents.shape, n_rois)
        np.savez(target, latents=latents.astype(np.float32), session=str(sess))
        print(f"{sess.name}: {latents.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
