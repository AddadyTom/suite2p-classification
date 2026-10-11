# Demo models: 32-feature small model and labeller-specific models

All on the confirmed 32-feature set (results/feature_sets_confirmed.md): 21 trace + compact, mrs, radius,
solidity + 7 image features (max_proj_lam_corr, Vcorr_in_mean, meanImg_neuropil_contrast, meanImgE_lam_corr,
meanImgE_lam_corr_patch, max_proj_ratio, Vcorr_in_minus_ring). Threshold and trees from 5-fold session-grouped
inner CV on the training sessions (F1 objective). Held-out sessions never used for training or tuning.
Scored with scripts/score_heldout.py.

## Variant A: Inbar5 held out (models/demo_inbar5/)

| Model | trained on | thr | Inbar5 precision | recall | F1 | AUC |
|---|---|---|---|---|---|---|
| small32 (both labs) | 28 sessions | 0.67 | 0.774 | 0.864 | 0.817 | 0.981 |
| stav_only32 | 17 Stav | 0.64 | 0.769 | 0.859 | 0.812 | 0.980 |
| inbar_only32 | 11 Inbar | 0.76 | 0.802 | 0.856 | **0.828** | 0.984 |

## Variant B: Stav22 + Inbar7/8/11/12 held out (models/demo_inbar7_8_11_12/), same hold-out as models/image/suite2p_image_lgb_heldout_inbar7_8_11_12

| Held-out session | 44 features (other session) | small32 | stav_only32 (17 Stav) | inbar_only32 (8 Inbar) |
|---|---|---|---|---|
| stav22 | 0.784 | 0.802 | **0.802** | 0.786 |
| Inbar7 | 0.908 | 0.907 | 0.878 | **0.903** |
| Inbar8 | 0.892 | 0.880 | **0.881** | 0.873 |
| Inbar11 | 0.838 | 0.806 | **0.794** | 0.741 |
| Inbar12 | 0.777 | 0.749 | **0.777** | 0.712 |
| pooled | 0.829 | 0.828 | 0.824 | 0.808 |

(F1 at each model's stored threshold; bold = better of the two single-lab models.)

Reading: the Inbar-only model beats the Stav-only model on Inbar5 (+0.016) and Inbar7 (+0.025), but loses on
Inbar8, Inbar11 and Inbar12, where it was trained on only 8 Inbar sessions. Together with
results/labeller_experiments.md: Stav's labels are consistently best predicted by Stav training; for Inbar the
own-lab advantage appears on some sessions and is outweighed elsewhere by the smaller training set.
