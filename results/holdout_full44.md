# Full 44-feature model, held-out scores

Model: `models/image/suite2p_image_lgb_heldout_inbar7_8_11_12.pkl` (baseline 27 + 17 image features).
Trained on 25 sessions: Stav4-14, Stav16-21, inbar1, Inbar2-6, Inbar9-10 (74,237 ROIs).
Stav1 dropped as a duplicate of Stav5; Stav2 and Stav15 have no feature table (no F.npy);
Stav3/21 excluded as before. Threshold 0.70 and 414 trees from 5-fold session-grouped inner CV.

| session | ROIs | labelled cells | precision | recall | F1 @ 0.70 | best F1 (thr) | AUC |
|---|---|---|---|---|---|---|---|
| stav22 | 1201 | 763 | 0.960 | 0.663 | 0.784 | 0.868 (0.07) | 0.911 |
| Inbar7 | 1611 | 260 | 0.886 | 0.931 | 0.908 | 0.911 (0.74) | 0.993 |
| Inbar8 | 3446 | 223 | 0.892 | 0.892 | 0.892 | 0.894 (0.71) | 0.994 |
| Inbar11 | 3422 | 184 | 0.895 | 0.788 | 0.838 | 0.838 (0.70) | 0.991 |
| Inbar12 | 1867 | 160 | 0.830 | 0.731 | 0.777 | 0.795 (0.56) | 0.984 |
| pooled | 11547 | 1590 | 0.912 | 0.760 | 0.829 | | 0.980 |

Inbar15 is a copy of Inbar7 and is not scored separately.

    PYTHONPATH=. python scripts/score_heldout.py --model models/image/suite2p_image_lgb_heldout_inbar7_8_11_12.pkl --test stav22 Inbar7 Inbar8 Inbar11 Inbar12
