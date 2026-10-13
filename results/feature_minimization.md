Backward elimination (trace features always kept), 28 sessions, 5 session folds, nested threshold. Excluded: stav22, stav3/21, Inbar5.

| # features | F1 | ΔF1 vs 44 (paired) | folds worse | dropped next |
|---|---|---|---|---|
| 44 | 0.8513 ± 0.0246 | +0.0000 | 0/5 | Vcorr_lam_corr_patch |
| 43 | 0.8504 ± 0.0227 | -0.0009 | 3/5 | area_to_radius_sq |
| 42 | 0.8519 ± 0.0225 | +0.0006 | 3/5 | meanImg_contrast_z |
| 41 | 0.8523 ± 0.0247 | +0.0010 | 1/5 | Vcorr_contrast_z |
| 40 | 0.8511 ± 0.0238 | -0.0003 | 2/5 | meanImgE_contrast_z |
| 39 | 0.8516 ± 0.0227 | +0.0003 | 3/5 | aspect_ratio |
| 38 | 0.8522 ± 0.0229 | +0.0009 | 3/5 | max_proj_lam_corr_patch |
| 37 | 0.8509 ± 0.0247 | -0.0004 | 1/5 | Vcorr_lam_corr |
| 36 | 0.8504 ± 0.0251 | -0.0010 | 2/5 | max_proj_contrast_z |
| 35 | 0.8513 ± 0.0252 | -0.0001 | 3/5 | meanImg_lam_corr_patch |
| 34 | 0.8513 ± 0.0244 | -0.0000 | 2/5 | meanImg_lam_corr |
| 33 | 0.8513 ± 0.0247 | -0.0000 | 3/5 | meanImg_ratio |
| 32 | 0.8522 ± 0.0231 | +0.0009 | 1/5 | Vcorr_in_minus_ring |
| 31 | 0.8485 ± 0.0213 | -0.0028 | 4/5 | solidity |
| 30 | 0.8508 ± 0.0231 | -0.0005 | 3/5 | meanImgE_lam_corr |
| 29 | 0.8488 ± 0.0221 | -0.0025 | 3/5 | max_proj_ratio |
| 28 | 0.8502 ± 0.0229 | -0.0012 | 3/5 | radius |
| 27 | 0.8470 ± 0.0222 | -0.0043 | 4/5 | compact |
| 26 | 0.8412 ± 0.0224 | -0.0102 | 4/5 | meanImg_neuropil_contrast |
| 25 | 0.8406 ± 0.0205 | -0.0108 | 4/5 | meanImgE_lam_corr_patch |
| 24 | 0.8328 ± 0.0216 | -0.0185 | 5/5 | Vcorr_in_mean |
| 23 | 0.8281 ± 0.0250 | -0.0232 | 5/5 | mrs |
| 22 | 0.7981 ± 0.0247 | -0.0532 | 5/5 | max_proj_lam_corr |
| 21 | 0.7753 ± 0.0321 | -0.0760 | 5/5 | — |
