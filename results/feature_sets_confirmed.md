28 sessions (excluded stav22, stav3/21, Inbar5), 3x5 session folds, nested threshold.

| # features | Precision | Recall | F1 | ΔF1 vs 44 (paired) | folds not worse |
|---|---|---|---|---|---|
| 44 | 0.8385 | 0.8716 | 0.8528 ± 0.0234 | — | 15/15 |
| 32 | 0.8402 | 0.8685 | 0.8527 ± 0.0221 | -0.0002 ± 0.0037 | 9/15 |
| 30 | 0.8388 | 0.8664 | 0.8506 ± 0.0232 | -0.0022 ± 0.0043 | 4/15 |
| 28 | 0.8386 | 0.8654 | 0.8501 ± 0.0219 | -0.0027 ± 0.0032 | 3/15 |

**32 features:** corr_f_fneu, max_width, peak_to_q95_ratio, peak_to_q99_ratio, q10, q25, q50, q75, q90, q95, q99, range_f, range_fcorr, skew_diff_fcorr, skew_f, skew_fcorr, skew_fneu, std_f, std_fcorr, mean_diff_f_fneu, range_ratio_f_fneu, compact, mrs, radius, solidity, meanImgE_lam_corr, max_proj_lam_corr, meanImgE_lam_corr_patch, max_proj_ratio, Vcorr_in_mean, Vcorr_in_minus_ring, meanImg_neuropil_contrast

**30 features:** corr_f_fneu, max_width, peak_to_q95_ratio, peak_to_q99_ratio, q10, q25, q50, q75, q90, q95, q99, range_f, range_fcorr, skew_diff_fcorr, skew_f, skew_fcorr, skew_fneu, std_f, std_fcorr, mean_diff_f_fneu, range_ratio_f_fneu, compact, mrs, radius, meanImgE_lam_corr, max_proj_lam_corr, meanImgE_lam_corr_patch, max_proj_ratio, Vcorr_in_mean, meanImg_neuropil_contrast

**28 features:** corr_f_fneu, max_width, peak_to_q95_ratio, peak_to_q99_ratio, q10, q25, q50, q75, q90, q95, q99, range_f, range_fcorr, skew_diff_fcorr, skew_f, skew_fcorr, skew_fneu, std_f, std_fcorr, mean_diff_f_fneu, range_ratio_f_fneu, compact, mrs, radius, max_proj_lam_corr, meanImgE_lam_corr_patch, Vcorr_in_mean, meanImg_neuropil_contrast

