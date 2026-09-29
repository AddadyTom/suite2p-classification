# Feature Engineering Iteration Report (Candidate Run)

## Performance Summary
- **F1-Score**: 0.8428 ± 0.0501
- **Precision**: 0.8484 ± 0.0303
- **Recall**: 0.8390 ± 0.0730

## Active Features (30)
- `number_of_bright_pixels`
- `bright_pixels_ratio`
- `solidity`
- `mrs`
- `skew_f`
- `std_f`
- `max_to_mean_f`
- `cv_f`
- `skew_fneu`
- `corr_f_fneu`
- `skew_fcorr`
- `std_fcorr`
- `q10`
- `q25`
- `q50`
- `q75`
- `q90`
- `q95`
- `q99`
- `avg_asym`
- `max_asym`
- `max_width`
- `range_fcorr`
- `range_f`
- `snr`
- `activity_ratio`
- `peak_density`
- `compact`
- `aspect_ratio`
- `peak_to_q99_ratio`

## SHAP Feature Importance (Top 15)
| Rank | Feature | Mean Absolute SHAP |
|---|---|---|
| 1 | `number_of_bright_pixels` | 0.92746 |
| 2 | `q99` | 0.85529 |
| 3 | `q50` | 0.48433 |
| 4 | `skew_fcorr` | 0.47603 |
| 5 | `mrs` | 0.43700 |
| 6 | `max_to_mean_f` | 0.41465 |
| 7 | `cv_f` | 0.39072 |
| 8 | `corr_f_fneu` | 0.38567 |
| 9 | `range_fcorr` | 0.30158 |
| 10 | `compact` | 0.27735 |
| 11 | `range_f` | 0.14704 |
| 12 | `bright_pixels_ratio` | 0.13750 |
| 13 | `solidity` | 0.11096 |
| 14 | `avg_asym` | 0.11043 |
| 15 | `std_f` | 0.09550 |


## Error Diagnostic Analysis

### False Positive Analysis (Noise predicted as Cells)
Features that failed to assign negative weight to reject noise:
| Rank | Feature | SHAP Diff (FP - TN) | FP Mean | TN Mean |
|---|---|---|---|---|
| 1 | `q99` | +1.2968 | 13.8061 | 5.2958 |
| 2 | `number_of_bright_pixels` | +0.9364 | 89.8386 | 76.6953 |
| 3 | `skew_fcorr` | +0.7837 | 1.5032 | 0.4325 |


### False Negative Analysis (Cells predicted as Noise)
Features that failed to assign positive weight to identify cells:
| Rank | Feature | SHAP Diff (TP - FN) | FN Mean | TP Mean |
|---|---|---|---|---|
| 1 | `q99` | +1.6987 | 7.9197 | 23.0652 |
| 2 | `range_fcorr` | +0.4924 | 27.7329 | 59.6453 |
| 3 | `skew_fcorr` | +0.4813 | 0.8859 | 1.9800 |


## Recommendations & Next Steps
1. **FP Culprits**: Examine if normalizing these features or adding high-frequency noise parameters will help reject noise artifacts.
2. **FN Culprits**: Look for spatial or trace metrics that differentiate the missed cells from noise (e.g. baseline-subtracted variances or asymmetric peak shapes).
