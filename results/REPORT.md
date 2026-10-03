# Image features & leak-free CV — results

**Data:** 29 sessions, 84,583 ROIs, 17.6% cells. All non-Yael sessions except `stav22` (held out in train_model.py), `Stav3/21` (suspect labels: 70% positive) and `Stav1` (byte-identical copy of Stav5: same F/stat/labels, with an `ops.npy` from another recording).
Labels: `iscell_final.npy` > `iscell_backup_before_AI.npy` > `iscell.npy`.

**Protocol:** GroupKFold by session, 5 folds × 3 shuffled repeats = 15 paired folds. In each training fold, an inner 3-fold session-grouped CV picks `n_estimators` (early stopping) and the F1-optimal threshold. The validation fold is never used for tuning. Metrics are pooled per fold, mean ± std over folds. ΔF1 is paired per fold. LightGBM hyperparameters are the same as in train_model.py.

## Results

| Feature set | # feat | Precision | Recall | F1 | ΔF1 vs baseline (paired) |
|---|---|---|---|---|---|
| baseline (27), old leaky protocol¹ | 27 | 0.820 ± 0.026 | 0.844 ± 0.030 | 0.832 ± 0.026 | |
| **baseline (27)** | 27 | 0.813 ± 0.029 | 0.846 ± 0.064 | **0.827 ± 0.027** | |
| + trace-norm | 38 | 0.814 ± 0.027 | 0.848 ± 0.065 | 0.828 ± 0.028 | +0.001 ± 0.005 (8/15 up) |
| trace-norm replaces raw q/range | 29 | 0.809 ± 0.028 | 0.849 ± 0.071 | 0.826 ± 0.029 | −0.001 ± 0.006 (7/15 up) |
| + image: ring contrast | 36 | 0.822 ± 0.029 | 0.867 ± 0.053 | 0.843 ± 0.023 | +0.015 ± 0.008 (14/15 up) |
| + image: lam correlation | 35 | 0.843 ± 0.023 | 0.860 ± 0.051 | 0.850 ± 0.023 | +0.023 ± 0.008 (15/15 up) |
| **+ image (all)** | 44 | 0.843 ± 0.024 | 0.865 ± 0.049 | **0.853 ± 0.025** | **+0.026 ± 0.008 (15/15 up)** |
| replace + image | 46 | 0.839 ± 0.026 | 0.865 ± 0.049 | 0.851 ± 0.025 | +0.024 ± 0.007 (15/15 up) |

¹ Early stopping and threshold both tuned on the validation fold, as in train_model.py, so it is optimistic. The README's 0.856 came from a different session set and labels.

**Thresholds:** nested CV picks about 0.67 for the baseline and about 0.6–0.67 with image features. eval_session.py (0.66/0.69) and apply_AI.py (0.61–0.69, keyed on feature count) hard-code their thresholds; these should come from the nested CV of the model actually shipped.

## Per-feature-group gain
* **Image features: +0.026 F1**, up in 15/15 paired folds and 27/29 sessions. The largest gains are on the hardest sessions: Inbar12 0.685 → 0.805, Inbar11 0.767 → 0.825, inbar1 0.755 → 0.805, Inbar3 0.693 → 0.742. lam-vs-image correlation carries most of it (+0.023 alone); ring contrast adds +0.015 alone.
* **Intensity-normalized trace features: ≈ 0** (+0.001 added, −0.001 as a replacement). They are as informative as the raw q/range features but not more, because the existing features are already normalized by a session-level noise scale. Replacing them is optional and doesn't change accuracy.

## SHAP — top features (mean |SHAP|, "replace + image" model, outer validation folds)
1. max_proj_lam_corr 1.08 · 2. mrs 0.87 · 3. Vcorr_in_mean 0.52 · 4. skew_fcorr 0.42 · 5. std_fcorr 0.33 · 6. meanImg_neuropil_contrast 0.29 · 7. std_f 0.27 · 8. corr_f_fneu 0.25 · 9. radius 0.24 · 10. q999_over_noise 0.23 · 11. compact 0.22 · 12. range_ratio_f_fneu 0.21 · 13. mean_diff_f_fneu 0.20 · 14. meanImgE_lam_corr_patch 0.20 · 15. solidity 0.17

## Per-session F1 (baseline → + image (all), mean over the 3 repeats)
| Session | Baseline | + image | Δ |
|---|---|---|---|
| Inbar12 | 0.685 | 0.805 | +0.119 |
| Inbar11 | 0.767 | 0.825 | +0.058 |
| inbar1 | 0.755 | 0.805 | +0.050 |
| Inbar3 | 0.693 | 0.742 | +0.049 |
| Stav17 | 0.811 | 0.859 | +0.048 |
| Stav8 | 0.807 | 0.852 | +0.045 |
| Inbar2 | 0.780 | 0.822 | +0.042 |
| Inbar6 | 0.820 | 0.861 | +0.040 |
| Stav6 | 0.827 | 0.865 | +0.038 |
| Stav10 | 0.838 | 0.876 | +0.037 |
| Inbar9 | 0.763 | 0.799 | +0.037 |
| Stav5 | 0.855 | 0.888 | +0.033 |
| Stav12 | 0.858 | 0.886 | +0.029 |
| Stav9 | 0.840 | 0.868 | +0.028 |
| Stav11 | 0.860 | 0.888 | +0.028 |
| Inbar10 | 0.786 | 0.813 | +0.027 |
| Inbar5 | 0.787 | 0.812 | +0.026 |
| Inbar8 | 0.864 | 0.887 | +0.022 |
| inbar4 | 0.848 | 0.868 | +0.021 |
| Stav18 | 0.810 | 0.830 | +0.020 |
| Inbar7 | 0.882 | 0.900 | +0.018 |
| Stav13 | 0.891 | 0.907 | +0.017 |
| Stav14 | 0.746 | 0.761 | +0.015 |
| Stav7 | 0.822 | 0.834 | +0.013 |
| Stav21 | 0.820 | 0.832 | +0.012 |
| Stav19 | 0.866 | 0.874 | +0.007 |
| Stav4 | 0.902 | 0.903 | +0.001 |
| Stav20 | 0.887 | 0.885 | −0.002 |
| Stav16 | 0.847 | 0.843 | −0.004 |

## Data issues found
* `Stav1` is a copy of `Stav5`, and its `ops.npy` belongs to another recording. The image features include a label-free ops/stat alignment check that catches this (score 0.05 vs ≥ 0.27 for matching sessions). In deployment, sessions that fail it fall back to the trace/morphology model.
* The old train_model.py never read `iscell_final.npy` (final-only sessions were skipped). For sessions without a backup, it used an `iscell.npy` that apply_AI.py had overwritten (Inbar6, inbar4: about 8–9% of ROIs differ from final). Both are fixed on this branch.
