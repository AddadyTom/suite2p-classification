CV: 28 sessions (excluded stav22, stav3/21, Inbar5), 3x5 session folds, nested threshold. stav22: model trained on all those sessions, stav22 never used.

| Feature set | CV precision | CV recall | CV F1 | ΔF1 vs with-shape (paired) |
|---|---|---|---|---|
| 32 (small model) | 0.840 | 0.868 | 0.8527 ± 0.0221 |  |
| 31 = 32 without radius (size) | 0.840 | 0.868 | 0.8520 ± 0.0223 | -0.0006 ± 0.0029 (7/15 folds not worse) |
| 32, radius clipped at 3.5 px | 0.841 | 0.867 | 0.8525 ± 0.0220 | -0.0002 ± 0.0016 (6/15 folds not worse) |
| 32, neuropil skew scaled by amplitude | 0.842 | 0.865 | 0.8517 ± 0.0217 | -0.0010 ± 0.0024 (4/15 folds not worse) |
| 32, radius clipped + scaled neuropil skew | 0.841 | 0.867 | 0.8520 ± 0.0217 | -0.0006 ± 0.0024 (5/15 folds not worse) |
| 31, no radius + scaled neuropil skew | 0.844 | 0.864 | 0.8521 ± 0.0221 | -0.0005 ± 0.0036 (5/15 folds not worse) |

| Feature set | stav22 threshold | precision | recall | F1 | ROI 219 probability | recall, small labelled cells (radius ≤ 3.5, n=149) | recall, larger cells |
|---|---|---|---|---|---|---|---|
| 32 (small model) | 0.67 | 0.954 | 0.678 | 0.792 | 0.75 | 0.55 | 0.71 |
| 31 = 32 without radius (size) | 0.59 | 0.943 | 0.720 | 0.816 | 0.88 | 0.62 | 0.74 |
| 32, radius clipped at 3.5 px | 0.61 | 0.938 | 0.713 | 0.810 | 0.83 | 0.59 | 0.74 |
| 32, neuropil skew scaled by amplitude | 0.64 | 0.941 | 0.687 | 0.794 | 0.73 | 0.57 | 0.71 |
| 32, radius clipped + scaled neuropil skew | 0.65 | 0.936 | 0.713 | 0.810 | 0.92 | 0.59 | 0.74 |
| 31, no radius + scaled neuropil skew | 0.65 | 0.948 | 0.713 | 0.814 | 0.94 | 0.57 | 0.75 |

## Same-threshold check on stav22 (32 vs 31 = 32 without radius)

The 31-feature model chose a lower threshold (0.59 vs 0.67), which by itself raises recall on stav22's liberally
labelled session. Compared at equal thresholds:

| Model | AUC | AUC, small ROIs | F1 @ 0.59 | small-cell recall @ 0.59 | F1 @ 0.67 | small-cell recall @ 0.67 |
|---|---|---|---|---|---|---|
| 32 | 0.896 | 0.849 | 0.801 | 0.58 | 0.792 | 0.55 |
| 31 (no radius) | 0.894 | 0.852 | 0.816 | 0.62 | 0.799 | 0.58 |

Ranking quality is unchanged; dropping radius mainly raises the scores of small ROIs, which helps stav22 at any
fixed threshold (+0.007 to +0.015 F1) while costing nothing in the 28-session CV (-0.0006 +- 0.003).
