CV: 28 sessions (excluded stav22, stav3/21, Inbar5), 3x5 session folds, nested threshold. stav22: model trained on all those sessions, stav22 never used.

| Feature set | CV precision | CV recall | CV F1 | ΔF1 vs with-shape (paired) |
|---|---|---|---|---|
| 44 (current) | 0.842 | 0.869 | 0.8530 ± 0.0218 |  |
| 38 = 44 without shape | 0.823 | 0.856 | 0.8366 ± 0.0236 | -0.0164 ± 0.0052 (0/15 folds not worse) |
| 32 (small model) | 0.840 | 0.868 | 0.8527 ± 0.0221 |  |
| 28 = 32 without shape | 0.810 | 0.853 | 0.8284 ± 0.0222 | -0.0243 ± 0.0056 (0/15 folds not worse) |

| Feature set | stav22 threshold | precision | recall | F1 | ROI 219 probability | recall, small labelled cells (radius ≤ 3.5, n=149) | recall, larger cells |
|---|---|---|---|---|---|---|---|
| 44 (current) | 0.66 | 0.951 | 0.664 | 0.782 | 0.73 | 0.54 | 0.70 |
| 38 = 44 without shape | 0.67 | 0.966 | 0.523 | 0.679 | 0.88 | 0.46 | 0.54 |
| 32 (small model) | 0.67 | 0.954 | 0.678 | 0.792 | 0.75 | 0.55 | 0.71 |
| 28 = 32 without shape | 0.68 | 0.948 | 0.529 | 0.680 | 0.96 | 0.50 | 0.54 |
