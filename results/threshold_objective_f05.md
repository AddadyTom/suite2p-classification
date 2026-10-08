Image model (44 features), 28 sessions (excluded stav22, stav3/21, Inbar5), 3x5 session folds. Same model per fold; only the threshold objective differs, both chosen on training sessions only.

| Threshold chosen by | threshold | Precision | Recall | F1 | F0.5 | false positives / fold | missed cells / fold |
|---|---|---|---|---|---|---|---|
| F1 threshold (current) | 0.66 | 0.842 ± 0.034 | 0.869 ± 0.056 | 0.853 | 0.846 | 488 | 371 |
| F0.5 threshold (FP = 2x FN) | 0.90 | 0.929 ± 0.021 | 0.735 ± 0.072 | 0.818 | 0.881 | 166 | 758 |

Paired change F0.5 vs F1 threshold: precision +0.088 (15/15 folds up), recall -0.134, F0.5 +0.035 (14/15 up), F1 -0.035; false positives -321, missed cells +387 per fold.

Precision / recall by threshold (held-out sessions, repeat 0 pooled): the dashboard slider moves along this.

| threshold | Precision | Recall | false positives per 100 predicted cells |
|---|---|---|---|
| 0.50 | 0.793 | 0.908 | 20.7 |
| 0.60 | 0.823 | 0.888 | 17.7 |
| 0.65 | 0.837 | 0.877 | 16.3 |
| 0.70 | 0.851 | 0.863 | 14.9 |
| 0.75 | 0.866 | 0.845 | 13.4 |
| 0.80 | 0.882 | 0.824 | 11.8 |
| 0.85 | 0.903 | 0.793 | 9.7 |
| 0.90 | 0.926 | 0.747 | 7.4 |
