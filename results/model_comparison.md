29 sessions, 5 session-grouped folds (evaluate_cv repeat 0), nested threshold / early stopping. ΔF1 is paired vs LightGBM (current) on manual44.

| Input set | Model | Precision | Recall | F1 | ΔF1 vs LightGBM manual44 | fit time / fold |
|---|---|---|---|---|---|---|
| manual44 | Average LGB+XGB+Cat | 0.8508 | 0.8614 | 0.8555 ± 0.0253 | +0.0026 ± 0.0028 (4/5 up) | 109s |
| manual44 | LightGBM (tuned) | 0.8480 | 0.8633 | 0.8551 ± 0.0246 | +0.0022 ± 0.0023 (4/5 up) | 54s |
| manual44 | LightGBM (current) | 0.8460 | 0.8604 | 0.8529 ± 0.0262 |  | 33s |
| manual44 | XGBoost | 0.8420 | 0.8640 | 0.8525 ± 0.0247 | -0.0004 ± 0.0034 (3/5 up) | 57s |
| manual44 | CatBoost | 0.8520 | 0.8539 | 0.8520 ± 0.0256 | -0.0009 ± 0.0019 (1/5 up) | 19s |
| manual44 | RandomForest | 0.8419 | 0.8622 | 0.8515 ± 0.0223 | -0.0014 ± 0.0044 (1/5 up) | 192s |
| manual44 | MLP | 0.8511 | 0.8428 | 0.8459 ± 0.0256 | -0.0070 ± 0.0047 (1/5 up) | 91s |
| manual44 | ExtraTrees | 0.8343 | 0.8484 | 0.8405 ± 0.0255 | -0.0123 ± 0.0027 (0/5 up) | 36s |
| manual44 | LogisticRegression | 0.8144 | 0.8467 | 0.8289 ± 0.0258 | -0.0239 ± 0.0051 (0/5 up) | 10s |
| manual44+cnn_prob | CatBoost | 0.8545 | 0.8678 | 0.8604 ± 0.0222 | +0.0075 ± 0.0064 (5/5 up) | 19s |
| manual44+cnn_prob | Average LGB+XGB+Cat | 0.8505 | 0.8724 | 0.8602 ± 0.0198 | +0.0073 ± 0.0083 (4/5 up) | 108s |
| manual44+cnn_prob | XGBoost | 0.8536 | 0.8664 | 0.8591 ± 0.0222 | +0.0062 ± 0.0059 (4/5 up) | 56s |
| manual44+cnn_prob | LightGBM (current) | 0.8492 | 0.8704 | 0.8587 ± 0.0208 | +0.0058 ± 0.0084 (4/5 up) | 33s |
| manual44+cnn_prob | MLP | 0.8530 | 0.8583 | 0.8547 ± 0.0234 | +0.0018 ± 0.0075 (3/5 up) | 89s |
| manual44+cnn_prob | LogisticRegression | 0.8468 | 0.8554 | 0.8498 ± 0.0252 | -0.0031 ± 0.0099 (2/5 up) | 11s |
| shape+image+rocket | LightGBM (current) | 0.8283 | 0.8679 | 0.8462 ± 0.0200 | -0.0066 ± 0.0079 (1/5 up) | 38s |
| shape+image+rocket | MLP | 0.8242 | 0.8563 | 0.8375 ± 0.0202 | -0.0153 ± 0.0098 (0/5 up) | 76s |
| shape+image+rocket | LogisticRegression | 0.8027 | 0.8451 | 0.8201 ± 0.0243 | -0.0328 ± 0.0077 (0/5 up) | 7s |
| manual44+rocket | LightGBM (current) | 0.8400 | 0.8686 | 0.8533 ± 0.0231 | +0.0004 ± 0.0044 (2/5 up) | 41s |
| manual44+rocket | MLP | 0.8316 | 0.8587 | 0.8443 ± 0.0208 | -0.0086 ± 0.0056 (1/5 up) | 131s |
| manual44+rocket | LogisticRegression | 0.8143 | 0.8461 | 0.8278 ± 0.0247 | -0.0251 ± 0.0077 (0/5 up) | 14s |
