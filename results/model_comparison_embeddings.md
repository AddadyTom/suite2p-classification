29 sessions, 5 session-grouped folds (evaluate_cv repeat 0), nested threshold / early stopping. ΔF1 is paired vs LightGBM (current) on manual44.

| Input set | Model | Precision | Recall | F1 | ΔF1 vs LightGBM manual44 | fit time / fold |
|---|---|---|---|---|---|---|
| manual44 | LightGBM (current) | 0.8460 | 0.8604 | 0.8529 ± 0.0262 |  | 17s |
| manual44+cnn_emb | Average LGB+XGB+Cat | 0.8526 | 0.8707 | 0.8608 ± 0.0209 | +0.0079 ± 0.0101 (4/5 up) | 98s |
| manual44+cnn_emb | CatBoost | 0.8611 | 0.8609 | 0.8604 ± 0.0225 | +0.0075 ± 0.0098 (4/5 up) | 14s |
| manual44+cnn_emb | LightGBM (current) | 0.8466 | 0.8744 | 0.8599 ± 0.0209 | +0.0070 ± 0.0108 (3/5 up) | 24s |
| manual44+cnn_emb | XGBoost | 0.8446 | 0.8772 | 0.8599 ± 0.0200 | +0.0070 ± 0.0107 (3/5 up) | 59s |
| manual44+cnn_emb | MLP | 0.8609 | 0.8561 | 0.8572 ± 0.0217 | +0.0043 ± 0.0136 (3/5 up) | 52s |
| manual44+cnn_emb | LogisticRegression | 0.8561 | 0.8539 | 0.8542 ± 0.0238 | +0.0013 ± 0.0121 (3/5 up) | 15s |
| manual44+cnn_pca16 | Average LGB+XGB+Cat | 0.8513 | 0.8721 | 0.8607 ± 0.0208 | +0.0079 ± 0.0095 (4/5 up) | 61s |
| manual44+cnn_pca16 | CatBoost | 0.8634 | 0.8584 | 0.8600 ± 0.0222 | +0.0071 ± 0.0107 (4/5 up) | 10s |
| manual44+cnn_pca16 | LightGBM (current) | 0.8493 | 0.8699 | 0.8590 ± 0.0210 | +0.0061 ± 0.0083 (3/5 up) | 16s |
| manual44+cnn_pca16 | XGBoost | 0.8472 | 0.8711 | 0.8582 ± 0.0208 | +0.0053 ± 0.0098 (3/5 up) | 35s |
| manual44+cnn_pca16 | MLP | 0.8687 | 0.8480 | 0.8575 ± 0.0238 | +0.0046 ± 0.0103 (3/5 up) | 27s |
| manual44+cnn_pca16 | LogisticRegression | 0.8514 | 0.8606 | 0.8552 ± 0.0229 | +0.0023 ± 0.0124 (3/5 up) | 7s |
| shape+image+tracenn | MLP | 0.8347 | 0.8398 | 0.8361 ± 0.0281 | -0.0168 ± 0.0032 (0/5 up) | 1891s |
| shape+image+tracenn | LightGBM (current) | 0.8260 | 0.8481 | 0.8356 ± 0.0228 | -0.0173 ± 0.0078 (0/5 up) | 23s |
| shape+image+tracenn | LogisticRegression | 0.8079 | 0.8375 | 0.8214 ± 0.0266 | -0.0314 ± 0.0069 (0/5 up) | 12s |
| manual44+tracenn | LightGBM (current) | 0.8380 | 0.8491 | 0.8427 ± 0.0251 | -0.0102 ± 0.0048 (0/5 up) | 24s |
| manual44+tracenn | MLP | 0.8545 | 0.8261 | 0.8388 ± 0.0270 | -0.0140 ± 0.0043 (0/5 up) | 50s |
| manual44+tracenn | LogisticRegression | 0.8072 | 0.8401 | 0.8221 ± 0.0273 | -0.0308 ± 0.0088 (0/5 up) | 15s |
