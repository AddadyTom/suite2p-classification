Mouse-grouped CV: 29 sessions from 16 mice (excluded stav22, stav3/21), 3x5 folds holding out whole mice; inner threshold/tree selection also mouse-grouped.

| Feature set | P | R | F1 (nested threshold) | ΔF1 vs 27 (paired) | P @ 0.7 | R @ 0.7 | F1 @ 0.7 |
|---|---|---|---|---|---|---|---|
| 27 (previous model) | 0.802 | 0.852 | 0.8246 ± 0.0255 | — | 0.810 | 0.842 | 0.8235 |
| 44 (image model) | 0.834 | 0.868 | 0.8486 ± 0.0215 | +0.0240 ± 0.0100 (15/15 up) | 0.847 | 0.857 | 0.8504 |
| 32 | 0.832 | 0.869 | 0.8487 ± 0.0206 | +0.0241 ± 0.0095 (15/15 up) | 0.846 | 0.858 | 0.8500 |
| 31 (small model) | 0.832 | 0.871 | 0.8493 ± 0.0198 | +0.0247 ± 0.0088 (15/15 up) | 0.845 | 0.858 | 0.8497 |

Labeller test, leave-one-mouse-out (31 features): each mouse scored by a Stav-only and an Inbar-only model, both trained without that mouse; F1 at each model's nested threshold.

| Mouse | lab | sessions | Stav-only F1 | Inbar-only F1 | own-lab advantage |
|---|---|---|---|---|---|
| IP80 | inbar | inbar1, Inbar2 | 0.765 | 0.753 | -0.012 |
| IP81 | inbar | Inbar3, inbar4 | 0.782 | 0.815 | +0.033 |
| IP83 | inbar | Inbar5, Inbar6 | 0.836 | 0.806 | -0.029 |
| OI151 | stav | Stav7, Stav8, Stav9 | 0.853 | 0.828 | +0.024 |
| OI152 | stav | Stav10, Stav11 | 0.883 | 0.843 | +0.040 |
| OI208 | stav | Stav4, Stav5 | 0.894 | 0.872 | +0.022 |
| OI230 | stav | Stav6 | 0.845 | 0.853 | -0.008 |
| SS10 | stav | Stav20 | 0.898 | 0.838 | +0.060 |
| SS37 | stav | Stav14, Stav16 | 0.800 | 0.771 | +0.029 |
| SS38 | stav | Stav12, Stav13 | 0.899 | 0.868 | +0.030 |
| SS6 | stav | Stav17, Stav18 | 0.843 | 0.818 | +0.025 |
| SS7 | stav | Stav19 | 0.868 | 0.841 | +0.027 |
| SS9 | stav | Stav21 | 0.831 | 0.811 | +0.020 |
| YP115 | inbar | Inbar11, Inbar12 | 0.784 | 0.740 | -0.044 |
| YP116 | inbar | Inbar10, Inbar9 | 0.795 | 0.801 | +0.006 |
| YP119 | inbar | Inbar7, Inbar8 | 0.882 | 0.883 | +0.001 |

Stav mice: own-lab model better for 9/10, mean advantage +0.027.

Inbar mice: own-lab model better for 3/6, mean advantage -0.008.
