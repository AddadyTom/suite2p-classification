# Labeller (lab) hold-out experiments

Script: scripts/holdout_eval.py. 44 features (baseline + image), LightGBM as in the shipped model; threshold and n_estimators from session-grouped inner CV on the training pool only; the held-out sessions are never used for training or tuning. Pools: A Stav only, B Inbar only, C Stav + Inbar, D Stav + Inbar with labs weighted equally.

## Summary (pooled F1 on the 4 held-out sessions)

| Held-out sessions | Stav only | Inbar only | Own-lab advantage |
|---|---|---|---|
| Stav14, Stav8, Stav18, Stav17 (hardest) | 0.833 | 0.808 | +0.025 |
| Stav4, Stav16, Stav14, Stav6 | 0.848 | 0.834 | +0.014 |
| Stav9, Stav13, Stav20, Stav19 | 0.886 | 0.855 | +0.031 |
| Stav6, Stav18, Stav16, Stav14 | 0.831 | 0.809 | +0.022 |
| Inbar3, Inbar12, inbar1, Inbar9 (hardest) | 0.741 | 0.782 | +0.041 |
| Inbar9, inbar4, Inbar8, Inbar11 | 0.838 | 0.827 | −0.011 |
| Inbar10, Inbar2, Inbar6, Inbar8 | 0.847 | 0.848 | +0.001 |
| Inbar12, Inbar2, Inbar11, inbar4 | 0.826 | 0.817 | −0.009 |

Random draws: numpy default_rng(2026), 4 sessions per draw. Caveat: lab and labeller are confounded (different mice, rigs, days); the Inbar-only pool has 8 sessions vs 13–17 for Stav.

## Hardest Inbar sessions, all pools

| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | predicted pos rate | labelled pos rate |
|---|---|---|---|---|---|---|---|---|
| A: Stav only (17 sess) | Inbar3 | 0.67 | 0.966 | 0.563 | 0.712 | 0.856 | 0.154 | 0.265 |
| A: Stav only (17 sess) | Inbar12 | 0.67 | 0.792 | 0.787 | 0.790 | 0.792 | 0.085 | 0.086 |
| A: Stav only (17 sess) | inbar1 | 0.67 | 0.613 | 0.966 | 0.750 | 0.876 | 0.082 | 0.052 |
| A: Stav only (17 sess) | Inbar9 | 0.67 | 0.811 | 0.770 | 0.790 | 0.804 | 0.081 | 0.086 |
| A: Stav only (17 sess) | ALL (pooled) | 0.67 | 0.859 | 0.651 | 0.741 | 0.768 | 0.107 | 0.142 |
| B: Inbar only (8 sess) | Inbar3 | 0.61 | 0.955 | 0.631 | 0.760 | 0.857 | 0.175 | 0.265 |
| B: Inbar only (8 sess) | Inbar12 | 0.61 | 0.818 | 0.812 | 0.815 | 0.824 | 0.085 | 0.086 |
| B: Inbar only (8 sess) | inbar1 | 0.61 | 0.680 | 0.977 | 0.802 | 0.890 | 0.075 | 0.052 |
| B: Inbar only (8 sess) | Inbar9 | 0.61 | 0.842 | 0.793 | 0.817 | 0.825 | 0.081 | 0.086 |
| B: Inbar only (8 sess) | ALL (pooled) | 0.61 | 0.880 | 0.703 | 0.782 | 0.803 | 0.113 | 0.142 |
| C: Stav + Inbar (25 sess) | Inbar3 | 0.60 | 0.963 | 0.611 | 0.748 | 0.868 | 0.168 | 0.265 |
| C: Stav + Inbar (25 sess) | Inbar12 | 0.60 | 0.833 | 0.781 | 0.806 | 0.819 | 0.080 | 0.086 |
| C: Stav + Inbar (25 sess) | inbar1 | 0.60 | 0.641 | 0.966 | 0.771 | 0.880 | 0.079 | 0.052 |
| C: Stav + Inbar (25 sess) | Inbar9 | 0.60 | 0.814 | 0.785 | 0.799 | 0.814 | 0.083 | 0.086 |
| C: Stav + Inbar (25 sess) | ALL (pooled) | 0.60 | 0.874 | 0.684 | 0.767 | 0.806 | 0.111 | 0.142 |
| D: Stav + Inbar, labs equal weight (25 sess) | Inbar3 | 0.66 | 0.968 | 0.596 | 0.737 | 0.870 | 0.163 | 0.265 |
| D: Stav + Inbar, labs equal weight (25 sess) | Inbar12 | 0.66 | 0.840 | 0.787 | 0.813 | 0.818 | 0.080 | 0.086 |
| D: Stav + Inbar, labs equal weight (25 sess) | inbar1 | 0.66 | 0.677 | 0.966 | 0.796 | 0.880 | 0.075 | 0.052 |
| D: Stav + Inbar, labs equal weight (25 sess) | Inbar9 | 0.66 | 0.848 | 0.785 | 0.815 | 0.817 | 0.079 | 0.086 |
| D: Stav + Inbar, labs equal weight (25 sess) | ALL (pooled) | 0.66 | 0.889 | 0.675 | 0.767 | 0.807 | 0.107 | 0.142 |

## Hardest Stav sessions, all pools

| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | predicted pos rate | labelled pos rate |
|---|---|---|---|---|---|---|---|---|
| A: Stav only (13 sess) | Stav14 | 0.64 | 0.649 | 0.924 | 0.762 | 0.820 | 0.334 | 0.235 |
| A: Stav only (13 sess) | Stav8 | 0.64 | 0.830 | 0.870 | 0.849 | 0.851 | 0.198 | 0.189 |
| A: Stav only (13 sess) | Stav18 | 0.64 | 0.834 | 0.828 | 0.831 | 0.834 | 0.227 | 0.229 |
| A: Stav only (13 sess) | Stav17 | 0.64 | 0.830 | 0.924 | 0.874 | 0.874 | 0.256 | 0.230 |
| A: Stav only (13 sess) | ALL (pooled) | 0.64 | 0.787 | 0.884 | 0.833 | 0.837 | 0.246 | 0.219 |
| B: Inbar only (12 sess) | Stav14 | 0.56 | 0.622 | 0.924 | 0.744 | 0.796 | 0.349 | 0.235 |
| B: Inbar only (12 sess) | Stav8 | 0.56 | 0.878 | 0.791 | 0.832 | 0.843 | 0.170 | 0.189 |
| B: Inbar only (12 sess) | Stav18 | 0.56 | 0.824 | 0.815 | 0.820 | 0.820 | 0.226 | 0.229 |
| B: Inbar only (12 sess) | Stav17 | 0.56 | 0.751 | 0.925 | 0.829 | 0.839 | 0.283 | 0.230 |
| B: Inbar only (12 sess) | ALL (pooled) | 0.56 | 0.761 | 0.861 | 0.808 | 0.809 | 0.248 | 0.219 |
| C: Stav + Inbar (25 sess) | Stav14 | 0.59 | 0.611 | 0.949 | 0.743 | 0.813 | 0.365 | 0.235 |
| C: Stav + Inbar (25 sess) | Stav8 | 0.59 | 0.825 | 0.865 | 0.844 | 0.855 | 0.198 | 0.189 |
| C: Stav + Inbar (25 sess) | Stav18 | 0.59 | 0.805 | 0.866 | 0.834 | 0.842 | 0.246 | 0.229 |
| C: Stav + Inbar (25 sess) | Stav17 | 0.59 | 0.761 | 0.963 | 0.851 | 0.869 | 0.291 | 0.230 |
| C: Stav + Inbar (25 sess) | ALL (pooled) | 0.59 | 0.749 | 0.909 | 0.821 | 0.834 | 0.266 | 0.219 |
| D: Stav + Inbar, labs equal weight (25 sess) | Stav14 | 0.70 | 0.659 | 0.932 | 0.772 | 0.814 | 0.332 | 0.235 |
| D: Stav + Inbar, labs equal weight (25 sess) | Stav8 | 0.70 | 0.872 | 0.828 | 0.849 | 0.854 | 0.179 | 0.189 |
| D: Stav + Inbar, labs equal weight (25 sess) | Stav18 | 0.70 | 0.836 | 0.834 | 0.835 | 0.839 | 0.228 | 0.229 |
| D: Stav + Inbar, labs equal weight (25 sess) | Stav17 | 0.70 | 0.786 | 0.946 | 0.859 | 0.871 | 0.277 | 0.230 |
| D: Stav + Inbar, labs equal weight (25 sess) | ALL (pooled) | 0.70 | 0.785 | 0.883 | 0.832 | 0.832 | 0.246 | 0.219 |

## Random draw 1: Test sessions: ['Stav4', 'Stav16', 'Stav14', 'Stav6']

| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | predicted pos rate | labelled pos rate |
|---|---|---|---|---|---|---|---|---|
| A: Stav only (13 sess) | Stav4 | 0.56 | 0.862 | 0.961 | 0.909 | 0.912 | 0.299 | 0.269 |
| A: Stav only (13 sess) | Stav16 | 0.56 | 0.791 | 0.907 | 0.845 | 0.867 | 0.299 | 0.261 |
| A: Stav only (13 sess) | Stav14 | 0.56 | 0.618 | 0.938 | 0.745 | 0.828 | 0.357 | 0.235 |
| A: Stav only (13 sess) | Stav6 | 0.56 | 0.836 | 0.875 | 0.855 | 0.864 | 0.251 | 0.240 |
| A: Stav only (13 sess) | ALL (pooled) | 0.56 | 0.785 | 0.922 | 0.848 | 0.860 | 0.296 | 0.252 |
| B: Inbar only (12 sess) | Stav4 | 0.56 | 0.830 | 0.957 | 0.889 | 0.897 | 0.310 | 0.269 |
| B: Inbar only (12 sess) | Stav16 | 0.56 | 0.783 | 0.818 | 0.800 | 0.805 | 0.273 | 0.261 |
| B: Inbar only (12 sess) | Stav14 | 0.56 | 0.622 | 0.924 | 0.744 | 0.796 | 0.349 | 0.235 |
| B: Inbar only (12 sess) | Stav6 | 0.56 | 0.842 | 0.866 | 0.854 | 0.856 | 0.247 | 0.240 |
| B: Inbar only (12 sess) | ALL (pooled) | 0.56 | 0.777 | 0.899 | 0.834 | 0.840 | 0.291 | 0.252 |

## Random draw 2: Test sessions: ['Stav9', 'Stav13', 'Stav20', 'Stav19']

| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | predicted pos rate | labelled pos rate |
|---|---|---|---|---|---|---|---|---|
| A: Stav only (13 sess) | Stav9 | 0.61 | 0.848 | 0.885 | 0.866 | 0.872 | 0.238 | 0.228 |
| A: Stav only (13 sess) | Stav13 | 0.61 | 0.901 | 0.924 | 0.912 | 0.914 | 0.284 | 0.276 |
| A: Stav only (13 sess) | Stav20 | 0.61 | 0.856 | 0.926 | 0.890 | 0.894 | 0.150 | 0.139 |
| A: Stav only (13 sess) | Stav19 | 0.61 | 0.932 | 0.819 | 0.872 | 0.889 | 0.153 | 0.174 |
| A: Stav only (13 sess) | ALL (pooled) | 0.61 | 0.883 | 0.889 | 0.886 | 0.889 | 0.201 | 0.199 |
| B: Inbar only (12 sess) | Stav9 | 0.56 | 0.813 | 0.854 | 0.833 | 0.834 | 0.239 | 0.228 |
| B: Inbar only (12 sess) | Stav13 | 0.56 | 0.838 | 0.922 | 0.878 | 0.886 | 0.304 | 0.276 |
| B: Inbar only (12 sess) | Stav20 | 0.56 | 0.819 | 0.892 | 0.854 | 0.856 | 0.151 | 0.139 |
| B: Inbar only (12 sess) | Stav19 | 0.56 | 0.908 | 0.800 | 0.850 | 0.854 | 0.153 | 0.174 |
| B: Inbar only (12 sess) | ALL (pooled) | 0.56 | 0.841 | 0.869 | 0.855 | 0.856 | 0.206 | 0.199 |

## Random draw 3: Test sessions: ['Stav6', 'Stav18', 'Stav16', 'Stav14']

| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | predicted pos rate | labelled pos rate |
|---|---|---|---|---|---|---|---|---|
| A: Stav only (13 sess) | Stav6 | 0.71 | 0.886 | 0.832 | 0.858 | 0.874 | 0.225 | 0.240 |
| A: Stav only (13 sess) | Stav18 | 0.71 | 0.844 | 0.812 | 0.828 | 0.834 | 0.220 | 0.229 |
| A: Stav only (13 sess) | Stav16 | 0.71 | 0.841 | 0.883 | 0.862 | 0.863 | 0.274 | 0.261 |
| A: Stav only (13 sess) | Stav14 | 0.71 | 0.675 | 0.898 | 0.770 | 0.831 | 0.313 | 0.235 |
| A: Stav only (13 sess) | ALL (pooled) | 0.71 | 0.813 | 0.849 | 0.831 | 0.831 | 0.250 | 0.239 |
| B: Inbar only (12 sess) | Stav6 | 0.56 | 0.842 | 0.866 | 0.854 | 0.856 | 0.247 | 0.240 |
| B: Inbar only (12 sess) | Stav18 | 0.56 | 0.824 | 0.815 | 0.820 | 0.820 | 0.226 | 0.229 |
| B: Inbar only (12 sess) | Stav16 | 0.56 | 0.783 | 0.818 | 0.800 | 0.805 | 0.273 | 0.261 |
| B: Inbar only (12 sess) | Stav14 | 0.56 | 0.622 | 0.924 | 0.744 | 0.796 | 0.349 | 0.235 |
| B: Inbar only (12 sess) | ALL (pooled) | 0.56 | 0.769 | 0.853 | 0.809 | 0.810 | 0.266 | 0.239 |

## Random draw 4: Test sessions: ['Inbar9', 'inbar4', 'Inbar8', 'Inbar11']

| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | predicted pos rate | labelled pos rate |
|---|---|---|---|---|---|---|---|---|
| A: Stav only (17 sess) | Inbar9 | 0.67 | 0.811 | 0.770 | 0.790 | 0.804 | 0.081 | 0.086 |
| A: Stav only (17 sess) | inbar4 | 0.67 | 0.898 | 0.802 | 0.847 | 0.857 | 0.204 | 0.228 |
| A: Stav only (17 sess) | Inbar8 | 0.67 | 0.891 | 0.883 | 0.887 | 0.890 | 0.064 | 0.065 |
| A: Stav only (17 sess) | Inbar11 | 0.67 | 0.825 | 0.793 | 0.809 | 0.812 | 0.052 | 0.054 |
| A: Stav only (17 sess) | ALL (pooled) | 0.67 | 0.870 | 0.808 | 0.838 | 0.838 | 0.098 | 0.106 |
| B: Inbar only (8 sess) | Inbar9 | 0.49 | 0.784 | 0.809 | 0.796 | 0.811 | 0.088 | 0.086 |
| B: Inbar only (8 sess) | inbar4 | 0.49 | 0.739 | 0.937 | 0.826 | 0.872 | 0.290 | 0.228 |
| B: Inbar only (8 sess) | Inbar8 | 0.49 | 0.849 | 0.910 | 0.879 | 0.895 | 0.069 | 0.065 |
| B: Inbar only (8 sess) | Inbar11 | 0.49 | 0.757 | 0.864 | 0.807 | 0.829 | 0.061 | 0.054 |
| B: Inbar only (8 sess) | ALL (pooled) | 0.49 | 0.765 | 0.899 | 0.827 | 0.855 | 0.124 | 0.106 |

## Random draw 5: Test sessions: ['Inbar10', 'Inbar2', 'Inbar6', 'Inbar8']

| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | predicted pos rate | labelled pos rate |
|---|---|---|---|---|---|---|---|---|
| A: Stav only (17 sess) | Inbar10 | 0.67 | 0.802 | 0.814 | 0.808 | 0.813 | 0.059 | 0.058 |
| A: Stav only (17 sess) | Inbar2 | 0.67 | 0.659 | 0.946 | 0.777 | 0.870 | 0.107 | 0.074 |
| A: Stav only (17 sess) | Inbar6 | 0.67 | 0.807 | 0.926 | 0.862 | 0.869 | 0.167 | 0.146 |
| A: Stav only (17 sess) | Inbar8 | 0.67 | 0.891 | 0.883 | 0.887 | 0.890 | 0.064 | 0.065 |
| A: Stav only (17 sess) | ALL (pooled) | 0.67 | 0.805 | 0.893 | 0.847 | 0.855 | 0.093 | 0.084 |
| B: Inbar only (8 sess) | Inbar10 | 0.61 | 0.765 | 0.828 | 0.795 | 0.816 | 0.063 | 0.058 |
| B: Inbar only (8 sess) | Inbar2 | 0.61 | 0.752 | 0.924 | 0.829 | 0.881 | 0.091 | 0.074 |
| B: Inbar only (8 sess) | Inbar6 | 0.61 | 0.781 | 0.963 | 0.862 | 0.887 | 0.180 | 0.146 |
| B: Inbar only (8 sess) | Inbar8 | 0.61 | 0.852 | 0.901 | 0.876 | 0.887 | 0.068 | 0.065 |
| B: Inbar only (8 sess) | ALL (pooled) | 0.61 | 0.790 | 0.914 | 0.848 | 0.866 | 0.097 | 0.084 |

## Random draw 6: Test sessions: ['Inbar12', 'Inbar2', 'Inbar11', 'inbar4']

| Training pool | Test session | thr | Precision | Recall | F1 | F1 at session's best thr | predicted pos rate | labelled pos rate |
|---|---|---|---|---|---|---|---|---|
| A: Stav only (17 sess) | Inbar12 | 0.67 | 0.792 | 0.787 | 0.790 | 0.792 | 0.085 | 0.086 |
| A: Stav only (17 sess) | Inbar2 | 0.67 | 0.659 | 0.946 | 0.777 | 0.870 | 0.107 | 0.074 |
| A: Stav only (17 sess) | Inbar11 | 0.67 | 0.825 | 0.793 | 0.809 | 0.812 | 0.052 | 0.054 |
| A: Stav only (17 sess) | inbar4 | 0.67 | 0.898 | 0.802 | 0.847 | 0.857 | 0.204 | 0.228 |
| A: Stav only (17 sess) | ALL (pooled) | 0.67 | 0.842 | 0.810 | 0.826 | 0.826 | 0.114 | 0.119 |
| B: Inbar only (8 sess) | Inbar12 | 0.53 | 0.724 | 0.787 | 0.754 | 0.761 | 0.093 | 0.086 |
| B: Inbar only (8 sess) | Inbar2 | 0.53 | 0.752 | 0.924 | 0.829 | 0.903 | 0.091 | 0.074 |
| B: Inbar only (8 sess) | Inbar11 | 0.53 | 0.705 | 0.832 | 0.763 | 0.780 | 0.063 | 0.054 |
| B: Inbar only (8 sess) | inbar4 | 0.53 | 0.772 | 0.930 | 0.843 | 0.871 | 0.275 | 0.228 |
| B: Inbar only (8 sess) | ALL (pooled) | 0.53 | 0.753 | 0.894 | 0.817 | 0.833 | 0.141 | 0.119 |
