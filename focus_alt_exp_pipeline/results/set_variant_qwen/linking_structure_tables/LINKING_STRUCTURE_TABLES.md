# Cross-dataset linking-structure results

All columns use the same analysis units within each dataset. Higher is better for both Pearson correlation and mean proper log score.

Focus-context word-ranking and negation Spearman results, paired ranks, and equal-context means are in `../spearman/SPEARMAN.md`.

## Pearson correlations

| Dataset | No linking structure | X but not Y | Set Top-K | Set Top-p | Ordering | Conjunction Top-K | Conjunction Top-p | Disjunction Top-K | Disjunction Top-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| van Tiel et al. (2016) | 0.162 | 0.133 | 0.276 | 0.259 | -0.007 | 0.277 | 0.250 | 0.019 | 0.028 |
| Gotzner et al. (2018) | 0.159 | 0.248 | 0.278 | 0.297 | 0.099 | 0.266 | 0.278 | 0.106 | 0.112 |
| Pankratz & van Tiel (2021) | -0.110 | 0.129 | -0.162 | -0.153 | -0.046 | -0.162 | -0.122 | -0.071 | -0.092 |
| Ronai & Xiang (2022) | 0.087 | 0.371 | 0.183 | 0.169 | 0.163 | 0.150 | 0.155 | 0.207 | 0.195 |
| R&X ESI | 0.026 | NA | 0.196 | 0.137 | 0.154 | 0.144 | 0.114 | 0.213 | 0.185 |
| R&X Eweak | 0.096 | NA | -0.046 | -0.010 | 0.200 | 0.203 | 0.175 | -0.046 | 0.012 |
| R&X Estrong | -0.209 | NA | -0.074 | -0.235 | 0.109 | 0.028 | -0.027 | 0.000 | -0.192 |
| R&X Eonly | 0.064 | NA | 0.207 | 0.133 | 0.081 | 0.115 | 0.106 | 0.173 | 0.121 |
| R&X Eonlystrong | -0.211 | NA | -0.067 | -0.160 | -0.066 | -0.082 | -0.181 | -0.068 | -0.001 |
| Novel Focus Alternative Study | 0.363 | 0.263 | 0.594 | 0.499 | 0.462 | 0.560 | 0.498 | 0.524 | 0.487 |

## Spearman correlations: Hu datasets and R&X conditions

Hu ranks scales within each dataset, after averaging van Tiel template probabilities. R&X ranks the 60 items separately within each condition. These use the same units as Pearson. Average ranks handle ties; constant predictions are undefined. Focus uses the separate within-context analysis above.

| Dataset | No linking structure | X but not Y | Set Top-K | Set Top-p | Ordering | Conjunction Top-K | Conjunction Top-p | Disjunction Top-K | Disjunction Top-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| van Tiel et al. (2016) | 0.100 | 0.139 | 0.206 | 0.153 | -0.059 | 0.176 | 0.107 | -0.006 | 0.015 |
| Gotzner et al. (2018) | 0.080 | 0.043 | 0.137 | 0.162 | 0.067 | 0.130 | 0.141 | 0.074 | 0.087 |
| Pankratz & van Tiel (2021) | 0.001 | 0.216 | -0.092 | -0.116 | -0.238 | -0.103 | -0.123 | -0.226 | -0.231 |
| Ronai & Xiang (2022) | 0.140 | 0.371 | 0.298 | 0.143 | 0.143 | 0.287 | 0.127 | 0.184 | 0.182 |
| R&X ESI | 0.166 | NA | 0.324 | 0.162 | 0.163 | 0.314 | 0.153 | 0.203 | 0.195 |
| R&X Eweak | -0.130 | NA | 0.018 | -0.077 | 0.071 | 0.091 | 0.040 | -0.001 | -0.060 |
| R&X Estrong | -0.261 | NA | -0.008 | -0.197 | 0.094 | 0.070 | 0.037 | -0.003 | -0.222 |
| R&X Eonly | 0.120 | NA | 0.307 | 0.111 | 0.109 | 0.253 | 0.100 | 0.191 | 0.142 |
| R&X Eonlystrong | -0.224 | NA | -0.037 | -0.195 | -0.042 | -0.048 | -0.146 | -0.040 | -0.146 |

## Mean proper log scores

| Dataset | No linking structure | X but not Y | Set Top-K | Set Top-p | Ordering | Conjunction Top-K | Conjunction Top-p | Disjunction Top-K | Disjunction Top-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| van Tiel et al. (2016) | -2.902 | -1.614 | -1.575 | -1.357 | -1.490 | -1.648 | -1.459 | -1.478 | -1.471 |
| Gotzner et al. (2018) | -4.621 | -2.014 | -3.880 | -3.792 | -2.748 | -3.893 | -3.818 | -2.745 | -2.734 |
| Pankratz & van Tiel (2021) | -2.743 | -1.304 | -2.702 | -2.929 | -1.386 | -2.373 | -2.102 | -1.762 | -2.418 |
| Ronai & Xiang (2022) | -3.056 | -1.328 | -2.209 | -2.487 | -1.068 | -2.266 | -2.241 | -1.288 | -1.356 |
| R&X ESI | -3.054 | NA | -2.409 | -2.694 | -1.324 | -2.468 | -2.462 | -1.529 | -1.597 |
| R&X Eweak | -1.785 | NA | -4.549 | -2.741 | -2.206 | -2.224 | -2.460 | -4.532 | -2.490 |
| R&X Estrong | -1.361 | NA | -7.618 | -7.077 | -1.977 | -2.006 | -1.945 | -7.596 | -7.449 |
| R&X Eonly | -5.552 | NA | -4.364 | -3.916 | -1.696 | -4.663 | -3.964 | -1.657 | -1.688 |
| R&X Eonlystrong | -1.796 | NA | -2.352 | -2.226 | -0.762 | -0.836 | -0.880 | -2.283 | -2.193 |
| Novel Focus Alternative Study | -3.219 | -3.848 | -1.569 | -2.330 | -1.564 | -1.745 | -2.435 | -1.528 | -1.477 |

## Coverage

| Dataset | No linking structure | X but not Y | Set Top-K | Set Top-p | Ordering | Conjunction Top-K | Conjunction Top-p | Disjunction Top-K | Disjunction Top-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| van Tiel et al. (2016) | available (N=39) | available (N=39) | available (N=39) | available (N=39) | available (N=39) | available (N=39) | available (N=39) | available (N=39) | available (N=39) |
| Gotzner et al. (2018) | available (N=67) | available (N=67) | available (N=67) | available (N=67) | available (N=67) | available (N=67) | available (N=67) | available (N=67) | available (N=67) |
| Pankratz & van Tiel (2021) | available (N=50) | available (N=50) | available (N=50) | available (N=50) | available (N=50) | available (N=50) | available (N=50) | available (N=50) | available (N=50) |
| Ronai & Xiang (2022) | available (N=57) | available (N=57) | available (N=57) | available (N=57) | available (N=57) | available (N=57) | available (N=57) | available (N=57) | available (N=57) |
| R&X ESI | available (N=60) | not applicable | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) |
| R&X Eweak | available (N=60) | not applicable | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) |
| R&X Estrong | available (N=60) | not applicable | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) |
| R&X Eonly | available (N=60) | not applicable | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) |
| R&X Eonlystrong | available (N=60) | not applicable | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) | available (N=60) |
| Novel Focus Alternative Study | available (N=480) | available (N=480) | available (N=480) | available (N=480) | available (N=480) | available (N=480) | available (N=480) | available (N=480) | available (N=480) |

X-but-not-Y is structurally unavailable for the five R&X conditions. NA in any other correlation cell means that the statistic is undefined.
