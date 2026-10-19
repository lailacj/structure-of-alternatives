# Cross-dataset linking-structure results

All columns use the same analysis units within each dataset. Higher is better for both Pearson correlation and summed proper log score. Log scores sum item scores without dividing by item count. Sums depend on sample size, so compare models on matching items within a dataset. They are not participant-count-weighted response log-likelihoods. Training selection is unchanged.

Focus-context word-ranking and negation Spearman results, paired ranks, and equal-context means are in `../spearman/SPEARMAN.md`.

## Pearson correlations

| Dataset | No linking structure | X but not Y | Set Top-K | Set Top-p | Ordering | Conjunction Top-K | Conjunction Top-p | Disjunction Top-K | Disjunction Top-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| van Tiel et al. (2016) | 0.160 | 0.133 | 0.224 | 0.099 | -0.024 | 0.238 | 0.135 | -0.023 | -0.039 |
| Gotzner et al. (2018) | 0.157 | 0.248 | 0.187 | -0.086 | 0.124 | 0.176 | -0.084 | 0.126 | 0.124 |
| Pankratz & van Tiel (2021) | -0.109 | 0.129 | -0.139 | -0.166 | -0.057 | -0.140 | -0.128 | -0.065 | -0.121 |
| Ronai & Xiang (2022) | 0.085 | 0.371 | 0.136 | 0.131 | 0.179 | 0.135 | 0.106 | 0.191 | 0.208 |
| R&X ESI | 0.026 | NA | 0.095 | 0.106 | 0.186 | 0.083 | 0.067 | 0.197 | 0.219 |
| R&X Eweak | 0.098 | NA | 0.017 | -0.108 | 0.171 | 0.179 | 0.118 | 0.014 | -0.071 |
| R&X Estrong | -0.209 | NA | -0.155 | -0.219 | 0.016 | 0.001 | -0.134 | -0.170 | -0.082 |
| R&X Eonly | 0.067 | NA | 0.111 | 0.098 | 0.131 | 0.086 | 0.037 | 0.152 | 0.182 |
| R&X Eonlystrong | -0.207 | NA | -0.061 | -0.131 | -0.089 | -0.130 | -0.220 | 0.017 | 0.077 |
| Novel Focus Alternative Study | 0.361 | 0.263 | 0.540 | 0.475 | 0.473 | 0.527 | 0.470 | 0.494 | 0.492 |

## Spearman correlations: Hu datasets and R&X conditions

Hu ranks scales within each dataset, after averaging van Tiel template probabilities. R&X ranks the 60 items separately within each condition. These use the same units as Pearson. Average ranks handle ties; constant predictions are undefined. Focus uses the separate within-context analysis above.

| Dataset | No linking structure | X but not Y | Set Top-K | Set Top-p | Ordering | Conjunction Top-K | Conjunction Top-p | Disjunction Top-K | Disjunction Top-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| van Tiel et al. (2016) | 0.100 | 0.139 | 0.166 | 0.067 | -0.040 | 0.148 | 0.020 | -0.032 | -0.006 |
| Gotzner et al. (2018) | 0.081 | 0.043 | 0.092 | -0.080 | 0.076 | 0.091 | -0.078 | 0.078 | 0.075 |
| Pankratz & van Tiel (2021) | -0.004 | 0.216 | -0.033 | 0.003 | -0.118 | -0.034 | -0.002 | -0.116 | -0.132 |
| Ronai & Xiang (2022) | 0.136 | 0.371 | 0.221 | 0.104 | 0.153 | 0.247 | 0.091 | 0.158 | 0.202 |
| R&X ESI | 0.163 | NA | 0.245 | 0.107 | 0.175 | 0.271 | 0.111 | 0.177 | 0.219 |
| R&X Eweak | -0.130 | NA | -0.069 | -0.209 | -0.083 | -0.067 | -0.132 | -0.074 | -0.171 |
| R&X Estrong | -0.259 | NA | -0.150 | -0.372 | 0.113 | 0.094 | -0.050 | -0.134 | -0.350 |
| R&X Eonly | 0.116 | NA | 0.221 | 0.029 | 0.156 | 0.215 | 0.021 | 0.179 | 0.207 |
| R&X Eonlystrong | -0.223 | NA | -0.147 | -0.277 | -0.036 | -0.104 | -0.228 | -0.139 | -0.207 |

## Summed proper log scores

| Dataset | No linking structure | X but not Y | Set Top-K | Set Top-p | Ordering | Conjunction Top-K | Conjunction Top-p | Disjunction Top-K | Disjunction Top-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| van Tiel et al. (2016) | -113.229 | -62.936 | -70.858 | -48.023 | -48.979 | -85.616 | -63.610 | -48.423 | -50.283 |
| Gotzner et al. (2018) | -309.738 | -134.909 | -394.037 | -458.384 | -190.374 | -394.303 | -458.404 | -190.235 | -190.466 |
| Pankratz & van Tiel (2021) | -137.042 | -65.196 | -166.461 | -139.888 | -61.911 | -163.818 | -108.959 | -66.037 | -96.007 |
| Ronai & Xiang (2022) | -174.336 | -75.688 | -142.247 | -175.024 | -63.311 | -143.390 | -175.186 | -63.059 | -65.223 |
| R&X ESI | -183.392 | NA | -162.730 | -198.655 | -68.489 | -150.801 | -184.629 | -81.356 | -84.740 |
| R&X Eweak | -107.097 | NA | -125.890 | -198.448 | -102.122 | -102.399 | -105.752 | -126.142 | -194.895 |
| R&X Estrong | -81.824 | NA | -307.326 | -456.516 | -86.907 | -77.445 | -86.595 | -357.789 | -461.567 |
| R&X Eonly | -333.149 | NA | -315.970 | -307.110 | -95.594 | -323.693 | -314.892 | -88.648 | -89.906 |
| R&X Eonlystrong | -107.959 | NA | -96.480 | -137.559 | -39.065 | -37.425 | -43.521 | -109.177 | -133.711 |
| Novel Focus Alternative Study | -1544.388 | -1847.238 | -1102.600 | -1633.002 | -779.714 | -1170.137 | -1694.071 | -729.351 | -732.908 |

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
