# Paired method comparisons on the held-out split

Each block compares methods on the same items for one model and one seed.
`p_holm` is Holm-corrected within the set of pairwise tests for that block.
`comparability` is reported twice: once for the whole block (all methods in
the omnibus test) and once for each pair actually contrasted. A pair labelled
`homogeneous` came from one recorded code state and one training configuration;
`mixed_code_state` or `unknown_code_state` means something besides the method
differs between the two runs, so that contrast is descriptive only.

## olmo2_1_49b · seed 42 · top1_correct (n = 274, comparability = mixed_code_state)

Rates: A = 4.01%, B = 21.53%, C = 2.55%, D = 0.73%
Cochran's Q = 111.72, df = 3, p = 4.68e-24

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +17.52 | [+12.04, +22.99] | 66 | 1.18e-09 | 4.73e-09 | yes |
| C-A | -1.46 | [-4.38, +1.46] | 18 | 0.481 | 0.481 | no |
| D-C | -1.82 | [-3.65, -0.36] | 5 | 0.0625 | 0.125 | no |
| D-B | -20.80 | [-25.55, -15.69] | 61 | 1.64e-15 | 9.85e-15 | yes |
| D-A | -3.28 | [-5.84, -0.73] | 13 | 0.0225 | 0.0674 | no |

## olmo2_1_49b · seed 42 · json_object_recovered (n = 274, comparability = mixed_code_state)

Rates: A = 30.29%, B = 46.35%, C = 2.92%, D = 1.09%
Cochran's Q = 223.19, df = 3, p = 4.09e-48

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +16.06 | [+7.66, +24.45] | 148 | 0.000373 | 0.000747 | yes |
| C-A | -27.37 | [-33.21, -21.53] | 87 | 7.03e-18 | 2.11e-17 | yes |
| D-C | -1.82 | [-3.65, -0.36] | 5 | 0.0625 | 0.0625 | no |
| D-B | -45.26 | [-51.46, -39.42] | 130 | 5.38e-34 | 3.23e-33 | yes |
| D-A | -29.20 | [-35.04, -23.72] | 86 | 2.74e-21 | 1.1e-20 | yes |

## olmo2_1_49b · seed 42 · runtime_object_parsed (n = 274, comparability = mixed_code_state)

Rates: A = 26.28%, B = 38.32%, C = 2.92%, D = 1.09%
Cochran's Q = 175.89, df = 3, p = 6.81e-38

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +12.04 | [+3.65, +20.08] | 137 | 0.00605 | 0.0121 | yes |
| C-A | -23.36 | [-28.83, -17.88] | 76 | 6.31e-15 | 1.89e-14 | yes |
| D-C | -1.82 | [-3.65, -0.36] | 5 | 0.0625 | 0.0625 | no |
| D-B | -37.23 | [-43.43, -31.39] | 108 | 1.29e-27 | 7.77e-27 | yes |
| D-A | -25.18 | [-30.66, -19.71] | 75 | 3.73e-18 | 1.49e-17 | yes |

## olmo2_1_49b · seed 42 · strict_schema_valid (n = 274, comparability = mixed_code_state)

Rates: A = 0.0%, B = 0.0%, C = 0.0%, D = 0.0%
Cochran's Q = 0.00, df = 3, p = 1

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## olmo2_1_49b · seed 42 · encoding_object_recovered (n = 274, comparability = mixed_code_state)

Rates: A = 12.04%, B = 9.85%, C = 2.92%, D = 1.09%
Cochran's Q = 39.22, df = 3, p = 1.56e-08

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -2.19 | [-7.30, +2.92] | 54 | 0.497 | 0.497 | no |
| C-A | -9.12 | [-13.50, -5.11] | 39 | 7.03e-05 | 0.000281 | yes |
| D-C | -1.82 | [-3.65, -0.36] | 5 | 0.0625 | 0.125 | no |
| D-B | -8.76 | [-12.77, -5.11] | 30 | 8.43e-06 | 4.22e-05 | yes |
| D-A | -10.95 | [-14.96, -6.93] | 36 | 2.27e-07 | 1.36e-06 | yes |

## olmo2_1_49b · seed 42 · format_tolerant_chart_correct (n = 274, comparability = mixed_code_state)

Rates: A = 26.28%, B = 37.96%, C = 2.55%, D = 0.73%
Cochran's Q = 181.74, df = 3, p = 3.71e-39

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +11.68 | [+3.65, +19.71] | 126 | 0.00554 | 0.0111 | yes |
| C-A | -23.72 | [-29.20, -18.25] | 77 | 3.42e-15 | 1.03e-14 | yes |
| D-C | -1.82 | [-3.65, -0.36] | 5 | 0.0625 | 0.0625 | no |
| D-B | -37.23 | [-43.07, -31.39] | 106 | 1.4e-28 | 8.39e-28 | yes |
| D-A | -25.55 | [-31.02, -20.44] | 74 | 2.94e-19 | 1.18e-18 | yes |

## olmo2_1_49b · seed 42 · text_level_chart_correct (n = 274, comparability = mixed_code_state)

Rates: A = 86.86%, B = 84.67%, C = 87.23%, D = 59.85%
Cochran's Q = 110.05, df = 3, p = 1.07e-23

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -2.19 | [-5.47, +0.73] | 18 | 0.238 | 0.714 | no |
| C-A | +0.36 | [-4.38, +5.11] | 45 | 1 | 1 | no |
| D-C | -27.37 | [-33.94, -20.44] | 109 | 1.23e-13 | 7.36e-13 | yes |
| D-B | -24.82 | [-31.02, -18.61] | 98 | 1.4e-12 | 5.6e-12 | yes |
| D-A | -27.01 | [-33.58, -20.44] | 110 | 3.8e-13 | 1.9e-12 | yes |

## olmo2_1_49b · seed 43 · top1_correct (n = 274, comparability = unknown_code_state)

Rates: A = 6.93%, B = 24.09%, C = 2.92%, D = 2.19%
Cochran's Q = 110.51, df = 3, p = 8.52e-24

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +17.15 | [+11.68, +22.63] | 65 | 2.05e-09 | 8.2e-09 | yes |
| C-A | -4.01 | [-7.66, -0.36] | 27 | 0.0522 | 0.104 | no |
| D-C | -0.73 | [-2.19, +0.73] | 4 | 0.625 | 0.625 | no |
| D-B | -21.90 | [-27.37, -16.79] | 68 | 5.87e-15 | 3.52e-14 | yes |
| D-A | -4.74 | [-8.39, -1.46] | 25 | 0.0146 | 0.0439 | yes |

## olmo2_1_49b · seed 43 · json_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 32.85%, B = 49.64%, C = 2.92%, D = 2.55%
Cochran's Q = 250.88, df = 3, p = 4.21e-54

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +16.79 | [+8.76, +24.82] | 128 | 5.88e-05 | 0.000118 | yes |
| C-A | -29.93 | [-35.77, -24.09] | 90 | 4.32e-21 | 1.3e-20 | yes |
| D-C | -0.36 | [-1.82, +0.73] | 3 | 1 | 1 | no |
| D-B | -47.08 | [-53.65, -40.88] | 137 | 1.66e-34 | 9.97e-34 | yes |
| D-A | -30.29 | [-36.13, -24.45] | 89 | 3.8e-22 | 1.52e-21 | yes |

## olmo2_1_49b · seed 43 · runtime_object_parsed (n = 274, comparability = unknown_code_state)

Rates: A = 27.74%, B = 42.7%, C = 2.92%, D = 2.55%
Cochran's Q = 205.36, df = 3, p = 2.93e-44

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +14.96 | [+7.30, +22.63] | 115 | 0.000166 | 0.000331 | yes |
| C-A | -24.82 | [-30.29, -19.34] | 80 | 5.4e-16 | 1.62e-15 | yes |
| D-C | -0.36 | [-1.82, +0.73] | 3 | 1 | 1 | no |
| D-B | -40.15 | [-46.35, -34.31] | 118 | 4.78e-29 | 2.87e-28 | yes |
| D-A | -25.18 | [-30.66, -19.71] | 79 | 7.98e-17 | 3.19e-16 | yes |

## olmo2_1_49b · seed 43 · strict_schema_valid (n = 274, comparability = unknown_code_state)

Rates: A = 0.0%, B = 0.0%, C = 0.0%, D = 0.0%
Cochran's Q = 0.00, df = 3, p = 1

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## olmo2_1_49b · seed 43 · encoding_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 10.22%, B = 9.85%, C = 2.92%, D = 2.55%
Cochran's Q = 29.34, df = 3, p = 1.9e-06

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -0.36 | [-5.11, +4.38] | 41 | 1 | 1 | no |
| C-A | -7.30 | [-11.31, -3.65] | 30 | 0.000325 | 0.00162 | yes |
| D-C | -0.36 | [-1.82, +0.73] | 3 | 1 | 1 | no |
| D-B | -7.30 | [-11.31, -3.65] | 30 | 0.000325 | 0.00162 | yes |
| D-A | -7.66 | [-11.31, -4.01] | 29 | 0.000104 | 0.000622 | yes |

## olmo2_1_49b · seed 43 · format_tolerant_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 28.83%, B = 41.61%, C = 2.92%, D = 2.19%
Cochran's Q = 204.94, df = 3, p = 3.6e-44

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +12.77 | [+5.47, +20.07] | 109 | 0.00103 | 0.00207 | yes |
| C-A | -25.91 | [-31.75, -20.07] | 81 | 2.26e-17 | 6.79e-17 | yes |
| D-C | -0.73 | [-2.19, +0.73] | 4 | 0.625 | 0.625 | no |
| D-B | -39.42 | [-45.62, -33.21] | 116 | 1.79e-28 | 1.07e-27 | yes |
| D-A | -26.64 | [-32.12, -21.17] | 79 | 2.72e-19 | 1.09e-18 | yes |

## olmo2_1_49b · seed 43 · text_level_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 87.59%, B = 83.58%, C = 93.43%, D = 56.93%
Cochran's Q = 158.37, df = 3, p = 4.13e-34

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -4.01 | [-7.66, -0.73] | 23 | 0.0347 | 0.0347 | yes |
| C-A | +5.84 | [+1.82, +9.85] | 32 | 0.007 | 0.014 | yes |
| D-C | -36.50 | [-42.70, -29.93] | 118 | 5.86e-23 | 3.51e-22 | yes |
| D-B | -26.64 | [-33.21, -20.07] | 107 | 3.51e-13 | 1.4e-12 | yes |
| D-A | -30.66 | [-37.59, -23.72] | 122 | 3.76e-15 | 1.88e-14 | yes |

## olmo2_1_49b · seed 44 · top1_correct (n = 274, comparability = unknown_code_state)

Rates: A = 5.11%, B = 23.36%, C = 2.92%, D = 2.92%
Cochran's Q = 104.46, df = 3, p = 1.71e-22

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +18.25 | [+12.77, +23.72] | 70 | 8e-10 | 3.2e-09 | yes |
| C-A | -2.19 | [-5.47, +1.09] | 20 | 0.263 | 0.79 | no |
| D-C | +0.00 | [-2.19, +2.19] | 8 | 1 | 1 | no |
| D-B | -20.44 | [-25.91, -15.33] | 66 | 2.63e-13 | 1.58e-12 | yes |
| D-A | -2.19 | [-5.47, +1.09] | 22 | 0.286 | 0.79 | no |

## olmo2_1_49b · seed 44 · json_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 32.12%, B = 47.08%, C = 2.92%, D = 3.28%
Cochran's Q = 221.24, df = 3, p = 1.08e-47

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +14.96 | [+6.57, +22.99] | 135 | 0.000527 | 0.00105 | yes |
| C-A | -29.20 | [-35.04, -23.36] | 90 | 7.53e-20 | 3.01e-19 | yes |
| D-C | +0.36 | [-1.82, +2.55] | 9 | 1 | 1 | no |
| D-B | -43.80 | [-50.00, -37.59] | 128 | 6.48e-32 | 3.89e-31 | yes |
| D-A | -28.83 | [-34.67, -22.63] | 93 | 2.08e-18 | 6.23e-18 | yes |

## olmo2_1_49b · seed 44 · runtime_object_parsed (n = 274, comparability = unknown_code_state)

Rates: A = 27.01%, B = 38.69%, C = 2.92%, D = 3.28%
Cochran's Q = 166.07, df = 3, p = 8.99e-36

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +11.68 | [+3.28, +20.07] | 136 | 0.00762 | 0.0152 | yes |
| C-A | -24.09 | [-29.56, -18.61] | 76 | 5.25e-16 | 2.1e-15 | yes |
| D-C | +0.36 | [-1.82, +2.55] | 9 | 1 | 1 | no |
| D-B | -35.40 | [-41.61, -29.56] | 105 | 2.45e-25 | 1.47e-24 | yes |
| D-A | -23.72 | [-29.56, -17.88] | 81 | 2.98e-14 | 8.93e-14 | yes |

## olmo2_1_49b · seed 44 · strict_schema_valid (n = 274, comparability = unknown_code_state)

Rates: A = 0.0%, B = 0.0%, C = 0.0%, D = 0.0%
Cochran's Q = 0.00, df = 3, p = 1

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## olmo2_1_49b · seed 44 · encoding_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 12.04%, B = 9.12%, C = 2.92%, D = 3.28%
Cochran's Q = 28.45, df = 3, p = 2.93e-06

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -2.92 | [-7.66, +1.82] | 46 | 0.302 | 0.604 | no |
| C-A | -9.12 | [-13.50, -5.11] | 37 | 4.13e-05 | 0.000248 | yes |
| D-C | +0.36 | [-1.82, +2.55] | 9 | 1 | 1 | no |
| D-B | -5.84 | [-9.85, -2.19] | 30 | 0.00522 | 0.0157 | yes |
| D-A | -8.76 | [-13.14, -4.38] | 38 | 0.000116 | 0.000581 | yes |

## olmo2_1_49b · seed 44 · format_tolerant_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 30.29%, B = 40.51%, C = 2.92%, D = 2.92%
Cochran's Q = 187.51, df = 3, p = 2.1e-40

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +10.22 | [+2.19, +18.25] | 124 | 0.015 | 0.03 | yes |
| C-A | -27.37 | [-33.21, -21.53] | 85 | 1.81e-18 | 7.22e-18 | yes |
| D-C | +0.00 | [-2.19, +2.19] | 8 | 1 | 1 | no |
| D-B | -37.59 | [-43.80, -31.75] | 111 | 4.79e-27 | 2.87e-26 | yes |
| D-A | -27.37 | [-33.21, -21.53] | 89 | 2.43e-17 | 7.29e-17 | yes |

## olmo2_1_49b · seed 44 · text_level_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 86.13%, B = 85.77%, C = 94.89%, D = 66.06%
Cochran's Q = 114.72, df = 3, p = 1.06e-24

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -0.36 | [-3.65, +2.55] | 19 | 1 | 1 | no |
| C-A | +8.76 | [+5.11, +12.77] | 32 | 1.93e-05 | 3.86e-05 | yes |
| D-C | -28.83 | [-34.67, -22.99] | 91 | 5.78e-19 | 3.47e-18 | yes |
| D-B | -19.71 | [-25.91, -13.50] | 82 | 1.13e-09 | 5.67e-09 | yes |
| D-A | -20.07 | [-26.64, -13.50] | 93 | 7.72e-09 | 3.09e-08 | yes |

## qwen3_1_7b · seed 42 · top1_correct (n = 274, comparability = unknown_code_state)

Rates: A = 0.36%, B = 1.09%, C = 1.46%, D = 0.0%
Cochran's Q = 5.00, df = 3, p = 0.172

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.73 | [-0.73, +2.19] | 4 | 0.625 | 1 | no |
| C-A | +1.09 | [-0.36, +2.92] | 5 | 0.375 | 1 | no |
| D-C | -1.46 | [-2.92, -0.36] | 4 | 0.125 | 0.75 | no |
| D-B | -1.09 | [-2.55, +0.00] | 3 | 0.25 | 1 | no |
| D-A | -0.36 | [-1.09, +0.00] | 1 | 1 | 1 | no |

## qwen3_1_7b · seed 42 · json_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 99.64%, B = 100.0%, C = 97.45%, D = 95.99%
Cochran's Q = 17.00, df = 3, p = 0.000707

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.36 | [+0.00, +1.09] | 1 | 1 | 1 | no |
| C-A | -2.19 | [-4.38, -0.36] | 8 | 0.0703 | 0.211 | no |
| D-C | -1.46 | [-4.38, +1.46] | 18 | 0.481 | 0.961 | no |
| D-B | -4.01 | [-6.57, -1.82] | 11 | 0.000977 | 0.00586 | yes |
| D-A | -3.65 | [-6.20, -1.46] | 12 | 0.00635 | 0.0317 | yes |

## qwen3_1_7b · seed 42 · runtime_object_parsed (n = 274, comparability = unknown_code_state)

Rates: A = 0.36%, B = 1.09%, C = 1.46%, D = 0.0%
Cochran's Q = 5.00, df = 3, p = 0.172

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.73 | [-0.73, +2.19] | 4 | 0.625 | 1 | no |
| C-A | +1.09 | [-0.36, +2.92] | 5 | 0.375 | 1 | no |
| D-C | -1.46 | [-2.92, -0.36] | 4 | 0.125 | 0.75 | no |
| D-B | -1.09 | [-2.55, +0.00] | 3 | 0.25 | 1 | no |
| D-A | -0.36 | [-1.09, +0.00] | 1 | 1 | 1 | no |

## qwen3_1_7b · seed 42 · strict_schema_valid (n = 274, comparability = unknown_code_state)

Rates: A = 0.36%, B = 1.09%, C = 1.46%, D = 0.0%
Cochran's Q = 5.00, df = 3, p = 0.172

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.73 | [-0.73, +2.19] | 4 | 0.625 | 1 | no |
| C-A | +1.09 | [-0.36, +2.92] | 5 | 0.375 | 1 | no |
| D-C | -1.46 | [-2.92, -0.36] | 4 | 0.125 | 0.75 | no |
| D-B | -1.09 | [-2.55, +0.00] | 3 | 0.25 | 1 | no |
| D-A | -0.36 | [-1.09, +0.00] | 1 | 1 | 1 | no |

## qwen3_1_7b · seed 42 · encoding_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 0.36%, B = 1.09%, C = 1.46%, D = 0.0%
Cochran's Q = 5.00, df = 3, p = 0.172

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.73 | [-0.73, +2.19] | 4 | 0.625 | 1 | no |
| C-A | +1.09 | [-0.36, +2.92] | 5 | 0.375 | 1 | no |
| D-C | -1.46 | [-2.92, -0.36] | 4 | 0.125 | 0.75 | no |
| D-B | -1.09 | [-2.55, +0.00] | 3 | 0.25 | 1 | no |
| D-A | -0.36 | [-1.09, +0.00] | 1 | 1 | 1 | no |

## qwen3_1_7b · seed 42 · format_tolerant_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 86.5%, B = 90.51%, C = 94.53%, D = 89.05%
Cochran's Q = 16.21, df = 3, p = 0.00103

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +4.01 | [+1.09, +6.93] | 17 | 0.0127 | 0.0509 | no |
| C-A | +8.03 | [+4.01, +12.05] | 34 | 0.000195 | 0.00117 | yes |
| D-C | -5.47 | [-9.12, -1.82] | 29 | 0.00813 | 0.0407 | yes |
| D-B | -1.46 | [-5.47, +2.92] | 34 | 0.608 | 0.783 | no |
| D-A | +2.55 | [-2.55, +7.66] | 49 | 0.392 | 0.783 | no |

## qwen3_1_7b · seed 42 · text_level_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 86.86%, B = 90.51%, C = 96.72%, D = 93.07%
Cochran's Q = 34.84, df = 3, p = 1.32e-07

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +3.65 | [+1.09, +6.57] | 16 | 0.0213 | 0.0425 | yes |
| C-A | +9.85 | [+6.57, +13.50] | 27 | 1.49e-08 | 8.94e-08 | yes |
| D-C | -3.65 | [-6.20, -1.46] | 12 | 0.00635 | 0.0254 | yes |
| D-B | +2.55 | [-0.73, +6.20] | 23 | 0.21 | 0.21 | no |
| D-A | +6.20 | [+1.82, +10.58] | 37 | 0.00763 | 0.0254 | yes |

## qwen3_1_7b · seed 43 · top1_correct (n = 274, comparability = unknown_code_state)

Rates: A = 0.0%, B = 1.09%, C = 80.66%, D = 89.42%
Cochran's Q = 657.80, df = 3, p = 2.97e-142

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +1.09 | [+0.00, +2.55] | 3 | 0.25 | 0.25 | no |
| C-A | +80.66 | [+75.91, +85.04] | 221 | 5.93e-67 | 2.37e-66 | yes |
| D-C | +8.76 | [+3.65, +13.87] | 52 | 0.0012 | 0.00239 | yes |
| D-B | +88.32 | [+84.31, +91.97] | 242 | 2.83e-73 | 1.41e-72 | yes |
| D-A | +89.42 | [+85.77, +93.07] | 245 | 3.54e-74 | 2.12e-73 | yes |

## qwen3_1_7b · seed 43 · json_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 100.0%, B = 99.64%, C = 83.94%, D = 98.54%
Cochran's Q = 115.13, df = 3, p = 8.64e-25

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -0.36 | [-1.09, +0.00] | 1 | 1 | 1 | no |
| C-A | -16.06 | [-20.44, -11.68] | 44 | 1.14e-13 | 6.82e-13 | yes |
| D-C | +14.60 | [+10.22, +18.98] | 44 | 1.13e-10 | 4.51e-10 | yes |
| D-B | -1.09 | [-2.92, +0.36] | 5 | 0.375 | 0.75 | no |
| D-A | -1.46 | [-2.92, -0.36] | 4 | 0.125 | 0.375 | no |

## qwen3_1_7b · seed 43 · runtime_object_parsed (n = 274, comparability = unknown_code_state)

Rates: A = 0.0%, B = 1.46%, C = 83.94%, D = 98.54%
Cochran's Q = 717.60, df = 3, p = 3.2e-155

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +1.46 | [+0.36, +2.92] | 4 | 0.125 | 0.125 | no |
| C-A | +83.94 | [+79.56, +88.32] | 230 | 1.16e-69 | 4.64e-69 | yes |
| D-C | +14.60 | [+10.22, +18.98] | 44 | 1.13e-10 | 2.25e-10 | yes |
| D-B | +97.08 | [+94.89, +98.91] | 266 | 1.69e-80 | 8.43e-80 | yes |
| D-A | +98.54 | [+97.08, +99.64] | 270 | 1.05e-81 | 6.33e-81 | yes |

## qwen3_1_7b · seed 43 · strict_schema_valid (n = 274, comparability = unknown_code_state)

Rates: A = 0.0%, B = 1.46%, C = 0.0%, D = 0.0%
Cochran's Q = 12.00, df = 3, p = 0.00738

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +1.46 | [+0.36, +2.92] | 4 | 0.125 | 0.75 | no |
| C-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | -1.46 | [-2.92, -0.36] | 4 | 0.125 | 0.75 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## qwen3_1_7b · seed 43 · encoding_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 0.0%, B = 1.46%, C = 83.94%, D = 98.54%
Cochran's Q = 717.60, df = 3, p = 3.2e-155

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +1.46 | [+0.36, +2.92] | 4 | 0.125 | 0.125 | no |
| C-A | +83.94 | [+79.56, +88.32] | 230 | 1.16e-69 | 4.64e-69 | yes |
| D-C | +14.60 | [+10.22, +18.98] | 44 | 1.13e-10 | 2.25e-10 | yes |
| D-B | +97.08 | [+94.89, +98.91] | 266 | 1.69e-80 | 8.43e-80 | yes |
| D-A | +98.54 | [+97.08, +99.64] | 270 | 1.05e-81 | 6.33e-81 | yes |

## qwen3_1_7b · seed 43 · format_tolerant_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 87.23%, B = 89.78%, C = 81.02%, D = 89.42%
Cochran's Q = 16.44, df = 3, p = 0.000919

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +2.55 | [-0.36, +5.47] | 17 | 0.143 | 0.43 | no |
| C-A | -6.20 | [-12.04, -0.36] | 65 | 0.0464 | 0.185 | no |
| D-C | +8.39 | [+3.65, +13.50] | 51 | 0.00177 | 0.0106 | yes |
| D-B | -0.36 | [-4.38, +4.01] | 33 | 1 | 1 | no |
| D-A | +2.19 | [-2.55, +6.93] | 44 | 0.451 | 0.903 | no |

## qwen3_1_7b · seed 43 · text_level_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 87.23%, B = 90.15%, C = 96.72%, D = 90.88%
Cochran's Q = 28.86, df = 3, p = 2.39e-06

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +2.92 | [+0.00, +5.84] | 16 | 0.0768 | 0.23 | no |
| C-A | +9.49 | [+6.20, +13.14] | 26 | 2.98e-08 | 1.79e-07 | yes |
| D-C | -5.84 | [-8.76, -2.92] | 18 | 0.000145 | 0.00058 | yes |
| D-B | +0.73 | [-2.92, +4.38] | 28 | 0.851 | 0.851 | no |
| D-A | +3.65 | [-0.73, +8.03] | 40 | 0.154 | 0.308 | no |

## qwen3_1_7b · seed 44 · top1_correct (n = 274, comparability = unknown_code_state)

Rates: A = 0.36%, B = 1.09%, C = 85.4%, D = 89.05%
Cochran's Q = 671.96, df = 3, p = 2.52e-145

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.73 | [-0.73, +2.19] | 4 | 0.625 | 0.625 | no |
| C-A | +85.04 | [+80.66, +89.05] | 235 | 8.55e-69 | 2.56e-68 | yes |
| D-C | +3.65 | [-1.46, +8.76] | 50 | 0.203 | 0.405 | no |
| D-B | +87.96 | [+83.94, +91.61] | 241 | 5.66e-73 | 2.83e-72 | yes |
| D-A | +88.69 | [+84.67, +92.34] | 243 | 1.41e-73 | 8.49e-73 | yes |

## qwen3_1_7b · seed 44 · json_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 100.0%, B = 100.0%, C = 87.96%, D = 100.0%
Cochran's Q = 99.00, df = 3, p = 2.55e-21

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | -12.04 | [-16.06, -8.39] | 33 | 2.33e-10 | 1.4e-09 | yes |
| D-C | +12.04 | [+8.39, +16.06] | 33 | 2.33e-10 | 1.4e-09 | yes |
| D-B | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## qwen3_1_7b · seed 44 · runtime_object_parsed (n = 274, comparability = unknown_code_state)

Rates: A = 0.36%, B = 1.09%, C = 87.96%, D = 100.0%
Cochran's Q = 744.51, df = 3, p = 4.69e-161

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.73 | [-0.73, +2.19] | 4 | 0.625 | 0.625 | no |
| C-A | +87.59 | [+83.58, +91.61] | 242 | 6.88e-71 | 2.06e-70 | yes |
| D-C | +12.04 | [+8.39, +16.06] | 33 | 2.33e-10 | 4.66e-10 | yes |
| D-B | +98.91 | [+97.45, +100.00] | 271 | 5.27e-82 | 2.64e-81 | yes |
| D-A | +99.64 | [+98.91, +100.00] | 273 | 1.32e-82 | 7.91e-82 | yes |

## qwen3_1_7b · seed 44 · strict_schema_valid (n = 274, comparability = unknown_code_state)

Rates: A = 0.36%, B = 1.09%, C = 0.0%, D = 0.0%
Cochran's Q = 6.00, df = 3, p = 0.112

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.73 | [-0.73, +2.19] | 4 | 0.625 | 1 | no |
| C-A | -0.36 | [-1.09, +0.00] | 1 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | -1.09 | [-2.55, +0.00] | 3 | 0.25 | 1 | no |
| D-A | -0.36 | [-1.09, +0.00] | 1 | 1 | 1 | no |

## qwen3_1_7b · seed 44 · encoding_object_recovered (n = 274, comparability = unknown_code_state)

Rates: A = 0.36%, B = 1.09%, C = 87.96%, D = 100.0%
Cochran's Q = 744.51, df = 3, p = 4.69e-161

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.73 | [-0.73, +2.19] | 4 | 0.625 | 0.625 | no |
| C-A | +87.59 | [+83.58, +91.61] | 242 | 6.88e-71 | 2.06e-70 | yes |
| D-C | +12.04 | [+8.39, +16.06] | 33 | 2.33e-10 | 4.66e-10 | yes |
| D-B | +98.91 | [+97.45, +100.00] | 271 | 5.27e-82 | 2.64e-81 | yes |
| D-A | +99.64 | [+98.91, +100.00] | 273 | 1.32e-82 | 7.91e-82 | yes |

## qwen3_1_7b · seed 44 · format_tolerant_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 86.86%, B = 89.05%, C = 85.4%, D = 89.05%
Cochran's Q = 3.32, df = 3, p = 0.344

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +2.19 | [-0.73, +5.11] | 16 | 0.21 | 1 | no |
| C-A | -1.46 | [-6.93, +4.01] | 58 | 0.694 | 1 | no |
| D-C | +3.65 | [-1.46, +8.76] | 50 | 0.203 | 1 | no |
| D-B | +0.00 | [-4.01, +4.01] | 34 | 1 | 1 | no |
| D-A | +2.19 | [-2.55, +6.93] | 46 | 0.461 | 1 | no |

## qwen3_1_7b · seed 44 · text_level_chart_correct (n = 274, comparability = unknown_code_state)

Rates: A = 86.86%, B = 89.05%, C = 97.45%, D = 89.05%
Cochran's Q = 33.79, df = 3, p = 2.2e-07

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +2.19 | [-0.73, +5.11] | 16 | 0.21 | 0.63 | no |
| C-A | +10.58 | [+7.30, +14.60] | 29 | 3.73e-09 | 2.24e-08 | yes |
| D-C | -8.39 | [-11.68, -5.11] | 25 | 1.55e-06 | 7.75e-06 | yes |
| D-B | +0.00 | [-4.01, +4.01] | 34 | 1 | 1 | no |
| D-A | +2.19 | [-2.55, +6.93] | 46 | 0.461 | 0.923 | no |

## qwen3_8_27b · seed 42 · top1_correct (n = 274, comparability = mixed_code_state)

Rates: A = 92.34%, B = 83.58%, C = 96.72%, D = 97.45%
Cochran's Q = 84.46, df = 3, p = 3.39e-18

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -8.76 | [-12.41, -5.11] | 28 | 3.03e-06 | 1.21e-05 | yes |
| C-A | +4.38 | [+2.19, +6.93] | 12 | 0.000488 | 0.000977 | yes |
| D-C | +0.73 | [+0.00, +1.82] | 2 | 0.5 | 0.5 | no |
| D-B | +13.87 | [+9.85, +17.88] | 38 | 7.28e-12 | 4.37e-11 | yes |
| D-A | +5.11 | [+2.55, +7.67] | 14 | 0.000122 | 0.000366 | yes |

## qwen3_8_27b · seed 42 · json_object_recovered (n = 274, comparability = mixed_code_state)

Rates: A = 100.0%, B = 100.0%, C = 100.0%, D = 100.0%
Cochran's Q = 0.00, df = 3, p = 1

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## qwen3_8_27b · seed 42 · runtime_object_parsed (n = 274, comparability = mixed_code_state)

Rates: A = 100.0%, B = 100.0%, C = 100.0%, D = 100.0%
Cochran's Q = 0.00, df = 3, p = 1

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## qwen3_8_27b · seed 42 · strict_schema_valid (n = 274, comparability = mixed_code_state)

Rates: A = 46.72%, B = 84.67%, C = 0.0%, D = 0.0%
Cochran's Q = 530.02, df = 3, p = 1.49e-114

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +37.96 | [+31.02, +44.89] | 136 | 6.91e-21 | 1.38e-20 | yes |
| C-A | -46.72 | [-52.55, -40.88] | 128 | 5.88e-39 | 2.35e-38 | yes |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | -84.67 | [-88.69, -80.29] | 232 | 2.9e-70 | 1.74e-69 | yes |
| D-A | -46.72 | [-52.55, -40.88] | 128 | 5.88e-39 | 2.35e-38 | yes |

## qwen3_8_27b · seed 42 · encoding_object_recovered (n = 274, comparability = mixed_code_state)

Rates: A = 100.0%, B = 100.0%, C = 100.0%, D = 100.0%
Cochran's Q = 0.00, df = 3, p = 1

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## qwen3_8_27b · seed 42 · format_tolerant_chart_correct (n = 274, comparability = mixed_code_state)

Rates: A = 92.34%, B = 83.58%, C = 96.72%, D = 97.45%
Cochran's Q = 84.46, df = 3, p = 3.39e-18

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -8.76 | [-12.41, -5.11] | 28 | 3.03e-06 | 1.21e-05 | yes |
| C-A | +4.38 | [+2.19, +6.93] | 12 | 0.000488 | 0.000977 | yes |
| D-C | +0.73 | [+0.00, +1.82] | 2 | 0.5 | 0.5 | no |
| D-B | +13.87 | [+9.85, +17.88] | 38 | 7.28e-12 | 4.37e-11 | yes |
| D-A | +5.11 | [+2.55, +7.67] | 14 | 0.000122 | 0.000366 | yes |

## qwen3_8_27b · seed 42 · text_level_chart_correct (n = 274, comparability = mixed_code_state)

Rates: A = 92.34%, B = 83.58%, C = 96.72%, D = 97.45%
Cochran's Q = 84.46, df = 3, p = 3.39e-18

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -8.76 | [-12.41, -5.11] | 28 | 3.03e-06 | 1.21e-05 | yes |
| C-A | +4.38 | [+2.19, +6.93] | 12 | 0.000488 | 0.000977 | yes |
| D-C | +0.73 | [+0.00, +1.82] | 2 | 0.5 | 0.5 | no |
| D-B | +13.87 | [+9.85, +17.88] | 38 | 7.28e-12 | 4.37e-11 | yes |
| D-A | +5.11 | [+2.55, +7.67] | 14 | 0.000122 | 0.000366 | yes |

## qwen3_8b · seed 42 · top1_correct (n = 274, comparability = mixed_code_state)

Rates: A = 91.61%, B = 82.12%, C = 83.94%, D = 58.39%
Cochran's Q = 133.12, df = 3, p = 1.15e-28

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -9.49 | [-13.50, -5.84] | 32 | 2.56e-06 | 7.67e-06 | yes |
| C-A | -7.66 | [-12.41, -2.92] | 47 | 0.00309 | 0.00618 | yes |
| D-C | -25.55 | [-31.02, -20.07] | 74 | 2.94e-19 | 1.47e-18 | yes |
| D-B | -23.72 | [-29.56, -17.88] | 81 | 2.98e-14 | 1.19e-13 | yes |
| D-A | -33.21 | [-39.78, -27.01] | 111 | 4.4e-20 | 2.64e-19 | yes |

## qwen3_8b · seed 42 · json_object_recovered (n = 274, comparability = mixed_code_state)

Rates: A = 100.0%, B = 100.0%, C = 86.13%, D = 74.09%
Cochran's Q = 165.40, df = 3, p = 1.25e-35

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | -13.87 | [-18.25, -9.85] | 38 | 7.28e-12 | 2.91e-11 | yes |
| D-C | -12.04 | [-16.42, -8.03] | 37 | 1.02e-08 | 2.05e-08 | yes |
| D-B | -25.91 | [-31.02, -20.80] | 71 | 8.47e-22 | 5.08e-21 | yes |
| D-A | -25.91 | [-31.02, -20.80] | 71 | 8.47e-22 | 5.08e-21 | yes |

## qwen3_8b · seed 42 · runtime_object_parsed (n = 274, comparability = mixed_code_state)

Rates: A = 100.0%, B = 100.0%, C = 86.13%, D = 74.09%
Cochran's Q = 165.40, df = 3, p = 1.25e-35

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | -13.87 | [-18.25, -9.85] | 38 | 7.28e-12 | 2.91e-11 | yes |
| D-C | -12.04 | [-16.42, -8.03] | 37 | 1.02e-08 | 2.05e-08 | yes |
| D-B | -25.91 | [-31.02, -20.80] | 71 | 8.47e-22 | 5.08e-21 | yes |
| D-A | -25.91 | [-31.02, -20.80] | 71 | 8.47e-22 | 5.08e-21 | yes |

## qwen3_8b · seed 42 · strict_schema_valid (n = 274, comparability = mixed_code_state)

Rates: A = 0.0%, B = 0.0%, C = 0.0%, D = 0.0%
Cochran's Q = 0.00, df = 3, p = 1

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-C | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-B | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| D-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |

## qwen3_8b · seed 42 · encoding_object_recovered (n = 274, comparability = mixed_code_state)

Rates: A = 100.0%, B = 100.0%, C = 86.13%, D = 74.09%
Cochran's Q = 165.40, df = 3, p = 1.25e-35

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | +0.00 | [+0.00, +0.00] | 0 | 1 | 1 | no |
| C-A | -13.87 | [-18.25, -9.85] | 38 | 7.28e-12 | 2.91e-11 | yes |
| D-C | -12.04 | [-16.42, -8.03] | 37 | 1.02e-08 | 2.05e-08 | yes |
| D-B | -25.91 | [-31.02, -20.80] | 71 | 8.47e-22 | 5.08e-21 | yes |
| D-A | -25.91 | [-31.02, -20.80] | 71 | 8.47e-22 | 5.08e-21 | yes |

## qwen3_8b · seed 42 · format_tolerant_chart_correct (n = 274, comparability = mixed_code_state)

Rates: A = 91.61%, B = 82.12%, C = 83.94%, D = 58.76%
Cochran's Q = 130.84, df = 3, p = 3.57e-28

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -9.49 | [-13.50, -5.84] | 32 | 2.56e-06 | 7.67e-06 | yes |
| C-A | -7.66 | [-12.41, -2.92] | 47 | 0.00309 | 0.00618 | yes |
| D-C | -25.18 | [-30.66, -20.07] | 73 | 5.72e-19 | 2.86e-18 | yes |
| D-B | -23.36 | [-29.20, -17.52] | 80 | 5.38e-14 | 2.15e-13 | yes |
| D-A | -32.85 | [-39.42, -26.64] | 110 | 8.01e-20 | 4.81e-19 | yes |

## qwen3_8b · seed 42 · text_level_chart_correct (n = 274, comparability = mixed_code_state)

Rates: A = 91.61%, B = 82.12%, C = 95.99%, D = 77.01%
Cochran's Q = 75.16, df = 3, p = 3.36e-16

| contrast | difference (pp) | 95% CI (pp) | discordant | p (exact) | p (Holm) | significant | code state of the pair |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| B-A | -9.49 | [-13.50, -5.84] | 32 | 2.56e-06 | 7.67e-06 | yes |
| C-A | +4.38 | [+1.09, +7.66] | 22 | 0.0169 | 0.0338 | yes |
| D-C | -18.98 | [-24.09, -14.23] | 58 | 2.26e-13 | 1.36e-12 | yes |
| D-B | -5.11 | [-9.49, -0.73] | 40 | 0.0385 | 0.0385 | yes |
| D-A | -14.60 | [-20.44, -8.76] | 70 | 1.65e-06 | 6.61e-06 | yes |
