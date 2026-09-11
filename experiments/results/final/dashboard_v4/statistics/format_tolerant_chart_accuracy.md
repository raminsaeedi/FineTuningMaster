# Chart-token agreement with and without the output-format requirement

`pipeline top-1` is the metric produced by the evaluation code: it reads the
chart from the parsed runtime object, so an item whose output fails the
runtime contract is scored as incorrect.
`format-tolerant chart accuracy` repeats the same comparison on the raw
extracted JSON object and ignores whether the surrounding record satisfies the
runtime contract.
`text-level chart accuracy` reads the first `chart_type` token out of the raw
text, so it also survives a response that the token budget cut off.
`truncated` counts responses whose outermost object was never closed.
A large pipeline-to-text gap means the condition failed the output contract or
the length budget rather than the chart decision.

| model | method | seed | n | JSON object found | truncated | pipeline top-1 (%) | format-tolerant (%) | text-level (%) | gap pipeline->text (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| olmo2_1_49b | A | 42 | 274 | 83 | 272 | 4.01 | 26.28 | 86.86 | 82.85 |
| olmo2_1_49b | A | 43 | 274 | 90 | 272 | 6.93 | 28.83 | 87.59 | 80.66 |
| olmo2_1_49b | A | 44 | 274 | 88 | 273 | 5.11 | 30.29 | 86.13 | 81.02 |
| olmo2_1_49b | B | 42 | 274 | 127 | 274 | 21.53 | 37.96 | 84.67 | 63.14 |
| olmo2_1_49b | B | 43 | 274 | 136 | 274 | 24.09 | 41.61 | 83.58 | 59.49 |
| olmo2_1_49b | B | 44 | 274 | 129 | 272 | 23.36 | 40.51 | 85.77 | 62.41 |
| olmo2_1_49b | C | 42 | 274 | 8 | 256 | 2.55 | 2.55 | 87.23 | 84.68 |
| olmo2_1_49b | C | 43 | 274 | 8 | 254 | 2.92 | 2.92 | 93.43 | 90.51 |
| olmo2_1_49b | C | 44 | 274 | 8 | 258 | 2.92 | 2.92 | 94.89 | 91.97 |
| olmo2_1_49b | D | 42 | 274 | 3 | 264 | 0.73 | 0.73 | 59.85 | 59.12 |
| olmo2_1_49b | D | 43 | 274 | 7 | 262 | 2.19 | 2.19 | 56.93 | 54.74 |
| olmo2_1_49b | D | 44 | 274 | 9 | 258 | 2.92 | 2.92 | 66.06 | 63.14 |
| qwen3_1_7b | A | 42 | 274 | 273 | 0 | 0.36 | 86.5 | 86.86 | 86.5 |
| qwen3_1_7b | A | 43 | 274 | 274 | 0 | 0.0 | 87.23 | 87.23 | 87.23 |
| qwen3_1_7b | A | 44 | 274 | 274 | 0 | 0.36 | 86.86 | 86.86 | 86.5 |
| qwen3_1_7b | B | 42 | 274 | 274 | 0 | 1.09 | 90.51 | 90.51 | 89.42 |
| qwen3_1_7b | B | 43 | 274 | 273 | 0 | 1.09 | 89.78 | 90.15 | 89.06 |
| qwen3_1_7b | B | 44 | 274 | 274 | 0 | 1.09 | 89.05 | 89.05 | 87.96 |
| qwen3_1_7b | C | 42 | 274 | 267 | 1 | 1.46 | 94.53 | 96.72 | 95.26 |
| qwen3_1_7b | C | 43 | 274 | 230 | 38 | 80.66 | 81.02 | 96.72 | 16.06 |
| qwen3_1_7b | C | 44 | 274 | 241 | 29 | 85.4 | 85.4 | 97.45 | 12.05 |
| qwen3_1_7b | D | 42 | 274 | 263 | 1 | 0.0 | 89.05 | 93.07 | 93.07 |
| qwen3_1_7b | D | 43 | 274 | 270 | 0 | 89.42 | 89.42 | 90.88 | 1.46 |
| qwen3_1_7b | D | 44 | 274 | 274 | 0 | 89.05 | 89.05 | 89.05 | 0.0 |
| qwen3_8_27b | A | 42 | 274 | 274 | 0 | 92.34 | 92.34 | 92.34 | -0.0 |
| qwen3_8_27b | B | 42 | 274 | 274 | 0 | 83.58 | 83.58 | 83.58 | -0.0 |
| qwen3_8_27b | C | 42 | 274 | 274 | 0 | 96.72 | 96.72 | 96.72 | -0.0 |
| qwen3_8_27b | D | 42 | 274 | 274 | 0 | 97.45 | 97.45 | 97.45 | -0.0 |
| qwen3_8b | A | 42 | 274 | 274 | 0 | 91.61 | 91.61 | 91.61 | -0.0 |
| qwen3_8b | B | 42 | 274 | 274 | 0 | 82.12 | 82.12 | 82.12 | -0.0 |
| qwen3_8b | C | 42 | 274 | 236 | 36 | 83.94 | 83.94 | 95.99 | 12.05 |
| qwen3_8b | D | 42 | 274 | 203 | 69 | 58.39 | 58.76 | 77.01 | 18.62 |
