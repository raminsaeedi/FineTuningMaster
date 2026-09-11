# Failure modes per condition

`parse failures` counts items whose raw text could not be turned into the
runtime object. `no primary chart` counts items with no readable primary
recommendation, which the top-1 metric scores as incorrect.

| model | method | seed | n | parse failures | dominant parse error | no primary chart | items with >1 distinct recommendation |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| olmo2_1_49b | A | 42 | 274 | 202 | no_json_found | 262 | 8 |
| olmo2_1_49b | A | 43 | 274 | 198 | no_json_found | 255 | 12 |
| olmo2_1_49b | A | 44 | 274 | 200 | no_json_found | 258 | 11 |
| olmo2_1_49b | B | 42 | 274 | 169 | no_json_found | 201 | 66 |
| olmo2_1_49b | B | 43 | 274 | 157 | no_json_found | 191 | 64 |
| olmo2_1_49b | B | 44 | 274 | 168 | no_json_found | 199 | 56 |
| olmo2_1_49b | C | 42 | 274 | 266 | no_json_found | 266 | 0 |
| olmo2_1_49b | C | 43 | 274 | 266 | no_json_found | 266 | 0 |
| olmo2_1_49b | C | 44 | 274 | 266 | no_json_found | 266 | 0 |
| olmo2_1_49b | D | 42 | 274 | 271 | no_json_found | 271 | 0 |
| olmo2_1_49b | D | 43 | 274 | 267 | no_json_found | 267 | 0 |
| olmo2_1_49b | D | 44 | 274 | 265 | no_json_found | 265 | 0 |
| qwen3_1_7b | A | 42 | 274 | 273 | schema_error: 1 validation error for DesignOutput
kpi_chart_mapping.0.encoding
  Input should be a valid dictionary [type=dict_type, input_value='category', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/dict_type | 273 | 1 |
| qwen3_1_7b | A | 43 | 274 | 274 | schema_error: 1 validation error for DesignOutput
kpi_chart_mapping.0.encoding
  Input should be a valid dictionary [type=dict_type, input_value='category', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/dict_type | 274 | 0 |
| qwen3_1_7b | A | 44 | 274 | 273 | schema_error: 1 validation error for DesignOutput
kpi_chart_mapping.0.encoding
  Input should be a valid dictionary [type=dict_type, input_value='category', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/dict_type | 273 | 1 |
| qwen3_1_7b | B | 42 | 274 | 271 | schema_error: 1 validation error for DesignOutput
kpi_chart_mapping.0.encoding
  Input should be a valid dictionary [type=dict_type, input_value='category', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/dict_type | 271 | 3 |
| qwen3_1_7b | B | 43 | 274 | 270 | schema_error: 1 validation error for DesignOutput
kpi_chart_mapping.0.encoding
  Input should be a valid dictionary [type=dict_type, input_value='category', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/dict_type | 270 | 4 |
| qwen3_1_7b | B | 44 | 274 | 271 | schema_error: 1 validation error for DesignOutput
kpi_chart_mapping.0.encoding
  Input should be a valid dictionary [type=dict_type, input_value='category', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/dict_type | 271 | 3 |
| qwen3_1_7b | C | 42 | 274 | 270 | no_json_found | 270 | 0 |
| qwen3_1_7b | C | 43 | 274 | 44 | no_json_found | 45 | 0 |
| qwen3_1_7b | C | 44 | 274 | 33 | no_json_found | 33 | 0 |
| qwen3_1_7b | D | 42 | 274 | 274 | no_json_found | 274 | 0 |
| qwen3_1_7b | D | 43 | 274 | 4 | no_json_found | 4 | 0 |
| qwen3_1_7b | D | 44 | 274 | 0 | - | 0 | 0 |
| qwen3_8_27b | A | 42 | 274 | 0 | - | 0 | 265 |
| qwen3_8_27b | B | 42 | 274 | 0 | - | 0 | 267 |
| qwen3_8_27b | C | 42 | 274 | 0 | - | 0 | 0 |
| qwen3_8_27b | D | 42 | 274 | 0 | - | 0 | 0 |
| qwen3_8b | A | 42 | 274 | 0 | - | 0 | 274 |
| qwen3_8b | B | 42 | 274 | 0 | - | 0 | 274 |
| qwen3_8b | C | 42 | 274 | 38 | no_json_found | 38 | 0 |
| qwen3_8b | D | 42 | 274 | 71 | no_json_found | 72 | 0 |
