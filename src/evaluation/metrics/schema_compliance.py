"""Schema-compliance metric.

Reports how often the model produced parseable JSON and how complete it was.
Validity is checked against the *raw* extracted object (not the Pydantic model,
which fills defaults), so a missing required key is correctly counted as missing.

  json_parse_rate      — % of outputs from which a JSON object was extracted
  schema_validity_rate — % of outputs containing all required keys
  completeness_score   — mean fraction of required keys present (0..1)
  field_coverage       — per-key presence rate across all outputs
"""

from __future__ import annotations

from src.core.constants import REQUIRED_KEYS
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.inference.postprocess import extract_json_dict


@METRICS.register("schema_compliance")
class SchemaCompliance(BaseMetric):
    name = "schema_compliance"

    def compute(self, results, references=None) -> dict:
        n = len(results)
        if n == 0:
            return {
                "json_parse_rate": None,
                "schema_validity_rate": None,
                "completeness_score": None,
                "field_coverage": {},
                "n": 0,
            }

        parsed_ok = 0
        valid_schema = 0
        completeness_sum = 0.0
        key_present = {k: 0 for k in REQUIRED_KEYS}

        for r in results:
            obj = extract_json_dict(r.raw_text)
            if obj is None:
                continue
            parsed_ok += 1
            present = [k for k in REQUIRED_KEYS if k in obj]
            for k in present:
                key_present[k] += 1
            completeness_sum += len(present) / len(REQUIRED_KEYS)
            if len(present) == len(REQUIRED_KEYS):
                valid_schema += 1

        return {
            "json_parse_rate": round(100.0 * parsed_ok / n, 2),
            "schema_validity_rate": round(100.0 * valid_schema / n, 2),
            "completeness_score": round(completeness_sum / n, 4),
            "field_coverage": {k: round(100.0 * c / n, 2) for k, c in key_present.items()},
            "n": n,
        }
