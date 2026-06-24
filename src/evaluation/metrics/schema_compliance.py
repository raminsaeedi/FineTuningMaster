"""Schema-compliance metric.

Reports how often the model produced parseable JSON and how complete / valid it
was. Three levels, from lenient to strict:

  json_parse_rate      — % of outputs from which a JSON object was extracted
  required_keys_rate   — % of outputs containing all required top-level keys
                         (presence only — the old, lenient "schema_validity")
  schema_validity_rate — % of outputs that ALSO validate against the full Pydantic
                         contract (correct types + valid TaskType/ChartType enums,
                         with NO lenient normalisation). This is the corrected,
                         scientifically meaningful schema-validity number.
  completeness_score   — mean fraction of required keys that are present AND
                         non-empty (an empty list/dict/string is NOT complete).
  field_coverage       — per-key presence rate across all outputs (diagnostic)

The full-schema check validates the *raw* extracted object, so enum near-misses
the inference parser would silently repair (e.g. "column chart" -> "bar") are
correctly counted as invalid here.
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.constants import REQUIRED_KEYS
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.core.schemas import DesignOutput
from src.inference.postprocess import extract_json_dict


def _nonempty(value: Any) -> bool:
    """A value counts toward completeness only if present and not empty."""
    if value is None:
        return False
    if isinstance(value, (list, dict, str)):
        return len(value) > 0
    return True


def completeness_fraction(obj: Optional[dict]) -> float:
    """Fraction of required keys that are present AND non-empty (0..1)."""
    if not obj:
        return 0.0
    present = sum(1 for k in REQUIRED_KEYS if _nonempty(obj.get(k)))
    return present / len(REQUIRED_KEYS)


def full_schema_valid(obj: Optional[dict]) -> bool:
    """True iff ``obj`` has all required keys AND validates the full Pydantic
    contract strictly (no lenient enum normalisation)."""
    if not obj or not all(k in obj for k in REQUIRED_KEYS):
        return False
    try:
        DesignOutput(**obj)  # validates typed sub-fields incl. TaskType/ChartType
        return True
    except Exception:
        return False


@METRICS.register("schema_compliance")
class SchemaCompliance(BaseMetric):
    name = "schema_compliance"

    def compute(self, results, references=None) -> dict:
        n = len(results)
        if n == 0:
            return {
                "json_parse_rate": None,
                "required_keys_rate": None,
                "schema_validity_rate": None,
                "completeness_score": None,
                "field_coverage": {},
                "n": 0,
            }

        parsed_ok = 0
        required_keys_ok = 0
        full_valid = 0
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
            if len(present) == len(REQUIRED_KEYS):
                required_keys_ok += 1
            if full_schema_valid(obj):
                full_valid += 1
            completeness_sum += completeness_fraction(obj)

        return {
            "json_parse_rate": round(100.0 * parsed_ok / n, 2),
            # Corrected schema validity = full Pydantic + enum validation.
            "schema_validity_rate": round(100.0 * full_valid / n, 2),
            # Old, lenient presence-only number, kept for transparency.
            "required_keys_rate": round(100.0 * required_keys_ok / n, 2),
            "completeness_score": round(completeness_sum / n, 4),
            "field_coverage": {k: round(100.0 * c / n, 2) for k, c in key_present.items()},
            "n": n,
        }
