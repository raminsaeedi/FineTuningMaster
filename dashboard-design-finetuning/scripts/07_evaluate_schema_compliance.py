"""
07_evaluate_schema_compliance.py

Evaluates whether model predictions conform to the expected JSON schema.

Input files:
  outputs/predictions/base_model_prediction.json
  outputs/predictions/finetuned_model_prediction.json

Output:
  outputs/predictions/schema_compliance_report.json

Uses only Python standard library. Does not crash on invalid JSON.
"""

import json
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
PRED_DIR     = os.path.join(BASE_DIR, "..", "outputs", "predictions")
BASE_FILE    = os.path.join(PRED_DIR, "base_model_prediction.json")
FT_FILE      = os.path.join(PRED_DIR, "finetuned_model_prediction.json")
REPORT_FILE  = os.path.join(PRED_DIR, "schema_compliance_report.json")

# ── Expected schema ───────────────────────────────────────────────────────────
# Each entry: (field_name, expected_python_type)
REQUIRED_FIELDS = [
    ("context_summary",      str),
    ("kpi_task_chart_mapping", list),
    ("layout_hierarchy",     str),
    ("labels_scales_colors", str),
    ("interactions",         list),
    ("design_rationales",    list),
]

TOTAL_FIELDS = len(REQUIRED_FIELDS)


# ── Core evaluation logic ─────────────────────────────────────────────────────

def load_prediction_file(filepath: str):
    """
    Load a prediction JSON file and extract the 'prediction' sub-dict.
    Returns (prediction_dict_or_None, error_message_or_None).
    """
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, f"File is not valid JSON: {e}"

    prediction = data.get("prediction")
    if prediction is None:
        return None, "Key 'prediction' is missing from the file."

    # If the prediction itself contains a parse_error, it was not valid JSON
    if isinstance(prediction, dict) and "parse_error" in prediction:
        return None, f"Model output was not valid JSON: {prediction.get('parse_error')}"

    if not isinstance(prediction, dict):
        return None, f"'prediction' is not a JSON object (got {type(prediction).__name__})."

    return prediction, None


def evaluate_schema(prediction: dict) -> dict:
    """
    Check each required field for presence and correct type.
    Returns a dict with detailed results.
    """
    present_fields   = []
    missing_fields   = []
    wrong_type_fields = []

    for field_name, expected_type in REQUIRED_FIELDS:
        if field_name not in prediction:
            missing_fields.append(field_name)
        elif not isinstance(prediction[field_name], expected_type):
            actual_type = type(prediction[field_name]).__name__
            wrong_type_fields.append({
                "field":         field_name,
                "expected_type": expected_type.__name__,
                "actual_type":   actual_type,
            })
            # Still count as present (field exists, just wrong type)
            present_fields.append(field_name)
        else:
            present_fields.append(field_name)

    completeness_score = round(len(present_fields) / TOTAL_FIELDS, 4)

    return {
        "present_fields":    present_fields,
        "missing_fields":    missing_fields,
        "wrong_type_fields": wrong_type_fields,
        "fields_present":    len(present_fields),
        "fields_total":      TOTAL_FIELDS,
        "completeness_score": completeness_score,
    }


def assess_file(label: str, filepath: str) -> dict:
    """
    Full assessment pipeline for one prediction file.
    Never raises — all errors are captured in the result dict.
    """
    result = {
        "label":              label,
        "file":               os.path.abspath(filepath),
        "valid_json":         False,
        "error":              None,
        "completeness_score": 0.0,
        "fields_present":     0,
        "fields_total":       TOTAL_FIELDS,
        "missing_fields":     list(f for f, _ in REQUIRED_FIELDS),
        "wrong_type_fields":  [],
        "present_fields":     [],
    }

    prediction, error = load_prediction_file(filepath)

    if error:
        result["error"] = error
        return result

    result["valid_json"] = True
    schema_result = evaluate_schema(prediction)
    result.update(schema_result)
    return result


def print_report(assessment: dict):
    """Pretty-print one model's assessment to the terminal."""
    label = assessment["label"]
    print(f"\n{label}:")
    print(f"  valid_json        : {assessment['valid_json']}")
    if assessment["error"]:
        print(f"  error             : {assessment['error']}")
    print(f"  completeness_score: {assessment['completeness_score']} "
          f"({assessment['fields_present']}/{assessment['fields_total']} fields)")
    if assessment["missing_fields"]:
        print(f"  missing_fields    : {', '.join(assessment['missing_fields'])}")
    else:
        print(f"  missing_fields    : none")
    if assessment["wrong_type_fields"]:
        for wt in assessment["wrong_type_fields"]:
            print(f"  wrong_type        : '{wt['field']}' "
                  f"(expected {wt['expected_type']}, got {wt['actual_type']})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    base_assessment = assess_file("Base model",        BASE_FILE)
    ft_assessment   = assess_file("Fine-tuned model",  FT_FILE)

    # Terminal output
    print("=" * 55)
    print("SCHEMA COMPLIANCE REPORT")
    print("=" * 55)
    print_report(base_assessment)
    print_report(ft_assessment)
    print("=" * 55)

    # Build and save report
    report = {
        "required_schema": {f: t.__name__ for f, t in REQUIRED_FIELDS},
        "base_model":      base_assessment,
        "finetuned_model": ft_assessment,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {os.path.abspath(REPORT_FILE)}")


if __name__ == "__main__":
    main()
