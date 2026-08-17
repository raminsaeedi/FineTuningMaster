"""End-to-end human-evaluation study lifecycle.

This module owns the final-study contract shared by the builder, rating app and
analysis script.  A study is one immutable dataset/model/seed tuple with four
methods (A--D).  The code deliberately keeps source-run validation and rating
validation close to the artifact writers so a partially compatible study cannot
be created accidentally.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.evaluation.human.assignment import build_assignment, build_eval_items
from src.evaluation.human.irr import krippendorff_alpha
from src.evaluation.human.rubric import RUBRIC_KEYS, RUBRIC_VERSION, rubric_hash
from src.evaluation.human.storage import load_all_ratings
from src.evaluation.stats import (
    cohen_dz,
    cochran_q,
    paired_bootstrap_diff,
    paired_rank_biserial,
    pairwise_mcnemar,
    pairwise_wilcoxon,
    friedman_test,
)


METHODS = ("A", "B", "C", "D")
METHOD_LABELS = {
    "A": "Prompt-only",
    "B": "RAG",
    "C": "QLoRA",
    "D": "QLoRA + RAG",
}
METHOD_ALIASES = {
    "a": "A",
    "promptonly": "A",
    "prompt": "A",
    "promptbaseline": "A",
    "b": "B",
    "rag": "B",
    "retrieval": "B",
    "retrievalaugmented": "B",
    "c": "C",
    "ft": "C",
    "qlora": "C",
    "finetuned": "C",
    "finetunedonly": "C",
    "d": "D",
    "ftrag": "D",
    "qlorarag": "D",
    "qloraandrag": "D",
    "finetunedrag": "D",
    "finetunedretrieval": "D",
    "retrievalfinetuned": "D",
}
STUDY_SCHEMA_VERSION = "human-eval-study-v2"
DEFAULT_OUTPUTS_ROOT = "experiments/outputs/final"
DEFAULT_RESULTS_ROOT = "experiments/results/human_eval"


class HumanEvaluationError(ValueError):
    """Actionable error raised when a study violates its scientific contract."""


class IncompleteStudyError(HumanEvaluationError):
    """Raised when final analysis is attempted before all ratings arrive."""


def _resolve(project_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanEvaluationError(f"Invalid JSON metadata: {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise HumanEvaluationError(f"Could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def normalize_method(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    compact = "".join(ch for ch in raw.lower() if ch.isalnum())
    if compact in METHOD_ALIASES:
        return METHOD_ALIASES[compact]
    return raw.upper() if raw.upper() in METHODS else None


def _nested(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping or {}
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _read_yaml_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HumanEvaluationError(f"Invalid YAML metadata: {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _hash_map(payload: Any) -> dict[str, str]:
    """Extract file/hash mappings from the several run metadata formats."""
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("files"), Mapping):
        result = {}
        for name, entry in payload["files"].items():
            if isinstance(entry, Mapping) and entry.get("sha256"):
                result[Path(str(name)).name.removesuffix(".jsonl")] = str(entry["sha256"])
        return result
    result = {}
    for key, value in payload.items():
        if isinstance(value, str) and len(value) >= 32:
            result[str(key).removesuffix(".jsonl")] = value
    return result


def _metadata_hashes(metadata_paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in metadata_paths if path.exists()}


def inspect_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Read optional Professor run metadata into a compact normalized record."""
    metadata_paths = [
        run_dir / "manifest.json",
        run_dir / "config_snapshot.yaml",
        run_dir / "cache_identity.json",
        run_dir / "dataset_hashes.json",
        run_dir / "kb_hashes.json",
    ]
    manifest = _json_load(metadata_paths[0]) if metadata_paths[0].exists() else {}
    cache_identity = _json_load(metadata_paths[2]) if metadata_paths[2].exists() else {}
    dataset_payload = _json_load(metadata_paths[3]) if metadata_paths[3].exists() else {}
    kb_payload = _json_load(metadata_paths[4]) if metadata_paths[4].exists() else {}
    config = _read_yaml_if_present(metadata_paths[1])

    manifest_hashes = _first(
        manifest.get("dataset_hashes"),
        manifest.get("data_file_sha256"),
        _nested(manifest, "cache_identity", "dataset_hashes"),
    )
    cache_hashes = cache_identity.get("dataset_hashes")
    dataset_hashes = _hash_map(dataset_payload)
    if not dataset_hashes:
        dataset_hashes = _hash_map(manifest_hashes)
    if not dataset_hashes:
        dataset_hashes = _hash_map(cache_hashes)
    standalone_test_hash = _first(
        manifest.get("test_set_hash"),
        manifest.get("test_hash"),
        cache_identity.get("test_set_hash"),
        cache_identity.get("test_hash"),
    )
    if standalone_test_hash and "test" not in dataset_hashes:
        dataset_hashes["test"] = str(standalone_test_hash)

    manifest_kb = _first(
        manifest.get("kb_hashes"),
        manifest.get("knowledge_base"),
    )
    kb_hashes = {}
    if isinstance(kb_payload, Mapping):
        kb_hashes = {str(k): str(v) for k, v in kb_payload.items() if v not in (None, "")}
    if not kb_hashes and isinstance(manifest_kb, Mapping):
        kb_hashes = {str(k): str(v) for k, v in manifest_kb.items() if v not in (None, "")}
    if not kb_hashes:
        for key in ("kb_hash", "kb_manifest_hash"):
            value = cache_identity.get(key)
            if value not in (None, ""):
                kb_hashes[key] = str(value)

    model_key = _first(
        manifest.get("model_key"),
        cache_identity.get("model_key"),
        config.get("model_key"),
        _nested(config, "model", "key"),
    )
    model_name = _first(
        manifest.get("model_hf_id"),
        manifest.get("model"),
        cache_identity.get("model_hf_id"),
        _nested(config, "model", "hf_id"),
        _nested(config, "model", "name"),
    )
    method_values = [
        manifest.get("method_key"),
        manifest.get("method"),
        cache_identity.get("method"),
        config.get("method_key"),
        _nested(config, "method", "key"),
        _nested(config, "method", "name"),
        _nested(config, "method", "type"),
    ]
    method_keys = [normalize_method(value) for value in method_values]
    method_keys = [value for value in method_keys if value is not None]
    return {
        "run_dir": str(run_dir.resolve()),
        "metadata_paths": [str(path.resolve()) for path in metadata_paths if path.exists()],
        "metadata_hashes": _metadata_hashes(metadata_paths),
        "manifest": manifest,
        "config": config,
        "cache_identity": cache_identity,
        "dataset_hashes": dataset_hashes,
        "kb_hashes": kb_hashes,
        "dataset_version": _first(
            manifest.get("dataset_version"),
            manifest.get("dataset"),
            cache_identity.get("dataset_version"),
            config.get("dataset"),
            _nested(config, "data", "name"),
            config.get("dataset_version"),
            _nested(config, "data", "dataset_version"),
        ),
        "model_key": model_key,
        "model_name": model_name,
        "method_values": method_values,
        "method_keys": sorted(set(method_keys)),
        "seed": _first(manifest.get("seed"), cache_identity.get("seed"), config.get("seed")),
        "run_id": _first(
            manifest.get("run_id"),
            manifest.get("experiment_id"),
            config.get("run_id"),
            config.get("experiment_id"),
        ),
        "config_hash": _first(
            manifest.get("config_hash"),
            cache_identity.get("config_hash"),
            (run_dir / "config_hash.txt").read_text(encoding="utf-8").strip()
            if (run_dir / "config_hash.txt").exists()
            else None,
        ),
        "test_file": _first(
            _nested(config, "data", "test_file"),
            config.get("test_file"),
        ),
        "test_hash": dataset_hashes.get("test"),
    }


def _model_matches(observed: Any, expected: str) -> bool:
    if observed is None:
        return True
    left = "".join(ch for ch in str(observed).lower() if ch.isalnum())
    right = "".join(ch for ch in str(expected).lower() if ch.isalnum())
    return left == right or left in right or right in left


def expected_dataset_hashes(project_root: Path, dataset: str) -> dict[str, str]:
    hashes_path = project_root / "data" / "frozen" / dataset / "hashes.json"
    if hashes_path.exists():
        payload = _json_load(hashes_path)
        result = _hash_map(payload)
        if result:
            return result
    result = {}
    for name in ("train", "val", "test"):
        path = project_root / "data" / "frozen" / dataset / f"{name}.jsonl"
        if path.exists():
            result[name] = sha256_file(path)
    return result


def validate_run_compatibility(
    run_infos: Mapping[str, Mapping[str, Any]],
    *,
    dataset: str,
    model: str,
    seed: int,
    project_root: Path,
) -> dict[str, Any]:
    """Reject mixed model/dataset/seed/method runs before study creation."""
    errors: list[str] = []
    warnings: list[str] = []
    observed_dataset_versions: dict[str, Any] = {}
    observed_model_keys: dict[str, Any] = {}
    observed_seeds: dict[str, Any] = {}
    observed_dataset_hashes: dict[str, dict[str, str]] = {}
    observed_kb_hashes: dict[str, dict[str, str]] = {}
    observed_test_files: dict[str, Any] = {}

    for method in METHODS:
        info = run_infos.get(method)
        if info is None:
            errors.append(f"Missing run information for method {method}.")
            continue
        run_dir = Path(str(info["run_dir"]))
        metadata = info.get("metadata") or {}
        prediction_rows = info.get("prediction_rows") or []
        if not metadata.get("metadata_paths"):
            warnings.append(f"{method}: no run metadata files found; path and prediction identity used only.")

        dataset_version = metadata.get("dataset_version")
        if dataset_version is not None:
            observed_dataset_versions[method] = dataset_version
            if str(dataset_version) != str(dataset):
                errors.append(
                    f"{method}: dataset version {dataset_version!r} does not match requested dataset {dataset!r}."
                )

        model_key = metadata.get("model_key")
        if model_key is not None:
            observed_model_keys[method] = model_key
            if not _model_matches(model_key, model):
                errors.append(f"{method}: model key {model_key!r} does not match requested model {model!r}.")
        model_name = metadata.get("model_name")
        if model_name is not None and not _model_matches(model_name, model):
            errors.append(f"{method}: model identity {model_name!r} does not match requested model {model!r}.")

        observed_seed = metadata.get("seed")
        if observed_seed is not None:
            try:
                observed_seed = int(observed_seed)
            except (TypeError, ValueError):
                errors.append(f"{method}: invalid seed metadata {observed_seed!r}.")
            else:
                observed_seeds[method] = observed_seed
                if observed_seed != int(seed):
                    errors.append(f"{method}: seed {observed_seed} does not match requested seed {seed}.")

        method_keys = set(metadata.get("method_keys") or [])
        if method_keys and method not in method_keys:
            errors.append(
                f"{method}: run metadata identifies method(s) {sorted(method_keys)}, expected {method}."
            )

        row_methods = set()
        for row in prediction_rows:
            if row.get("method_name") not in (None, ""):
                normalized = normalize_method(row.get("method_name"))
                if normalized is None:
                    errors.append(f"{method}: unknown prediction method_name {row.get('method_name')!r}.")
                else:
                    row_methods.add(normalized)
        if row_methods and row_methods != {method}:
            errors.append(f"{method}: predictions identify method(s) {sorted(row_methods)}, expected {method}.")

        hashes = {str(k): str(v) for k, v in (metadata.get("dataset_hashes") or {}).items()}
        if hashes:
            observed_dataset_hashes[method] = hashes
        test_file = metadata.get("test_file")
        if test_file:
            observed_test_files[method] = test_file
            resolved_test = _resolve(project_root, str(test_file))
            expected_test_hash = expected_dataset_hashes(project_root, dataset).get("test")
            if not resolved_test.exists():
                errors.append(f"{method}: configured test file does not exist: {test_file}")
            elif expected_test_hash and sha256_file(resolved_test) != expected_test_hash:
                errors.append(
                    f"{method}: configured test file {test_file!r} is not the frozen {dataset} test set."
                )
        kb_hashes = {str(k): str(v) for k, v in (metadata.get("kb_hashes") or {}).items()}
        if kb_hashes:
            observed_kb_hashes[method] = kb_hashes

    expected_hashes = expected_dataset_hashes(project_root, dataset)
    for method, hashes in observed_dataset_hashes.items():
        for key, expected in expected_hashes.items():
            actual = hashes.get(key)
            if actual and actual != expected:
                errors.append(
                    f"{method}: dataset hash for {key!r} differs from frozen {dataset} ({actual} != {expected})."
                )
    for key in set().union(*(set(values) for values in observed_dataset_hashes.values())):
        values = {method: values[key] for method, values in observed_dataset_hashes.items() if key in values}
        if len(set(values.values())) > 1:
            errors.append(f"Dataset hash mismatch for {key!r}: {values}.")

    # A/B/C/D do not all use a knowledge base.  RAG-bearing runs B and D must
    # agree, while null KB metadata on A/C is expected.
    rag_hashes = [observed_kb_hashes[m] for m in ("B", "D") if m in observed_kb_hashes]
    if len(rag_hashes) == 2 and rag_hashes[0] != rag_hashes[1]:
        errors.append(f"KB hash mismatch between B and D: {rag_hashes[0]} != {rag_hashes[1]}.")

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise HumanEvaluationError(f"Run compatibility validation failed:\n{detail}")
    return {
        "dataset_versions": observed_dataset_versions,
        "model_keys": observed_model_keys,
        "seeds": observed_seeds,
        "dataset_hashes": observed_dataset_hashes,
        "test_files": observed_test_files,
        "kb_hashes": observed_kb_hashes,
        "expected_dataset_hashes": expected_hashes,
        "warnings": warnings,
    }


def professor_prediction_paths(
    outputs_root: Path, dataset: str, model: str, seed: int
) -> dict[str, Path]:
    return {
        method: outputs_root / dataset / model / method / f"seed_{seed}" / "predictions.jsonl"
        for method in METHODS
    }


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HumanEvaluationError(f"Missing predictions for {path.parent.parent.name}: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HumanEvaluationError(f"Could not read predictions: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HumanEvaluationError(f"Malformed prediction JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict) or not row.get("item_id"):
            raise HumanEvaluationError(f"Prediction at {path}:{line_number} lacks item_id.")
        item_id = str(row["item_id"])
        if item_id in seen:
            raise HumanEvaluationError(f"Duplicate prediction item_id {item_id!r} in {path}.")
        seen.add(item_id)
        rows.append(row)
    return rows


def _read_item_ids(path: Path) -> list[str]:
    if not path.exists():
        raise HumanEvaluationError(f"Human-evaluation item list not found: {path}")
    suffix = path.suffix.lower()
    ids: list[str] = []
    if suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "item_id" not in reader.fieldnames:
                    raise HumanEvaluationError(f"Item CSV must contain item_id: {path}")
                ids = [str(row["item_id"]).strip() for row in reader if str(row.get("item_id", "")).strip()]
        except OSError as exc:
            raise HumanEvaluationError(f"Could not read item list {path}: {exc}") from exc
    elif suffix == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HumanEvaluationError(f"Malformed item-list JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict) and row.get("item_id"):
                ids.append(str(row["item_id"]))
    else:
        payload = _json_load(path)
        if isinstance(payload, Mapping):
            payload = payload.get("item_ids") or payload.get("items") or []
        if isinstance(payload, list):
            ids = [str(row.get("item_id") if isinstance(row, Mapping) else row) for row in payload]
            ids = [item_id for item_id in ids if item_id and item_id != "None"]
    if not ids:
        raise HumanEvaluationError(f"Item list contains no item IDs: {path}")
    duplicates = sorted(item_id for item_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise HumanEvaluationError(f"Item list contains duplicate IDs: {duplicates[:5]}")
    return ids


def default_item_list(project_root: Path, dataset: str) -> Path:
    configured = project_root / "src" / "config" / "data" / f"{dataset}.yaml"
    config = _read_yaml_if_present(configured)
    configured_path = config.get("human_eval_file")
    if configured_path:
        candidate = _resolve(project_root, str(configured_path))
        if candidate.exists():
            return candidate
    candidate = project_root / "data" / "frozen" / dataset / "human_eval_test_items_40.csv"
    if candidate.exists():
        return candidate
    raise HumanEvaluationError(
        f"No canonical human-evaluation item list for dataset {dataset!r}. "
        "Pass --item-list explicitly."
    )


def default_test_file(project_root: Path, dataset: str) -> Path:
    configured = project_root / "src" / "config" / "data" / f"{dataset}.yaml"
    config = _read_yaml_if_present(configured)
    configured_path = _nested(config, "test_file") or _nested(config, "data", "test_file")
    candidate = _resolve(project_root, str(configured_path)) if configured_path else (
        project_root / "data" / "frozen" / dataset / "test.jsonl"
    )
    if not candidate.exists():
        raise HumanEvaluationError(f"Authoritative test brief file not found: {candidate}")
    return candidate


def _read_test_briefs(path: Path) -> dict[str, dict[str, Any]]:
    briefs: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HumanEvaluationError(f"Malformed test JSON at {path}:{line_number}: {exc}") from exc
        item_id = str(row.get("item_id", ""))
        brief = row.get("brief")
        if not item_id or not isinstance(brief, Mapping):
            raise HumanEvaluationError(f"Test row {path}:{line_number} lacks item_id or brief.")
        if item_id in briefs:
            raise HumanEvaluationError(f"Duplicate test item_id {item_id!r} in {path}.")
        # Keep only what the rater-facing renderer uses.  In particular, do not
        # copy source evidence/reference recommendation fields from extra.
        briefs[item_id] = {
            "item_id": item_id,
            "users": brief.get("users", ""),
            "goals": list(brief.get("goals", []) or []),
            "kpis": list(brief.get("kpis", []) or []),
            "columns": [dict(column) for column in (brief.get("columns", []) or [])],
            "constraints": brief.get("constraints"),
        }
    return briefs


def _ensure_new_study_dir(path: Path) -> None:
    if (path / "study_manifest.json").exists():
        raise HumanEvaluationError(
            f"Study already exists at {path}. Refusing to overwrite it; choose a new --out-dir."
        )
    ratings = path / "ratings"
    if ratings.exists() and any(ratings.glob("*.jsonl")):
        raise HumanEvaluationError(
            f"Existing ratings found at {ratings}. Refusing to overwrite a rating study."
        )


def _validate_assignment(assignment: Mapping[str, Any], item_ids: Sequence[str], ratings_per_output: int) -> None:
    config = assignment.get("config") or {}
    expected_units = {(item_id, method) for item_id in item_ids for method in METHODS}
    if set(config.get("methods", [])) != set(METHODS):
        raise HumanEvaluationError(f"Assignment methods are not exactly A/B/C/D: {config.get('methods')}")
    if int(config.get("n_units", -1)) != len(expected_units):
        raise HumanEvaluationError("Assignment unit count does not match item × method count.")
    per_unit: dict[str, set[str]] = defaultdict(set)
    for rater, tasks in (assignment.get("raters") or {}).items():
        for task in tasks:
            item_id = str(task.get("item_id"))
            method = str(task.get("method"))
            unit_id = str(task.get("unit_id"))
            if (item_id, method) not in expected_units:
                raise HumanEvaluationError(f"Assignment contains unknown unit {unit_id!r}.")
            per_unit[unit_id].add(str(rater))
    expected_ids = {f"{item_id}__{method}" for item_id, method in expected_units}
    if set(per_unit) != expected_ids:
        raise HumanEvaluationError("Assignment does not contain every expected output unit.")
    bad = {unit: sorted(raters) for unit, raters in per_unit.items() if len(raters) != ratings_per_output}
    if bad:
        raise HumanEvaluationError(f"Assignment has wrong distinct-rater counts: {list(bad.items())[:3]}")


def build_study(
    *,
    project_root: Path,
    dataset: str,
    model: str,
    seed: int,
    outputs_root: str | Path = DEFAULT_OUTPUTS_ROOT,
    n_items: int = 40,
    n_raters: int = 6,
    rater_ids: Sequence[str] | None = None,
    ratings_per_output: int = 3,
    assignment_seed: int = 42,
    out_dir: str | Path | None = None,
    item_list: str | Path | None = None,
    test_file: str | Path | None = None,
) -> dict[str, Any]:
    if n_items <= 0 or n_raters <= 0 or ratings_per_output <= 0:
        raise HumanEvaluationError("n-items, n-raters and ratings-per-output must be positive.")
    raters = list(rater_ids) if rater_ids else [f"rater_{i:02d}" for i in range(1, n_raters + 1)]
    if len(set(raters)) != len(raters):
        raise HumanEvaluationError("Rater IDs must be unique.")
    if len(raters) != n_raters and rater_ids is None:
        raise HumanEvaluationError("Internal rater count mismatch.")
    if ratings_per_output > len(raters):
        raise HumanEvaluationError(
            f"ratings-per-output ({ratings_per_output}) cannot exceed number of raters ({len(raters)})."
        )

    outputs_root_path = _resolve(project_root, outputs_root)
    prediction_paths = professor_prediction_paths(outputs_root_path, dataset, model, seed)
    run_infos: dict[str, dict[str, Any]] = {}
    for method, prediction_path in prediction_paths.items():
        rows = _read_prediction_rows(prediction_path)
        run_dir = prediction_path.parent
        run_infos[method] = {
            "run_dir": str(run_dir.resolve()),
            "prediction_path": prediction_path,
            "prediction_hash": sha256_file(prediction_path),
            "prediction_rows": rows,
            "metadata": inspect_run_metadata(run_dir),
        }

    compatibility = validate_run_compatibility(
        run_infos,
        dataset=dataset,
        model=model,
        seed=seed,
        project_root=project_root,
    )

    item_list_path = _resolve(project_root, item_list) if item_list else default_item_list(project_root, dataset)
    all_item_ids = _read_item_ids(item_list_path)
    if n_items > len(all_item_ids):
        raise HumanEvaluationError(
            f"Requested {n_items} items but item list contains only {len(all_item_ids)} IDs: {item_list_path}"
        )
    item_ids = all_item_ids[:n_items]
    test_path = _resolve(project_root, test_file) if test_file else default_test_file(project_root, dataset)
    if not test_path.exists():
        raise HumanEvaluationError(f"Authoritative test brief file not found: {test_path}")
    briefs = _read_test_briefs(test_path)
    missing_briefs = [item_id for item_id in item_ids if item_id not in briefs]
    if missing_briefs:
        raise HumanEvaluationError(
            f"Canonical item list contains IDs missing from authoritative test.jsonl: {missing_briefs[:10]}"
        )

    method_to_predictions = {method: info["prediction_rows"] for method, info in run_infos.items()}
    missing_by_method = {
        method: [item_id for item_id in item_ids if item_id not in {row["item_id"] for row in rows}]
        for method, rows in method_to_predictions.items()
    }
    missing_by_method = {method: ids for method, ids in missing_by_method.items() if ids}
    if missing_by_method:
        raise HumanEvaluationError(
            "Missing predictions for canonical human-evaluation items: "
            + "; ".join(f"{method}: {ids[:10]}" for method, ids in missing_by_method.items())
        )

    items = build_eval_items(
        method_to_predictions,
        briefs,
        n_items=n_items,
        seed=assignment_seed,
        item_ids=item_ids,
    )
    if [item["item_id"] for item in items] != item_ids:
        raise HumanEvaluationError("Builder did not preserve canonical item-list order.")
    assignment = build_assignment(
        item_ids,
        list(METHODS),
        raters,
        ratings_per_output=ratings_per_output,
        seed=assignment_seed,
    )
    _validate_assignment(assignment, item_ids, ratings_per_output)

    final_design = n_items == 40 and len(raters) == 6 and ratings_per_output == 3
    study_type = "final" if final_design else "pilot"
    default_out = project_root / DEFAULT_RESULTS_ROOT / dataset / model / f"seed_{seed}"
    study_dir = _resolve(project_root, out_dir) if out_dir else default_out
    _ensure_new_study_dir(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "ratings").mkdir(parents=True, exist_ok=True)
    (study_dir / "analysis").mkdir(parents=True, exist_ok=True)

    # Only brief + recommendation are written to the rater-facing item file.
    for item in items:
        item["brief"] = dict(item["brief"])
        item["brief"].pop("extra", None)

    with (study_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    _json_dump(study_dir / "assignment.json", assignment)

    source_runs = {}
    source_prediction_paths = {}
    source_prediction_hashes = {}
    source_run_ids = {}
    source_config_hashes = {}
    source_metadata_hashes = {}
    for method, info in run_infos.items():
        metadata = info["metadata"]
        source_prediction_paths[method] = str(info["prediction_path"].resolve())
        source_prediction_hashes[method] = info["prediction_hash"]
        source_run_ids[method] = metadata.get("run_id")
        source_config_hashes[method] = metadata.get("config_hash")
        source_metadata_hashes[method] = metadata.get("metadata_hashes", {})
        source_runs[method] = {
            "run_dir": metadata.get("run_dir"),
            "run_id": metadata.get("run_id"),
            "config_hash": metadata.get("config_hash"),
            "model_key": metadata.get("model_key"),
            "model_name": metadata.get("model_name"),
            "dataset_version": metadata.get("dataset_version"),
            "seed": metadata.get("seed"),
            "method_keys": metadata.get("method_keys", []),
            "dataset_hashes": metadata.get("dataset_hashes", {}),
            "kb_hashes": metadata.get("kb_hashes", {}),
            "metadata_hashes": metadata.get("metadata_hashes", {}),
        }

    manifest = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_type": study_type,
        "dataset": dataset,
        "dataset_version": dataset,
        "dataset_hashes": compatibility["expected_dataset_hashes"],
        "source_dataset_hashes": compatibility["dataset_hashes"],
        "model": model,
        "model_key": model,
        "seed": int(seed),
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "source_prediction_paths": source_prediction_paths,
        "source_prediction_hashes": source_prediction_hashes,
        "source_run_ids": source_run_ids,
        "source_config_hashes": source_config_hashes,
        "source_metadata_hashes": source_metadata_hashes,
        "source_runs": source_runs,
        "item_ids": item_ids,
        "item_list_source": str(item_list_path.resolve()),
        "item_list_sha256": sha256_file(item_list_path),
        "test_file": str(test_path.resolve()),
        "test_file_sha256": sha256_file(test_path),
        "n_items": len(item_ids),
        "n_raters": len(raters),
        "rater_ids": raters,
        "ratings_per_output": int(ratings_per_output),
        "assignment_seed": int(assignment_seed),
        "rubric_version": RUBRIC_VERSION,
        "rubric_hash": rubric_hash(),
        "rubric_dimensions": list(RUBRIC_KEYS),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_expected_outputs": len(item_ids) * len(METHODS),
        "total_expected_ratings": len(item_ids) * len(METHODS) * ratings_per_output,
        "compatibility_warnings": compatibility["warnings"],
        "status": "built",
    }
    _json_dump(study_dir / "study_manifest.json", manifest)
    return {"study_dir": study_dir, "manifest": manifest, "items": items, "assignment": assignment}


def load_study(study_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = study_dir / "study_manifest.json"
    items_path = study_dir / "items.jsonl"
    assignment_path = study_dir / "assignment.json"
    missing = [str(path) for path in (manifest_path, items_path, assignment_path) if not path.exists()]
    if missing:
        raise HumanEvaluationError(f"Study directory missing required files: {missing}")
    manifest = _json_load(manifest_path)
    if manifest.get("schema_version") != STUDY_SCHEMA_VERSION:
        raise HumanEvaluationError(
            f"Unsupported human-evaluation study schema: {manifest.get('schema_version')!r}"
        )
    items = []
    for line_number, line in enumerate(items_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HumanEvaluationError(f"Malformed items.jsonl at line {line_number}: {exc}") from exc
        items.append(row)
    assignment = _json_load(assignment_path)
    expected_ids = list(manifest.get("item_ids", []))
    actual_ids = [str(item.get("item_id")) for item in items]
    if actual_ids != expected_ids:
        raise HumanEvaluationError("items.jsonl item IDs differ from study_manifest.json.")
    _validate_assignment(assignment, expected_ids, int(manifest.get("ratings_per_output", 0)))
    if manifest.get("rubric_hash") != rubric_hash():
        raise HumanEvaluationError("Rubric hash changed after study creation; do not analyze mixed rubrics.")
    return manifest, items, assignment


def verify_source_predictions_unchanged(
    manifest: Mapping[str, Any], *, project_root: Path
) -> None:
    run_infos: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        raw_path = (manifest.get("source_prediction_paths") or {}).get(method)
        if not raw_path:
            raise HumanEvaluationError(f"Study manifest lacks source prediction path for method {method}.")
        path = Path(str(raw_path))
        if not path.exists():
            raise HumanEvaluationError(f"Source predictions disappeared for {method}: {path}")
        rows = _read_prediction_rows(path)
        run_infos[method] = {
            "run_dir": str(path.parent.resolve()),
            "prediction_rows": rows,
            "metadata": inspect_run_metadata(path.parent),
        }
        expected_hash = (manifest.get("source_prediction_hashes") or {}).get(method)
        actual_hash = sha256_file(path)
        if expected_hash and actual_hash != expected_hash:
            raise HumanEvaluationError(
                f"Source predictions changed for {method}: study has {expected_hash}, current file has {actual_hash}. Rebuild study."
            )
        expected_metadata = ((manifest.get("source_metadata_hashes") or {}).get(method) or {})
        current_metadata = run_infos[method]["metadata"].get("metadata_hashes", {})
        for metadata_path, expected in expected_metadata.items():
            actual = current_metadata.get(metadata_path)
            if actual != expected:
                raise HumanEvaluationError(
                    f"Source metadata changed for {method}: {metadata_path}. Rebuild study before using ratings."
                )
    validate_run_compatibility(
        run_infos,
        dataset=str(manifest.get("dataset")),
        model=str(manifest.get("model_key") or manifest.get("model")),
        seed=int(manifest.get("seed")),
        project_root=project_root,
    )


def _expected_unit_map(assignment: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    expected: dict[str, dict[str, Any]] = {}
    raters: dict[str, set[str]] = defaultdict(set)
    for rater, tasks in (assignment.get("raters") or {}).items():
        for task in tasks:
            unit_id = str(task.get("unit_id"))
            if unit_id in expected and expected[unit_id] != task:
                # Multiple task entries are expected across raters; compare only
                # identity fields, not object equality.
                old = expected[unit_id]
                if (old.get("item_id"), old.get("method")) != (task.get("item_id"), task.get("method")):
                    raise HumanEvaluationError(f"Assignment reuses unit_id with different identity: {unit_id}")
            expected[unit_id] = dict(task)
            raters[unit_id].add(str(rater))
    return expected, raters


def validate_ratings(
    manifest: Mapping[str, Any],
    assignment: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate every rating against the immutable assignment contract."""
    methods = list(manifest.get("methods") or METHODS)
    item_ids = list(manifest.get("item_ids") or [])
    expected_units, expected_raters = _expected_unit_map(assignment)
    expected_raters_by_id = set(manifest.get("rater_ids") or (assignment.get("raters") or {}).keys())
    expected_total = int(manifest.get("total_expected_ratings", 0))
    expected_per_unit = int(manifest.get("ratings_per_output", 0))
    valid_rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    invalid_scores: list[dict[str, Any]] = []
    unknown_items: list[dict[str, Any]] = []
    unknown_methods: list[dict[str, Any]] = []
    unknown_units: list[dict[str, Any]] = []
    field_errors: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for index, raw in enumerate(rows):
        row = dict(raw)
        rater_id = str(row.get("rater_id", ""))
        unit_id = str(row.get("unit_id", ""))
        key = (rater_id, unit_id)
        if key in seen_keys:
            duplicates.append({"row": index, "rater_id": rater_id, "unit_id": unit_id})
            continue
        seen_keys.add(key)
        task = expected_units.get(unit_id)
        row_valid = True
        if rater_id not in expected_raters_by_id:
            field_errors.append({"row": index, "error": f"unknown rater_id: {rater_id}"})
            row_valid = False
        if task is None:
            unknown_units.append({"row": index, "unit_id": unit_id})
            row_valid = False
        else:
            expected_item = str(task.get("item_id"))
            expected_method = str(task.get("method"))
            row_item = str(row.get("item_id", ""))
            if row_item not in set(item_ids) or row_item != expected_item:
                unknown_items.append({"row": index, "item_id": row_item, "expected_item_id": expected_item})
                row_valid = False
            raw_method = row.get("method")
            row_method = normalize_method(raw_method)
            if raw_method not in (None, "") and row_method is None:
                unknown_methods.append({"row": index, "method": raw_method, "expected_method": expected_method})
                row_valid = False
            elif row_method is not None and row_method != expected_method:
                unknown_methods.append({"row": index, "method": raw_method, "expected_method": expected_method})
                row_valid = False
            if expected_method not in methods:
                unknown_methods.append({"row": index, "method": expected_method})
                row_valid = False

        scores = row.get("scores")
        score_error: dict[str, Any] = {}
        if not isinstance(scores, Mapping):
            score_error["scores"] = "scores must be an object"
        else:
            missing = sorted(set(RUBRIC_KEYS) - set(scores))
            extra = sorted(set(scores) - set(RUBRIC_KEYS))
            if missing:
                score_error["missing_dimensions"] = missing
            if extra:
                score_error["unknown_dimensions"] = extra
            invalid = {}
            for dim in RUBRIC_KEYS:
                value = scores.get(dim)
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                    invalid[dim] = value
            if invalid:
                score_error["invalid_scores"] = invalid
        if score_error:
            invalid_scores.append({"row": index, "rater_id": rater_id, "unit_id": unit_id, **score_error})
            row_valid = False
        if row_valid:
            normalized = dict(row)
            normalized["rater_id"] = rater_id
            normalized["unit_id"] = unit_id
            normalized["item_id"] = str(task["item_id"])
            normalized["method"] = str(task["method"])
            valid_rows.append(normalized)

    valid_by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    valid_by_rater: dict[str, int] = Counter()
    for row in valid_rows:
        valid_by_unit[row["unit_id"]].append(row)
        valid_by_rater[row["rater_id"]] += 1

    missing_units = sorted(unit_id for unit_id in expected_units if not valid_by_unit.get(unit_id))
    wrong_unit_counts = {
        unit_id: len({row["rater_id"] for row in unit_rows})
        for unit_id, unit_rows in valid_by_unit.items()
        if len({row["rater_id"] for row in unit_rows}) != expected_per_unit
    }
    missing_ratings = {
        unit_id: max(0, expected_per_unit - len({row["rater_id"] for row in valid_by_unit.get(unit_id, [])}))
        for unit_id in expected_units
        if len({row["rater_id"] for row in valid_by_unit.get(unit_id, [])}) != expected_per_unit
    }
    expected_per_rater = {
        rater: sum(1 for tasks in (assignment.get("raters") or {}).get(rater, []) for _ in [tasks])
        for rater in expected_raters_by_id
    }
    completion = {
        "expected_ratings": expected_total,
        "received_ratings": len(valid_rows),
        "received_rows": len(rows),
        "completion_percentage": round((len(valid_rows) / expected_total * 100) if expected_total else 0.0, 3),
        "ratings_per_rater": {
            rater: {"expected": expected_per_rater.get(rater, 0), "received": valid_by_rater.get(rater, 0)}
            for rater in sorted(expected_raters_by_id)
        },
        "ratings_per_unit": {
            unit_id: len({row["rater_id"] for row in valid_by_unit.get(unit_id, [])})
            for unit_id in sorted(expected_units)
        },
        "distinct_raters_per_output": {
            unit_id: len({row["rater_id"] for row in valid_by_unit.get(unit_id, [])})
            for unit_id in sorted(expected_units)
        },
        "missing_units": missing_units,
        "missing_ratings": missing_ratings,
        "duplicate_ratings": duplicates,
        "invalid_scores": invalid_scores,
        "unknown_items": unknown_items,
        "unknown_methods": unknown_methods,
        "unknown_units": unknown_units,
        "field_errors": field_errors,
        "complete": not (
            len(valid_rows) != expected_total
            or duplicates
            or invalid_scores
            or unknown_items
            or unknown_methods
            or unknown_units
            or field_errors
            or missing_ratings
            or wrong_unit_counts
        ),
    }
    completion["wrong_unit_counts"] = wrong_unit_counts
    return {"completion": completion, "valid_rows": valid_rows}


def _mean(values: Sequence[float]) -> float | None:
    return round(float(statistics.mean(values)), 4) if values else None


def _sd(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.stdev(values)), 4) if len(values) > 1 else 0.0


def _cell_scores(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["item_id"]), str(row["method"]))].append(dict(row))
    return cells


def _per_item_rows(
    cells: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for (item_id, method), ratings in sorted(cells.items(), key=lambda pair: (pair[0][0], pair[0][1])):
        values = {dim: [float(row["scores"][dim]) for row in ratings] for dim in RUBRIC_KEYS}
        means = {dim: _mean(vals) for dim, vals in values.items()}
        composite_values = [float(statistics.mean(row["scores"][dim] for dim in RUBRIC_KEYS)) for row in ratings]
        rows.append({
            "item_id": item_id,
            "method": method,
            **means,
            "composite_score": _mean(composite_values),
            "n_ratings": len(ratings),
        })
    return rows


def _paired_scores(
    per_item: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    outcome: str,
) -> tuple[list[str], dict[str, list[float]]]:
    by_key = {(str(row["item_id"]), str(row["method"])): row for row in per_item}
    items = sorted({str(row["item_id"]) for row in per_item})
    common = [item for item in items if all((item, method) in by_key for method in methods)]
    return common, {method: [float(by_key[(item, method)][outcome]) for item in common] for method in methods}


def _inferential_outcome(
    per_item: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    outcome: str,
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    common_items, scores = _paired_scores(per_item, methods, outcome)
    result: dict[str, Any] = {
        "outcome": outcome,
        "n_common_items": len(common_items),
        "paired_items": common_items,
        "methods": list(methods),
    }
    if len(common_items) < 2:
        result["friedman"] = {
            "test": "friedman",
            "applicable": False,
            "reason": "At least two paired items are required.",
            "n": len(common_items),
        }
        result["pairwise_wilcoxon_holm"] = []
        return result
    result["friedman"] = friedman_test(scores)
    pairwise = pairwise_wilcoxon(scores)
    for index, row in enumerate(pairwise):
        a = scores[row["method_a"]]
        b = scores[row["method_b"]]
        row["rank_biserial"] = paired_rank_biserial(a, b)
        row["cohen_dz"] = cohen_dz(a, b)
        row["bootstrap_ci"] = paired_bootstrap_diff(
            a,
            b,
            n_boot=bootstrap_resamples,
            seed=seed + index,
        )
    result["pairwise_wilcoxon_holm"] = pairwise
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _system_means(rows: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    result: dict[str, Any] = {"methods": {}}
    csv_rows = []
    outcomes = list(RUBRIC_KEYS) + ["composite_mean"]
    for method in methods:
        ratings = by_method.get(method, [])
        method_payload: dict[str, Any] = {}
        for dim in RUBRIC_KEYS:
            values = [float(row["scores"][dim]) for row in ratings]
            method_payload[dim] = {"mean": _mean(values), "sd": _sd(values), "n": len(values)}
            csv_rows.append({"method": method, "dimension": dim, "mean": _mean(values), "sd": _sd(values), "n": len(values)})
        composite = [float(statistics.mean(row["scores"][dim] for dim in RUBRIC_KEYS)) for row in ratings]
        method_payload["composite_mean"] = {"mean": _mean(composite), "sd": _sd(composite), "n": len(composite)}
        csv_rows.append({"method": method, "dimension": "composite_mean", "mean": _mean(composite), "sd": _sd(composite), "n": len(composite)})
        result["methods"][method] = method_payload
    result["dimensions"] = outcomes
    return result, csv_rows


def _acceptability(
    per_item: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    *,
    expected_ratings_per_output: int,
) -> dict[str, Any]:
    by_key = {(str(row["item_id"]), str(row["method"])): row for row in per_item}
    # Per-item rows contain means, so a majority decision must be computed from
    # the original ratings.  Caller attaches the decision vectors below.
    return {"methods": list(methods), "expected_ratings_per_output": expected_ratings_per_output}


def compute_human_chart_acceptability(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    item_ids: Sequence[str],
    *,
    expected_ratings_per_output: int,
) -> dict[str, Any]:
    votes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        votes[(str(row["item_id"]), str(row["method"]))].append(
            int(row["scores"]["chart_appropriateness"] >= 4)
        )
    decisions: dict[tuple[str, str], int] = {}
    for key, values in votes.items():
        if len(values) < expected_ratings_per_output:
            continue
        positives = sum(values)
        # Strict majority keeps a two-rater tie undecided in pilot studies.
        if positives * 2 > len(values):
            decisions[key] = 1
        elif (len(values) - positives) * 2 > len(values):
            decisions[key] = 0
    common_items = [
        item_id for item_id in item_ids
        if all((item_id, method) in decisions for method in methods)
    ]
    outcomes = {
        method: [decisions[(item_id, method)] for item_id in common_items]
        for method in methods
    }
    rates = {}
    for method in methods:
        vector = outcomes.get(method, [])
        rates[method] = {
            "acceptable": int(sum(vector)),
            "n_items": len(vector),
            "rate": round(sum(vector) / len(vector), 4) if vector else None,
        }
    if len(common_items) >= 1 and len(methods) >= 3:
        omnibus = cochran_q(outcomes)
        pairwise = pairwise_mcnemar(outcomes)
    else:
        omnibus = {
            "test": "cochran_q",
            "applicable": False,
            "reason": "No complete paired item set for human chart acceptability.",
            "k": len(methods),
            "n": len(common_items),
        }
        pairwise = []
    return {
        "label": "human chart acceptability",
        "definition": "chart_appropriateness >= 4 per rater, aggregated by strict majority per item and method",
        "threshold": 4,
        "methods": list(methods),
        "n_items": len(common_items),
        "item_ids": common_items,
        "acceptability_rate": rates,
        "cochran_q": omnibus,
        "pairwise_mcnemar_holm": pairwise,
    }


def _summary_markdown(
    manifest: Mapping[str, Any],
    completion: Mapping[str, Any],
    alphas: Mapping[str, Any],
    system_means: Mapping[str, Any],
    human_stats: Mapping[str, Any],
    acceptability: Mapping[str, Any],
) -> str:
    lines = [
        "# Human evaluation summary",
        "",
        "## Study identity",
        "",
        f"- Dataset: `{manifest.get('dataset')}`",
        f"- Model: `{manifest.get('model')}`",
        f"- Seed: `{manifest.get('seed')}`",
        f"- Study type: `{manifest.get('study_type')}`",
        f"- Items: {manifest.get('n_items')}",
        f"- Raters: {manifest.get('n_raters')}",
        f"- Ratings per output: {manifest.get('ratings_per_output')}",
        f"- Total ratings: {completion.get('received_ratings')} / {completion.get('expected_ratings')}",
        "",
        "## Krippendorff's ordinal alpha",
        "",
        "| Dimension | α |",
        "|---|---:|",
    ]
    for dim in RUBRIC_KEYS:
        value = alphas.get(dim)
        lines.append(f"| `{dim}` | {value if value is not None else 'n/a'} |")
    lines.extend(["", "## System means", "", "| Method | Dimension | Mean ± SD | n |", "|---|---|---:|---:|"])
    for method in manifest.get("methods", METHODS):
        for dim in list(RUBRIC_KEYS) + ["composite_mean"]:
            value = ((system_means.get("methods") or {}).get(method) or {}).get(dim) or {}
            mean = value.get("mean")
            sd = value.get("sd")
            rendered = "n/a" if mean is None else f"{mean:.4f} ± {sd:.4f}"
            lines.append(f"| `{method}` | `{dim}` | {rendered} | {value.get('n', 0)} |")
    lines.extend(["", "## Inferential statistics", ""])
    for outcome, result in (human_stats.get("outcomes") or {}).items():
        friedman = result.get("friedman") or {}
        lines.append(
            f"- `{outcome}`: Friedman statistic={friedman.get('statistic', 'n/a')}, "
            f"p={friedman.get('p_value', 'n/a')}, n={result.get('n_common_items', 0)}."
        )
        for pair in result.get("pairwise_wilcoxon_holm", []):
            ci = pair.get("bootstrap_ci") or {}
            lines.append(
                f"  - `{pair.get('method_a')} vs {pair.get('method_b')}`: "
                f"p={pair.get('p_value')}, Holm p={pair.get('p_holm')}, "
                f"rank-biserial={pair.get('rank_biserial')}, Cohen d_z={pair.get('cohen_dz')}, "
                f"paired bootstrap CI=[{ci.get('ci_low')}, {ci.get('ci_high')}]."
            )
    lines.extend(["", "## Human chart acceptability", ""])
    lines.append("This is human chart acceptability, not objective chart correctness.")
    for method, value in (acceptability.get("acceptability_rate") or {}).items():
        lines.append(
            f"- `{method}`: {value.get('rate', 'n/a')} "
            f"({value.get('acceptable', 0)}/{value.get('n_items', 0)})."
        )
    q = acceptability.get("cochran_q") or {}
    lines.append(f"- Cochran-Q: statistic={q.get('statistic', 'n/a')}, p={q.get('p_value', 'n/a')}.")
    for pair in acceptability.get("pairwise_mcnemar_holm", []):
        lines.append(
            f"  - `{pair.get('method_a')} vs {pair.get('method_b')}`: "
            f"p={pair.get('p_value')}, Holm p={pair.get('p_holm')}"
        )
    lines.append("")
    return "\n".join(lines)


def run_analysis(
    *,
    study_dir: Path,
    project_root: Path,
    allow_incomplete: bool = False,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    if bootstrap_resamples <= 0:
        raise HumanEvaluationError("bootstrap-resamples must be positive.")
    manifest, items, assignment = load_study(study_dir)
    verify_source_predictions_unchanged(manifest, project_root=project_root)
    rows = load_all_ratings(study_dir / "ratings")
    validation = validate_ratings(manifest, assignment, rows)
    analysis_dir = study_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(analysis_dir / "rating_completion.json", validation["completion"])
    if not validation["completion"]["complete"] and not allow_incomplete:
        raise IncompleteStudyError(
            "Rating study incomplete or invalid. See analysis/rating_completion.json; "
            "use --allow-incomplete only for pilot/debug analysis."
        )

    valid_rows = validation["valid_rows"]
    methods = list(manifest.get("methods") or METHODS)
    cells = _cell_scores(valid_rows)
    per_item = _per_item_rows(cells, methods)
    per_item_fields = ["item_id", "method", *RUBRIC_KEYS, "composite_score", "n_ratings"]
    _write_csv(analysis_dir / "per_item_scores.csv", per_item_fields, per_item)

    alphas: dict[str, float | None] = {}
    for dim in RUBRIC_KEYS:
        units: dict[str, list[int]] = defaultdict(list)
        for row in valid_rows:
            units[str(row["unit_id"])].append(int(row["scores"][dim]))
        alphas[dim] = krippendorff_alpha(list(units.values()), level="ordinal")
    alpha_payload = {
        "level": "ordinal",
        "dimensions": {dim: {"alpha": value} for dim, value in alphas.items()},
        "krippendorff_alpha_ordinal": alphas,
    }
    _json_dump(analysis_dir / "irr_alphas.json", alpha_payload)
    _write_csv(
        analysis_dir / "irr_alphas.csv",
        ["dimension", "alpha"],
        [{"dimension": dim, "alpha": value} for dim, value in alphas.items()],
    )

    system_means, system_mean_rows = _system_means(valid_rows, methods)
    _json_dump(analysis_dir / "system_means.json", system_means)
    _write_csv(analysis_dir / "system_means.csv", ["method", "dimension", "mean", "sd", "n"], system_mean_rows)

    outcome_names = list(RUBRIC_KEYS) + ["composite_score"]
    outcomes = {
        outcome: _inferential_outcome(
            per_item,
            methods,
            outcome,
            bootstrap_resamples=bootstrap_resamples,
            seed=42 + index * 1000,
        )
        for index, outcome in enumerate(outcome_names)
    }
    human_stats = {
        "methods": methods,
        "outcomes": outcomes,
        # Compatibility aliases for callers that used the old aggregate-only file.
        "n_common_items": outcomes["composite_score"]["n_common_items"],
        "friedman": outcomes["composite_score"]["friedman"],
        "pairwise_wilcoxon_holm": outcomes["composite_score"]["pairwise_wilcoxon_holm"],
    }
    _json_dump(analysis_dir / "human_stats.json", human_stats)

    acceptability = compute_human_chart_acceptability(
        valid_rows,
        methods,
        list(manifest.get("item_ids") or []),
        expected_ratings_per_output=int(manifest.get("ratings_per_output", 0)),
    )
    _json_dump(analysis_dir / "human_chart_acceptability.json", acceptability)
    summary = _summary_markdown(manifest, validation["completion"], alphas, system_means, human_stats, acceptability)
    (analysis_dir / "human_eval_summary.md").write_text(summary, encoding="utf-8")
    return {
        "manifest": manifest,
        "completion": validation["completion"],
        "irr_alphas": alpha_payload,
        "system_means": system_means,
        "human_stats": human_stats,
        "human_chart_acceptability": acceptability,
        "analysis_dir": analysis_dir,
    }
