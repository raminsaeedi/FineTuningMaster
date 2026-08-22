"""Run the smoke profile or the complete professor matrix.

Examples::

    python experiments/scripts/run_final_matrix.py --profile smoke \
        --model qwen2_5_0_5b --all-methods --seed 42 \
        --with-dependencies --resume
    python experiments/scripts/run_final_matrix.py --profile final \
        --all-models --all-methods --seeds 42 43 44 \
        --with-dependencies --resume

Final artifacts use ``<dataset>/<model>/<method>/seed_<n>`` below the output
root, e.g. ``experiments/outputs/final/dashboard_v4/qwen3_8b/C/seed_42``. The
dataset is selected with ``--dataset`` (default: ``dashboard_v4``) and reaches
training, inference, cache identity, manifests, adapter compatibility,
aggregation and packaging. The runner never mixes datasets, models or seeds, and
method D refuses to start without its same-dataset, same-model, same-seed method
C adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402
from src.utils.config import load_cfg  # noqa: E402
from src.utils.config_hash import hash_config  # noqa: E402
from src.utils.artifacts import cache_identity  # noqa: E402

DEFAULT_MATRIX = _PROJECT_ROOT / "src" / "config" / "matrix" / "final.yaml"
FINAL_MODELS = ("qwen3_1_7b", "qwen3_8b", "qwen3_14b", "llama3_1_8b")
ADDITIONAL_MODELS = ("olmo2_1_49b", "qwen3_8_27b")
SUPPORTED_FINAL_MODELS = FINAL_MODELS + ADDITIONAL_MODELS
SMOKE_MODEL = "qwen2_5_0_5b"
DEFAULT_DATASET = "dashboard_v4"
METHOD_KEYS = ("A", "B", "C", "D")
METHODS = {
    "A": {"experiment": "E01_qwen0_5b_prompt", "description": "prompt-only baseline"},
    "B": {"experiment": "E02_qwen0_5b_rag", "description": "retrieval-augmented"},
    "C": {"experiment": "E03_qwen0_5b_ft", "description": "QLoRA fine-tuned"},
    "D": {"experiment": "E04_qwen0_5b_ft_rag", "description": "QLoRA fine-tuned + retrieval"},
}


# ---------------------------------------------------------------------------
# Legacy matrix helpers. Kept stable for callers and unit tests.
def load_matrix(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Matrix config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f) or {}
    if not matrix.get("runs"):
        raise SystemExit(f"{path} defines no runs.")
    return matrix


def _run_by_key(matrix: dict, key: str) -> Optional[dict]:
    return next((r for r in matrix["runs"] if str(r.get("key")) == str(key)), None)


def ordered_runs(matrix: dict) -> list[dict]:
    """Return adapter producers before their consumers."""
    runs = list(matrix["runs"])
    producers = [r for r in runs if not r.get("adapter_from")]
    consumers = [r for r in runs if r.get("adapter_from")]
    for consumer in consumers:
        if _run_by_key(matrix, consumer["adapter_from"]) is None:
            raise SystemExit(
                f"Run {consumer.get('key')} declares adapter_from="
                f"{consumer['adapter_from']}, which is not defined in the matrix."
            )
    return producers + consumers


def experiment_id(experiment: str, seed: int) -> str:
    return f"{experiment}_{seed}"


def run_dir(output_root: Path, experiment: str, seed: int) -> Path:
    return output_root / experiment_id(experiment, seed)


def adapter_dir(output_root: Path, experiment: str, seed: int) -> Path:
    return run_dir(output_root, experiment, seed) / "adapter"


def adapter_is_trained(path: Path) -> bool:
    """A usable adapter has a config plus weights, not just a directory."""
    if not (path / "adapter_config.json").exists():
        return False
    return any((path / n).exists() for n in ("adapter_model.safetensors", "adapter_model.bin"))


def run_is_complete(path: Path) -> bool:
    metrics_path = path / "metrics_auto.json"
    if not metrics_path.exists():
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    def complete(values: object) -> bool:
        if not isinstance(values, dict):
            return False
        requested = values.get("n_requested")
        predicted = values.get("n_predictions")
        missing = values.get("n_missing")
        return (
            isinstance(requested, int)
            and isinstance(predicted, int)
            and requested > 0
            and predicted == requested
            and missing == 0
        )

    if not complete(payload.get("coverage")):
        return False
    variants = payload.get("variant_coverage")
    return isinstance(variants, dict) and all(
        complete(values) for values in variants.values()
    )


# ---------------------------------------------------------------------------
# Profile paths and commands
def _resolved(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _hydra_path(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    # Hydra treats the colon in a Windows drive letter as override syntax.
    # Quote paths that contain Hydra-significant characters so external
    # Windows paths and directories with spaces remain valid overrides.
    if any(char in value for char in " \t,:[]{}=#"):
        return json.dumps(value)
    return value


def profile_run_dir(
    output_root: Path, model: str, method: str, seed: int,
    dataset: str = DEFAULT_DATASET,
) -> Path:
    """``<root>/<dataset>/<model>/<method>/seed_<n>`` — the one layout used everywhere."""
    return output_root / dataset / model / method / f"seed_{seed}"


def profile_adapter_dir(
    output_root: Path, model: str, method: str, seed: int = 42,
    dataset: str = DEFAULT_DATASET,
) -> Path:
    return profile_run_dir(output_root, model, method, seed, dataset) / "adapter"


def _manifest_matches(
    path: Path, *, model: str, method: str, seed: int, profile: str,
    dataset: str | None = None,
) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if dataset and str(manifest.get("dataset_version")) != dataset:
        return False
    return (
        manifest.get("profile") == profile
        and str(manifest.get("model_key")) == model
        and str(manifest.get("method_key")) == method
        and int(manifest.get("seed", -1)) == int(seed)
    )


def _compatible_adapter(
    path: Path,
    *,
    model: str,
    seed: int,
    profile: str,
    dataset: str | None = None,
    training_config_hash: str | None = None,
    model_config_hash: str | None = None,
) -> bool:
    if not adapter_is_trained(path):
        return False
    run_path = path.parent
    if not _manifest_matches(
        run_path, model=model, method="C", seed=seed, profile=profile, dataset=dataset
    ):
        return False
    metadata_path = path / "training_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if training_config_hash and metadata.get("training_config_hash") != training_config_hash:
        return False
    if model_config_hash and metadata.get("model_config_hash") != model_config_hash:
        return False
    return True


def _compatible_complete(
    path: Path,
    *,
    model: str,
    method: str,
    seed: int,
    profile: str,
    dataset: str | None = None,
    cache_identity_hash: str | None = None,
) -> bool:
    if not run_is_complete(path) or not _manifest_matches(
        path, model=model, method=method, seed=seed, profile=profile, dataset=dataset
    ):
        return False
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            return False
        if cache_identity_hash and manifest.get("cache_identity_hash") != cache_identity_hash:
            return False
        return True
    except (OSError, ValueError):
        return False


def _execute(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print("  " + " ".join(cmd[1:]))
    started = time.time()
    rc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT)).returncode
    print(f"  -> {'ok' if rc == 0 else f'FAILED (exit {rc})'} in "
          f"{(time.time() - started) / 60:.1f} min")
    return rc


def train_cmd(experiment: str, seed: int, overrides: list[str]) -> list[str]:
    """Legacy command builder retained for external callers."""
    script = str(_PROJECT_ROOT / "experiments" / "scripts" / "train.py")
    return [sys.executable, script, "--experiment", experiment, "--override",
            f"seed={seed}", *overrides]


def experiment_cmd(experiment: str, seed: int, overrides: list[str]) -> list[str]:
    """Legacy command builder retained for external callers."""
    script = str(_PROJECT_ROOT / "experiments" / "scripts" / "run_experiment.py")
    return [sys.executable, script, "--experiment", experiment, "--override",
            f"seed={seed}", *overrides]


def _profile_cmd(script_name: str, experiment: str, overrides: list[str], *, resume: bool = False) -> list[str]:
    script = str(_PROJECT_ROOT / "experiments" / "scripts" / script_name)
    cmd = [sys.executable, script, "--experiment", experiment]
    if resume:
        cmd.append("--resume")
    cmd.extend(["--override", *overrides])
    return cmd


# ---------------------------------------------------------------------------
# Selection and smoke data
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run smoke or final multi-model experiments")
    p.add_argument("--profile", choices=("smoke", "tiny", "final"), default="final")
    p.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="Matrix definition")
    p.add_argument("--model", default=None, help="One model profile")
    p.add_argument("--all-models", action="store_true", help="Run all four final model profiles")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="Frozen dataset config name, e.g. dashboard_v4 or dashboard_v3")
    p.add_argument("--method", default=None, help="One method key: A, B, C or D")
    p.add_argument("--methods", nargs="+", default=None, help="Selected method keys")
    p.add_argument("--all-methods", action="store_true", help="Run methods A, B, C and D")
    p.add_argument("--seed", type=int, default=None, help="One seed")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="Seed list")
    p.add_argument("--with-dependencies", action="store_true",
                   help="Train method C automatically when method D needs it")
    p.add_argument("--resume", action="store_true", help="Resume compatible interrupted training")
    # Legacy spellings retained for the old matrix file and handbook.
    p.add_argument("--only", nargs="+", default=None, metavar="KEY")
    p.add_argument("--output-root", default=None)
    p.add_argument("--output-model-path", default=None,
                   help="Separate root for adapters/checkpoints (defaults to output root)")
    p.add_argument("--results-dir", default=None,
                   help="Directory for aggregated results")
    p.add_argument("--input-model-weights", default=None,
                   help="Existing adapter directory for C/D inference")
    p.add_argument("--kb-chunks-path", default=None,
                   help="Knowledge-base chunks file for methods B/D (default: $FTM_KB_CHUNKS_PATH)")
    p.add_argument("--smoke-source", default=None,
                   help="Validation JSONL used to build the smoke slice")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--n-eval-items", type=int, default=2)
    p.add_argument("--n-train-items", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=1)
    args = p.parse_args(argv)
    if not args.kb_chunks_path:
        args.kb_chunks_path = os.environ.get("FTM_KB_CHUNKS_PATH") or None
    return args


def _selected_models(args: argparse.Namespace, matrix: dict) -> list[str]:
    if args.all_models and args.model:
        raise SystemExit("Use --all-models or --model, not both.")
    if args.profile in {"smoke", "tiny"}:
        model = args.model or matrix.get("smoke_model", SMOKE_MODEL)
        if args.all_models:
            raise SystemExit(f"The {args.profile} profile supports one explicit model only.")
        if not (_PROJECT_ROOT / "src" / "config" / "model" / f"{model}.yaml").is_file():
            raise SystemExit(f"Unknown model profile: {model}")
        return [model]
    if args.all_models:
        return list(matrix.get("final_models") or FINAL_MODELS)
    model = args.model or matrix.get("model") or FINAL_MODELS[0]
    if model not in SUPPORTED_FINAL_MODELS:
        raise SystemExit(
            f"Final profile model '{model}' is not one of: "
            f"{', '.join(SUPPORTED_FINAL_MODELS)}"
        )
    return [model]


def _selected_methods(args: argparse.Namespace) -> list[str]:
    if sum(bool(value) for value in (args.method, args.methods, args.all_methods, args.only)) > 1:
        raise SystemExit("Choose one of --method, --methods, --all-methods or --only.")
    if args.all_methods or not any((args.method, args.methods, args.only)):
        values = list(METHOD_KEYS)
    elif args.method:
        values = [args.method]
    elif args.methods:
        values = list(args.methods)
    else:
        values = list(args.only)
    values = [str(value).upper() for value in values]
    unknown = [value for value in values if value not in METHOD_KEYS]
    if unknown:
        raise SystemExit(f"Unknown method key(s): {', '.join(unknown)}")
    return [key for key in METHOD_KEYS if key in values]


def _selected_seeds(args: argparse.Namespace, matrix: dict) -> list[int]:
    if args.seed is not None and args.seeds is not None:
        raise SystemExit("Use --seed or --seeds, not both.")
    if args.seed is not None:
        return [int(args.seed)]
    if args.seeds is not None:
        return [int(seed) for seed in args.seeds]
    if args.profile in {"smoke", "tiny"}:
        return [42]
    return [int(seed) for seed in matrix.get("seeds", [42, 43, 44])]


def _smoke_slice(
    n_items: int, out_path: Path, source: Path | None = None,
    dataset: str = DEFAULT_DATASET,
) -> int:
    source = source or (_PROJECT_ROOT / "data" / "frozen" / dataset / "val.jsonl")
    if not source.exists():
        raise SystemExit(f"Validation split not found: {source}")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda row: str(row.get("item_id", "")))
    selected = records[:n_items]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    return len(selected)


def _print_runtime_profile(models: list[str]) -> None:
    """Report the fixed loading profile before any subprocess starts."""
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        total = int(torch.cuda.get_device_properties(0).total_memory) if cuda_available else None
        free = int(torch.cuda.mem_get_info(0)[0]) if cuda_available else None
    except Exception:
        cuda_available, gpu, total, free = False, "unavailable", None, None
    print(f"  GPU          : {gpu}")
    print(f"  total VRAM   : {total if total is not None else 'n/a'} bytes")
    print(f"  free VRAM    : {free if free is not None else 'n/a'} bytes")
    print("  dtype        : auto (bf16 AMP on Ampere/Ada; fp16 AMP + fp32 LoRA on V100/P100)")
    print("  quantization : QLoRA 4-bit NF4 for C/D")
    print("  device_map   : auto")


def _base_overrides(
    *, profile: str, output_root_arg: str, model: str, method: str, seed: int,
    extra: list[str], dataset: str = DEFAULT_DATASET,
    smoke_eval_file: str | None = None, smoke_items: int = 3,
    output_model_path_arg: str | None = None,
    input_model_weights: str | None = None,
    kb_chunks_path: str | None = None,
    include_input_model_weights: bool = True,
) -> list[str]:
    run_id = f"{dataset}_{model}_{method}_seed_{seed}"
    # The dataset group override must precede any data.* key override. When the
    # caller already supplies one (the shell launcher does), it is not repeated:
    # Hydra rejects the same group override twice.
    values = [] if any(str(item).startswith("data=") for item in extra) else [
        f"data={dataset}"
    ]
    values += [
        f"output_root={_hydra_path(output_root_arg)}",
        f"profile={profile}",
        # Tiny runs are development artifacts but use the collision-proof
        # final-style layout so their adapter can be reused for sports eval.
        f"run_layout={'final' if profile == 'tiny' else profile}",
        f"model={model}",
        f"model_key={model}",
        f"method_key={method}",
        f"experiment_name={model}_{method}",
        f"experiment_id={run_id}",
        f"seed={seed}",
    ]
    if output_model_path_arg:
        values.append(f"output_model_path={_hydra_path(output_model_path_arg)}")
    if input_model_weights and include_input_model_weights and method in {"C", "D"}:
        values.append(f"method.adapter_path={_hydra_path(input_model_weights)}")
    if kb_chunks_path and method in {"B", "D"}:
        values.append(f"method.retriever.chunks_path={_hydra_path(kb_chunks_path)}")
    # Caller extras first, then the smoke slice: Hydra applies overrides in
    # order, so the small evaluation slice must come last or the caller's
    # data.test_file would silently replace it.
    values.extend(extra)
    if smoke_eval_file:
        values.extend([
            f"data.test_file={_hydra_path(smoke_eval_file)}",
            "data.paraphrased_file=null",
            "data.missing_info_file=null",
            f"data.max_samples={smoke_items}",
        ])
    return values


def _planned_hashes(experiment: str, overrides: list[str]) -> tuple[str, str]:
    cfg = load_cfg(experiment=experiment, overrides=overrides)
    return hash_config(cache_identity(cfg)), hash_config(cfg)


def _quarantine_stale_cache(
    stage_dir: Path, expected_hash: str, expected_config_hash: str
) -> None:
    """Preserve, then remove from the active path, a cache from another run."""
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        old_hash = json.loads(manifest_path.read_text(encoding="utf-8")).get("cache_identity_hash")
    except (OSError, ValueError):
        return
    stale_files = [
        path for path in stage_dir.glob("predictions*.jsonl")
        if path.is_file()
    ] + [
        path for path in stage_dir.glob("errors*.jsonl")
        if path.is_file()
    ]
    if not stale_files:
        return
    config_hashes = set()
    for path in stale_files:
        if not path.name.startswith("predictions"):
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line).get("config_hash")
                    if value:
                        config_hashes.add(str(value))
        except (OSError, ValueError, TypeError):
            config_hashes.add("<invalid>")
    if (not old_hash or old_hash == expected_hash) and (
        not config_hashes or config_hashes == {expected_config_hash}
    ):
        return
    target = stage_dir / "_stale_cache" / old_hash
    target.mkdir(parents=True, exist_ok=True)
    for path in stale_files:
        shutil.move(str(path), str(target / path.name))
    print(f"[CACHE INVALIDATED] preserved stale files under {target}")


def _resume_checkpoint_compatible(checkpoint_root: Path, expected_training_hash: str) -> bool:
    """Only pass --resume when the newest checkpoint matches this config."""
    checkpoints = sorted(
        (path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-", 1)[1]) if path.name.split("-", 1)[1].isdigit() else -1,
        reverse=True,
    )
    if not checkpoints:
        return False
    candidates = [checkpoints[0] / "resume_metadata.json", checkpoint_root.parent / "resume_metadata.json"]
    for metadata_path in candidates:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        return metadata.get("training_config_hash") == expected_training_hash
    return False


def _aggregate_final(output_root: Path, results_dir: Path, dataset: str) -> int:
    script = str(_PROJECT_ROOT / "experiments" / "scripts" / "aggregate_results.py")
    outputs = output_root / dataset
    return _execute(
        [sys.executable, script, "--outputs-root", str(outputs), "--out-dir", str(results_dir)],
        "AGGREGATE FINAL RESULTS",
    )


def _verify_smoke_stage(stage_dir: Path, method: str, seed: int) -> list[str]:
    """Check the small smoke contract after a subprocess reports success."""
    required = (
        "predictions.jsonl", "metrics_auto.json", "manifest.json",
        "config_snapshot.yaml", "config_hash.txt", "cache_identity.json",
    )
    problems = [f"{method}: missing {name}" for name in required if not (stage_dir / name).exists()]
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                problems.append(f"{method}: manifest status is {manifest.get('status')!r}")
            if manifest.get("seed") != seed:
                problems.append(f"{method}: manifest seed mismatch")
            if not manifest.get("cache_identity_hash"):
                problems.append(f"{method}: cache identity missing")
        except (OSError, ValueError):
            problems.append(f"{method}: manifest is not valid JSON")
    metrics_path = stage_dir / "metrics_auto.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not payload.get("n_predictions"):
                problems.append(f"{method}: no predictions were scored")
        except (OSError, ValueError):
            problems.append(f"{method}: metrics_auto.json is not valid JSON")
    return problems


def _write_summary(output_root: Path, payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    matrix = load_matrix(Path(args.matrix))
    dataset = str(args.dataset or matrix.get("dataset") or DEFAULT_DATASET)
    dataset_config = _PROJECT_ROOT / "src" / "config" / "data" / f"{dataset}.yaml"
    if not dataset_config.exists():
        raise SystemExit(
            f"Unknown dataset '{dataset}': {dataset_config} does not exist. "
            f"Available: "
            + ", ".join(sorted(
                path.stem for path in (_PROJECT_ROOT / "src" / "config" / "data").glob("dashboard_v*.yaml")
            ))
        )
    override_dataset = next(
        (str(item).split("=", 1)[1] for item in args.override if str(item).startswith("data=")),
        None,
    )
    if override_dataset and override_dataset != dataset:
        raise SystemExit(
            f"Dataset mismatch: --dataset {dataset} but override 'data={override_dataset}'. "
            "Results and adapters would be filed under a different dataset than the one used."
        )
    models = _selected_models(args, matrix)
    methods = _selected_methods(args)
    seeds = _selected_seeds(args, matrix)
    # Direct runner calls also honor the machine-specific HPC environment;
    # explicit CLI paths remain highest priority.
    output_root_arg = args.output_root or os.environ.get("FTM_OUTPUT_DATA_PATH") or (
        "experiments/outputs/smoke" if args.profile == "smoke"
        else "experiments/outputs/tiny" if args.profile == "tiny"
        else matrix.get("output_root", "experiments/outputs/final")
    )
    output_root = _resolved(output_root_arg)
    output_model_root_arg = (
        args.output_model_path
        or os.environ.get("FTM_OUTPUT_MODEL_PATH")
        or output_root_arg
    )
    output_model_root = _resolved(output_model_root_arg)
    results_dir = _resolved(
        args.results_dir
        or os.environ.get("FTM_RESULTS_PATH")
        or (_PROJECT_ROOT / "experiments" / "results" / args.profile / dataset)
    )
    explicit_input_adapter = (
        _resolved(args.input_model_weights) if args.input_model_weights else None
    )

    smoke_file = None
    smoke_items = args.n_eval_items
    if args.profile == "smoke":
        smoke_root = profile_run_dir(
            output_root, SMOKE_MODEL, "A", seeds[0], dataset
        ).parents[1]
        smoke_path = smoke_root / "smoke_eval_items.jsonl"
        if args.dry_run:
            n_eval = args.n_eval_items
        else:
            smoke_source = _resolved(args.smoke_source) if args.smoke_source else None
            n_eval = _smoke_slice(args.n_eval_items, smoke_path, smoke_source, dataset)
        smoke_file = str(smoke_path)
        smoke_items = n_eval

    print("=" * 70)
    print(f"{args.profile.upper()} EXPERIMENT MATRIX")
    print("=" * 70)
    print(f"  dataset      : {dataset}")
    print(f"  models       : {', '.join(models)}")
    print(f"  methods      : {', '.join(methods)}")
    print(f"  seeds        : {seeds}")
    print(f"  output data  : {output_root}")
    print(f"  output model : {output_model_root}")
    print(f"  results dir  : {results_dir}")
    _print_runtime_profile(models)
    print("=" * 70)

    summary: list[dict[str, Any]] = []
    failures = 0
    effective_methods = set(methods)
    if "D" in effective_methods and args.with_dependencies:
        effective_methods.add("C")
    ordered_methods = [key for key in METHOD_KEYS if key in effective_methods]

    for model in models:
        for seed in seeds:
            c_adapter = profile_adapter_dir(output_model_root, model, "C", seed, dataset)
            c_run = profile_run_dir(output_model_root, model, "C", seed, dataset)
            input_adapter_ready = bool(
                explicit_input_adapter and adapter_is_trained(explicit_input_adapter)
            )
            # In inference-only mode an explicitly supplied adapter is a valid
            # C/D input and does not require a locally produced C run.
            c_needed = "C" in effective_methods and not (
                args.skip_training and input_adapter_ready
            )
            smoke_train_items = args.n_train_items if args.profile == "smoke" else None
            train_extra = list(args.override)
            if smoke_train_items is not None:
                train_extra.extend([
                    f"data.max_samples={smoke_train_items}",
                    "model.max_seq_length=512",
                    "training.sft.num_train_epochs=1",
                    "training.sft.per_device_train_batch_size=1",
                    "training.sft.gradient_accumulation_steps=1",
                    "training.sft.gradient_checkpointing=false",
                    f"+training.sft.max_steps={args.max_steps}",
                    "training.sft.logging_steps=1",
                    "training.sft.save_steps=1000",
                ])
            train_overrides = _base_overrides(
                profile=args.profile,
                output_root_arg=str(output_root_arg),
                model=model,
                method="C",
                seed=seed,
                extra=train_extra,
                dataset=dataset,
                smoke_eval_file=smoke_file,
                smoke_items=smoke_items,
                output_model_path_arg=(
                    str(output_model_root_arg) if args.output_model_path else None
                ),
                kb_chunks_path=args.kb_chunks_path,
                include_input_model_weights=False,
            )
            planned_cfg = load_cfg(experiment=METHODS["C"]["experiment"], overrides=train_overrides)
            expected_training_hash = hash_config(planned_cfg.training)
            expected_model_hash = hash_config(planned_cfg.model)
            c_ready = input_adapter_ready if args.skip_training else _compatible_adapter(
                c_adapter,
                model=model,
                seed=seed,
                profile=args.profile,
                dataset=dataset,
                training_config_hash=expected_training_hash,
                model_config_hash=expected_model_hash,
            )

            if c_needed:
                if c_ready and not args.force:
                    print(f"[SKIP] C adapter already compatible: {c_adapter}")
                    summary.append({"model": model, "method": "C", "seed": seed, "stage": "train", "status": "skipped"})
                elif args.skip_training:
                    print(f"[FAIL] --skip-training but C adapter is unavailable: {c_adapter}")
                    summary.append({"model": model, "method": "C", "seed": seed, "stage": "train", "status": "FAILED(no-adapter)"})
                    failures += 1
                    c_ready = False
                else:
                    resume_checkpoint = args.resume and _resume_checkpoint_compatible(
                        c_run / "checkpoints", expected_training_hash
                    )
                    if args.resume and any(c_run.glob("checkpoints/checkpoint-*")) and not resume_checkpoint:
                        print("[RESUME] incompatible old checkpoint found; starting a fresh compatible C run")
                    cmd = _profile_cmd(
                        "train.py", METHODS["C"]["experiment"], train_overrides,
                        resume=resume_checkpoint,
                    )
                    if args.dry_run:
                        print("[DRY-RUN] " + " ".join(cmd[1:]))
                        summary.append({"model": model, "method": "C", "seed": seed, "stage": "train", "status": "dry-run"})
                    else:
                        rc = _execute(cmd, f"TRAIN C ({model}) seed={seed}")
                        failures += int(rc != 0)
                        c_ready = rc == 0 and adapter_is_trained(c_adapter)
                        summary.append({"model": model, "method": "C", "seed": seed, "stage": "train", "status": "ok" if rc == 0 else f"FAILED({rc})"})
                        if not c_ready:
                            print(f"[FAIL] C adapter was not produced: {c_adapter}")

            for method in ordered_methods:
                if method == "C" and "C" not in methods:
                    # C may have been injected solely as D's dependency.
                    continue
                if method == "D" and not c_ready and not input_adapter_ready and not args.dry_run:
                    message = f"D requires compatible C adapter for {model} seed {seed}: {c_adapter}"
                    print(f"[FAIL] {message}")
                    summary.append({"model": model, "method": "D", "seed": seed, "stage": "run", "status": "FAILED(no-adapter)"})
                    failures += 1
                    continue
                if method == "C" and not c_ready and not args.dry_run:
                    print(f"[FAIL] C run skipped because its adapter is unavailable: {c_adapter}")
                    continue

                extra = list(args.override)
                if args.profile == "smoke":
                    extra.extend([
                        "model.max_seq_length=512",
                        "method.generate.max_new_tokens=64",
                        "method.generate.do_sample=false",
                        "training.sft.num_train_epochs=1",
                        "training.sft.per_device_train_batch_size=1",
                        "training.sft.gradient_accumulation_steps=1",
                        "training.sft.gradient_checkpointing=false",
                        "training.sft.logging_steps=1",
                        "training.sft.save_steps=1000",
                        f"+training.sft.max_steps={args.max_steps}",
                    ])
                if method == "D":
                    extra.extend([
                        "method.adapter_source_experiment=E03_qwen0_5b_ft",
                        "method.adapter_source_method_key=C",
                    ])
                overrides = _base_overrides(
                    profile=args.profile,
                    output_root_arg=str(output_root_arg),
                    model=model,
                    method=method,
                    seed=seed,
                    extra=extra,
                    dataset=dataset,
                    smoke_eval_file=smoke_file,
                    smoke_items=smoke_items,
                    output_model_path_arg=(
                        str(output_model_root_arg) if args.output_model_path else None
                    ),
                    input_model_weights=args.input_model_weights,
                    kb_chunks_path=args.kb_chunks_path,
                )
                stage_dir = profile_run_dir(output_root, model, method, seed, dataset)
                expected_cache_identity_hash, expected_config_hash = _planned_hashes(
                    METHODS[method]["experiment"], overrides
                )
                if _compatible_complete(
                    stage_dir,
                    model=model,
                    method=method,
                    seed=seed,
                    profile=args.profile,
                    dataset=dataset,
                    cache_identity_hash=expected_cache_identity_hash,
                ) and not args.force:
                    print(f"[SKIP] {method} already complete: {stage_dir}")
                    summary.append({"model": model, "method": method, "seed": seed, "stage": "run", "status": "skipped"})
                    continue
                if not args.dry_run:
                    _quarantine_stale_cache(
                        stage_dir, expected_cache_identity_hash, expected_config_hash
                    )
                cmd = _profile_cmd("run_experiment.py", METHODS[method]["experiment"], overrides)
                if args.dry_run:
                    print("[DRY-RUN] " + " ".join(cmd[1:]))
                    summary.append({"model": model, "method": method, "seed": seed, "stage": "run", "status": "dry-run"})
                    continue
                rc = _execute(cmd, f"RUN {method} ({model}) seed={seed}")
                failures += int(rc != 0)
                status = "ok" if rc == 0 else f"FAILED({rc})"
                if rc == 0 and args.profile == "smoke":
                    smoke_problems = _verify_smoke_stage(stage_dir, method, seed)
                    if smoke_problems:
                        failures += 1
                        status = "FAILED(smoke-contract)"
                        for problem in smoke_problems:
                            print(f"[FAIL] {problem}")
                summary.append({"model": model, "method": method, "seed": seed, "stage": "run", "status": status})

    payload = {
        "profile": args.profile,
        "dataset": dataset,
        "models": models,
        "methods": methods,
        "seeds": seeds,
        "output_root": str(output_root),
        "runs": summary,
        "failures": failures,
    }
    if not args.dry_run:
        _write_summary(output_root / dataset, payload)

    print("\n" + "=" * 70)
    print("MATRIX SUMMARY")
    print("=" * 70)
    for row in summary:
        print(f"  {dataset:<14} {row['model']:<16} {row['method']:<6} seed={row['seed']:<5} {row['stage']:<6} {row['status']}")
    print("=" * 70)

    if failures:
        print(f"{failures} stage(s) failed. Re-run the same command with --resume.")
        raise SystemExit(1)

    if not args.dry_run:
        if _aggregate_final(output_root, results_dir, dataset) != 0:
            raise SystemExit(1)
        print(f"{args.profile.title()} results: {results_dir}")

    if args.profile == "smoke":
        print("PASS_QWEN_0_5B_END_TO_END_SMOKE")
        print(f"Smoke artifacts: {output_root / dataset / SMOKE_MODEL}")
    elif args.profile == "tiny":
        print("PASS_QWEN_0_5B_TINY_GPU_PIPELINE")
        print(f"Tiny artifacts: {output_root / dataset / SMOKE_MODEL}")
    else:
        print("PASS_MULTI_MODEL_EXPERIMENT_RELEASE_READY_FOR_PROFESSOR")


if __name__ == "__main__":
    main()
