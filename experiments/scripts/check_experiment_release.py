"""Verify this clone can actually run the experiments.

    python experiments/scripts/check_experiment_release.py

Run this immediately after cloning and installing. It answers one question --
"will the experiments run here, on the intended data?" -- and says exactly what
is wrong when the answer is no.

Checks, in order:

    python version              required interpreter version
    required packages           inference + training stacks
    CUDA                        informational locally, required with --require-cuda
    frozen dataset files        every file the selected dataset declares exists
    dataset SHA-256             contents match data/frozen/<dataset>/hashes.json
    split counts                match the dataset manifest counts
    RAG knowledge base          guidelines present, chunks built, manifest matches
    experiment configs          every experiment config composes
    model config                the selected model config resolves
    output directory            writable
    no staging dependency       no final config points at staging or processed data

Exit code 0 means PASS. Any FAIL exits 1 with a short reason; warnings never
fail the run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MIN_PYTHON = (3, 11)  # pyproject: requires-python = ">=3.11,<3.14"

DEFAULT_DATASET = "dashboard_v4"

# Fallback split counts for datasets whose manifest does not declare them.
FALLBACK_COUNTS = {
    "dashboard_v3": {"train.jsonl": 1281, "val.jsonl": 264, "test.jsonl": 274},
    "dashboard_v4": {"train.jsonl": 2932, "val.jsonl": 613, "test.jsonl": 274},
}


def frozen_dir(dataset: str) -> Path:
    return _PROJECT_ROOT / "data" / "frozen" / dataset


def expected_counts(dataset: str) -> dict:
    """Split counts declared by the dataset manifest, else the pinned fallback."""
    manifest = frozen_dir(dataset) / "manifest.json"
    if manifest.exists():
        try:
            counts = json.loads(manifest.read_text(encoding="utf-8")).get("counts") or {}
        except Exception:
            counts = {}
        mapped = {}
        for name, keys in (
            ("train.jsonl", ("train",)),
            ("val.jsonl", ("validation", "val")),
            ("test.jsonl", ("test",)),
        ):
            for key in keys:
                if isinstance(counts.get(key), int):
                    mapped[name] = int(counts[key])
                    break
        if len(mapped) == 3:
            return mapped
    return dict(FALLBACK_COUNTS.get(dataset, {}))

# Base stack needed for inference + evaluation; the training extra is separate
# because the local machine may legitimately not have it.
CORE_PACKAGES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("pydantic", "pydantic"),
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
]
TRAIN_PACKAGES = [
    ("peft", "peft"),
    ("trl", "trl"),
    ("bitsandbytes", "bitsandbytes"),
    ("accelerate", "accelerate"),
    ("datasets", "datasets"),
]

EXPERIMENTS = [
    "E01_qwen0_5b_prompt",
    "E02_qwen0_5b_rag",
    "E03_qwen0_5b_ft",
    "E04_qwen0_5b_ft_rag",
]
FINAL_MODELS = ["qwen3_1_7b", "qwen3_8b", "qwen3_14b", "llama3_1_8b"]
SMOKE_MODEL = "qwen2_5_0_5b"

# Paths a final experiment config must never depend on: staging is pre-freeze and
# gitignored, processed/ is the superseded v1 pipeline.
FORBIDDEN_DATA_PREFIXES = ("data/staging", "data/processed", "data/raw_legacy")


class Report:
    """Collects check outcomes and renders them as one aligned block."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = 0
        self.warned = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.rows.append(("PASS", name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.rows.append(("WARN", name, detail))
        self.warned += 1

    def fail(self, name: str, detail: str) -> None:
        self.rows.append(("FAIL", name, detail))
        self.failed += 1

    def render(self) -> None:
        width = max(len(name) for _, name, _ in self.rows)
        for status, name, detail in self.rows:
            line = f"  [{status}] {name.ljust(width)}"
            if detail:
                line += f"  {detail}"
            print(line)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ----------------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------------
def check_python(report: Report) -> None:
    actual = sys.version_info[:3]
    version = ".".join(str(p) for p in actual)
    if actual[:2] < MIN_PYTHON:
        report.fail(
            "python version",
            f"{version} < required {'.'.join(str(p) for p in MIN_PYTHON)}",
        )
    else:
        report.ok("python version", version)


def _probe(packages: list[tuple[str, str]]) -> list[str]:
    """Return the distributions whose import module cannot be located.

    Deliberately uses ``find_spec`` rather than importing: pulling torch,
    bitsandbytes, accelerate and datasets into one process is exactly the
    DLL load-order hazard the trainer works around on Windows, and a preflight
    must never be the thing that crashes.
    """
    missing = []
    for module, dist in packages:
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(dist)
        except Exception:
            missing.append(dist)
    return missing


def check_packages(report: Report, require_training: bool) -> None:
    missing = _probe(CORE_PACKAGES)
    if missing:
        report.fail("core packages", f"missing: {', '.join(missing)} -> pip install -e .")
    else:
        report.ok("core packages", f"{len(CORE_PACKAGES)} present")

    missing_train = _probe(TRAIN_PACKAGES)
    if not missing_train:
        report.ok("training packages", f"{len(TRAIN_PACKAGES)} present")
    elif require_training:
        report.fail(
            "training packages",
            f"missing: {', '.join(missing_train)} -> pip install -e \".[train]\"",
        )
    else:
        report.warn(
            "training packages",
            f"missing: {', '.join(missing_train)} (needed only for methods C/D)",
        )


def check_cuda(report: Report, require_cuda: bool) -> None:
    try:
        import torch

        build = getattr(getattr(torch, "version", None), "cuda", None)
        report.ok("pytorch", f"{torch.__version__} (CUDA build: {build or 'none - CPU-only wheel'})")
        if not build:
            message = (
                "CPU-only PyTorch wheel installed; QLoRA training cannot run. Reinstall the "
                "CUDA build: poetry run pip install torch==2.6.0 "
                "--index-url https://download.pytorch.org/whl/cu124"
            )
            if require_cuda:
                report.fail("cuda", message)
            else:
                report.warn("cuda", message)
            return
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            report.ok(
                "cuda",
                f"{torch.cuda.get_device_name(0)}; VRAM {free // (1 << 20)}/{total // (1 << 20)} MiB free/total",
            )
            return
        message = "no GPU visible (CPU inference works; training will be very slow)"
    except Exception as exc:
        message = f"could not query torch.cuda: {exc}"

    if require_cuda:
        report.fail("cuda", message)
    else:
        report.warn("cuda", message)


def check_frozen_dataset(report: Report, dataset: str) -> Optional[dict]:
    FROZEN_DIR = frozen_dir(dataset)
    HASHES_FILE = FROZEN_DIR / "hashes.json"
    config_file = _PROJECT_ROOT / "src" / "config" / "data" / f"{dataset}.yaml"
    if not config_file.exists():
        report.fail("dataset config", f"missing {config_file}")
        return None
    report.ok("dataset config", str(config_file.relative_to(_PROJECT_ROOT)))
    if not FROZEN_DIR.exists():
        report.fail("frozen dataset", f"missing directory {FROZEN_DIR}")
        return None
    if not HASHES_FILE.exists():
        report.fail("frozen dataset", f"missing {HASHES_FILE}")
        return None

    try:
        with HASHES_FILE.open("r", encoding="utf-8") as f:
            hashes = json.load(f)
    except Exception as exc:
        report.fail("frozen dataset", f"unreadable hashes.json: {exc}")
        return None

    records = hashes.get("records") or hashes.get("files") or {}
    if not records:
        report.fail("frozen dataset", "hashes.json lists no files")
        return None

    missing = [name for name in records if not (FROZEN_DIR / name).exists()]
    if missing:
        report.fail("frozen dataset", f"missing files: {', '.join(sorted(missing))}")
        return None

    report.ok("frozen dataset", f"{len(records)} files present in {FROZEN_DIR.name}")
    return records


def check_hashes(report: Report, records: Optional[dict], dataset: str) -> None:
    FROZEN_DIR = frozen_dir(dataset)
    if not records:
        report.fail("dataset sha256", "skipped (dataset check failed)")
        return

    mismatched = []
    for name, entry in sorted(records.items()):
        expected = entry.get("sha256") if isinstance(entry, dict) else str(entry)
        if not expected:
            continue
        actual = sha256_of(FROZEN_DIR / name)
        if actual.lower() != str(expected).lower():
            mismatched.append(name)

    if mismatched:
        report.fail(
            "dataset sha256",
            f"content differs from hashes.json: {', '.join(mismatched)} "
            f"(the frozen dataset must not be edited)",
        )
    else:
        report.ok("dataset sha256", f"{len(records)} files verified")


def check_counts(report: Report, dataset: str) -> None:
    FROZEN_DIR = frozen_dir(dataset)
    EXPECTED_COUNTS = expected_counts(dataset)
    if not EXPECTED_COUNTS:
        report.warn("split counts", f"no declared counts for {dataset}")
        return
    wrong = []
    for name, expected in EXPECTED_COUNTS.items():
        path = FROZEN_DIR / name
        if not path.exists():
            wrong.append(f"{name}: missing")
            continue
        actual = count_lines(path)
        if actual != expected:
            wrong.append(f"{name}: {actual} != {expected}")
    if wrong:
        report.fail("split counts", "; ".join(wrong))
    else:
        report.ok(
            "split counts",
            "/".join(str(EXPECTED_COUNTS[n]) for n in ("train.jsonl", "val.jsonl", "test.jsonl")),
        )


def check_knowledge_base(report: Report) -> None:
    kb_dir = _PROJECT_ROOT / "data" / "knowledge_base"
    guidelines = sorted((kb_dir / "guidelines").glob("*.md")) if kb_dir.exists() else []
    if not guidelines:
        report.fail(
            "rag guidelines",
            f"no guideline documents in {kb_dir / 'guidelines'} "
            f"(methods B and D cannot run)",
        )
        return
    report.ok("rag guidelines", f"{len(guidelines)} documents")

    chunks = kb_dir / "chunks.jsonl"
    if not chunks.exists():
        report.fail(
            "rag knowledge base",
            "chunks.jsonl not built -> python experiments/scripts/build_kb.py",
        )
        return

    try:
        from src.data_pipeline.kb_builder import verify_kb

        ok, problems = verify_kb(kb_dir)
        if ok:
            report.ok("rag knowledge base", f"{count_lines(chunks)} chunks verified")
        else:
            report.fail(
                "rag knowledge base",
                f"{'; '.join(problems)} -> python experiments/scripts/build_kb.py",
            )
    except ImportError:
        # Verification helper not available: fall back to presence only.
        report.warn(
            "rag knowledge base",
            f"{count_lines(chunks)} chunks present (no manifest verification available)",
        )


def _model_keys(profile: str, model_override: Optional[str], all_models: bool) -> list[str]:
    if profile == "smoke":
        return [model_override or SMOKE_MODEL]
    if all_models:
        return list(FINAL_MODELS)
    return [model_override or FINAL_MODELS[0]]


def check_configs(
    report: Report,
    model_override: Optional[str],
    *,
    profile: str = "smoke",
    all_models: bool = False,
    dataset: str = DEFAULT_DATASET,
) -> list[dict]:
    try:
        from src.utils.config import load_cfg
    except Exception as exc:
        report.fail("experiment configs", f"cannot import config loader: {exc}")
        return

    broken = []
    staging_hits = []
    profiles: list[dict] = []
    for model_key in _model_keys(profile, model_override, all_models):
        model_broken = []
        model_names = set()
        for experiment in EXPERIMENTS:
            try:
                cfg = load_cfg(
                    experiment=experiment,
                    overrides=[f"data={dataset}", f"model={model_key}"],
                )
            except Exception as exc:
                model_broken.append(f"{experiment}: {type(exc).__name__}")
                continue

            model_names.add(str(cfg.model.get("name", "")))
            if experiment == EXPERIMENTS[0]:
                profiles.append({
                    "key": model_key,
                    "hf_id": str(cfg.model.get("hf_id") or cfg.model.get("name") or ""),
                    "revision": cfg.model.get("revision"),
                    "requires_hf_token": bool(cfg.model.get("requires_hf_token", False)),
                })

            data_cfg = cfg.get("data", {})
            for key in ("train_file", "val_file", "test_file"):
                value = data_cfg.get(key)
                if not value:
                    continue
                normalized = str(value).replace("\\", "/")
                if normalized.startswith(FORBIDDEN_DATA_PREFIXES):
                    staging_hits.append(f"{model_key}/{experiment}.{key} -> {value}")

        if model_broken:
            broken.extend(f"{model_key}: {item}" for item in model_broken)
        elif len(model_names) == 1:
            report.ok(f"model config {model_key}", next(iter(model_names)))
        else:
            report.fail(f"model config {model_key}", "no model resolved")

    if broken:
        report.fail("experiment configs", "; ".join(broken))
    else:
        report.ok("experiment configs", f"{len(EXPERIMENTS)} configs compose for {len(profiles)} model(s)")

    if staging_hits:
        report.fail("no staging dependency", "; ".join(staging_hits))
    else:
        report.ok("no staging dependency", "selected profiles use data/frozen only")
    return profiles


def check_hf_access(report: Report, profiles: list[dict], *, verify_remote: bool = True) -> None:
    """Check gated credentials and repository metadata without downloading weights."""
    token = os.environ.get("HF_TOKEN")
    gated = [profile["key"] for profile in profiles if profile.get("requires_hf_token")]
    if gated and not token:
        report.fail("HF_TOKEN", f"missing for gated model profile(s): {', '.join(gated)}")
    elif gated:
        report.ok("HF_TOKEN", f"present for {len(gated)} gated model profile(s)")
    else:
        report.ok("HF_TOKEN", "no selected model profile requires a gated token")

    if not verify_remote:
        report.warn("model repository access", "remote metadata check skipped")
        return
    try:
        from huggingface_hub import HfApi
    except Exception:
        report.warn("model repository access", "huggingface_hub unavailable; weights were not downloaded")
        return

    api = HfApi()
    for profile in profiles:
        if profile.get("requires_hf_token") and not token:
            continue
        try:
            kwargs = {"token": token} if token else {}
            if profile.get("revision"):
                kwargs["revision"] = profile["revision"]
            api.model_info(profile["hf_id"], **kwargs)
            report.ok(f"model repository {profile['key']}", "metadata reachable; weights not downloaded")
        except Exception as exc:
            report.warn(
                f"model repository {profile['key']}",
                f"metadata check unavailable ({type(exc).__name__}); weights were not downloaded",
            )


def check_transformers_model_support(report: Report, profiles: list[dict]) -> None:
    """Verify Qwen3/Llama config classes exist without loading model weights."""
    try:
        from transformers.models.llama import LlamaConfig
        from transformers.models.qwen3 import Qwen3Config
    except Exception as exc:
        report.fail("transformers model support", f"Qwen3/Llama classes unavailable: {type(exc).__name__}")
        return
    families = {str(profile["key"]).split("_", 1)[0] for profile in profiles}
    details = []
    if any(key.startswith("qwen3") for key in families) or any(
        str(profile["key"]).startswith("qwen3") for profile in profiles
    ):
        details.append(Qwen3Config.model_type)
    if any(str(profile["key"]).startswith("llama3") for profile in profiles):
        details.append(LlamaConfig.model_type)
    report.ok("transformers model support", ", ".join(details) or "Qwen3/Llama classes available")


def check_output_dir(report: Report, output_root: str) -> None:
    root = Path(output_root)
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report.ok("output directory", str(root))
    except Exception as exc:
        report.fail("output directory", f"not writable: {root} ({exc})")


def check_robustness_splits(report: Report, dataset: str) -> None:
    """Robustness splits are optional, but a missing declared file must be visible."""
    try:
        from src.utils.config import load_cfg

        cfg = load_cfg(overrides=[f"data={dataset}"])
    except Exception as exc:
        report.warn("robustness splits", f"config not composable: {type(exc).__name__}")
        return
    missing = []
    present = []
    for key in ("paraphrased_file", "missing_info_file"):
        raw = cfg.data.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        (present if path.exists() else missing).append(str(raw))
    if missing:
        suffix = dataset.replace("dashboard_", "")
        report.warn(
            "robustness splits",
            f"declared but missing: {', '.join(missing)} -> "
            f"python experiments/scripts/build_perturbations_v3.py --source "
            f"data/frozen/{dataset}/test.jsonl --out-dir data/eval/robustness_{suffix}",
        )
    else:
        report.ok("robustness splits", f"{len(present)} file(s) present")


def check_adapter_path_logic(
    report: Report, dataset: str, profile: str, output_root: str
) -> None:
    """Confirm D resolves the same-dataset/model/seed C adapter, without training."""
    try:
        from src.utils.adapter import resolve_adapter_path

        cfg = {
            "output_root": output_root,
            "profile": profile,
            "run_layout": profile,
            "model_key": "qwen3_8b",
            "method_key": "D",
            "seed": 43,
            "model": {"key": "qwen3_8b"},
            "data": {"dataset_version": dataset},
            "method": {
                "name": "ft_rag",
                "type": "fine_tuned_rag",
                "adapter_source_experiment": "E03_qwen0_5b_ft",
                "adapter_source_method_key": "C",
            },
        }
        resolved = Path(resolve_adapter_path(cfg, _PROJECT_ROOT))
    except Exception as exc:
        report.fail("adapter path logic", f"{type(exc).__name__}: {exc}")
        return
    expected = (dataset, "qwen3_8b", "C", "seed_43", "adapter")
    if resolved.parts[-5:] == expected:
        report.ok("adapter path logic", "D -> same dataset/model/seed C adapter")
    else:
        report.fail("adapter path logic", f"unexpected resolution: {resolved}")


# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify this clone can run the experiments")
    p.add_argument("--require-cuda", action="store_true",
                   help="Fail (not warn) when no GPU is visible")
    p.add_argument("--require-training", action="store_true",
                   help="Fail (not warn) when the training extra is not installed")
    p.add_argument("--model", default=None,
                   help="Check against a specific model config name")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="Frozen dataset config name (default: dashboard_v4)")
    p.add_argument("--profile", choices=("smoke", "final"), default="smoke",
                   help="Preflight profile (smoke or final)")
    p.add_argument("--all-models", action="store_true",
                   help="Check all four final model profiles")
    p.add_argument("--output-root", default=None,
                   help="Output root to test for writability")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = Report()

    print("=" * 70)
    print("EXPERIMENT RELEASE CHECK")
    print("=" * 70)

    print(f"  dataset: {args.dataset}")
    check_python(report)
    final_profile = args.profile == "final"
    check_packages(report, require_training=args.require_training or final_profile)
    check_cuda(report, require_cuda=args.require_cuda or final_profile)
    records = check_frozen_dataset(report, args.dataset)
    check_hashes(report, records, args.dataset)
    check_counts(report, args.dataset)
    check_robustness_splits(report, args.dataset)
    check_knowledge_base(report)
    profiles = check_configs(
        report, args.model, profile=args.profile, all_models=args.all_models,
        dataset=args.dataset,
    )
    check_transformers_model_support(report, profiles)
    check_hf_access(report, profiles)
    output_root = args.output_root or (
        "experiments/outputs/final" if final_profile else "experiments/outputs/smoke"
    )
    check_output_dir(report, output_root)
    check_adapter_path_logic(report, args.dataset, args.profile, output_root)

    report.render()
    print("=" * 70)

    if report.failed:
        print(f"FAIL: {report.failed} check(s) failed. Fix the lines marked FAIL above.")
        raise SystemExit(1)

    if report.warned:
        print(f"PASS ({report.warned} warning(s) -- see WARN lines above)")
    else:
        print("PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
