"""Verify this clone can actually run the experiments.

    python experiments/scripts/check_experiment_release.py

Run this immediately after cloning and installing. It answers one question --
"will the experiments run here, on the intended data?" -- and says exactly what
is wrong when the answer is no.

Checks, in order:

    python version              required interpreter version
    required packages           inference + training stacks
    CUDA                        informational locally, required with --require-cuda
    frozen dataset files        every file dashboard_v3 declares exists
    dataset SHA-256             contents match data/frozen/dashboard_v3/hashes.json
    split counts                train/val/test == 1281/264/274
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
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MIN_PYTHON = (3, 10)

FROZEN_DIR = _PROJECT_ROOT / "data" / "frozen" / "dashboard_v3"
HASHES_FILE = FROZEN_DIR / "hashes.json"

EXPECTED_COUNTS = {"train.jsonl": 1281, "val.jsonl": 264, "test.jsonl": 274}

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

        if torch.cuda.is_available():
            report.ok("cuda", torch.cuda.get_device_name(0))
            return
        message = "no GPU visible (CPU inference works; training will be very slow)"
    except Exception as exc:
        message = f"could not query torch.cuda: {exc}"

    if require_cuda:
        report.fail("cuda", message)
    else:
        report.warn("cuda", message)


def check_frozen_dataset(report: Report) -> Optional[dict]:
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


def check_hashes(report: Report, records: Optional[dict]) -> None:
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


def check_counts(report: Report) -> None:
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


def check_configs(report: Report, model_override: Optional[str]) -> None:
    try:
        from src.utils.config import load_cfg
    except Exception as exc:
        report.fail("experiment configs", f"cannot import config loader: {exc}")
        return

    broken = []
    staging_hits = []
    model_names = set()

    for experiment in EXPERIMENTS:
        overrides = [f"model={model_override}"] if model_override else []
        try:
            cfg = load_cfg(experiment=experiment, overrides=overrides)
        except Exception as exc:
            broken.append(f"{experiment}: {type(exc).__name__}")
            continue

        model_names.add(str(cfg.model.get("name", "")))

        data_cfg = cfg.get("data", {})
        for key in ("train_file", "val_file", "test_file"):
            value = data_cfg.get(key)
            if not value:
                continue
            normalized = str(value).replace("\\", "/")
            if normalized.startswith(FORBIDDEN_DATA_PREFIXES):
                staging_hits.append(f"{experiment}.{key} -> {value}")

    if broken:
        report.fail("experiment configs", "; ".join(broken))
    else:
        report.ok("experiment configs", f"{len(EXPERIMENTS)} configs compose")

    if staging_hits:
        report.fail("no staging dependency", "; ".join(staging_hits))
    else:
        report.ok("no staging dependency", "final configs use data/frozen only")

    if not model_names:
        report.fail("model config", "no model resolved")
    elif len(model_names) > 1:
        report.warn("model config", f"experiments disagree: {sorted(model_names)}")
    else:
        report.ok("model config", model_names.pop())


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


# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify this clone can run the experiments")
    p.add_argument("--require-cuda", action="store_true",
                   help="Fail (not warn) when no GPU is visible")
    p.add_argument("--require-training", action="store_true",
                   help="Fail (not warn) when the training extra is not installed")
    p.add_argument("--model", default=None,
                   help="Check against a specific model config name")
    p.add_argument("--output-root", default="experiments/outputs/final",
                   help="Output root to test for writability")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = Report()

    print("=" * 70)
    print("EXPERIMENT RELEASE CHECK")
    print("=" * 70)

    check_python(report)
    check_packages(report, require_training=args.require_training)
    check_cuda(report, require_cuda=args.require_cuda)
    records = check_frozen_dataset(report)
    check_hashes(report, records)
    check_counts(report)
    check_knowledge_base(report)
    check_configs(report, args.model)
    check_output_dir(report, args.output_root)

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
