"""Build the v3 robustness perturbation sets from the frozen dashboard_v3 test split.

Reads ``data/frozen/dashboard_v3/test.jsonl`` STRICTLY read-only -- the frozen
package must stay byte-identical -- and writes two perturbed variants into a
separate evaluation directory, each row paired to its source item via
``item_id`` / ``extra.base_item_id`` and tagged with ``extra.variant``:

    data/eval/robustness_v3/test_paraphrased.jsonl   (paraphrase_consistency / paraphrase_accuracy)
    data/eval/robustness_v3/test_missing_info.jsonl  (missing-info robustness)

The reference ``recommendation`` is copied unchanged -- it is the gold the
perturbed input is scored against. The transformations are the same
deterministic, dependency-free ones already used for v2
(``src.data_pipeline.perturbations``): no LLM, no network, no randomness, and
rows are emitted in sorted ``item_id`` order, so re-running produces
byte-identical output files.

Usage:
    python experiments/scripts/build_perturbations_v3.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.frozen_validation import sha256_of_file
from src.data_pipeline.perturbations import drop_info_brief, paraphrase_brief
from src.utils.io import read_jsonl, write_json, write_jsonl

GENERATOR = "experiments/scripts/build_perturbations_v3.py"
GENERATOR_VERSION = "robustness_v3_v1"
DEFAULT_SOURCE = "data/frozen/dashboard_v3/test.jsonl"
DEFAULT_OUT_DIR = "data/eval/robustness_v3"
DEFAULT_SEED = 42
MANIFEST_NAME = "manifest.json"

# (output filename, brief transform, variant tag) -- fixed order, so the
# manifest and the printed summary are stable across runs.
VARIANTS: tuple[tuple[str, Callable[[dict], dict], str], ...] = (
    ("test_paraphrased.jsonl", paraphrase_brief, "paraphrased"),
    ("test_missing_info.jsonl", drop_info_brief, "missing_info"),
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _rel(path: Path) -> str:
    """Project-relative POSIX path (falls back to absolute for outside paths)."""
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _assert_writable_target(out_dir: Path) -> None:
    """The frozen dataset package is immutable -- never emit anything into it."""
    frozen = (_PROJECT_ROOT / "data" / "frozen").resolve()
    target = out_dir.resolve()
    if target == frozen or frozen in target.parents:
        raise SystemExit(f"Refusing to write into the frozen dataset package: {out_dir}")


def _perturb_rows(rows: List[dict], transform: Callable[[dict], dict], variant: str) -> List[dict]:
    """Apply ``transform`` to every brief, keeping the item_id stable for pairing.

    Rows are emitted sorted by ``item_id`` so the output order never depends on
    the read order of the source file.
    """
    out: List[dict] = []
    for row in sorted(rows, key=lambda r: str(r.get("item_id", ""))):
        brief = dict(row.get("brief", {}))
        base_id = row.get("item_id") or brief.get("item_id", "")
        new_brief = transform(brief)
        new_brief["item_id"] = base_id  # keep id stable so pairs line up
        extra = dict(new_brief.get("extra") or {})
        extra["variant"] = variant
        extra["base_item_id"] = base_id
        new_brief["extra"] = extra
        out.append({
            "item_id": base_id,
            "split": row.get("split"),
            "brief": new_brief,
            "recommendation": row.get("recommendation", {}),
        })
    return out


def _n_briefs_changed(rows: List[dict], transform: Callable[[dict], dict]) -> int:
    """How many briefs the transform actually alters (perturbation coverage)."""
    return sum(1 for row in rows if transform(dict(row.get("brief", {}))) != row.get("brief", {}))


def build_perturbation_sets(
    source: str | Path = DEFAULT_SOURCE,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Write the perturbed variants plus a manifest; return the manifest dict."""
    source_path = _resolve(source)
    target_dir = _resolve(out_dir)
    if not source_path.exists():
        raise SystemExit(f"Missing {source_path}. The frozen dashboard_v3 package must exist first.")
    _assert_writable_target(target_dir)

    source_sha = sha256_of_file(source_path)
    rows = read_jsonl(source_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Any] = {}
    for fname, transform, variant in VARIANTS:
        perturbed = _perturb_rows(rows, transform, variant)
        write_jsonl(perturbed, target_dir / fname)
        outputs[fname] = {
            "variant": variant,
            "n_records": len(perturbed),
            "n_briefs_changed": _n_briefs_changed(rows, transform),
            "sha256": sha256_of_file(target_dir / fname),
        }

    if sha256_of_file(source_path) != source_sha:
        raise SystemExit(f"Source changed during the build: {source_path} must stay read-only.")

    manifest = {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "description": (
            "Deterministic offline robustness perturbations of the frozen dashboard_v3 test "
            "split. Briefs are transformed by src.data_pipeline.perturbations (meaning-preserving "
            "synonym paraphrase / information drop); the gold recommendation is copied unchanged "
            "and item_id is preserved so the robustness metrics can pair original vs perturbed "
            "predictions. No LLM, no network, no randomness."
        ),
        "dataset_version": "dashboard_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "seed_note": (
            "Recorded for provenance only: the perturbations are rule-based and contain no "
            "randomness, so the output is identical for any seed."
        ),
        "perturbation_module": "src.data_pipeline.perturbations",
        "pairing_key": "item_id (mirrored in brief.extra.base_item_id)",
        "source": {
            "path": _rel(source_path),
            "sha256": source_sha,
            "n_records": len(rows),
            "read_only": True,
        },
        "outputs": outputs,
    }
    write_json(manifest, target_dir / MANIFEST_NAME)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate deterministic v3 paraphrase / missing-info robustness sets")
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="frozen test split to perturb (read-only)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="directory for the perturbed variants (never inside data/frozen)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="recorded in the manifest; the transforms are seed-independent")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_perturbation_sets(args.source, args.out_dir, args.seed)
    out_dir = _resolve(args.out_dir)

    print("=" * 66)
    print("DASHBOARD V3 ROBUSTNESS PERTURBATION SETS BUILT")
    print("=" * 66)
    print(f"  Source   : {manifest['source']['path']} ({manifest['source']['n_records']} items, read-only)")
    print(f"  sha256   : {manifest['source']['sha256']}")
    print(f"  Out dir  : {_rel(out_dir)}")
    print(f"  Seed     : {manifest['seed']}")
    for fname, info in manifest["outputs"].items():
        print(f"  {fname}: {info['n_records']} items "
              f"({info['n_briefs_changed']} briefs altered)")
        print(f"    sha256 : {info['sha256']}")
    print(f"  Manifest : {_rel(out_dir / MANIFEST_NAME)}")
    print("=" * 66)


if __name__ == "__main__":
    main()
