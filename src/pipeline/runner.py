"""Local infer -> eval orchestration for a single experiment.

Given a composed config it: resolves the method from the registry, runs cached
inference over the test split (and any available perturbation variants), then
computes the configured metrics plus robustness and writes ``metrics_auto.json``.
Re-running is cheap: inference is cached per item and the whole run is idempotent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import src.evaluation  # noqa: F401  (registers metrics under METRICS)
import src.methods  # noqa: F401  (registers methods under METHODS)
from src.core.registry import METHODS, METRICS
from src.core.schemas import GenerationResult
from src.data_pipeline.dataset import load_gold_items
from src.evaluation.metrics.robustness import compute_robustness
from src.evaluation.reporting import write_per_run_reports
from src.inference.postprocess import reparse
from src.inference.runner import InferenceRunner
from src.utils.artifacts import cache_identity, experiment_dir
from src.utils.io import read_jsonl, write_json

logger = logging.getLogger(__name__)

VARIANTS = {
    "paraphrased": "paraphrased_file",
    "missing_info": "missing_info_file",
}


def _coverage(results: Optional[List[GenerationResult]], expected_ids: List[str]) -> dict:
    predicted_ids = {result.item_id for result in (results or [])}
    expected_set = set(expected_ids)
    missing_ids = [item_id for item_id in expected_ids if item_id not in predicted_ids]
    n_requested = len(expected_ids)
    return {
        "n_requested": n_requested,
        "n_predictions": len(predicted_ids & expected_set),
        "n_missing": len(missing_ids),
        "prediction_coverage_rate": round(
            100.0 * (n_requested - len(missing_ids)) / n_requested, 2
        ) if n_requested else None,
        "missing_item_ids": missing_ids,
    }


def _resolve(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root / p


class ExperimentRunner:
    def __init__(self, cfg: Any, project_root: Path) -> None:
        self.cfg = cfg
        self.project_root = project_root
        self.exp_dir = experiment_dir(cfg, project_root)
        self.data_cfg = cfg.get("data", {})

    # ------------------------------------------------------------------
    def _load_test_items(self):
        test_file = _resolve(str(self.data_cfg.get("test_file")), self.project_root)
        items = load_gold_items(test_file)
        max_samples = self.data_cfg.get("eval_max_samples")
        if max_samples is None:
            max_samples = self.data_cfg.get("max_samples")
        if max_samples:
            items = items[: int(max_samples)]
        return items

    def _make_method(self):
        method_cls = METHODS.get(str(self.cfg.method.name))
        return method_cls(self.cfg)

    # ------------------------------------------------------------------
    def run_inference(self) -> None:
        items = self._load_test_items()
        briefs = [it.brief for it in items]

        method = self._make_method()
        identity = cache_identity(self.cfg)
        jobs = [(briefs, self.exp_dir / "predictions.jsonl", "original")]

        # Optional perturbation variants — only if their files exist.
        for variant, cfg_key in VARIANTS.items():
            file_str = self.data_cfg.get(cfg_key)
            if not file_str:
                continue
            vpath = _resolve(str(file_str), self.project_root)
            if not vpath.exists():
                continue
            v_items = load_gold_items(vpath)
            v_briefs = [it.brief for it in v_items]
            jobs.append((v_briefs, self.exp_dir / f"predictions_{variant}.jsonl", variant))

        # Original and robustness variants use identical model/retriever setup.
        # Keep generation order and cache semantics, but avoid loading the same
        # base model and adapter up to three times in one experiment.
        InferenceRunner(
            method,
            self.exp_dir / "predictions.jsonl",
            cache_identity=identity,
        ).run_many(jobs)

    # ------------------------------------------------------------------
    def _load_predictions(self, name: str) -> Optional[List[GenerationResult]]:
        path = self.exp_dir / name
        if not path.exists():
            return None
        # Reparse from raw_text so the current parser applies to cached outputs.
        return [reparse(GenerationResult(**r)) for r in read_jsonl(path)]

    def run_eval(self) -> dict:
        items = self._load_test_items()
        references = [
            {
                "item_id": it.item_id,
                "brief": it.brief.model_dump(mode="json"),
                "recommendation": it.recommendation.model_dump(mode="json"),
            }
            for it in items
        ]
        results = self._load_predictions("predictions.jsonl")
        if results is None:
            raise FileNotFoundError(
                f"No predictions at {self.exp_dir / 'predictions.jsonl'}. Run inference first."
            )

        expected_ids = [reference["item_id"] for reference in references]
        coverage = _coverage(results, expected_ids)
        missing_ids = coverage["missing_item_ids"]
        n_requested = coverage["n_requested"]

        variant_results: dict[str, Optional[List[GenerationResult]]] = {}
        variant_coverage: dict[str, dict] = {}
        for variant, config_key in VARIANTS.items():
            configured_path = self.data_cfg.get(config_key)
            if not configured_path:
                continue
            source_path = _resolve(str(configured_path), self.project_root)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Configured {variant} evaluation data not found: {source_path}"
                )
            expected_variant_ids = [item.item_id for item in load_gold_items(source_path)]
            loaded = self._load_predictions(f"predictions_{variant}.jsonl")
            variant_results[variant] = loaded
            variant_coverage[variant] = _coverage(loaded, expected_variant_ids)

        metric_names = list(self.cfg.eval.get("metrics", []))
        metrics: dict = {}
        for name in metric_names:
            metric = METRICS.get(name)(self.cfg)
            metrics[name] = metric.compute(results, references)

        metrics["robustness"] = compute_robustness(
            results,
            variant_results.get("paraphrased"),
            variant_results.get("missing_info"),
            references=references,
        )

        payload = {
            "experiment_id": str(self.cfg.get("experiment_id", "")),
            "method": str(self.cfg.method.name),
            "model": str(self.cfg.model.get("name", "")),
            "seed": int(self.cfg.get("seed", 42)),
            "n_predictions": len(results),
            "n_requested": n_requested,
            "coverage": coverage,
            "variant_coverage": variant_coverage,
            "metrics": metrics,
        }
        write_json(payload, self.exp_dir / "metrics_auto.json")
        # Additive Phase-1 reporting artifacts (do not mutate metrics_auto.json /
        # predictions.jsonl): a layered metrics.json and a per-item scored file.
        report_results = list(results)
        report_results.extend(
            GenerationResult(
                item_id=item_id,
                method_name=str(self.cfg.method.name),
                model_name=str(self.cfg.model.get("name", "")),
                raw_text="",
                parse_error="missing_prediction",
                seed=int(self.cfg.get("seed", 42)),
            )
            for item_id in missing_ids
        )
        write_per_run_reports(self.exp_dir, payload, report_results, references)
        incomplete = []
        if missing_ids:
            incomplete.append(f"original: {len(missing_ids)} of {n_requested}")
        incomplete.extend(
            f"{variant}: {values['n_missing']} of {values['n_requested']}"
            for variant, values in variant_coverage.items()
            if values["n_missing"]
        )
        if incomplete:
            detail = "; ".join(incomplete)
            if len(incomplete) == 1 and incomplete[0].startswith("original:"):
                detail = incomplete[0].removeprefix("original: ")
            raise RuntimeError(
                f"{detail} expected predictions are missing; coverage-aware metrics were "
                "written, but the run remains incomplete."
            )
        return payload

    def run(self) -> dict:
        self.run_inference()
        return self.run_eval()
