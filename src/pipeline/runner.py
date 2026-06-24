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
from src.inference.postprocess import reparse
from src.inference.runner import InferenceRunner
from src.utils.artifacts import experiment_dir
from src.utils.io import read_jsonl, write_json

logger = logging.getLogger(__name__)

VARIANTS = {
    "paraphrased": "paraphrased_file",
    "missing_info": "missing_info_file",
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
        out_path = self.exp_dir / "predictions.jsonl"
        InferenceRunner(method, out_path).run(briefs, variant="original")

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
            v_method = self._make_method()
            v_out = self.exp_dir / f"predictions_{variant}.jsonl"
            InferenceRunner(v_method, v_out).run(v_briefs, variant=variant)

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

        metric_names = list(self.cfg.eval.get("metrics", []))
        metrics: dict = {}
        for name in metric_names:
            metric = METRICS.get(name)(self.cfg)
            metrics[name] = metric.compute(results, references)

        metrics["robustness"] = compute_robustness(
            results,
            self._load_predictions("predictions_paraphrased.jsonl"),
            self._load_predictions("predictions_missing_info.jsonl"),
            references=references,
        )

        payload = {
            "experiment_id": str(self.cfg.get("experiment_id", "")),
            "method": str(self.cfg.method.name),
            "model": str(self.cfg.model.get("name", "")),
            "seed": int(self.cfg.get("seed", 42)),
            "n_predictions": len(results),
            "metrics": metrics,
        }
        write_json(payload, self.exp_dir / "metrics_auto.json")
        return payload

    def run(self) -> dict:
        self.run_inference()
        return self.run_eval()
