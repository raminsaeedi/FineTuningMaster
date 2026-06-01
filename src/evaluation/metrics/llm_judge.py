"""G-Eval: LLM-as-judge metric (complements, never replaces, human eval).

A strong LLM scores each recommendation on the rubric dimensions (1-5) via a
chain-of-thought prompt. This is a cheap, scalable proxy for human ratings and
enables a correlation check against them.

The judge call is pluggable: pass ``judge_fn`` (used in tests), or rely on the
default OpenAI-compatible client via urllib (no extra dependency), configured by
env vars OPENAI_API_KEY and optional OPENAI_BASE_URL. With no judge available the
metric returns ``{"available": False}`` so normal runs are unaffected.

Judge config (under the eval config): ``judge: {provider, model, dimensions}``.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from src.evaluation.human.rubric import RUBRIC_KEYS
from src.evaluation.metrics.base import index_references


def _get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def build_judge_prompt(brief_text: str, output_text: str, dimensions: List[str]) -> str:
    dims = "\n".join(f"- {d}" for d in dimensions)
    return (
        "You are an expert dashboard-design reviewer. Rate the RECOMMENDATION for "
        "the given BRIEF on each dimension from 1 (poor) to 5 (excellent). Think "
        "briefly, then output ONLY a JSON object mapping each dimension to an "
        "integer 1-5.\n\n"
        f"Dimensions:\n{dims}\n\n"
        f"BRIEF:\n{brief_text}\n\n"
        f"RECOMMENDATION:\n{output_text}\n\n"
        "JSON scores:"
    )


def parse_judge_scores(text: str, dimensions: List[str]) -> Optional[Dict[str, int]]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    scores = {}
    for d in dimensions:
        if d in obj:
            try:
                scores[d] = max(1, min(5, int(round(float(obj[d])))))
            except (TypeError, ValueError):
                continue
    return scores or None


def _openai_judge_fn(model: str):
    """Build a judge callable hitting an OpenAI-compatible chat endpoint."""
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return None

    def judge(brief_text: str, output_text: str, dimensions: List[str]) -> Optional[Dict[str, int]]:
        prompt = build_judge_prompt(brief_text, output_text, dimensions)
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return None
        return parse_judge_scores(content, dimensions)

    return judge


def _brief_to_text(brief: dict) -> str:
    if not brief:
        return "(brief unavailable)"
    goals = ", ".join(brief.get("goals", []) or [])
    kpis = ", ".join(brief.get("kpis", []) or [])
    return f"Users: {brief.get('users','')}\nGoals: {goals}\nKPIs: {kpis}\nConstraints: {brief.get('constraints')}"


@METRICS.register("g_eval")
class GEval(BaseMetric):
    name = "g_eval"

    def __init__(self, cfg: Any = None, judge_fn: Optional[Callable] = None) -> None:
        super().__init__(cfg)
        jc = _get(_get(cfg, "eval", {}) if cfg else {}, "judge", {}) or {}
        self.dimensions = list(_get(jc, "dimensions", RUBRIC_KEYS))
        self.model = str(_get(jc, "model", "gpt-4o-mini"))
        self._judge_fn = judge_fn or _openai_judge_fn(self.model)

    def compute(self, results, references=None) -> dict:
        if self._judge_fn is None:
            return {"available": False, "reason": "no judge configured (set OPENAI_API_KEY or pass judge_fn)"}

        ref_by_id = index_references(references or [])
        per_dim: Dict[str, List[int]] = {d: [] for d in self.dimensions}
        n = 0
        for r in results:
            ref = ref_by_id.get(r.item_id, {})
            brief_text = _brief_to_text(ref.get("brief", {}))
            output_text = r.raw_text or ""
            scores = self._judge_fn(brief_text, output_text, self.dimensions)
            if not scores:
                continue
            n += 1
            for d, v in scores.items():
                per_dim[d].append(v)

        if n == 0:
            return {"available": True, "n": 0, "per_dimension": {}, "overall_mean": None}
        per_dim_mean = {d: round(statistics.mean(v), 3) for d, v in per_dim.items() if v}
        all_scores = [v for vs in per_dim.values() for v in vs]
        return {
            "available": True,
            "n": n,
            "per_dimension": per_dim_mean,
            "overall_mean": round(statistics.mean(all_scores), 3) if all_scores else None,
        }
