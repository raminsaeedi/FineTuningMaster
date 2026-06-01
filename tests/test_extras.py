"""Tests for the optional extras: DoRA/RSLoRA flags, GaLore, dense retriever,
constrained-decoding schema, and the G-Eval judge metric.

These verify the logic without the heavy optional dependencies (peft/outlines/
sentence-transformers/galore-torch/API), by checking config assembly, registry
wiring, and injecting stubs where a model/embedder/judge would be needed.
"""

from src.core.schemas import DesignOutput, GenerationResult


# ---- DoRA / RSLoRA via flags -------------------------------------------------

def test_build_lora_kwargs_flags():
    from src.training.sft_trainer import build_lora_kwargs

    kw = build_lora_kwargs({"r": 8, "lora_alpha": 16, "use_dora": True,
                            "target_modules": ["q_proj", "v_proj"]})
    assert kw["r"] == 8 and kw["use_dora"] is True and kw["use_rslora"] is False
    assert kw["target_modules"] == ["q_proj", "v_proj"]


def test_trainers_registered():
    import src.training  # noqa: F401

    from src.core.registry import TRAINERS

    assert {"qlora_sft", "galore_sft"} <= set(TRAINERS.keys())


# ---- Dense retriever (stub embedder, no download) ----------------------------

def test_dense_retriever_ranks_with_stub_embedder():
    import numpy as np

    from src.retrievers.dense import DenseRetriever

    class StubEmbedder:
        # Map texts to 2-D vectors by keyword so ranking is deterministic.
        def encode(self, texts, normalize_embeddings=True):
            vecs = []
            for t in texts:
                v = np.array([1.0 if "line" in t.lower() else 0.0,
                              1.0 if "color" in t.lower() else 0.0])
                n = np.linalg.norm(v) or 1.0
                vecs.append(v / n)
            return np.array(vecs)

    r = DenseRetriever({"top_k": 1})
    r.chunks = [
        {"id": "a", "text": "use a line chart for trends"},
        {"id": "b", "text": "ensure color contrast for accessibility"},
    ]
    r.embedder = StubEmbedder()
    r.setup()
    hits = r.retrieve("line chart over time", k=1)
    assert hits and hits[0]["id"] == "a"


def test_dense_retriever_registered():
    import src.retrievers  # noqa: F401

    from src.core.registry import RETRIEVERS

    assert {"tfidf", "dense"} <= set(RETRIEVERS.keys())


# ---- Constrained decoding schema (no outlines needed) ------------------------

def test_design_output_json_schema():
    from src.inference.decoders import design_output_json_schema

    schema = design_output_json_schema()
    assert schema["type"] == "object"
    assert "kpi_chart_mapping" in schema["properties"]


# ---- G-Eval (stub judge) -----------------------------------------------------

def _result(item_id: str) -> GenerationResult:
    return GenerationResult(item_id=item_id, method_name="m", model_name="x",
                            raw_text='{"context_summary": {}}', parsed=DesignOutput())


def test_g_eval_with_stub_judge():
    from src.evaluation.metrics.llm_judge import GEval

    def stub_judge(brief_text, output_text, dimensions):
        return {d: 4 for d in dimensions}

    refs = [{"item_id": "a", "brief": {"users": "U", "goals": ["g"], "kpis": ["Revenue"]}}]
    metric = GEval(cfg=None, judge_fn=stub_judge)
    out = metric.compute([_result("a")], refs)
    assert out["available"] is True and out["n"] == 1
    assert out["overall_mean"] == 4.0


def test_g_eval_without_judge_is_unavailable():
    from src.evaluation.metrics.llm_judge import GEval

    metric = GEval(cfg=None, judge_fn=None)
    metric._judge_fn = None  # ensure no env-based judge picked up
    out = metric.compute([_result("a")], [{"item_id": "a", "brief": {}}])
    assert out["available"] is False


def test_parse_judge_scores():
    from src.evaluation.metrics.llm_judge import parse_judge_scores

    txt = 'reasoning... {"chart_appropriateness": 5, "overall_usefulness": "3"}'
    scores = parse_judge_scores(txt, ["chart_appropriateness", "overall_usefulness", "layout_quality"])
    assert scores["chart_appropriateness"] == 5 and scores["overall_usefulness"] == 3
    assert "layout_quality" not in scores
