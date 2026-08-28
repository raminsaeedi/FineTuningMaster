"""Focused public-interface tests for supervised RAG retrieval metrics."""

import json

import pytest
import yaml

from src.core.schemas import GenerationResult


def _prediction(item_id: str, chunk_ids: list[str]) -> GenerationResult:
    return GenerationResult(
        item_id=item_id,
        method_name="rag",
        model_name="test-model",
        retrieved_docs=[{"id": chunk_id} for chunk_id in chunk_ids],
    )


def _write_qrels(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_retrieval_relevance_is_explicitly_unavailable_without_qrels():
    from src.evaluation.metrics.retrieval_relevance import RetrievalRelevance

    result = RetrievalRelevance(cfg={"eval": {}}).compute([], [])

    assert result["available"] is False
    assert result["reason"] == "eval.retrieval_qrels_path is not configured"
    assert result["recall_at_3"] is None
    assert result["mrr_at_3"] is None
    assert result["ndcg_at_3"] is None


def test_non_rag_is_not_applicable_but_rag_with_zero_docs_is_scored(tmp_path):
    from omegaconf import OmegaConf

    from src.evaluation.metrics.retrieval_relevance import RetrievalRelevance

    qrels_path = tmp_path / "qrels.jsonl"
    _write_qrels(qrels_path, [
        {"item_id": "q1", "relevant_chunks": [{"chunk_id": "gold", "relevance": 2}]},
    ])
    eval_cfg = {"retrieval_qrels_path": str(qrels_path)}

    non_rag_results = [
        RetrievalRelevance(
            cfg=OmegaConf.create({"eval": eval_cfg, "method": {"name": method_name}})
        ).compute([], [])
        for method_name in ("prompt_only", "ft")
    ]
    rag_zero_docs = RetrievalRelevance(
        cfg=OmegaConf.create({
            "eval": eval_cfg,
            "method": {"name": "rag", "retriever": {"name": "tfidf"}},
        })
    ).compute([_prediction("q1", [])], [])

    for non_rag in non_rag_results:
        assert non_rag["available"] is False
        assert non_rag["applicable"] is False
        assert non_rag["reason"] == "not_applicable: cfg.method.retriever is not configured"
        assert non_rag["recall_at_3"] is None

    assert rag_zero_docs["available"] is True
    assert rag_zero_docs["applicable"] is True
    assert rag_zero_docs["recall_at_3"] == 0.0
    assert rag_zero_docs["mrr_at_3"] == 0.0
    assert rag_zero_docs["ndcg_at_3"] == 0.0
    assert rag_zero_docs["query_coverage"] == 0.0
    assert rag_zero_docs["n_predictions"] == 1


def test_retrieval_relevance_scores_ranked_hits_and_keeps_missing_queries(tmp_path):
    from src.evaluation.metrics.retrieval_relevance import RetrievalRelevance

    qrels_path = tmp_path / "qrels.jsonl"
    _write_qrels(qrels_path, [
        {
            "item_id": "q1",
            "relevant_chunks": [
                {"chunk_id": "high", "relevance": 2},
                {"chunk_id": "low", "relevance": 1},
            ],
        },
        {"item_id": "q2", "relevant_chunks": [{"chunk_id": "only", "relevance": 2}]},
        {"item_id": "q3", "relevant_chunks": [{"chunk_id": "other", "relevance": 1}]},
    ])
    metric = RetrievalRelevance(cfg={"eval": {"retrieval_qrels_path": str(qrels_path)}})

    result = metric.compute([
        _prediction("q1", ["irrelevant", "low", "high"]),
        _prediction("q3", []),
    ], [])

    assert result["available"] is True
    assert result["recall_at_3"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["mrr_at_3"] == pytest.approx(1 / 6, abs=1e-6)
    assert result["ndcg_at_3"] == pytest.approx(0.1956, abs=1e-4)
    assert result["query_coverage"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["n_qrels"] == 3
    assert result["n_predictions"] == 2
    assert result["n_queries_with_hits"] == 1
    assert result["n_missing_predictions"] == 1
    assert result["confidence_intervals"]["recall_at_3"]["n"] == 3
    assert result["confidence_intervals"]["mrr_at_3"]["n_boot"] == 10_000
    assert result["confidence_intervals"]["ndcg_at_3"]["ci_level"] == 0.95
    assert result["top_3_retrieval_support_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["n_with_3_unique_retrieved_ids"] == 1


def test_retrieval_relevance_never_infers_labels_from_text_overlap(tmp_path):
    from src.evaluation.metrics.retrieval_relevance import RetrievalRelevance

    qrels_path = tmp_path / "qrels.jsonl"
    _write_qrels(qrels_path, [
        {"item_id": "q1", "relevant_chunks": [{"chunk_id": "gold", "relevance": 2}]},
    ])
    prediction = GenerationResult(
        item_id="q1",
        method_name="rag",
        model_name="test-model",
        retrieved_docs=[{"id": "wrong", "text": "exactly the same words as the gold passage"}],
    )

    result = RetrievalRelevance(
        cfg={"eval": {"retrieval_qrels_path": str(qrels_path)}}
    ).compute([prediction], [])

    assert result["recall_at_3"] == 0.0
    assert result["mrr_at_3"] == 0.0
    assert result["ndcg_at_3"] == 0.0


def test_retrieval_relevance_preserves_rank_when_a_hit_has_no_id(tmp_path):
    from src.evaluation.metrics.retrieval_relevance import RetrievalRelevance

    qrels_path = tmp_path / "qrels.jsonl"
    _write_qrels(qrels_path, [
        {"item_id": "q1", "relevant_chunks": [{"chunk_id": "gold", "relevance": 1}]},
    ])
    prediction = GenerationResult(
        item_id="q1",
        method_name="rag",
        model_name="test-model",
        retrieved_docs=[{"text": "missing ID"}, {"id": "gold"}],
    )

    result = RetrievalRelevance(
        cfg={"eval": {"retrieval_qrels_path": str(qrels_path)}}
    ).compute([prediction], [])

    assert result["mrr_at_3"] == 0.5
    assert result["ndcg_at_3"] == pytest.approx(0.63093, abs=1e-5)
    assert result["top_3_retrieval_support_rate"] == 0.0


def test_retrieval_relevance_duplicates_consume_rank_without_extra_credit(tmp_path):
    from src.evaluation.metrics.retrieval_relevance import RetrievalRelevance

    qrels_path = tmp_path / "qrels.jsonl"
    _write_qrels(qrels_path, [{
        "item_id": "q1",
        "relevant_chunks": [
            {"chunk_id": "a", "relevance": 2},
            {"chunk_id": "b", "relevance": 1},
        ],
    }])

    result = RetrievalRelevance(
        cfg={"eval": {"retrieval_qrels_path": str(qrels_path)}}
    ).compute([_prediction("q1", ["a", "a", "wrong"])], [])

    assert result["recall_at_3"] == 0.5
    assert result["mrr_at_3"] == 1.0
    assert result["ndcg_at_3"] == pytest.approx(0.826235, abs=1e-6)
    assert result["n_with_3_unique_retrieved_ids"] == 0


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"item_id": "q1", "relevant_chunks": [{"chunk_id": "a", "relevance": 1}]},
                {"item_id": "q1", "relevant_chunks": [{"chunk_id": "b", "relevance": 1}]},
            ],
            "duplicate item_id 'q1' at line 2",
        ),
        (
            [{
                "item_id": "q1",
                "relevant_chunks": [
                    {"chunk_id": "a", "relevance": 2},
                    {"chunk_id": "a", "relevance": 1},
                ],
            }],
            "duplicate chunk_id 'a' for item_id 'q1' at line 1",
        ),
        (
            [{"item_id": "q1", "relevant_chunks": [{"chunk_id": "a", "relevance": 0}]}],
            "relevance must be integer 1 or 2 at line 1",
        ),
        (
            [{"item_id": "q1", "relevant_chunks": [{"chunk_id": "a", "relevance": 3}]}],
            "relevance must be integer 1 or 2 at line 1",
        ),
        (
            [{"item_id": "q1", "relevant_chunks": [{"chunk_id": "a", "relevance": 1.0}]}],
            "relevance must be integer 1 or 2 at line 1",
        ),
        (
            [{"item_id": "q1", "relevant_chunks": []}],
            "relevant_chunks must be a non-empty list at line 1",
        ),
    ],
)
def test_retrieval_relevance_rejects_invalid_qrels_clearly(tmp_path, rows, message):
    from src.evaluation.metrics.retrieval_relevance import RetrievalRelevance

    qrels_path = tmp_path / "qrels.jsonl"
    _write_qrels(qrels_path, rows)
    metric = RetrievalRelevance(cfg={"eval": {"retrieval_qrels_path": str(qrels_path)}})

    with pytest.raises(ValueError, match=message):
        metric.compute([], [])


def test_retrieval_relevance_is_registered_in_optional_full_profile():
    import src.evaluation.metrics as metrics

    from src.core.registry import METRICS

    with open("src/config/eval/full.yaml", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)

    assert "retrieval_relevance" in METRICS
    assert "retrieval_relevance" in metrics.__all__
    assert "retrieval_relevance" in profile["metrics"]
    assert profile["retrieval_qrels_path"] is None
