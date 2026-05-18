"""
pipeline/inference.py — Stage 3: Run inference for base and/or fine-tuned model.

Saves per-example predictions to:
  {experiment_dir}/predictions/base_model.jsonl
  {experiment_dir}/predictions/finetuned_model.jsonl

Each line records: brief_id, variant, raw_output, parsed JSON, validity,
completeness, predicted chart types, reference chart types, and latency.

Usage:
    python pipeline/inference.py \\
        --experiment-dir outputs/experiments/qlora_qwen05b_20250513_143022_a3f1 \\
        [--mode base|finetuned|both]    (default: both) \\
        [--test-file data/test.jsonl]   (overrides config) \\
        [--max-new-tokens 1024]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.experiment import load_experiment_config, save_metrics
from utils.helpers import (
    SYSTEM_PROMPT,
    extract_json_from_text,
    format_instruction_prompt,
    load_jsonl,
    print_device_info,
    save_jsonl,
    setup_logging,
    validate_recommendation,
)

logger = setup_logging()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(config: dict):
    """Load the base model without any fine-tuning adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model"]["name"]
    cache_dir = config["model"].get("cache_dir")
    torch_dtype_str = config["model"].get("torch_dtype", "float16")
    dtype = torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float16

    logger.info(f"Loading base model (no adapter): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True,
        dtype=dtype, cache_dir=cache_dir,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(experiment_dir: Path, config: dict):
    """Load the fine-tuned model from experiment_dir/final_adapter/.

    Branches on the presence of adapter_config.json:
      - Present  → PEFT/LoRA model  (QLoRA, DoRA, RSLoRA, LoftQ, ORPO+LoRA)
      - Absent   → full model save  (GaLore — trained all weights, no adapter)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = experiment_dir / "final_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter / model weights not found at {adapter_dir}.\n"
            "Run pipeline/train.py first."
        )

    # Read base model name from training metadata if available
    meta_path = adapter_dir / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_name = meta.get("base_model", config["model"]["name"])
    else:
        model_name = config["model"]["name"]

    cache_dir = config["model"].get("cache_dir")
    torch_dtype_str = config["model"].get("torch_dtype", "float16")
    dtype = torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.exists():
        # LoRA-based algorithms: QLoRA, DoRA, RSLoRA, LoftQ, ORPO+LoRA
        from peft import PeftModel
        logger.info(
            f"PEFT adapter detected — loading base model + LoRA: "
            f"base={model_name}, adapter={adapter_dir}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
            dtype=dtype, cache_dir=cache_dir,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    else:
        # Full-model algorithms: GaLore (saves all weights directly)
        logger.info(
            f"No adapter_config.json — loading full model weights (GaLore): "
            f"{adapter_dir}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(adapter_dir), device_map="auto", trust_remote_code=True,
            dtype=dtype,
        )

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def _extract_chart_types(parsed: dict | None) -> list[str]:
    """Extract recommended chart type strings from kpi_task_chart_mapping."""
    if not parsed:
        return []
    mapping = parsed.get("kpi_task_chart_mapping", [])
    if not isinstance(mapping, list):
        return []
    return [
        str(entry.get("recommended_chart", "")).strip()
        for entry in mapping
        if isinstance(entry, dict) and entry.get("recommended_chart")
    ]


def _format_prompt_with_rag(brief: dict, tokenizer, rag_context: str) -> str:
    """
    Build instruction prompt with RAG context injected into the system message.

    The retrieved guidelines are appended to the base system prompt so the
    model can use domain knowledge when generating recommendations.
    """
    from utils.helpers import _build_user_message
    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Relevant Design Guidelines ---\n{rag_context}\n"
        f"--- End of Guidelines ---"
    )
    user_message = _build_user_message(brief)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        f"### System:\n{system}\n\n"
        f"### User:\n{user_message}\n\n"
        f"### Assistant:\n"
    )


def generate_prediction(
    brief: dict,
    model,
    tokenizer,
    config: dict,
    brief_id: int = 0,
    variant: str = "original",
    reference_chart_types: list[str] | None = None,
    rag_context: str | None = None,
) -> dict:
    """
    Run inference for a single brief and return a structured result row.

    The returned dict is one line of the predictions/*.jsonl files:
      brief_id, variant, raw_output, parsed, valid_json, valid_schema,
      completeness, chart_types_predicted, chart_types_reference, latency_s

    If rag_context is provided (non-empty string), it is injected into the
    system prompt before tokenization.
    """
    import torch

    inf_cfg = config.get("inference", {})
    max_seq = config["model"].get("max_seq_length", 2048)
    max_new = inf_cfg.get("max_new_tokens", 1024)

    if rag_context:
        prompt = _format_prompt_with_rag(brief, tokenizer, rag_context)
    else:
        prompt = format_instruction_prompt(brief, tokenizer)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max_seq - max_new,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=inf_cfg.get("temperature", 0.1),
            top_p=inf_cfg.get("top_p", 0.9),
            do_sample=inf_cfg.get("do_sample", True),
            repetition_penalty=inf_cfg.get("repetition_penalty", 1.15),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = round(time.time() - t0, 3)

    generated_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    parsed = extract_json_from_text(raw_output)
    valid_json = parsed is not None
    valid_schema, missing_keys = validate_recommendation(parsed) if parsed else (False, [])

    required_fields = config.get("evaluation", {}).get("required_fields", list())
    completeness = (
        (len(required_fields) - len(missing_keys)) / len(required_fields)
        if required_fields else (1.0 if valid_schema else 0.0)
    )

    chart_types_predicted = _extract_chart_types(parsed)

    return {
        "brief_id": brief_id,
        "variant": variant,
        "raw_output": raw_output,
        "parsed": parsed,
        "valid_json": valid_json,
        "valid_schema": valid_schema,
        "completeness": round(completeness, 4),
        "missing_keys": missing_keys,
        "chart_types_predicted": chart_types_predicted,
        "chart_types_reference": reference_chart_types or [],
        "latency_s": latency,
    }


def run_inference_loop(
    test_data: list[dict],
    model,
    tokenizer,
    config: dict,
    variant: str = "original",
) -> list[dict]:
    """Run generate_prediction for every example in test_data.

    When config["rag"]["enabled"] is True, retrieves relevant guideline
    context for each brief and injects it into the system prompt.
    """
    # ── RAG setup (optional) ──────────────────────────────────────────────────
    rag_cfg     = config.get("rag", {})
    rag_enabled = rag_cfg.get("enabled", False)
    retriever   = None

    if rag_enabled:
        try:
            from rag import RAGRetriever
            retriever = RAGRetriever(top_k=rag_cfg.get("top_k", 3))
            if not retriever.is_ready():
                logger.warning(
                    "RAG is enabled but the index is missing. "
                    "Run `python -m rag.indexer` to build it. "
                    "Proceeding without RAG context."
                )
                retriever = None
            else:
                logger.info(
                    f"RAG enabled — top_k={rag_cfg.get('top_k', 3)}, "
                    f"index loaded from rag/index/"
                )
        except Exception as exc:
            logger.warning(f"RAG initialisation failed ({exc}); proceeding without RAG.")
            retriever = None

    # ── Inference loop ────────────────────────────────────────────────────────
    results = []
    n = len(test_data)
    for i, item in enumerate(test_data):
        logger.info(f"  [{i+1}/{n}] brief_id={i} variant={variant}")

        # Extract reference chart types if present in the ground truth
        rec        = item.get("recommendation", {})
        ref_charts = _extract_chart_types(rec)

        # Build RAG context string for this brief (if enabled and index ready)
        rag_context: str | None = None
        if retriever is not None:
            brief = item.get("brief", {})
            query = " ".join(filter(None, [
                str(brief.get("title", "")),
                str(brief.get("business_goals", "")),
                " ".join(brief.get("kpis", [])),
            ]))
            try:
                rag_context = retriever.retrieve(query)
            except Exception as exc:
                logger.debug(f"RAG retrieve failed for brief {i}: {exc}")

        result = generate_prediction(
            brief=item["brief"],
            model=model,
            tokenizer=tokenizer,
            config=config,
            brief_id=i,
            variant=variant,
            reference_chart_types=ref_charts,
            rag_context=rag_context,
        )
        results.append(result)

        status = "valid" if result["valid_schema"] else f"missing={result['missing_keys']}"
        rag_tag = " [RAG]" if rag_context else ""
        print(f"  [{i+1:2d}/{n}] {status:<40} {result['latency_s']}s{rag_tag}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 — run inference")
    p.add_argument("--experiment-dir", required=True,
                   help="Path to a completed experiment directory")
    p.add_argument("--mode", choices=["base", "finetuned", "both"], default="both")
    p.add_argument("--test-file", default=None,
                   help="Override test file from config")
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    config = load_experiment_config(experiment_dir)

    if args.max_new_tokens:
        config.setdefault("inference", {})["max_new_tokens"] = args.max_new_tokens

    test_file = args.test_file or config["data"].get("test_file", "data/test.jsonl")
    if not Path(test_file).exists():
        # Try relative to project root
        test_file = str(_PROJECT_ROOT / test_file)
    test_data = load_jsonl(test_file)

    print_device_info()

    predictions_dir = experiment_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    modes_to_run = ["base", "finetuned"] if args.mode == "both" else [args.mode]

    for mode in modes_to_run:
        logger.info(f"\n{'='*60}\nRunning inference: mode={mode}\n{'='*60}")
        if mode == "base":
            model, tokenizer = load_base_model(config)
        else:
            model, tokenizer = load_finetuned_model(experiment_dir, config)

        results = run_inference_loop(test_data, model, tokenizer, config, variant="original")

        out_file = predictions_dir / f"{mode}_model.jsonl"
        save_jsonl(results, str(out_file))
        logger.info(f"Predictions saved: {out_file}")

        # Free GPU memory between base and finetuned runs
        del model
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\nInference complete.")
    print(f"  Predictions in: {predictions_dir}")
    print(f"\nNext step:")
    print(f"  python pipeline/evaluate.py --experiment-dir {experiment_dir}")


if __name__ == "__main__":
    main()
