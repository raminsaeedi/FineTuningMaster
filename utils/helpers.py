"""
utils/helpers.py
================
Shared utility functions for the fine-tuning pipeline.
Covers: config loading, prompt formatting, JSON parsing, logging setup.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ============================================================
# Logging Setup
# ============================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure root logger with console (and optional file) handler.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR
        log_file:  Optional path to write logs to a file

    Returns:
        Configured logger instance
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================
# Config Loading
# ============================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Dictionary with all configuration values
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from: {config_path}")
    return config


# ============================================================
# Prompt Formatting
# ============================================================

SYSTEM_PROMPT = """You are an expert dashboard design consultant. 
Given a dashboard brief, you generate structured, professional design recommendations.
Always respond with valid JSON following the exact schema provided."""

def format_instruction_prompt(brief: Dict[str, Any], tokenizer=None) -> str:
    """
    Format a dashboard brief into a chat-style instruction prompt.

    This uses the model's built-in chat template if a tokenizer is provided,
    otherwise falls back to a simple text format.

    Args:
        brief:     Dictionary with dashboard brief fields
        tokenizer: HuggingFace tokenizer (optional, for apply_chat_template)

    Returns:
        Formatted prompt string
    """
    user_message = _build_user_message(brief)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
        # add_generation_prompt=True adds the assistant turn opener
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    # Fallback: simple text format
    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### User:\n{user_message}\n\n"
        f"### Assistant:\n"
    )


def format_training_example(brief: Dict[str, Any],
                             recommendation: Dict[str, Any],
                             tokenizer=None) -> str:
    """
    Format a complete training example (prompt + expected output).

    Args:
        brief:          Dashboard brief dictionary
        recommendation: Expected structured recommendation dictionary
        tokenizer:      HuggingFace tokenizer (optional)

    Returns:
        Full training text (prompt + JSON response)
    """
    prompt = format_instruction_prompt(brief, tokenizer)
    response_json = json.dumps(recommendation, ensure_ascii=False, indent=2)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        # For chat-template models, the response is appended after the prompt
        return prompt + response_json + tokenizer.eos_token

    return prompt + response_json + "\n"


def _build_user_message(brief: Dict[str, Any]) -> str:
    """Build the user message text from a brief dictionary."""
    lines = [
        "Please generate a structured dashboard design recommendation for the following brief:",
        "",
        f"**Dashboard Title:** {brief.get('title', 'N/A')}",
        f"**Target Audience:** {brief.get('target_audience', 'N/A')}",
        f"**Business Goals:** {brief.get('business_goals', 'N/A')}",
        f"**KPIs:** {', '.join(brief.get('kpis', []))}",
        f"**Data Context:** {brief.get('data_context', 'N/A')}",
        f"**Update Frequency:** {brief.get('update_frequency', 'N/A')}",
        f"**User Expertise:** {brief.get('user_expertise', 'N/A')}",
        "",
        "Respond ONLY with a valid JSON object containing these exact keys:",
        "  1. context_summary",
        "  2. kpi_task_chart_mapping",
        "  3. layout_hierarchy",
        "  4. labels_scales_colors",
        "  5. interactions",
        "  6. design_rationales",
    ]
    return "\n".join(lines)


# ============================================================
# JSON Parsing & Validation
# ============================================================

REQUIRED_KEYS = {
    "context_summary",
    "kpi_task_chart_mapping",
    "layout_hierarchy",
    "labels_scales_colors",
    "interactions",
    "design_rationales",
}


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse a JSON object from model output text.

    Handles cases where the model wraps JSON in markdown code blocks
    or adds extra text before/after the JSON.

    Args:
        text: Raw model output string

    Returns:
        Parsed dictionary or None if parsing fails
    """
    # 1. Try to find JSON inside ```json ... ``` blocks
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        candidate = code_block_match.group(1)
        result = _try_parse_json(candidate)
        if result is not None:
            return result

    # 2. Try to find the first { ... } block in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        result = _try_parse_json(candidate)
        if result is not None:
            return result

    # 3. Try parsing the whole text directly
    return _try_parse_json(text.strip())


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt JSON parsing, return None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def validate_recommendation(rec: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that a recommendation contains all required keys.

    Args:
        rec: Parsed recommendation dictionary

    Returns:
        (is_valid, list_of_missing_keys)
    """
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    return len(missing) == 0, missing


# ============================================================
# File I/O Helpers
# ============================================================

def load_jsonl(filepath: str) -> list[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.

    Args:
        filepath: Path to .jsonl file

    Returns:
        List of parsed JSON objects
    """
    records = []
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(records)} records from {filepath}")
    return records


def save_jsonl(records: list[Dict[str, Any]], filepath: str) -> None:
    """
    Save a list of dictionaries to a JSONL file.

    Args:
        records:  List of dictionaries to save
        filepath: Output path for .jsonl file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(records)} records to {filepath}")


def ensure_dirs(config: Dict[str, Any]) -> None:
    """
    Create all output directories defined in config['paths'].

    Args:
        config: Loaded configuration dictionary
    """
    paths = config.get("paths", {})
    for key, path_str in paths.items():
        Path(path_str).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path_str}")


# ============================================================
# Device Detection
# ============================================================

def get_device_info() -> Dict[str, Any]:
    """
    Detect available compute device and return info dict.

    Returns:
        Dictionary with device type, name, and VRAM info
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {"device": device, "name": device_name, "vram_gb": round(vram_gb, 1)}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"device": "mps", "name": "Apple Silicon", "vram_gb": None}
        else:
            return {"device": "cpu", "name": "CPU", "vram_gb": None}
    except ImportError:
        return {"device": "unknown", "name": "PyTorch not installed", "vram_gb": None}


def print_device_info() -> None:
    """Print device information to console."""
    info = get_device_info()
    print("\n" + "=" * 50)
    print("  COMPUTE DEVICE INFORMATION")
    print("=" * 50)
    print(f"  Device : {info['device'].upper()}")
    print(f"  Name   : {info['name']}")
    if info["vram_gb"]:
        print(f"  VRAM   : {info['vram_gb']} GB")
    print("=" * 50 + "\n")
