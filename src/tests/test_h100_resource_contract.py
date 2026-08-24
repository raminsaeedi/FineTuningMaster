"""Resource contract for the direct 27B H100 launcher."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("prompt_only", "rag", "ft", "ft_rag")


def test_all_comparison_methods_use_same_512_output_token_limit():
    limits = {}
    for method in METHODS:
        profile = yaml.safe_load(
            (ROOT / "src" / "config" / "method" / f"{method}.yaml").read_text(
                encoding="utf-8"
            )
        )
        limits[method] = profile["generate"]["max_new_tokens"]
    assert limits == {method: 512 for method in METHODS}


def test_h100_launcher_supports_method_and_robustness_selection():
    launcher = (ROOT / "run_h100_qwen3_8_27b.sh").read_text(encoding="utf-8")
    assert "FTM_METHODS" in launcher
    assert "FTM_ROBUSTNESS" in launcher
    assert "--no-paraphrased" in launcher
    assert "--no-missing-info" in launcher
