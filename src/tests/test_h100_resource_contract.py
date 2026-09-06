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
    assert "FTM_AUTO_RESUME_ATTEMPTS" in launcher


def test_c_and_d_use_4bit_inference_without_changing_model_profile():
    for method in ("ft", "ft_rag"):
        profile = yaml.safe_load(
            (ROOT / "src" / "config" / "method" / f"{method}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert profile["inference"] == {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        assert profile["allow_training_config_mismatch"] is False

    model_profile = yaml.safe_load(
        (ROOT / "src" / "config" / "model" / "qwen3_8_27b.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "inference" not in model_profile
    assert "inference_load_in_4bit" not in model_profile


def test_h100_inference_mode_requires_adapter_and_skips_training():
    launcher = (ROOT / "run_h100_qwen3_8_27b.sh").read_text(encoding="utf-8")
    assert "--inference" in launcher
    assert "--mode inference" in launcher
    assert 'DEFAULT_METHODS="C D"' in launcher
    assert 'DEFAULT_ROBUSTNESS=0' in launcher
    assert "adapter_config.json" in launcher
    assert "adapter_model.safetensors" in launcher
    assert '--input-model-weights "${ADAPTER_DIR}"' in launcher
    assert '"model.max_seq_length=${INFERENCE_MAX_SEQ_LENGTH}"' in launcher
    assert '"method.generate.max_new_tokens=${INFERENCE_MAX_NEW_TOKENS}"' in launcher
    assert '"method.allow_training_config_mismatch=true"' in launcher
    assert '"method.allow_inference_context_length_mismatch=true"' in launcher
