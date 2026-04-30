"""
scripts/01_check_environment.py
================================
Checks that your environment is ready for fine-tuning.

Checks performed:
    1. Python version (3.10+ required)
    2. PyTorch installation and version
    3. CUDA availability and GPU name
    4. Library versions: transformers, datasets, peft, trl, accelerate
    5. Folder existence: data/processed/ and outputs/models/

Output format:
    [OK]      Everything is fine
    [WARNING] Works but with limitations (e.g. no GPU)
    [MISSING] Library not installed – shows install command
    [ERROR]   Critical problem that must be fixed

Final recommendation:
    - CUDA available  -> "You can try local fine-tuning."
    - No CUDA         -> "Use Google Colab for training; local machine can
                          still run dataset preparation and small tests."

Usage:
    python scripts/01_check_environment.py
"""

import os
import sys
from pathlib import Path

# Project root = one level above this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Formatting helpers
# ============================================================

def ok(label: str, detail: str = "") -> None:
    msg = f"[OK]      {label}"
    if detail:
        msg += f": {detail}"
    print(msg)


def warn(label: str, detail: str = "") -> None:
    msg = f"[WARNING] {label}"
    if detail:
        msg += f". {detail}"
    print(msg)


def missing(label: str, install_cmd: str = "") -> None:
    msg = f"[MISSING] {label} is not installed."
    if install_cmd:
        msg += f"\n          Fix: {install_cmd}"
    print(msg)


def error(label: str, detail: str = "") -> None:
    msg = f"[ERROR]   {label}"
    if detail:
        msg += f": {detail}"
    print(msg)


def section(title: str) -> None:
    print()
    print(f"--- {title} ---")


# ============================================================
# Individual checks
# ============================================================

def check_python() -> bool:
    """Check Python version. Requires 3.10+."""
    section("Python")
    version_str = sys.version.split()[0]
    major = sys.version_info.major
    minor = sys.version_info.minor

    if major == 3 and minor >= 10:
        ok("Python version", version_str)
        return True
    else:
        error(
            "Python version",
            f"{version_str} detected – Python 3.10 or 3.11 is required.\n"
            "          Download: https://www.python.org/downloads/"
        )
        return False


def check_pytorch() -> tuple:
    """
    Check PyTorch installation.

    Returns:
        (torch_ok: bool, cuda_available: bool)
    """
    section("PyTorch & CUDA")

    try:
        import torch
        ok("PyTorch", torch.__version__)
    except ImportError:
        missing(
            "PyTorch",
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        warn("CUDA check skipped", "PyTorch must be installed first")
        return False, False

    # CUDA check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_ver = torch.version.cuda or "unknown"
        ok("CUDA available", f"version {cuda_ver}")
        ok("GPU detected",   f"{gpu_name}  (VRAM: {vram_gb:.1f} GB)")

        if vram_gb < 6:
            warn(
                "Low VRAM detected",
                f"Only {vram_gb:.1f} GB available. "
                "Set batch_size: 1 and load_in_4bit: true in config/train_config.yaml"
            )
        return True, True

    # Apple Silicon MPS
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        ok("MPS (Apple Silicon)", "GPU acceleration available via Metal")
        warn(
            "bitsandbytes not supported on Apple Silicon",
            "Set load_in_4bit: false in config/train_config.yaml"
        )
        return True, False  # Has acceleration but not CUDA

    # CPU only
    else:
        warn(
            "CUDA not available",
            "Training will be very slow on CPU. "
            "Use Google Colab for training (free T4 GPU)."
        )
        return True, False


def check_libraries() -> dict:
    """
    Check all required ML libraries.

    Returns:
        Dict mapping library name -> (installed: bool, version: str)
    """
    section("Required Libraries")

    # (display_name, import_name, pip_name)
    libraries = [
        ("transformers", "transformers", "transformers>=4.40.0"),
        ("datasets",     "datasets",     "datasets>=2.18.0"),
        ("peft",         "peft",         "peft>=0.10.0"),
        ("trl",          "trl",          "trl>=0.8.6"),
        ("accelerate",   "accelerate",   "accelerate>=0.29.0"),
        ("bitsandbytes", "bitsandbytes", "bitsandbytes>=0.43.0"),
        ("pyyaml",       "yaml",         "pyyaml>=6.0"),
        ("jsonlines",    "jsonlines",    "jsonlines>=4.0.0"),
    ]

    results = {}
    for display, import_name, pip_name in libraries:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "installed")
            ok(display, version)
            results[display] = (True, version)
        except ImportError:
            missing(display, f"pip install {pip_name}")
            results[display] = (False, "")

    return results


def check_folders() -> dict:
    """
    Check that required project folders exist.

    Returns:
        Dict mapping folder path -> exists (bool)
    """
    section("Project Folders")

    folders = {
        "data/processed": PROJECT_ROOT / "data" / "processed",
        "data/raw":       PROJECT_ROOT / "data" / "raw",
        "outputs/models": PROJECT_ROOT / "outputs" / "models",
    }

    results = {}
    for label, path in folders.items():
        if path.exists():
            # Count files if any
            files = list(path.iterdir())
            detail = f"exists ({len(files)} file(s))" if files else "exists (empty)"
            ok(label, detail)
            results[label] = True
        else:
            warn(
                f"{label} does not exist yet",
                f"It will be created automatically when you run the relevant script"
            )
            results[label] = False

    return results


# ============================================================
# Final recommendation
# ============================================================

def print_recommendation(cuda_available: bool, lib_results: dict) -> None:
    """Print a final summary and recommendation."""
    print()
    print("=" * 60)
    print("  SUMMARY & RECOMMENDATION")
    print("=" * 60)

    missing_libs = [name for name, (installed, _) in lib_results.items() if not installed]

    if missing_libs:
        print()
        print("  Missing libraries:")
        for lib in missing_libs:
            print(f"    - {lib}")
        print()
        print("  Install all at once:")
        print("    pip install -r requirements.txt")

    print()
    if cuda_available:
        print("  CUDA is available.")
        print("  You can try local fine-tuning.")
        print()
        print("  Recommended next steps:")
        print("    python scripts/02_generate_synthetic_dataset.py")
        print("    python scripts/03_prepare_dataset.py")
        print("    python scripts/04_train_lora.py")
    else:
        print("  CUDA is NOT available on this machine.")
        print("  Use Google Colab for training;")
        print("  local machine can still run dataset preparation and small tests.")
        print()
        print("  What you CAN do locally (no GPU needed):")
        print("    python scripts/02_generate_synthetic_dataset.py")
        print("    python scripts/03_prepare_dataset.py")
        print()
        print("  For training, open in Google Colab:")
        print("    notebooks/colab_finetuning_qwen_0_5b.ipynb")
        print("    -> Runtime > Change runtime type > T4 GPU")

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  ENVIRONMENT CHECK")
    print("  dashboard-design-finetuning")
    print("=" * 60)

    python_ok          = check_python()
    torch_ok, cuda_ok  = check_pytorch()
    lib_results        = check_libraries()
    folder_results     = check_folders()

    print_recommendation(cuda_ok, lib_results)


if __name__ == "__main__":
    main()
