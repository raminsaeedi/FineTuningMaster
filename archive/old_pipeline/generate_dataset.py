"""
generate_dataset.py — Thin shim for backward compatibility.

The actual implementation lives in data/generator.py.
Run this script directly, or use the module form:
    python -m data.generator
"""

from data.generator import (  # noqa: F401  (re-exports for importers)
    generate_dataset,
    generate_example,
    INDUSTRIES,
    KPIS_BY_INDUSTRY,
)
from data.generator import main

if __name__ == "__main__":
    main()
