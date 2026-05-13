"""Quick project health check — run from repo root."""
import json, os, sys

BASE = "dashboard-design-finetuning"

print("=" * 55)
print("PROJECT HEALTH CHECK")
print("=" * 55)

# 1. JSONL files
print("\n[1] JSONL file validity")
jsonl_files = {
    "raw/synthetic": f"{BASE}/data/raw/synthetic_dashboard_recommendations.jsonl",
    "processed/train": f"{BASE}/data/processed/train.jsonl",
    "processed/validation": f"{BASE}/data/processed/validation.jsonl",
}
for label, path in jsonl_files.items():
    if not os.path.exists(path):
        print(f"  MISSING : {label} -> {path}")
        continue
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    bad = []
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except Exception as e:
            bad.append(i + 1)
    if bad:
        print(f"  BAD JSON: {label} — bad lines: {bad[:5]}")
    else:
        first = json.loads(lines[0])
        print(f"  OK      : {label} ({len(lines)} lines) keys={list(first.keys())}")

# 2. Key directories
print("\n[2] Required directories")
dirs = [
    f"{BASE}/data/raw",
    f"{BASE}/data/processed",
    f"{BASE}/outputs/models",
    f"{BASE}/outputs/logs",
    f"{BASE}/outputs/predictions",
    f"{BASE}/rag/index",
]
for d in dirs:
    status = "OK" if os.path.isdir(d) else "MISSING"
    print(f"  {status:7} : {d}")

# 3. Key files
print("\n[3] Key files")
files = [
    f"{BASE}/scripts/01_check_environment.py",
    f"{BASE}/scripts/02_generate_synthetic_dataset.py",
    f"{BASE}/scripts/03_prepare_dataset.py",
    f"{BASE}/scripts/04_train_lora.py",
    f"{BASE}/scripts/05_inference_base_model.py",
    f"{BASE}/scripts/06_inference_finetuned_model.py",
    f"{BASE}/scripts/07_evaluate_schema_compliance.py",
    f"{BASE}/rag/build_index.py",
    f"{BASE}/rag/retrieve_context.py",
    f"{BASE}/rag/index/chunks.json",
    f"{BASE}/rag/index/tfidf_index.pkl",
    f"{BASE}/requirements.txt",
    f"{BASE}/README.md",
]
for f in files:
    status = "OK" if os.path.isfile(f) else "MISSING"
    print(f"  {status:7} : {f}")

# 4. Python version
print("\n[4] Python version")
v = sys.version_info
print(f"  Python {v.major}.{v.minor}.{v.micro} — {'OK (3.10+)' if v.major==3 and v.minor>=10 else 'WARNING: need 3.10+'}")

# 5. Installed packages
print("\n[5] Installed packages (training-independent)")
for pkg in ["json", "os", "random", "datasets", "sklearn"]:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "stdlib")
        print(f"  OK      : {pkg} ({ver})")
    except ImportError:
        print(f"  MISSING : {pkg}")

print("\n[6] Training packages (needed only for scripts 04-06)")
for pkg in ["transformers", "peft", "trl", "accelerate", "torch"]:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "?")
        print(f"  OK      : {pkg} ({ver})")
    except ImportError:
        print(f"  NOT INSTALLED (install before running scripts 04-06): {pkg}")

print("\n" + "=" * 55)
print("Check complete.")
print("=" * 55)
