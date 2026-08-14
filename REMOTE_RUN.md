# Remote GPU run — moved

This file described an older workflow and is no longer maintained.

The single authoritative guide for running the experiments on a remote Linux GPU
server is:

**[RUN_PROFESSOR.md](RUN_PROFESSOR.md)**

Short version:

```bash
git clone https://github.com/raminsaeedi/FineTuningMaster.git
cd FineTuningMaster
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt && pip install -e .
export HF_TOKEN="hf_your_token_here"
python experiments/scripts/build_kb.py
python experiments/scripts/check_experiment_release.py --profile final --all-models --dataset dashboard_v4
./run_experiment.sh --profile final --dataset dashboard_v4 --all-models --all-methods --seeds 42 43 44 --with-dependencies --resume
```
