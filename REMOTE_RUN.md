# Remote GPU run — moved

This file described an older workflow and is no longer maintained.

The single authoritative guide for running the experiments on a remote Linux GPU
server is:

**[RUN_PROFESSOR.md](RUN_PROFESSOR.md)**

Short version:

```bash
git clone https://github.com/raminsaeedi/FineTuningMaster.git
cd FineTuningMaster
./scripts/bootstrap_remote.sh
export HF_TOKEN="hf_your_token_here"
./.venv/bin/python experiments/scripts/build_kb.py
./.venv/bin/python experiments/scripts/check_experiment_release.py --profile final --all-models --dataset dashboard_v4
./run_experiment.sh --profile final --dataset dashboard_v4 --all-models --all-methods --seeds 42 43 44 --with-dependencies --resume
```
