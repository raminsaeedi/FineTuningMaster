"""Static safety contract for generated SLURM array submissions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_each_submission_uses_private_task_and_script_files():
    script = (ROOT / "scripts" / "submit_slurm.sh").read_text(encoding="utf-8")
    assert 'SUBMIT_TAG="${DATASET}_$(date +%Y%m%d_%H%M%S)_${BASHPID}"' in script
    assert 'tasks_${SUBMIT_TAG}.txt' in script
    assert 'run_matrix_${SUBMIT_TAG}.sbatch' in script
    assert 'package_${SUBMIT_TAG}.sbatch' in script


def test_generated_jobs_use_tres_gpu_resources_consistently():
    script = (ROOT / "scripts" / "submit_slurm.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gpus=$GPUS" in script
    assert "#SBATCH --cpus-per-gpu=$CPUS" in script
    assert "#SBATCH --gres=gpu:$GPUS" not in script
