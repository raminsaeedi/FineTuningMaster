# Run the project on DGX Spark (simple instructions)

This project has a separate DGX Spark setup path for the ARM64 NVIDIA GB10.
It uses the tested `torch 2.13.0+cu130` wheel. The experiment code, QLoRA
configuration, datasets, methods A--D, metrics and result format are unchanged.

## 1. Copy the Spark changes from the Windows laptop

Open **PowerShell on the laptop**, not a DGX shell. Run these commands:

```powershell
scp "C:\Projects\MasterArbeit\master-thesis-finetuning\scripts\bootstrap_dgx_spark.sh" student@192.168.178.32:/home/student/ramin/FineTuningMaster-main/scripts/
scp "C:\Projects\MasterArbeit\master-thesis-finetuning\scripts\bootstrap_remote.sh" student@192.168.178.32:/home/student/ramin/FineTuningMaster-main/scripts/
scp "C:\Projects\MasterArbeit\master-thesis-finetuning\run_professor.sh" student@192.168.178.32:/home/student/ramin/FineTuningMaster-main/
scp "C:\Projects\MasterArbeit\master-thesis-finetuning\experiments\scripts\check_experiment_release.py" student@192.168.178.32:/home/student/ramin/FineTuningMaster-main/experiments/scripts/
scp "C:\Projects\MasterArbeit\master-thesis-finetuning\experiments\scripts\run_final_matrix.py" student@192.168.178.32:/home/student/ramin/FineTuningMaster-main/experiments/scripts/
```

Enter the DGX password when asked. Do not copy `.env` or any token.

## 2. Open one SSH session to the DGX

On the laptop PowerShell:

```powershell
ssh student@192.168.178.32
```

Then, inside the DGX session:

```bash
cd /home/student/ramin/FineTuningMaster-main
chmod +x run_professor.sh run_experiment.sh scripts/*.sh scripts/lib/*.sh
# The Windows copy may contain CRLF line endings; Bash needs LF.
sed -i 's/\r$//' run_professor.sh run_experiment.sh scripts/*.sh scripts/lib/*.sh
```

The old `/home/student/ramin/FineTuningMaster-main/venv` is not used. The new
environment is created as `.venv`, so the broken old environment is left alone.

## 3. Install and test the DGX environment

Still inside the DGX session, run:

```bash
bash scripts/bootstrap_remote.sh
```

The script installs the portable project dependencies, the CUDA 13 PyTorch
wheel, and performs three checks:

1. PyTorch sees the NVIDIA GB10.
2. A real CUDA matrix operation succeeds.
3. `bitsandbytes` performs a real 4-bit GPU forward pass.

The expected check line is:

```text
  environment  : OK
```

If this step fails, do not start training. Copy the complete error output.

## 4. First smoke test: mini dataset, all four methods

Run this first. It is a functional test, not a thesis result:

```bash
bash run_professor.sh --dataset dashboard_v4_tiny --model qwen2_5_0_5b --seed 42 --methods "A B C D" --no-package
```

The important checks are that method C trains and method D finds and uses the
same-seed method-C adapter. The result directory is:

```text
experiments/outputs/final/dashboard_v4_tiny/qwen2_5_0_5b/
```

## 5. Then run the real dataset for one model and one seed

After the smoke test completes successfully:

```bash
bash run_professor.sh --dataset dashboard_v4 --model qwen3_1_7b --seed 42 --methods "A B C D" --no-package
```

This is the sensible first full DGX run. It does not run all models or all
seeds, so it limits GPU time while checking the real thesis dataset and the
complete A--D pipeline.

## 6. Monitor from the laptop

Keep the SSH session running. In a second laptop PowerShell window, connect
again and monitor the GPU:

```powershell
ssh -t student@192.168.178.32 "watch -n 2 nvidia-smi"
```

To copy results back after a run, use laptop PowerShell:

```powershell
scp -r student@192.168.178.32:/home/student/ramin/FineTuningMaster-main/experiments/results/final/dashboard_v4 C:\Projects\MasterArbeit\master-thesis-finetuning\experiments\results\final\
```

Do not press **Update Now** in the DGX Dashboard while a run is active.
