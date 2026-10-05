#!/bin/bash
# Submit from the repo root: sbatch jobs/new_nru_copying_1000.sh
#SBATCH --job-name=new_nru_copying_1000
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=rtx8000:1
## Or --gpus-per-task=rtx8000:1    to request a different GPU model
## Or --gpus-per-task=1            for any GPU model
#SBATCH --time=30:00:00

cd ~/CODE/non-linear-rnn
# .env holds WANDB_API_KEY (see .env.example); wandb picks it up from the environment
[ -f .env ] || { echo "missing .env, copy .env.example and fill in your key"; exit 1; }
set -a; source .env; set +a
source .venv/bin/activate
uv run train.py configs/new_nru_copying_1000.json
