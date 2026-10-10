#!/bin/bash
#SBATCH --job-name=play # define job name
#SBATCH --gpus-per-node=1     # define gpu limmit in 1 node
#SBATCH --ntasks=1            # define number tasks
#SBATCH --cpus-per-task=8   # There are 24 CPU cores
#SBATCH --time=20-20:10:00     # Max running time = 40 days
#SBATCH --output=slurm-%j.out
#SBATCH --export=ALL

set -euo pipefail

# Submit from the repository root or from self_play/.
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$submit_dir/self_play/run_main.sh" ]]; then
    work_dir="$submit_dir/self_play"
elif [[ -f "$submit_dir/run_main.sh" ]]; then
    work_dir="$submit_dir"
else
    echo "Submit this job from the repository root or self_play/" >&2
    exit 1
fi
cd "$work_dir"

# Replace these values before submitting. Keep filled-in tokens out of commits.
export HF_TOKEN="PASTE_HF_TOKEN_HERE"
export WANDB_API_KEY="PASTE_WANDB_API_KEY_HERE"
if [[ "$HF_TOKEN" == "PASTE_HF_TOKEN_HERE" || "$WANDB_API_KEY" == "PASTE_WANDB_API_KEY_HERE" ]]; then
    echo "Set HF_TOKEN and WANDB_API_KEY in run_slurm.sbatch before submitting." >&2
    exit 1
fi

source /data2/shared/apps/conda/etc/profile.d/conda.sh
conda activate env_llm
module load shared 2>/dev/null || true


export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command -v python >/dev/null || { echo "python is unavailable" >&2; exit 1; }
command -v accelerate >/dev/null || { echo "accelerate is unavailable" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA GPU tools are unavailable" >&2; exit 1; }

export TOKENIZERS_PARALLELISM=false
export SOLVER_PORT="${SOLVER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 30000))}"
export SELF_PLAY_OUTPUT_DIR="${SELF_PLAY_OUTPUT_DIR:-$PWD/runs/slurm-${SLURM_JOB_ID:-local}}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Working directory: $PWD"
echo "Visible GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "Solver server port: $SOLVER_PORT"
echo "Output directory: $SELF_PLAY_OUTPUT_DIR"
nvidia-smi

exec bash run_main.sh
