#!/bin/bash
#SBATCH --job-name=self-play
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=self-play-%j.out
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

# Activate the cluster's Python environment before submission, or add its
# module/conda activation commands here.
command -v python >/dev/null || { echo "python is unavailable" >&2; exit 1; }
command -v accelerate >/dev/null || { echo "accelerate is unavailable" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA GPU tools are unavailable" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
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
