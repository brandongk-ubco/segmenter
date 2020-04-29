#!/bin/bash
#SBATCH --time=02:50:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

if command -v module; then
  module load singularity
fi

launch --adaptor singularity "$@"

if [ $? -eq 123 ]; then
  echo "Restarting job."
  sbatch ${BASH_SOURCE[0]} "$@"
fi
