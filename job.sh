#!/bin/bash
#SBATCH --time=01:20:00
#SBATCH --account=def-lasserre
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

export LIMIT_SECONDS=3600 

module load singularity

if [ -z ${OUTDIR+x} ]; then
  export OUTDIR=$(pwd)/output
fi

mkdir -p "$OUTDIR"

singularity exec \
  -B ./src:/src \
  -B ~/nvidiadriver:/nvlib \
  -B ~/nvidiadriver:/nvbin \
  -B /project/def-lasserre/bgk/datasets/severstal:/data \
  -B "$OUTDIR":/output \
  --pwd /src image.sif \
  python train.py

if [ $? -eq 123 ]; then
  echo "Restarting job."
  sbatch ${BASH_SOURCE[0]}
fi
