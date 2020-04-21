#!/bin/bash
#SBATCH --time=01:20:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

module load singularity

singularity exec \
  -B ./src:/src \
  --nv \
  -B "${OUTDIR}${OUTFOLDER}":/output \
  --pwd /src \
  image.sif python "${COMMAND}.py"

if [ $? -eq 123 ]; then
  echo "Restarting job."
  sbatch ${BASH_SOURCE[0]}
fi
