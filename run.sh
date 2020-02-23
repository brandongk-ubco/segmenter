#!/bin/bash

set -euo pipefail

module load singularity
export PYTHONUNBUFFERED=true

if [ -z ${OUTDIR+x} ]; then
  export OUTDIR="/scratch/${USER}/results/severstal/"
fi

if [ -z ${INDIR+x} ]; then
  export INDIR="/scratch/${USER}/datasets/severstal/"
fi

if [ -z ${OUTFOLDER+x} ]; then
  export OUTFOLDER="output"
fi

if [ -z ${COMMAND+x} ]; then
  export COMMAND="train"
fi

mkdir -p "${OUTDIR}${OUTFOLDER}"

echo "Using output directory ${OUTDIR}${OUTFOLDER}"

singularity exec \
  -B ./src:/src \
  -B ~/nvidiadriver:/nvlib \
  -B ~/nvidiadriver:/nvbin \
  -B "$INDIR":/data \
  -B "${OUTDIR}${OUTFOLDER}":/output \
  --pwd /src image.sif \
  python "${COMMAND}.py"

if [ $? -eq 123 ]; then
  echo "Restarting job."
  sbatch ${BASH_SOURCE[0]}
fi
