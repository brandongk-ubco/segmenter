#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

nvidia-smi > ~/nvidia-smi.log
