#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J per_dueling_ddqn
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
##BSUB -R "select[gpu32gb]"
##BSUB -R "select[sxm2]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -B
#BSUB -N
module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
echo "Running script..."
python3 main.py
