#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J dqn_train_gamma_0_97
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -B
#BSUB -N
module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
echo "Running script..."
python3 main.py
