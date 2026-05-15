#!/bin/bash
#SBATCH --job-name=ml-hyperopt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --time=1-00:00:00
#SBATCH --output=%j-tensorboard.out

source /home/iskoulid/.slurmrc

module purge

mamba activate dig

python hyperopt_mpnn.py
