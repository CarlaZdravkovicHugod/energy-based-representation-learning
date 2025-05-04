#!/bin/sh

# A100 GPU queue, there is also gpua40 and gpua10
#BSUB -q gpua100

# job name
#BSUB -J Autoencoder

# 4 cpus, 1 machine, 1 gpu, 24 hours (the max)
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

# at least 32 GB RAM
#BSUB -R "rusage[mem=16GB]"

# stdout/stderr files for debugging (%J is substituted for job ID)
#BSUB -o logs/my_run_%J.out
#BSUB -e logs/my_run_%J.err

# your training script here, e.g.
# activate environment ...
source .venv/bin/activate
export NEPTUNE_PROJECT_NAME="unsupervised-learning-of-energy-representations/unsupervised-learning-of-energy-representations"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNzZkZDFhNy03NTdjLTRjMjAtYTAyZS05NzU4NzEwZGI2N2EifQ=="
PYTHONPATH="." python src/ae.py --train_dir "data/" --description 2056Autoencoder_With_SSIM_50_Epochs_NoSkipConnections_HPC