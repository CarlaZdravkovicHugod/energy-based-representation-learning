#!/bin/sh

# A100 GPU queue, there is also gpua40 and gpua10
#BSUB -q gpua40

# job name
#BSUB -J 2DMRI_Data_Many_Steps

# 4 cpus, 1 machine, 1 gpu, 24 hours (the max)
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

# at least 32 GB RAM
#BSUB -R "rusage[mem=32GB]"

# stdout/stderr files for debugging (%J is substituted for job ID)
#BSUB -o logs/my_run_%J.out
#BSUB -e logs/my_run_%J.err

# your training script here, e.g.
# activate environment ...
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH="." python src/train.py --config="src/config/2DMRI_config_many_steps.yml"