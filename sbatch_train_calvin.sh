#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J protox

# Cluster Settings
#SBATCH -n 4       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 21:00:00 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4


# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-robolear/ft4740/code/beso_calvin/logs/outputs/%x_%j.out
#SBATCH --error=/home/hk-project-robolear/ft4740/code/beso_calvin/logs/outputs/%x_%j.err


# -------------------------------

# Activate the virtualenv / conda environment
conda activate mdt_env

export TORCH_USE_CUDA_DSA=1
# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
#CUDA_VISIBLE_DEVICES=0,1,2,3  

srun python mdt/training.py 
# model=mdtv_agent datamodule/datasets=vision_lang max_epochs=20 rollout_lh_skip_epochs=19
# model.masked_beta=0.01 model.cont_alpha=0.01
# model.action_decoder.model.inner_model.num_experts=6 seed=142 
# --multirun seed=42,142,242
# model.action_decoder.model.inner_model.num_experts=4 seed=42
#python -m torch.distributed.launch \
#    --nnodes 1 \
#    --node-rank 0 \
#    --nproc-per-node 4 \
#    mdt/training.py \
#    --launcher pytorch

# THIS WAS BUILT FROM THE DEFAULLT SBATCH TEMPLATE
