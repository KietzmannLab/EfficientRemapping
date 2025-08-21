#!/bin/bash
#SBATCH --job-name=temporal_contrastive_sweep
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/temporal_contrastive_sweep_%j.out
#SBATCH --error=logs/temporal_contrastive_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=psulewski@uni-osnabrueck.de

# Create logs directory if it doesn't exist
mkdir -p logs

echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate efficient

# Set up proxy (required for UOS network)
export http_proxy=http://rhn-proxy.rz.uos.de:3128
export https_proxy=http://rhn-proxy.rz.uos.de:3128

# Set environment variables for CUDA
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Change to your project directory
cd /home/student/p/psulewski/EfficientRemapping

# Print some info for debugging
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Create directory structure for models if it doesn't exist
mkdir -p /share/klab/psulewski/psulewski/EfficientRemapping/models/temporal_contrastive

# Base parameters
DATASET_PATH="/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5"
INPUT_SIZE=16384
HIDDEN_SIZE=2048
NUM_EPOCHS=800
BATCH_SIZE=512
SAVE_DIR="/share/klab/psulewski/psulewski/EfficientRemapping/models/temporal_contrastive"

echo "Starting temporal contrastive learning experiments..."

# Experiment 1: Standard InfoNCE parameters
echo "Experiment 1: Standard InfoNCE (temp=0.07, n_back=3)"
python src/train_temporal_contrastive.py \
    --dataset_path $DATASET_PATH \
    --input_size $INPUT_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --temperature 0.07 \
    --n_back 3 \
    --projection_dim 128 \
    --negative_samples 8 \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --model_name "contrastive_standard_t07_nb3" \
    --save_dir $SAVE_DIR \
    --seed 42 \
    --use_wandb

echo "Experiment 1 completed. Starting Experiment 2..."

# Experiment 2: Warmer temperature for easier learning
echo "Experiment 2: Warmer temperature (temp=0.2, n_back=3)"
python src/train_temporal_contrastive.py \
    --dataset_path $DATASET_PATH \
    --input_size $INPUT_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --temperature 0.2 \
    --n_back 3 \
    --projection_dim 128 \
    --negative_samples 8 \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --model_name "contrastive_warm_t02_nb3" \
    --save_dir $SAVE_DIR \
    --seed 43 \
    --use_wandb

echo "Experiment 2 completed. Starting Experiment 3..."

# Experiment 3: Longer temporal window
echo "Experiment 3: Longer temporal window (temp=0.07, n_back=5)"
python src/train_temporal_contrastive.py \
    --dataset_path $DATASET_PATH \
    --input_size $INPUT_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --temperature 0.07 \
    --n_back 5 \
    --projection_dim 128 \
    --negative_samples 8 \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --model_name "contrastive_long_t07_nb5" \
    --save_dir $SAVE_DIR \
    --seed 44 \
    --use_wandb

echo "Experiment 3 completed. Starting Experiment 4..."

# Experiment 4: Higher dimensional projection space
echo "Experiment 4: Higher dim projection (temp=0.07, n_back=3, proj_dim=256)"
python src/train_temporal_contrastive.py \
    --dataset_path $DATASET_PATH \
    --input_size $INPUT_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --temperature 0.07 \
    --n_back 3 \
    --projection_dim 256 \
    --negative_samples 8 \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --model_name "contrastive_highdim_t07_nb3_pd256" \
    --save_dir $SAVE_DIR \
    --seed 45 \
    --use_wandb

echo "All experiments completed at: $(date)"