#!/bin/bash
#SBATCH --job-name=temporal_contrastive_fast
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/temporal_contrastive_fast_%j.out
#SBATCH --error=logs/temporal_contrastive_fast_%j.err
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

echo "Starting FAST temporal contrastive learning..."
echo "Optimizations: Higher LR + Smaller batch + Shorter training"

# Run the training script with FAST settings
python src/train_temporal_contrastive.py \
    --dataset_path /share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5 \
    --input_size 16384 \
    --hidden_size 2048 \
    --temperature 0.3 \
    --n_back 3 \
    --projection_dim 128 \
    --negative_samples 8 \
    --contrastive_layer last \
    --num_epochs 500 \
    --batch_size 256 \
    --learning_rate 1e-7 \
    --time_steps_img 6 \
    --time_steps_cords 3 \
    --model_name "temporal_contrastive_fast_lr1e7_bs256" \
    --save_dir "/share/klab/psulewski/psulewski/EfficientRemapping/models/temporal_contrastive" \
    --log_interval 10 \
    --save_interval 50 \
    --seed 42 \
    --use_wandb \
    --use_cosine_schedule \
    --use_mixed_precision \

echo "Job finished at: $(date)"

echo ""
echo "Fast training optimizations applied:"
echo "- Learning rate: 1e-4 → 5e-4 (5x higher)"
echo "- Batch size: 1024 → 256 (4x smaller, better for contrastive learning)"
echo "- Epochs: 1500 → 500 (3x fewer for quick results)"
echo "- Temperature: 0.5 → 0.3 (more discriminative)"
echo "- Cosine LR schedule (better convergence)"
echo "- Mixed precision (FP16 speed boost)"