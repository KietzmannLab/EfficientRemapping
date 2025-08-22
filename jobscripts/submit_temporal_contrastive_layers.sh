#!/bin/bash
#SBATCH --job-name=temporal_contrastive_layers
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=96:00:00
#SBATCH --output=logs/temporal_contrastive_layers_%j.out
#SBATCH --error=logs/temporal_contrastive_layers_%j.err
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

# Base parameters for layer comparison
DATASET_PATH="/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5"
INPUT_SIZE=16384
HIDDEN_SIZE=2048
NUM_EPOCHS=1000
BATCH_SIZE=512
SAVE_DIR="/share/klab/psulewski/psulewski/EfficientRemapping/models/temporal_contrastive"

echo "Starting layer comparison experiments for temporal contrastive learning..."

# Experiment 1: First layer (low-level features - edges, textures)
echo "Experiment 1: First layer contrastive loss"
python src/train_temporal_contrastive.py \
    --dataset_path $DATASET_PATH \
    --input_size $INPUT_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --temperature 0.5 \
    --n_back 3 \
    --projection_dim 128 \
    --negative_samples 16 \
    --contrastive_layer first \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --model_name "contrastive_first_layer_t05_nb3" \
    --save_dir $SAVE_DIR \
    --seed 42 \
    --use_wandb

echo "Experiment 1 completed. Starting Experiment 2..."

# Experiment 2: Middle layer (mid-level features - shapes, patterns)
echo "Experiment 2: Middle layer contrastive loss"
python src/train_temporal_contrastive.py \
    --dataset_path $DATASET_PATH \
    --input_size $INPUT_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --temperature 0.5 \
    --n_back 3 \
    --projection_dim 128 \
    --negative_samples 16 \
    --contrastive_layer middle \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --model_name "contrastive_middle_layer_t05_nb3" \
    --save_dir $SAVE_DIR \
    --seed 43 \
    --use_wandb

echo "Experiment 2 completed. Starting Experiment 3..."

# Experiment 3: Last layer (high-level features - scene identity)
echo "Experiment 3: Last layer contrastive loss"
python src/train_temporal_contrastive.py \
    --dataset_path $DATASET_PATH \
    --input_size $INPUT_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --temperature 0.5 \
    --n_back 3 \
    --projection_dim 128 \
    --negative_samples 16 \
    --contrastive_layer last \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --model_name "contrastive_last_layer_t05_nb3" \
    --save_dir $SAVE_DIR \
    --seed 44 \
    --use_wandb

echo "Experiment 3 completed. Starting Experiment 4..."

# Experiment 4: Specific layer 0 (bottom layer)
echo "Experiment 4: Layer 0 (bottom) contrastive loss"
python src/train_temporal_contrastive.py \
    --dataset_path $DATASET_PATH \
    --input_size $INPUT_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --temperature 0.5 \
    --n_back 3 \
    --projection_dim 128 \
    --negative_samples 16 \
    --contrastive_layer layer_0 \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --model_name "contrastive_layer0_t05_nb3" \
    --save_dir $SAVE_DIR \
    --seed 45 \
    --use_wandb

echo "All layer comparison experiments completed at: $(date)"

echo ""
echo "Layer comparison summary:"
echo "- First layer: Low-level features (edges, textures) - should show fine-grained temporal consistency"
echo "- Middle layer: Mid-level features (shapes, patterns) - intermediate consistency"  
echo "- Last layer: High-level features (scene identity) - scene-level consistency"
echo "- Layer 0: Bottom layer specific - most detailed features"
echo ""
echo "Expected results:"
echo "- First/Layer0: High temporal stability for local visual features"
echo "- Middle: Balance between local and global temporal features"
echo "- Last: Strongest scene-level temporal representations"