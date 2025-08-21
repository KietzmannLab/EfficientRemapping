#!/bin/bash
#SBATCH --job-name=temporal_stability_RNN
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/temporal_stability_%j.out
#SBATCH --error=logs/temporal_stability_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=psulewski@uni-osnabrueck.de

# Create logs directory if it doesn't exist
mkdir -p logs

echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

## Please add any modules you want to load here, as an example we have commented out the modules
## that you may need such as cuda, cudnn, miniconda3, uncomment them if that is your use case 
## term handler the function is executed once the job gets the TERM signal


spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate efficient
# print conda info
conda info

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
echo "CUDA devices available:"
nvidia-smi

# Create directory structure for models if it doesn't exist
mkdir -p /share/klab/psulewski/psulewski/EfficientRemapping/models/temporal_stability

# Run the training script
python src/train_temporal_stability.py \
    --dataset_path /share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5 \
    --input_size 16384 \
    --hidden_size 2048 \
    --temporal_loss_type l2 \
    --temporal_alpha 0.1 \
    --num_epochs 1500 \
    --batch_size 1024 \
    --learning_rate 7e-4 \
    --time_steps_img 6 \
    --time_steps_cords 3 \
    --model_name "temporal_stability_l2_alpha01" \
    --save_dir "/share/klab/psulewski/psulewski/EfficientRemapping/models/temporal_stability" \
    --log_interval 10 \
    --save_interval 50 \
    --seed 42

echo "Job finished at: $(date)"