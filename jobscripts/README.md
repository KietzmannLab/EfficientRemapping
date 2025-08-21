# Job Scripts for EfficientRemapping

This directory contains Slurm job scripts for running experiments on the Klab HPC infrastructure.

## Available Scripts

### `submit_temporal_stability.sh`
Trains an RNN model with temporal stability loss as an alternative objective to energy efficiency.

**Usage:**
1. Update the following placeholders in the script:
   - `your.email@uni-osnabrueck.de` → your actual email address
   - `your_env_name` → your conda environment name  
   - `your_username` → your username on the cluster

2. Make the script executable:
   ```bash
   chmod +x jobscripts/submit_temporal_stability.sh
   ```

3. Submit the job:
   ```bash
   sbatch jobscripts/submit_temporal_stability.sh
   ```

4. Monitor the job:
   ```bash
   squeue -u $USER
   ```

**Parameters:**
- **Partition:** `klab-gpu` (uses H100 80GB GPU)
- **Resources:** 8 CPUs, 64GB RAM, 1 GPU
- **Time limit:** 48 hours
- **Model:** Temporal stability loss with L2 objective
- **Dataset:** MS-COCO with DeepGaze fixations

**Output:**
- Logs: `logs/temporal_stability_<job_id>.out/err`
- Models: `models/temporal_stability/`
- Wandb logging enabled

## Resource Recommendations

- **Single experiment:** Use H100 GPU (`klab-gpu` partition)
- **Multiple experiments:** Use L40S GPUs (`klab-l40s` partition) to run parallel jobs
- **CPU-only baseline:** Use `klab-cpu` partition

## Monitoring

Check GPU usage during training:
```bash
watch -n 1 nvidia-smi
```

View job logs in real-time:
```bash
tail -f logs/temporal_stability_<job_id>.out
```