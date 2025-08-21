#!/usr/bin/env python3
"""
Training script for RNN with temporal stability loss.
Alternative training objective to test whether energy efficiency specifically drives predictive remapping.
"""

import argparse
import torch
import torch.optim as optim
import numpy as np
import wandb
from RNNWithTemporalStability import RNNWithTemporalStability
from H5dataset import H5dataset
from functions import get_device, Timer
import os
import random

def train_temporal_stability_model(config):
    """
    Train RNN with temporal stability loss.
    
    Args:
        config: Dictionary containing training configuration
    """
    
    # Set device
    device = get_device()
    print(f"Training on device: {device}")
    
    # Set random seeds for reproducibility
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
    
    # Load dataset
    print(f"Loading dataset: {config['dataset_path']}")
    train_set = H5dataset('train', config['dataset_path'], device=device, use_color=False)
    validation_set = H5dataset('val', config['dataset_path'], device=device, use_color=False)
    
    # Initialize model with temporal stability loss
    model = RNNWithTemporalStability(
        activation_func=torch.nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=config['learning_rate'],
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        title=config['model_name'],
        device=device,
        temporal_loss_type=config['temporal_loss_type'],
        temporal_alpha=config['temporal_alpha'],
        use_fixation=True,
        use_conv=False,
        use_lstm=False,
        warp_imgs=True,
        use_resNet=False,
        time_steps_img=config['time_steps_img'],
        time_steps_cords=config['time_steps_cords'],
        mnist=False,
        twolayer=True,
        dropout=config['dropout']
    )
    
    # Initialize wandb logging if enabled
    if config['use_wandb']:
        wandb.init(
            project='efficient_remapping_temporal_stability',
            config=config,
            name=config['model_name']
        )
    
    # Training parameters
    scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer, 
        step_size=config['scheduler_step'], 
        gamma=config['scheduler_gamma']
    )
    
    timer = Timer()
    best_val_loss = float('inf')
    
    print(f"Starting training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        timer.lap()
        
        # Training phase
        model.model.train()
        train_loss = 0
        num_train_batches = 0
        
        # Create training batches
        train_loader = train_set.create_batches(
            batch_size=config['batch_size'], 
            shuffle=True
        )
        
        for batch_idx, (batch, fixations) in enumerate(train_loader):
            # Zero gradients
            model.zero_grad()
            
            # Forward pass with temporal stability loss
            loss, _, _ = model.run(batch, fixations, loss_fn='temporal_stability')
            
            # Backward pass
            model.step(loss)
            
            train_loss += loss.item()
            num_train_batches += 1
            
            if batch_idx % config['log_interval'] == 0:
                print(f'Epoch {epoch+1}/{config["num_epochs"]}, '
                      f'Batch {batch_idx}, '
                      f'Train Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
        
        # Validation phase
        model.model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            val_loader = validation_set.create_batches(
                batch_size=config['batch_size'], 
                shuffle=False
            )
            
            for batch, fixations in val_loader:
                loss, _, _ = model.run(batch, fixations, loss_fn='temporal_stability')
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        epoch_time = timer.get() / 1000.0  # Convert to seconds
        
        print(f'Epoch {epoch+1}/{config["num_epochs"]} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
        
        if config['use_wandb']:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch_time': epoch_time
            })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{config['save_dir']}/{config['model_name']}_best.pth"
            os.makedirs(config['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved: {save_path}')
        
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % config['save_interval'] == 0:
            save_path = f"{config['save_dir']}/{config['model_name']}_epoch_{epoch+1}.pth"
            os.makedirs(config['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'Checkpoint saved: {save_path}')
    
    print(f'Training completed! Best validation loss: {best_val_loss:.6f}')
    
    if config['use_wandb']:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train RNN with temporal stability loss')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, 
                        default='/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5',
                        help='Path to H5 dataset')
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=128*128,
                        help='Input size (default: 128*128)')
    parser.add_argument('--hidden_size', type=int, default=2048,
                        help='Hidden size (default: 2048)')
    parser.add_argument('--temporal_loss_type', type=str, default='l2',
                        choices=['l2', 'cosine', 'combined'],
                        help='Type of temporal stability loss')
    parser.add_argument('--temporal_alpha', type=float, default=0.1,
                        help='Temporal stability loss weight')
    parser.add_argument('--time_steps_img', type=int, default=6,
                        help='Time steps for image processing')
    parser.add_argument('--time_steps_cords', type=int, default=3,
                        help='Time steps for coordinate processing')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=7e-4,
                        help='Learning rate')
    parser.add_argument('--scheduler_step', type=int, default=50,
                        help='Learning rate scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.75,
                        help='Learning rate scheduler gamma')
    
    # Logging and saving
    parser.add_argument('--model_name', type=str, 
                        default='temporal_stability_model',
                        help='Model name for saving')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Batch logging interval')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Model save interval (epochs)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Convert args to config dictionary
    config = vars(args)
    
    # Add additional config parameters
    config.update({
        'temporal_loss_type': args.temporal_loss_type,
        'temporal_alpha': args.temporal_alpha
    })
    
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train the model
    train_temporal_stability_model(config)


if __name__ == "__main__":
    main()