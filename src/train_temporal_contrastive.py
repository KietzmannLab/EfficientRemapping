#!/usr/bin/env python3
"""
Training script for RNN with temporal contrastive learning loss.
Uses InfoNCE loss with positive pairs from same scene across time and negative pairs from different scenes.
"""

import argparse
import torch
import torch.optim as optim
import numpy as np
import wandb
from RNN import State
from TemporalContrastiveLoss import BatchTemporalContrastiveLoss
from H5dataset import H5dataset
from functions import get_device, Timer
import os
import random

class RNNWithTemporalContrastive(State):
    """
    RNN with temporal contrastive learning integration.
    """
    
    def __init__(self, activation_func, optimizer, lr, input_size, hidden_size, 
                 title, device, temperature=0.07, n_back=3, projection_dim=128,
                 negative_samples=8, **kwargs):
        """Initialize RNN with temporal contrastive learning."""
        
        # Initialize base RNN
        super().__init__(
            activation_func=activation_func,
            optimizer=optimizer,
            lr=lr,
            input_size=input_size,
            hidden_size=hidden_size,
            title=title,
            device=device,
            **kwargs
        )
        
        # Initialize temporal contrastive loss
        self.contrastive_loss_fn = BatchTemporalContrastiveLoss(
            temperature=temperature,
            n_back=n_back,
            projection_dim=projection_dim,
            hidden_dim=hidden_size,
            device=device,
            negative_samples=negative_samples
        )
        
    def run(self, batch, fixations, loss_fn='temporal_contrastive', state=None):
        """Run batch through model with temporal contrastive loss."""
        batch = batch.to(self.device)
        fixations = fixations.to(self.device)
        
        # Handle batch shape formatting (same as original)
        if len(batch.shape) == 2:
            batch_size = batch.shape[1]
            batch = batch.permute(1,0).reshape(batch_size, 1, 140, 56)
        else:
            batch_size = batch.shape[0]
            if len(batch.shape) > 3:
                batch = batch.reshape(batch_size, 3, 256, 256)
            else:
                batch = batch.reshape(batch_size, 1, 256, 256)
        
        # Initialize hidden state
        h = self.model.init_state(batch_size)
        total_loss = torch.zeros(1, dtype=torch.float, requires_grad=True)
        total_loss = total_loss.to(self.device)
        
        # Handle fixation shape formatting
        if len(fixations.shape) == 2:
            fixations = fixations.permute(1,0).reshape(batch_size, 10, 2)
        
        # Apply foveal transform
        images = self.foveal_transform(batch, fixations)
        if len(images.shape) == 3:
            images = images.permute(1, 0, 2)
        else:
            images = images.permute(1, 0, 2, 3, 4)
        
        # Convert to relative coordinates (same as original)
        if not self.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
            if self.mnist:
                fixations[:, :, 1] = fixations[:, :, 1] / 0.4
        
        # Reset contrastive loss for new sequence
        self.contrastive_loss_fn.reset_sequence()
        
        # Forward pass through sequence
        recurrent_state = None
        for i, image in enumerate(images):
            if self.model.use_conv:
                image = image.reshape(image.shape[0], 128, 128)
                
            for t in range(self.model.time_steps_img):
                if t >= self.model.time_steps_img - self.model.time_steps_cords:
                    # Forward pass with efference copy
                    h, l_a, recurrent_state = self.model(
                        image, fixation=fixations[:, i], state=h, 
                        recurrent_state=recurrent_state
                    )
                else:
                    # Forward pass without efference copy
                    h, l_a, recurrent_state = self.model(
                        image, fixation=torch.zeros_like(fixations[:, i]), 
                        state=h, recurrent_state=recurrent_state
                    )
                
                # Compute temporal contrastive loss on hidden states
                # Use the last layer's hidden state for contrastive learning
                if isinstance(recurrent_state, list) and len(recurrent_state) > 0:
                    if isinstance(recurrent_state[0], list) and len(recurrent_state[0]) > 0:
                        # Get the last layer's hidden state
                        last_hidden = recurrent_state[0][-1]
                        if isinstance(last_hidden, tuple):
                            last_hidden = last_hidden[0]  # For LSTM
                        contrastive_loss = self.contrastive_loss_fn(last_hidden)
                        total_loss = total_loss + contrastive_loss
                elif h is not None:
                    # Fallback to using h if recurrent_state is not available
                    contrastive_loss = self.contrastive_loss_fn(h)
                    total_loss = total_loss + contrastive_loss
        
        return total_loss, total_loss.detach(), None

def train_temporal_contrastive_model(config):
    """
    Train RNN with temporal contrastive learning.
    
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
    
    # Initialize model with temporal contrastive learning
    model = RNNWithTemporalContrastive(
        activation_func=torch.nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=config['learning_rate'],
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        title=config['model_name'],
        device=device,
        temperature=config['temperature'],
        n_back=config['n_back'],
        projection_dim=config['projection_dim'],
        negative_samples=config['negative_samples'],
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
            project='efficient_remapping_temporal_contrastive',
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
            
            # Forward pass with temporal contrastive loss
            loss, _, _ = model.run(batch, fixations, loss_fn='temporal_contrastive')
            
            # Backward pass
            model.step(loss)
            
            train_loss += loss.item()
            num_train_batches += 1
            
            if batch_idx % config['log_interval'] == 0:
                # Get contrastive learning stats
                stats = model.contrastive_loss_fn.get_stats()
                print(f'Epoch {epoch+1}/{config["num_epochs"]}, '
                      f'Batch {batch_idx}, '
                      f'Train Loss: {loss.item():.6f}, '
                      f'Timestep: {stats["current_timestep"]}, '
                      f'Global Negs: {stats["global_negatives"]}')
        
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
                loss, _, _ = model.run(batch, fixations, loss_fn='temporal_contrastive')
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
            # Get final contrastive stats for logging
            final_stats = model.contrastive_loss_fn.get_stats()
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch_time': epoch_time,
                'contrastive_timestep': final_stats['current_timestep'],
                'global_negatives': final_stats['global_negatives'],
                'temperature': final_stats['temperature'],
                'n_back': final_stats['n_back']
            })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{config['save_dir']}/{config['model_name']}_best.pth"
            os.makedirs(config['save_dir'], exist_ok=True)
            
            # Save both model and contrastive projection head
            state_dict = {
                'model': model.model.state_dict(),
                'contrastive_projection': model.contrastive_loss_fn.projection_head.state_dict(),
                'config': config
            }
            torch.save(state_dict, save_path)
            print(f'New best model saved: {save_path}')
        
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % config['save_interval'] == 0:
            save_path = f"{config['save_dir']}/{config['model_name']}_epoch_{epoch+1}.pth"
            os.makedirs(config['save_dir'], exist_ok=True)
            
            state_dict = {
                'model': model.model.state_dict(),
                'contrastive_projection': model.contrastive_loss_fn.projection_head.state_dict(),
                'config': config
            }
            torch.save(state_dict, save_path)
            print(f'Checkpoint saved: {save_path}')
    
    print(f'Training completed! Best validation loss: {best_val_loss:.6f}')
    
    if config['use_wandb']:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train RNN with temporal contrastive learning')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, 
                        default='/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5',
                        help='Path to H5 dataset')
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=128*128,
                        help='Input size (default: 128*128)')
    parser.add_argument('--hidden_size', type=int, default=2048,
                        help='Hidden size (default: 2048)')
    parser.add_argument('--time_steps_img', type=int, default=6,
                        help='Time steps for image processing')
    parser.add_argument('--time_steps_cords', type=int, default=3,
                        help='Time steps for coordinate processing')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    
    # Contrastive learning parameters
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature parameter for contrastive loss')
    parser.add_argument('--n_back', type=int, default=3,
                        help='Number of timesteps back for positive pairs')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Dimension of projection head output')
    parser.add_argument('--negative_samples', type=int, default=8,
                        help='Number of negative samples per positive pair')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--scheduler_step', type=int, default=50,
                        help='Learning rate scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.75,
                        help='Learning rate scheduler gamma')
    
    # Logging and saving
    parser.add_argument('--model_name', type=str, 
                        default='temporal_contrastive_model',
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
    
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train the model
    train_temporal_contrastive_model(config)

if __name__ == "__main__":
    main()