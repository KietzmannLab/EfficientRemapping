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
                 negative_samples=8, contrastive_layer='last', **kwargs):
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
        
        # Store layer selection parameter
        self.contrastive_layer = contrastive_layer
        print(f"Contrastive loss will be applied to: {contrastive_layer} layer")
        
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
                
                # Compute temporal contrastive loss on selected layer
                selected_hidden = self._get_selected_layer_hidden(h, recurrent_state)
                if selected_hidden is not None:
                    contrastive_loss = self.contrastive_loss_fn(selected_hidden)
                    total_loss = total_loss + contrastive_loss
        
        return total_loss, total_loss.detach(), None
    
    def _get_selected_layer_hidden(self, h, recurrent_state):
        """
        Get hidden states from the selected layer for contrastive learning.
        
        Args:
            h: Top-level hidden state
            recurrent_state: Hierarchical recurrent states
            
        Returns:
            selected_hidden: Hidden states from selected layer, or None if not available
        """
        if isinstance(recurrent_state, list) and len(recurrent_state) > 0:
            if isinstance(recurrent_state[0], list) and len(recurrent_state[0]) > 0:
                layers = recurrent_state[0]
                
                if self.contrastive_layer == 'first':
                    # First layer (lowest level - edges, textures)
                    selected_hidden = layers[0]
                elif self.contrastive_layer == 'middle':
                    # Middle layer (mid-level features)
                    middle_idx = len(layers) // 2
                    selected_hidden = layers[middle_idx]
                elif self.contrastive_layer == 'last':
                    # Last layer (highest level - scene features)
                    selected_hidden = layers[-1]
                elif self.contrastive_layer.startswith('layer_'):
                    # Specific layer index (e.g., 'layer_1', 'layer_2')
                    try:
                        layer_idx = int(self.contrastive_layer.split('_')[1])
                        if 0 <= layer_idx < len(layers):
                            selected_hidden = layers[layer_idx]
                        else:
                            print(f"Warning: layer_{layer_idx} not available, using last layer")
                            selected_hidden = layers[-1]
                    except (ValueError, IndexError):
                        print(f"Warning: invalid layer specification '{self.contrastive_layer}', using last layer")
                        selected_hidden = layers[-1]
                else:
                    print(f"Warning: unknown layer '{self.contrastive_layer}', using last layer")
                    selected_hidden = layers[-1]
                
                # Handle LSTM tuple (hidden, cell) -> take hidden
                if isinstance(selected_hidden, tuple):
                    selected_hidden = selected_hidden[0]
                    
                return selected_hidden
                
        # Fallback to top-level hidden state
        elif h is not None:
            return h
            
        return None

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
        contrastive_layer=config['contrastive_layer'],
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
    
    # Advanced learning rate scheduling for faster convergence
    if config.get('use_cosine_schedule', False):
        # Cosine annealing with warmup for contrastive learning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        print(f"Using cosine annealing LR schedule (min_lr: {config['learning_rate'] * 0.01:.2e})")
    else:
        # Original step scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            model.optimizer, 
            step_size=config['scheduler_step'], 
            gamma=config['scheduler_gamma']
        )
        print(f"Using step LR schedule (step: {config['scheduler_step']}, gamma: {config['scheduler_gamma']})")
    
    # Mixed precision training for speed
    scaler = torch.cuda.amp.GradScaler() if config.get('use_mixed_precision', False) and device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision training (FP16) for faster convergence")
    
    # Initialize negative buffer for immediate learning (avoids cold start)
    if config.get('fast_start', True):
        model.contrastive_loss_fn.initialize_negative_buffer_with_random(num_random=200)
    
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
            
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    loss, _, _ = model.run(batch, fixations, loss_fn='temporal_contrastive')
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(model.optimizer)
                scaler.update()
            else:
                # Standard precision
                loss, _, _ = model.run(batch, fixations, loss_fn='temporal_contrastive')
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
    
    # Contrastive learning parameters (FIXED defaults)
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature parameter for contrastive loss (increased for stability)')
    parser.add_argument('--n_back', type=int, default=3,
                        help='Number of timesteps back for positive pairs')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Dimension of projection head output')
    parser.add_argument('--negative_samples', type=int, default=16,
                        help='Number of negative samples per positive pair (increased for better learning)')
    parser.add_argument('--contrastive_layer', type=str, default='last',
                        choices=['first', 'middle', 'last', 'layer_0', 'layer_1', 'layer_2', 'layer_3'],
                        help='Which layer to apply contrastive loss to (first=low-level, last=high-level)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size (smaller is better for contrastive learning)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (can be higher for contrastive learning)')
    parser.add_argument('--scheduler_step', type=int, default=50,
                        help='Learning rate scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.75,
                        help='Learning rate scheduler gamma')
    
    # Speed optimization parameters
    parser.add_argument('--use_cosine_schedule', action='store_true',
                        help='Use cosine annealing LR schedule (better for contrastive learning)')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use FP16 mixed precision training for faster convergence')
    
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