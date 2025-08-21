"""
Temporal Contrastive Learning Loss for Fixation Sequences.

This loss function learns representations where:
- Positive pairs: Current crop with its predecessor n_back timesteps ago from same scene
- Negative pairs: Current crop with random crops from different scenes

Key components:
1. Temporal positive pair sampling (same scene, different timesteps)
2. Negative pair sampling (different scenes)
3. InfoNCE-style contrastive loss
4. Batch-aware implementation for efficient training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random

class TemporalContrastiveLoss(nn.Module):
    """
    Temporal contrastive learning for fixation sequences.
    
    Learns representations where crops from the same scene at different timesteps
    are more similar than crops from different scenes.
    """
    
    def __init__(self, temperature=0.07, n_back=3, projection_dim=128, 
                 hidden_dim=512, device='cpu', negative_samples=8):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
            n_back: Number of timesteps back for positive pairs
            projection_dim: Dimension of projection head output
            hidden_dim: Dimension of input hidden states
            device: PyTorch device
            negative_samples: Number of negative samples per positive pair
        """
        super(TemporalContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.n_back = n_back
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.negative_samples = negative_samples
        
        # Projection head to map hidden states to contrastive space
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, projection_dim),
            nn.LayerNorm(projection_dim)
        ).to(device)
        
        # History buffer for temporal positive pairs
        # Store: (projected_features, scene_id, timestep)
        self.history_buffer = deque(maxlen=n_back + 10)  # Extra buffer for safety
        
        # Negative sample buffer across different scenes
        self.negative_buffer = deque(maxlen=1000)  # Store features from different scenes
        
        # Track current scene for positive/negative sampling
        self.current_scene_id = None
        self.timestep = 0
        
        print(f"TemporalContrastiveLoss initialized:")
        print(f"  Temperature: {temperature}")
        print(f"  N-back timesteps: {n_back}")
        print(f"  Projection dim: {projection_dim}")
        print(f"  Negative samples: {negative_samples}")
        
    def reset_sequence(self, scene_id=None):
        """
        Reset for new sequence/scene.
        
        Args:
            scene_id: Identifier for the current scene (for positive/negative sampling)
        """
        # Store previous scene's features as negatives before resetting
        if self.current_scene_id is not None and len(self.history_buffer) > 0:
            # Sample some features from current scene as future negatives
            scene_features = [item[0] for item in self.history_buffer 
                            if item[1] == self.current_scene_id]
            if len(scene_features) > 0:
                # Add random subset to negative buffer
                n_to_add = min(len(scene_features), 10)
                sampled_features = random.sample(scene_features, n_to_add)
                for feat in sampled_features:
                    self.negative_buffer.append(feat.detach())
        
        # Reset for new scene
        self.history_buffer.clear()
        self.current_scene_id = scene_id
        self.timestep = 0
        
    def project_features(self, hidden_states):
        """
        Project hidden states to contrastive learning space.
        
        Args:
            hidden_states: [batch_size, hidden_dim] or [hidden_dim]
            
        Returns:
            projected_features: [batch_size, projection_dim] or [projection_dim]
        """
        if len(hidden_states.shape) == 1:
            hidden_states = hidden_states.unsqueeze(0)
        
        projected = self.projection_head(hidden_states)
        # L2 normalize for cosine similarity
        projected = F.normalize(projected, dim=-1)
        
        return projected
        
    def get_positive_pairs(self, current_features):
        """
        Get positive pairs from history buffer.
        
        Args:
            current_features: [batch_size, projection_dim] current projected features
            
        Returns:
            positive_pairs: List of (anchor, positive) tensors
            valid_indices: Indices in batch that have valid positive pairs
        """
        positive_pairs = []
        valid_indices = []
        
        # Look for features from n_back timesteps ago in same scene
        target_timestep = self.timestep - self.n_back
        
        for i, (feat, scene_id, timestep) in enumerate(self.history_buffer):
            if (scene_id == self.current_scene_id and 
                timestep == target_timestep):
                # Found positive pair
                for batch_idx in range(current_features.shape[0]):
                    positive_pairs.append((
                        current_features[batch_idx],  # anchor
                        feat  # positive (from n_back timesteps ago)
                    ))
                    valid_indices.append(batch_idx)
                break
                
        return positive_pairs, valid_indices
        
    def get_negative_samples(self, n_negatives):
        """
        Sample negative features from different scenes.
        
        Args:
            n_negatives: Number of negative samples to return
            
        Returns:
            negative_features: [n_negatives, projection_dim] negative samples
        """
        if len(self.negative_buffer) == 0:
            # No negatives available, return zeros
            return torch.zeros(n_negatives, self.projection_dim, device=self.device)
        
        # Sample negatives from buffer
        available_negatives = list(self.negative_buffer)
        if len(available_negatives) >= n_negatives:
            sampled_negatives = random.sample(available_negatives, n_negatives)
        else:
            # Repeat samples if not enough negatives
            sampled_negatives = available_negatives * ((n_negatives // len(available_negatives)) + 1)
            sampled_negatives = sampled_negatives[:n_negatives]
            
        return torch.stack(sampled_negatives)
        
    def compute_contrastive_loss(self, anchor, positive, negatives):
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            anchor: [projection_dim] anchor features
            positive: [projection_dim] positive features  
            negatives: [n_negatives, projection_dim] negative features
            
        Returns:
            loss: scalar contrastive loss
        """
        # Compute similarities
        pos_similarity = torch.dot(anchor, positive) / self.temperature
        
        # Compute negative similarities
        neg_similarities = torch.mv(negatives, anchor) / self.temperature
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        all_similarities = torch.cat([pos_similarity.unsqueeze(0), neg_similarities])
        loss = -pos_similarity + torch.logsumexp(all_similarities, dim=0)
        
        return loss
        
    def forward(self, hidden_states, scene_id=None):
        """
        Compute temporal contrastive loss.
        
        Args:
            hidden_states: [batch_size, hidden_dim] raw neural states
            scene_id: Current scene identifier
            
        Returns:
            loss: scalar contrastive loss (0 if no valid pairs available)
        """
        # Update scene tracking
        if scene_id is not None and scene_id != self.current_scene_id:
            self.reset_sequence(scene_id)
        
        # Project features to contrastive space
        projected_features = self.project_features(hidden_states)
        
        # Get positive pairs from history
        positive_pairs, valid_indices = self.get_positive_pairs(projected_features)
        
        # If no positive pairs available, store current features and return zero loss
        if len(positive_pairs) == 0:
            # Store current features in history
            for batch_idx in range(projected_features.shape[0]):
                self.history_buffer.append((
                    projected_features[batch_idx].detach(),
                    self.current_scene_id,
                    self.timestep
                ))
            
            self.timestep += 1
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get negative samples
        negatives = self.get_negative_samples(self.negative_samples)
        
        # Compute contrastive loss for each positive pair
        total_loss = 0.0
        n_pairs = 0
        
        for anchor, positive in positive_pairs:
            if negatives.shape[0] > 0:  # Only compute if negatives available
                loss = self.compute_contrastive_loss(anchor, positive, negatives)
                total_loss += loss
                n_pairs += 1
        
        # Store current features in history for future positive pairs
        for batch_idx in range(projected_features.shape[0]):
            self.history_buffer.append((
                projected_features[batch_idx].detach(),
                self.current_scene_id,
                self.timestep
            ))
        
        self.timestep += 1
        
        # Return average loss over pairs
        if n_pairs > 0:
            return total_loss / n_pairs
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
    def get_loss_components(self, hidden_states, scene_id=None):
        """
        Get detailed loss components for monitoring.
        
        Args:
            hidden_states: [batch_size, hidden_dim] raw neural states
            scene_id: Current scene identifier
            
        Returns:
            dict with loss components and statistics
        """
        projected_features = self.project_features(hidden_states)
        positive_pairs, valid_indices = self.get_positive_pairs(projected_features)
        
        return {
            'n_positive_pairs': len(positive_pairs),
            'n_negatives_available': len(self.negative_buffer),
            'history_buffer_size': len(self.history_buffer),
            'current_timestep': self.timestep,
            'current_scene_id': self.current_scene_id,
            'projection_norm': projected_features.norm(dim=-1).mean().item()
        }


class BatchTemporalContrastiveLoss(nn.Module):
    """
    Batch-aware temporal contrastive loss that integrates with existing RNN training.
    Works with the same batching approach as energy efficiency RNN.
    """
    
    def __init__(self, temperature=0.07, n_back=3, projection_dim=128,
                 hidden_dim=512, device='cpu', negative_samples=8):
        super(BatchTemporalContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.n_back = n_back
        self.projection_dim = projection_dim
        self.device = device
        self.negative_samples = negative_samples
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, projection_dim),
            nn.LayerNorm(projection_dim)
        ).to(device)
        
        # Global history buffer across all batches for negative sampling
        self.global_negative_buffer = deque(maxlen=2000)
        
        # Current batch temporal history: [timestep][batch_idx] -> features
        self.current_batch_history = []
        self.current_timestep = 0
        
        print(f"BatchTemporalContrastiveLoss initialized for energy efficiency batching:")
        print(f"  Temperature: {temperature}, N-back: {n_back}")
        print(f"  Projection dim: {projection_dim}, Negative samples: {negative_samples}")
        
    def reset_sequence(self):
        """Reset for new batch - called at start of each batch processing."""
        # Store some features from current batch as future negatives
        if len(self.current_batch_history) > 0:
            # Sample features from random timesteps and batch items
            for timestep_features in self.current_batch_history[-5:]:  # Last 5 timesteps
                if len(timestep_features) > 0:
                    # Sample random subset of batch items
                    n_to_sample = min(len(timestep_features), 10)
                    sampled_indices = random.sample(range(len(timestep_features)), n_to_sample)
                    for idx in sampled_indices:
                        self.global_negative_buffer.append(timestep_features[idx].detach())
        
        # Reset current batch history
        self.current_batch_history = []
        self.current_timestep = 0
        
    def forward(self, hidden_states):
        """
        Compute temporal contrastive loss for current timestep.
        
        Args:
            hidden_states: [batch_size, hidden_dim] hidden states from RNN
            
        Returns:
            loss: scalar contrastive loss
        """
        batch_size = hidden_states.shape[0]
        
        # Project features to contrastive space
        projected = self.project_features(hidden_states)
        
        # Ensure we have enough history for current batch
        while len(self.current_batch_history) <= self.current_timestep:
            self.current_batch_history.append([])
        
        # Store current features
        current_features = [projected[b] for b in range(batch_size)]
        self.current_batch_history[self.current_timestep] = current_features
        
        total_loss = 0.0
        n_pairs = 0
        
        # Look for positive pairs from n_back timesteps ago in current batch
        positive_timestep = self.current_timestep - self.n_back
        if positive_timestep >= 0 and positive_timestep < len(self.current_batch_history):
            positive_features = self.current_batch_history[positive_timestep]
            
            # Each sequence in batch forms positive pair with its past self
            for b in range(min(batch_size, len(positive_features))):
                anchor = current_features[b]
                positive = positive_features[b]
                
                # Collect negatives from different sources
                negatives = []
                
                # 1. Other sequences in current batch (different scenes)
                for other_b in range(batch_size):
                    if other_b != b:
                        negatives.append(current_features[other_b])
                
                # 2. Other sequences from positive timestep
                for other_b in range(len(positive_features)):
                    if other_b != b:
                        negatives.append(positive_features[other_b])
                
                # 3. Random features from global negative buffer (different batches/scenes)
                if len(self.global_negative_buffer) > 0:
                    n_global_negs = min(self.negative_samples // 2, len(self.global_negative_buffer))
                    global_negs = random.sample(list(self.global_negative_buffer), n_global_negs)
                    negatives.extend(global_negs)
                
                # 4. Random features from other timesteps in current batch
                for t in range(len(self.current_batch_history)):
                    if t != self.current_timestep and t != positive_timestep:
                        timestep_features = self.current_batch_history[t]
                        if len(timestep_features) > 0:
                            random_feat = random.choice(timestep_features)
                            negatives.append(random_feat)
                
                # Sample negatives if we have too many
                if len(negatives) > self.negative_samples:
                    negatives = random.sample(negatives, self.negative_samples)
                
                # Compute InfoNCE loss if we have enough negatives
                if len(negatives) >= 2:  # Minimum threshold
                    negatives_tensor = torch.stack(negatives)
                    
                    # InfoNCE loss computation
                    pos_sim = torch.dot(anchor, positive) / self.temperature
                    neg_sims = torch.mv(negatives_tensor, anchor) / self.temperature
                    
                    all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                    loss = -pos_sim + torch.logsumexp(all_sims, dim=0)
                    
                    total_loss += loss
                    n_pairs += 1
        
        self.current_timestep += 1
        
        # Return average loss or zero if no pairs
        if n_pairs > 0:
            return total_loss / n_pairs
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
    def project_features(self, hidden_states):
        """Project and normalize features for contrastive learning."""
        projected = self.projection_head(hidden_states)
        return F.normalize(projected, dim=-1)
        
    def get_stats(self):
        """Get statistics for monitoring."""
        return {
            'current_timestep': self.current_timestep,
            'batch_history_length': len(self.current_batch_history),
            'global_negatives': len(self.global_negative_buffer),
            'n_back': self.n_back,
            'temperature': self.temperature
        }


# Integration with existing RNN loss system
def temporal_contrastive_loss(loss_terms):
    """
    Temporal contrastive loss function for the existing loss parsing system.
    Compatible with energy efficiency RNN batching approach.
    
    Args:
        loss_terms: List containing [hidden_states, ...]
    
    Returns:
        loss_fn: Function that computes temporal contrastive loss
        loss_arg: Arguments for the loss function
    """
    if len(loss_terms) < 1:
        raise ValueError("Temporal contrastive loss requires hidden states")
    
    hidden_states = loss_terms[0]  # First element should be hidden states
    
    def loss_fn(states):
        # This will be called by the existing loss computation system
        return states  # Return states for contrastive comparison
    
    return loss_fn, hidden_states


def parse_temporal_contrastive_loss(loss_string, loss_terms):
    """
    Parse temporal contrastive loss string for integration with existing system.
    Add this to functions.py parse_loss function.
    """
    if 'temporal_contrastive' in loss_string:
        return temporal_contrastive_loss(loss_terms)
    else:
        # Fallback to existing parsing
        from functions import parse_loss
        return parse_loss(loss_string, loss_terms)


# Factory function for easy integration
def create_temporal_contrastive_loss(loss_type='single', **kwargs):
    """
    Factory function to create temporal contrastive loss.
    
    Args:
        loss_type: 'single' for single sequence, 'batch' for batch processing
        **kwargs: Arguments for loss initialization
        
    Returns:
        TemporalContrastiveLoss instance
    """
    if loss_type == 'single':
        return TemporalContrastiveLoss(**kwargs)
    elif loss_type == 'batch':
        return BatchTemporalContrastiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")