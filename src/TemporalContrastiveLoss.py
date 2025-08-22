"""
Fixed Temporal Contrastive Learning Loss for Fixation Sequences.

Key fixes:
1. Increased default temperature from 0.07 to 0.5 for stable gradients
2. Proper InfoNCE implementation using F.cross_entropy
3. Minimum 8 negatives required for meaningful contrastive learning
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
    
    def __init__(self, temperature=0.5, n_back=3, projection_dim=128, 
                 hidden_dim=512, device='cpu', negative_samples=16):
        """
        Args:
            temperature: Temperature parameter for contrastive loss (increased from 0.07)
            n_back: Number of timesteps back for positive pairs
            projection_dim: Dimension of projection head output
            hidden_dim: Dimension of input hidden states
            device: PyTorch device
            negative_samples: Number of negative samples per positive pair (increased from 8)
        """
        super(TemporalContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.n_back = n_back
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.negative_samples = negative_samples
        self.min_negatives = 8  # Minimum negatives required
        
        # Projection head to map hidden states to contrastive space
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, projection_dim),
            nn.LayerNorm(projection_dim)
        ).to(device)
        
        # History buffer for temporal positive pairs
        self.history_buffer = deque(maxlen=n_back + 10)
        
        # Negative sample buffer across different scenes
        self.negative_buffer = deque(maxlen=1000)
        
        # Track current scene for positive/negative sampling
        self.current_scene_id = None
        self.timestep = 0
        
        print(f"TemporalContrastiveLoss initialized (FIXED):")
        print(f"  Temperature: {temperature} (increased for stability)")
        print(f"  N-back timesteps: {n_back}")
        print(f"  Projection dim: {projection_dim}")
        print(f"  Negative samples: {negative_samples} (min: {self.min_negatives})")
        
    def reset_sequence(self, scene_id=None):
        """Reset for new sequence/scene."""
        # Store previous scene's features as negatives before resetting
        if self.current_scene_id is not None and len(self.history_buffer) > 0:
            scene_features = [item[0] for item in self.history_buffer 
                            if item[1] == self.current_scene_id]
            if len(scene_features) > 0:
                n_to_add = min(len(scene_features), 10)
                sampled_features = random.sample(scene_features, n_to_add)
                for feat in sampled_features:
                    self.negative_buffer.append(feat.detach())
        
        self.history_buffer.clear()
        self.current_scene_id = scene_id
        self.timestep = 0
        
    def project_features(self, hidden_states):
        """Project hidden states to contrastive learning space."""
        if len(hidden_states.shape) == 1:
            hidden_states = hidden_states.unsqueeze(0)
        
        projected = self.projection_head(hidden_states)
        # L2 normalize for cosine similarity
        projected = F.normalize(projected, dim=-1)
        
        return projected
        
    def get_positive_pairs(self, current_features):
        """Get positive pairs from history buffer."""
        positive_pairs = []
        valid_indices = []
        
        target_timestep = self.timestep - self.n_back
        
        for i, (feat, scene_id, timestep) in enumerate(self.history_buffer):
            if (scene_id == self.current_scene_id and 
                timestep == target_timestep):
                for batch_idx in range(current_features.shape[0]):
                    positive_pairs.append((
                        current_features[batch_idx],  # anchor
                        feat  # positive
                    ))
                    valid_indices.append(batch_idx)
                break
                
        return positive_pairs, valid_indices
        
    def get_negative_samples(self, n_negatives):
        """Sample negative features from different scenes."""
        if len(self.negative_buffer) < self.min_negatives:
            # Not enough negatives - return None to skip contrastive loss
            return None
        
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
        Compute InfoNCE contrastive loss using proper F.cross_entropy.
        
        Args:
            anchor: [projection_dim] anchor features
            positive: [projection_dim] positive features  
            negatives: [n_negatives, projection_dim] negative features
            
        Returns:
            loss: scalar contrastive loss
        """
        # Compute similarities (cosine similarity since features are normalized)
        pos_similarity = torch.dot(anchor, positive) / self.temperature
        neg_similarities = torch.mv(negatives, anchor) / self.temperature
        
        # Stack positive and negative similarities
        # Positive should be first (label = 0)
        logits = torch.cat([pos_similarity.unsqueeze(0), neg_similarities])
        
        # Create label for positive (index 0)
        labels = torch.zeros(1, dtype=torch.long, device=self.device)
        
        # Compute cross-entropy loss (proper InfoNCE)
        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        
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
        
        # Skip if insufficient negatives
        if negatives is None:
            # Store current features and return zero loss
            for batch_idx in range(projected_features.shape[0]):
                self.history_buffer.append((
                    projected_features[batch_idx].detach(),
                    self.current_scene_id,
                    self.timestep
                ))
            
            self.timestep += 1
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute contrastive loss for each positive pair
        total_loss = 0.0
        n_pairs = 0
        
        for anchor, positive in positive_pairs:
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


class BatchTemporalContrastiveLoss(nn.Module):
    """
    Fixed batch-aware temporal contrastive loss.
    """
    
    def __init__(self, temperature=0.5, n_back=3, projection_dim=128,
                 hidden_dim=512, device='cpu', negative_samples=16):
        super(BatchTemporalContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.n_back = n_back
        self.projection_dim = projection_dim
        self.device = device
        self.negative_samples = negative_samples
        self.min_negatives = 8  # Minimum negatives required
        
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
        
        print(f"BatchTemporalContrastiveLoss initialized (FIXED):")
        print(f"  Temperature: {temperature} (increased for stability)")
        print(f"  N-back: {n_back}, Projection dim: {projection_dim}")
        print(f"  Negative samples: {negative_samples} (min: {self.min_negatives})")
        
    def reset_sequence(self):
        """Reset for new batch - called at start of each batch processing."""
        # Store some features from current batch as future negatives
        if len(self.current_batch_history) > 0:
            for timestep_features in self.current_batch_history[-5:]:
                if len(timestep_features) > 0:
                    n_to_sample = min(len(timestep_features), 10)
                    sampled_indices = random.sample(range(len(timestep_features)), n_to_sample)
                    for idx in sampled_indices:
                        self.global_negative_buffer.append(timestep_features[idx].detach())
        
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
                
                # FIXED: Only collect negatives from DIFFERENT SCENES (not current batch)
                negatives = []
                
                # Only use global negative buffer from previous batches/scenes
                # This ensures ALL negatives are from different scenes
                if len(self.global_negative_buffer) >= self.min_negatives:
                    # Sample all negatives from global buffer (guaranteed different scenes)
                    n_available = len(self.global_negative_buffer)
                    n_to_sample = min(self.negative_samples, n_available)
                    global_negs = random.sample(list(self.global_negative_buffer), n_to_sample)
                    negatives.extend(global_negs)
                
                # Only compute loss if we have enough negatives
                if len(negatives) >= self.min_negatives:
                    negatives_tensor = torch.stack(negatives)
                    
                    # Proper InfoNCE loss computation using F.cross_entropy
                    pos_sim = torch.dot(anchor, positive) / self.temperature
                    neg_sims = torch.mv(negatives_tensor, anchor) / self.temperature
                    
                    # Positive similarity should be first (label = 0)
                    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                    labels = torch.zeros(1, dtype=torch.long, device=self.device)
                    
                    loss = F.cross_entropy(logits.unsqueeze(0), labels)
                    
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
            'temperature': self.temperature,
            'min_negatives_required': self.min_negatives
        }


# Factory function for easy integration
def create_temporal_contrastive_loss(loss_type='batch', **kwargs):
    """
    Factory function to create temporal contrastive loss.
    
    Args:
        loss_type: 'single' for single sequence, 'batch' for batch processing
        **kwargs: Arguments for loss initialization
        
    Returns:
        TemporalContrastiveLoss instance
    """
    # Set fixed defaults
    defaults = {
        'temperature': 0.5,  # Fixed: increased from 0.07
        'negative_samples': 16,  # Fixed: increased from 8
    }
    defaults.update(kwargs)
    
    if loss_type == 'single':
        return TemporalContrastiveLoss(**defaults)
    elif loss_type == 'batch':
        return BatchTemporalContrastiveLoss(**defaults)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")