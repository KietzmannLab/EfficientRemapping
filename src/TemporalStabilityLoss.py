"""
Temporal stability loss functions for the efficient remapping codebase.
Alternative objective to test whether energy efficiency specifically drives predictive remapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalStabilityLoss(nn.Module):
    """
    Temporal stability loss as alternative training objective.
    Based on König et al. (2006) temporal coherence principle.
    """
    
    def __init__(self, stability_type='l2', alpha=0.1, beta=0.05, 
                 temporal_window=3, device='cpu'):
        super(TemporalStabilityLoss, self).__init__()
        self.stability_type = stability_type
        self.alpha = alpha
        self.beta = beta
        self.temporal_window = temporal_window
        self.device = device
        
        # State history for temporal comparison
        self.hidden_states_history = []
        self.reset_sequence()
        
    def reset_sequence(self):
        """Reset temporal history (call at start of new sequence)"""
        self.hidden_states_history = []
        
    def forward(self, current_hidden_states, fixation_coords=None):
        """
        Compute temporal stability loss
        
        Args:
            current_hidden_states: Current hidden states [batch_size, hidden_dim]
            fixation_coords: Current fixation coordinates [batch_size, 2] (optional)
        
        Returns:
            temporal_loss: Scalar tensor
        """
        # Store current states
        self.hidden_states_history.append(current_hidden_states.detach())
        
        # Keep only recent history
        if len(self.hidden_states_history) > self.temporal_window:
            self.hidden_states_history.pop(0)
            
        # Need at least 2 timesteps for temporal comparison
        if len(self.hidden_states_history) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.stability_type == 'l2':
            return self._l2_temporal_loss(current_hidden_states)
        elif self.stability_type == 'cosine':
            return self._cosine_temporal_loss(current_hidden_states)
        elif self.stability_type == 'combined':
            return self._combined_temporal_loss(current_hidden_states, fixation_coords)
        else:
            raise ValueError(f"Unknown stability type: {self.stability_type}")
    
    def _l2_temporal_loss(self, current_states):
        """L2 temporal consistency loss"""
        previous_states = self.hidden_states_history[-2]
        return self.alpha * F.mse_loss(current_states, previous_states)
    
    def _cosine_temporal_loss(self, current_states):
        """Cosine similarity temporal consistency loss"""
        previous_states = self.hidden_states_history[-2]
        cosine_sim = F.cosine_similarity(current_states, previous_states, dim=1)
        return self.alpha * (1 - cosine_sim).mean()
    
    def _combined_temporal_loss(self, current_states, fixation_coords=None):
        """Combined temporal stability with spatial consistency"""
        total_loss = 0.0
        
        # Basic temporal stability
        previous_states = self.hidden_states_history[-2]
        total_loss += self.alpha * F.mse_loss(current_states, previous_states)
        
        # Multi-step consistency (if enough history)
        if len(self.hidden_states_history) >= 3:
            for i in range(len(self.hidden_states_history) - 2):
                older_states = self.hidden_states_history[i]
                weight = self.beta * (0.5 ** (len(self.hidden_states_history) - 1 - i))
                total_loss += weight * F.mse_loss(current_states, older_states)
        
        return total_loss


def temporal_stability_loss(loss_terms):
    """
    Temporal stability loss function for the existing loss parsing system.
    
    Args:
        loss_terms: List containing [hidden_states, ...]
    
    Returns:
        loss_fn: Function that computes temporal stability loss
        loss_arg: Arguments for the loss function
    """
    if len(loss_terms) < 1:
        raise ValueError("Temporal stability loss requires hidden states")
    
    hidden_states = loss_terms[0]  # First element should be hidden states
    
    def loss_fn(states):
        # This will be called by the existing loss computation system
        return states  # Return states for temporal comparison
    
    return loss_fn, hidden_states


def parse_temporal_loss(loss_string, loss_terms):
    """
    Parse temporal stability loss string for integration with existing system.
    Add this to functions.py parse_loss function.
    """
    if 'temporal_stability' in loss_string:
        return temporal_stability_loss(loss_terms)
    else:
        # Fallback to existing parsing
        return parse_loss(loss_string, loss_terms)