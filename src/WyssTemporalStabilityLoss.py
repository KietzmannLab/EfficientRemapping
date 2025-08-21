"""
Temporal Stability Loss Implementation based on Wyss, König, Verschure (2006).
Three-term objective function for hierarchical visual processing networks.

Mathematical formulation:
ω_l = -∑_i ⟨(A_l^i(t) - A_l^i(t - s_l'))²⟩_t / var_t(A_l^i) 
      - β ∑_{i≠j} (ρ_{ij}^t)² 
      - C ∑_i ⟨A_l^i⟩_t

Where:
- A_l^i(t): Activity of unit i at level l, time t
- s_l' = 2^(l-1): Time delay (increases with hierarchy level)
- ρ_{ij}^t: Temporal correlation between units i,j
- β = 5/N_l, C = 20/N_l: Level-specific weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np

class RunningStats:
    """
    Running statistics tracker for temporal correlations, means, and variances.
    Uses exponential decay with τ = 1000 time steps as in the paper.
    """
    
    def __init__(self, num_units, device='cpu', decay_rate=0.001):
        """
        Args:
            num_units: Number of units in the layer
            device: PyTorch device
            decay_rate: 1/τ where τ = 1000 time steps
        """
        self.num_units = num_units
        self.device = device
        self.decay_rate = decay_rate
        self.alpha = 1 - decay_rate  # Exponential decay factor
        
        # Running statistics
        self.mean = torch.zeros(num_units, device=device)
        self.var = torch.ones(num_units, device=device)  # Initialize to 1 to avoid division by zero
        
        # Correlation matrix for all unit pairs
        self.correlation_matrix = torch.zeros(num_units, num_units, device=device)
        self.covariance_matrix = torch.zeros(num_units, num_units, device=device)
        
        # Keep track of updates
        self.num_updates = 0
        
    def update(self, activity):
        """
        Update running statistics with new activity vector.
        
        Args:
            activity: [batch_size, num_units] or [num_units]
        """
        if len(activity.shape) == 2:
            # Average over batch dimension
            activity = activity.mean(dim=0)
        
        # Update count
        self.num_updates += 1
        
        # Update running mean
        self.mean = self.alpha * self.mean + self.decay_rate * activity
        
        # Update running variance
        centered_activity = activity - self.mean
        self.var = self.alpha * self.var + self.decay_rate * (centered_activity ** 2)
        
        # Update covariance matrix for correlations
        # Cov(i,j) = E[(X_i - μ_i)(X_j - μ_j)]
        outer_product = torch.outer(centered_activity, centered_activity)
        self.covariance_matrix = self.alpha * self.covariance_matrix + self.decay_rate * outer_product
        
        # Update correlation matrix: ρ_ij = Cov(i,j) / (σ_i * σ_j)
        std_outer = torch.outer(torch.sqrt(self.var + 1e-8), torch.sqrt(self.var + 1e-8))
        self.correlation_matrix = self.covariance_matrix / (std_outer + 1e-8)
        
    def get_correlations_squared(self, exclude_diagonal=True):
        """
        Get sum of squared correlations for decorrelation term.
        
        Args:
            exclude_diagonal: Whether to exclude self-correlations (always 1.0)
            
        Returns:
            Sum of squared correlations
        """
        if exclude_diagonal:
            # Create mask to exclude diagonal
            mask = ~torch.eye(self.num_units, device=self.device).bool()
            return (self.correlation_matrix[mask] ** 2).sum()
        else:
            return (self.correlation_matrix ** 2).sum()


class WyssTemporalStabilityLoss(nn.Module):
    """
    Wyss et al. (2006) temporal stability loss with three terms:
    1. Temporal stability: minimize activity differences over time
    2. Decorrelation: minimize correlations between units
    3. Sparsity: minimize average activity
    """
    
    def __init__(self, level_idx=0, num_units=512, device='cpu', 
                 statistics_decay=0.001, history_length=None):
        """
        Args:
            level_idx: Hierarchy level (0-based), affects time constants
            num_units: Number of units in the layer
            device: PyTorch device
            statistics_decay: Decay rate for running statistics (1/τ)
            history_length: Length of activity history buffer (default: 2^level_idx)
        """
        super(WyssTemporalStabilityLoss, self).__init__()
        
        self.level_idx = level_idx
        self.num_units = num_units
        self.device = device
        
        # Time constants from paper
        self.time_delay = max(1, 2 ** (level_idx - 1))  # s_l' = 2^(l-1), min 1
        self.time_constant = 2 ** level_idx  # s_l = 2^l
        
        # Coefficients from paper
        self.beta = 5.0 / num_units  # β = 5/N_l
        self.c = 20.0 / num_units    # C = 20/N_l
        
        # Activity history buffer
        if history_length is None:
            history_length = max(10, self.time_delay + 5)  # Buffer extra for safety
        self.activity_history = deque(maxlen=history_length)
        
        # Running statistics tracker
        self.running_stats = RunningStats(num_units, device, statistics_decay)
        
        # Previous output for leaky integration
        self.prev_output = torch.zeros(num_units, device=device)
        
        print(f"WyssTemporalStabilityLoss Level {level_idx}:")
        print(f"  Time delay: {self.time_delay}")
        print(f"  Beta (decorrelation): {self.beta:.6f}")
        print(f"  C (sparsity): {self.c:.6f}")
        print(f"  History length: {history_length}")
        
    def reset_sequence(self):
        """Reset for new sequence (clear history)"""
        self.activity_history.clear()
        self.prev_output.zero_()
        
    def compute_unit_activity(self, hidden_states):
        """
        Compute unit activity with saturating activation f(x) = 1 - e^(-x²)
        
        Args:
            hidden_states: [batch_size, hidden_dim] raw neural activations
            
        Returns:
            activity: [batch_size, num_units] unit activities
        """
        # Apply saturating nonlinearity from paper
        energy_squared = hidden_states ** 2
        activity = 1.0 - torch.exp(-energy_squared)
        return activity
        
    def compute_level_output(self, activity):
        """
        Leaky integration with mean-correction and normalization.
        
        Args:
            activity: [batch_size, num_units] current activity
            
        Returns:
            output: [batch_size, num_units] integrated output
        """
        # Average over batch for single output
        if len(activity.shape) == 2:
            activity_mean = activity.mean(dim=0)
        else:
            activity_mean = activity
            
        # Normalize activity: A'(t) = (A(t) - ⟨A⟩_t) / √var_t(A)
        activity_centered = activity_mean - self.running_stats.mean
        activity_normalized = activity_centered / (torch.sqrt(self.running_stats.var) + 1e-8)
        
        # Leaky integration: τ = 1/s_l
        tau = 1.0 / self.time_constant
        output = tau * activity_normalized + (1 - tau) * self.prev_output
        
        # Update previous output
        self.prev_output = output.detach()
        
        return output.unsqueeze(0) if len(activity.shape) == 2 else output
        
    def temporal_stability_term(self, current_activity, past_activity):
        """
        Term 1: Temporal stability
        Minimize ∑_i ⟨(A_l^i(t) - A_l^i(t - s_l'))²⟩_t / var_t(A_l^i)
        
        Args:
            current_activity: [batch_size, num_units] current activity
            past_activity: [batch_size, num_units] past activity
            
        Returns:
            stability_loss: scalar tensor
        """
        # Compute squared differences
        diff_squared = (current_activity - past_activity) ** 2
        
        # Average over batch and units
        diff_mean = diff_squared.mean(dim=0)  # [num_units]
        
        # Normalize by variance to account for different unit scales
        normalized_diff = diff_mean / (self.running_stats.var + 1e-8)
        
        # Sum over units
        stability_loss = normalized_diff.sum()
        
        return stability_loss
        
    def decorrelation_term(self):
        """
        Term 2: Decorrelation
        Minimize β ∑_{i≠j} (ρ_{ij}^t)²
        
        Returns:
            decorrelation_loss: scalar tensor
        """
        correlations_squared_sum = self.running_stats.get_correlations_squared(exclude_diagonal=True)
        return self.beta * correlations_squared_sum
        
    def sparsity_term(self):
        """
        Term 3: Sparsity regularization
        Minimize C ∑_i ⟨A_l^i⟩_t
        
        Returns:
            sparsity_loss: scalar tensor
        """
        mean_activity_sum = self.running_stats.mean.sum()
        return self.c * mean_activity_sum
        
    def forward(self, hidden_states):
        """
        Compute complete three-term temporal stability loss.
        
        Args:
            hidden_states: [batch_size, hidden_dim] raw neural states
            
        Returns:
            total_loss: scalar tensor (note: we minimize -ω_l to maximize ω_l)
        """
        # Convert hidden states to unit activities
        activity = self.compute_unit_activity(hidden_states)
        
        # Apply leaky integration
        integrated_output = self.compute_level_output(activity)
        
        # Update running statistics (detach to prevent accumulating gradients)
        self.running_stats.update(integrated_output.detach())
        
        # Store in history (keep gradients for temporal comparison)
        self.activity_history.append(integrated_output)
        
        # Check if we have enough history for temporal comparison
        if len(self.activity_history) < self.time_delay + 1:
            # Return small loss to avoid affecting early training
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get past activity at appropriate time delay
        past_activity = self.activity_history[-(self.time_delay + 1)]
        current_activity = integrated_output
        
        # Compute three loss terms
        stability_loss = self.temporal_stability_term(current_activity, past_activity)
        decorrelation_loss = self.decorrelation_term()
        sparsity_loss = self.sparsity_term()
        
        # Total loss (we minimize -ω_l to maximize ω_l from the paper)
        total_loss = stability_loss + decorrelation_loss + sparsity_loss
        
        return total_loss
        
    def get_loss_components(self, hidden_states):
        """
        Get individual loss components for debugging/monitoring.
        
        Args:
            hidden_states: [batch_size, hidden_dim] raw neural states
            
        Returns:
            dict with individual loss components
        """
        if len(self.activity_history) < self.time_delay + 1:
            return {
                'total_loss': 0.0,
                'stability': 0.0,
                'decorrelation': 0.0,
                'sparsity': 0.0
            }
            
        # Get current and past activities
        activity = self.compute_unit_activity(hidden_states)
        integrated_output = self.compute_level_output(activity)
        past_activity = self.activity_history[-(self.time_delay + 1)]
        
        # Compute individual terms
        stability = self.temporal_stability_term(integrated_output, past_activity)
        decorrelation = self.decorrelation_term()
        sparsity = self.sparsity_term()
        total = stability + decorrelation + sparsity
        
        return {
            'total_loss': total.item(),
            'stability': stability.item(),
            'decorrelation': decorrelation.item(),
            'sparsity': sparsity.item(),
            'beta': self.beta,
            'c': self.c,
            'mean_activity': self.running_stats.mean.mean().item(),
            'mean_variance': self.running_stats.var.mean().item()
        }


def create_hierarchical_losses(hidden_sizes, device='cpu'):
    """
    Create hierarchical temporal stability losses for multi-level network.
    
    Args:
        hidden_sizes: List of hidden sizes for each level
        device: PyTorch device
        
    Returns:
        List of WyssTemporalStabilityLoss instances
    """
    losses = []
    for level_idx, hidden_size in enumerate(hidden_sizes):
        loss_fn = WyssTemporalStabilityLoss(
            level_idx=level_idx,
            num_units=hidden_size,
            device=device
        )
        losses.append(loss_fn)
        
    return losses