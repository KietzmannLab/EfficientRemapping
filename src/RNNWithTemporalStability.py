"""
RNN State class with temporal stability loss integration.
Maintains the same interface as the original RNN.State class.
"""

import torch
import torch.nn.functional as F
from RNN import State
from WyssTemporalStabilityLoss import WyssTemporalStabilityLoss

class RNNWithTemporalStability(State):
    """
    Modified RNN State class with temporal stability loss integration.
    Maintains the same interface as the original RNN.State class.
    """
    
    def __init__(self, activation_func, optimizer, lr, input_size, hidden_size, 
                 title, device, level_idx=0, use_fixation=True, use_conv=False, 
                 use_lstm=False, warp_imgs=False, use_resNet=False, time_steps_img=6, 
                 time_steps_cords=3, mnist=False, twolayer=False, dropout=0,
                 deterministic=True, weights_init=None, prevbatch=False,
                 conv=False, seed=None, disentangled_loss=False, 
                 useReservoir=False, **kwargs):
        """
        Initialize RNN with temporal stability loss.
        Same parameters as original RNN.State plus temporal stability options.
        """
        
        # Initialize the base RNN with all original parameters
        super().__init__(
            activation_func=activation_func,
            optimizer=optimizer,
            lr=lr,
            input_size=input_size,
            hidden_size=hidden_size,
            title=title,
            device=device,
            deterministic=deterministic,
            weights_init=weights_init,
            prevbatch=prevbatch,
            conv=conv,
            use_fixation=use_fixation,
            seed=seed,
            use_conv=use_conv,
            use_lstm=use_lstm,
            warp_imgs=warp_imgs,
            use_resNet=use_resNet,
            time_steps_img=time_steps_img,
            time_steps_cords=time_steps_cords,
            mnist=mnist,
            twolayer=twolayer,
            dropout=dropout,
            disentangled_loss=disentangled_loss,
            useReservoir=useReservoir
        )
        
        # Initialize Wyss temporal stability loss
        self.temporal_loss_fn = WyssTemporalStabilityLoss(
            level_idx=level_idx,
            num_units=hidden_size,
            device=device
        )
        
    def run(self, batch, fixations, loss_fn='temporal_stability', state=None):
        """
        Run batch through model with temporal stability loss.
        
        Args:
            batch: Input batch
            fixations: Fixation coordinates
            loss_fn: Loss function string (use 'temporal_stability')
            state: Model state (unused, for compatibility)
        
        Returns:
            loss: Total temporal stability loss
            loss_detached: Detached loss for logging
            metadata: Training metadata
        """
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
        
        # Reset temporal history for new sequence
        self.temporal_loss_fn.reset_sequence()
        
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
                
                # Compute Wyss temporal stability loss on hidden states
                # Use the last layer's hidden state for temporal consistency
                if isinstance(recurrent_state, list) and len(recurrent_state) > 0:
                    if isinstance(recurrent_state[0], list) and len(recurrent_state[0]) > 0:
                        # Get the last layer's hidden state
                        last_hidden = recurrent_state[0][-1]
                        if isinstance(last_hidden, tuple):
                            last_hidden = last_hidden[0]  # For LSTM
                        temporal_loss = self.temporal_loss_fn(last_hidden)
                        total_loss = total_loss + temporal_loss
                elif h is not None:
                    # Fallback to using h if recurrent_state is not available
                    temporal_loss = self.temporal_loss_fn(h)
                    total_loss = total_loss + temporal_loss
        
        return total_loss, total_loss.detach(), None
    
    def test(self, dataset, loss_fn='temporal_stability'):
        """Test model with temporal stability loss"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (batch, fixations) in enumerate(dataset):
                loss, _, _ = self.run(batch, fixations, loss_fn)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def state_dict(self):
        """Return the state dict of the underlying model"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into the underlying model"""
        return self.model.load_state_dict(state_dict)