#!/usr/bin/env python3
"""
Test script for RNN with temporal stability loss.
Evaluates the model trained with temporal stability objective.
"""

import sys
import torch
import argparse
import numpy as np
from RNNWithTemporalStability import RNNWithTemporalStability
from Dataset import Dataset
import functions

def test_temporal_stability_model(model_path, dataset_path, device='cuda', batch_size=32):
    """
    Test a model trained with temporal stability loss.
    
    Args:
        model_path: Path to saved model checkpoint
        dataset_path: Path to test dataset
        device: Computing device ('cuda' or 'cpu')
        batch_size: Batch size for testing
    
    Returns:
        test_loss: Average test loss
    """
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = Dataset(dataset_path, train=False)
    
    # Initialize model with temporal stability loss
    # Use same parameters as in the paper
    model = RNNWithTemporalStability(
        activation_func=torch.relu,
        optimizer=torch.optim.Adam,
        lr=0.001,  # Will be overridden when loading
        input_size=128*128,  # Standard size from paper
        hidden_size=2048,    # Hidden size from paper
        title="temporal_stability_test",
        device=device,
        temporal_loss_type='l2',
        temporal_alpha=0.1,
        use_fixation=True,
        use_conv=False,
        use_lstm=False,
        warp_imgs=True,
        use_resNet=False,
        time_steps_img=6,    # From paper
        time_steps_cords=3,  # From paper
        mnist=False,
        twolayer=True,
        dropout=0
    )
    
    # Load trained model weights
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.model.eval()
    
    # Test the model
    print("Testing model with temporal stability loss...")
    test_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        # Create data loader
        loader = dataset.create_batches(batch_size=batch_size, shuffle=False)
        
        for batch, fixations in loader:
            # Run batch through model with temporal stability loss
            loss, _, _ = model.run(batch, fixations, loss_fn='temporal_stability')
            test_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Processed {num_batches} batches, avg loss: {test_loss/num_batches:.6f}")
    
    avg_test_loss = test_loss / num_batches if num_batches > 0 else 0
    print(f"\nFinal Test Results:")
    print(f"Average temporal stability loss: {avg_test_loss:.6f}")
    print(f"Total batches processed: {num_batches}")
    
    return avg_test_loss


def main():
    parser = argparse.ArgumentParser(description='Test RNN with temporal stability loss')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Test the model
    test_loss = test_temporal_stability_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    print(f"\nTesting completed. Final loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()