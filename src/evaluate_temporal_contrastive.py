#!/usr/bin/env python3
"""
Evaluation script for temporal contrastive models.
Evaluates L1 preactivation loss across all layers/timesteps and XY decoding performance
with confidence intervals using existing infrastructure.
"""

import argparse
import torch
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
import RNN
import functions
from H5dataset import H5dataset
from ClosedFormDecoding import getFeatures


def evaluate_preactivation_loss(net, test_set, batch_size=1024, confidence=0.99):
    """
    Evaluate L1 preactivation loss across all layers and timesteps with confidence intervals.
    Uses existing RNN infrastructure (net.run with 'l1_all' loss function).
    
    Args:
        net: Trained RNN model
        test_set: Test dataset
        batch_size: Batch size for evaluation
        confidence: Confidence level for intervals
        
    Returns:
        dict with loss statistics and confidence intervals
    """
    print("Evaluating L1 preactivation loss (energy efficiency)...")
    
    net.model.eval()
    losses = []
    
    with torch.no_grad():
        loader = test_set.create_batches(batch_size=batch_size, shuffle=False)
        
        for batch_idx, (batch, fixations) in enumerate(loader):
            # Use existing infrastructure: net.run with 'l1_all' loss function
            loss, _, _ = net.run(batch, fixations, 'l1_all', None)
            losses.append(loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches, current avg loss: {np.mean(losses):.6f}")
    
    # Calculate statistics with confidence intervals (following codebase method from plot.py:1108-1109)
    losses = np.array(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    sem_loss = scipy.stats.sem(losses)
    
    # 99% Confidence interval using normal distribution (as in existing codebase)
    ci_lower, ci_upper = scipy.stats.norm.interval(
        confidence=confidence, 
        loc=mean_loss, 
        scale=sem_loss
    )
    
    results = {
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'sem_loss': sem_loss,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence': confidence,
        'n_samples': len(losses),
        'raw_losses': losses
    }
    
    print(f"\nL1 Preactivation Loss Results:")
    print(f"Mean loss: {mean_loss:.6f}")
    print(f"Standard deviation: {std_loss:.6f}")
    print(f"Standard error: {sem_loss:.6f}")
    print(f"{confidence*100}% Confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"Number of test batches: {len(losses)}")
    
    return results


def evaluate_xy_decoding(net, train_set, test_set, confidence=0.99):
    """
    Evaluate XY position decoding performance using closed-form linear regression.
    Uses existing ClosedFormDecoding infrastructure - no pre-training needed.
    
    Args:
        net: Trained RNN model 
        train_set: Training dataset for fitting decoder
        test_set: Test dataset for evaluation
        confidence: Confidence level for intervals
        
    Returns:
        dict with decoding performance statistics and confidence intervals
    """
    print("\nEvaluating XY position decoding performance using closed-form approach...")
    
    results = {}
    
    # Test layers 1 and 2 (following existing infrastructure)
    for layer in [1, 2]:
        results[f'layer_{layer}'] = {}
        
        # Test different timesteps
        for timestep in range(6):
            print(f"Testing layer {layer}, timestep {timestep}")
            
            # Test both current position and next position decoding
            for mode, condition in [('global', 'current'), ('next_relative', 'next')]:
                print(f"  Mode: {mode} (predicting {condition} fixation)")
                
                # Extract features using existing infrastructure
                X_train, Y_train = getFeatures(net, train_set, layer_idx=[layer], timestep=timestep)
                X_test, Y_test = getFeatures(net, test_set, layer_idx=[layer], timestep=timestep)
                
                # Handle relative coordinates if needed
                if mode == 'next_relative':
                    # Convert to relative coordinates (simplified version of changeToRelativeCoordinates)
                    if timestep is None:
                        # For all timesteps, convert to relative coordinates
                        Y_train_rel = np.zeros_like(Y_train)
                        Y_test_rel = np.zeros_like(Y_test)
                        seq_len = 7
                        for i in range(len(Y_train)):
                            seq_idx = i % (seq_len * 6)  # 6 timesteps per fixation
                            fix_idx = seq_idx // 6
                            if fix_idx < seq_len - 1:  # Not the last fixation
                                next_fix_start = ((i // (seq_len * 6)) * seq_len + fix_idx + 1) * 6
                                if next_fix_start < len(Y_train):
                                    Y_train_rel[i] = Y_train[next_fix_start] - Y_train[i]
                        Y_train = Y_train_rel
                        Y_test = Y_test_rel
                    else:
                        # For specific timestep, convert to next fixation coordinates
                        Y_train_rel = np.zeros_like(Y_train)
                        Y_test_rel = np.zeros_like(Y_test)
                        seq_len = 7
                        for i in range(len(Y_train)):
                            fix_idx = i % seq_len
                            if fix_idx < seq_len - 1:  # Not the last fixation
                                next_fix_idx = i + 1
                                if next_fix_idx < len(Y_train):
                                    Y_train_rel[i] = Y_train[next_fix_idx] - Y_train[i]
                        Y_train = Y_train_rel
                        Y_test = Y_test_rel
                
                # Normalize features
                norm_factors = np.mean(X_train, axis=0)
                norm_factors += np.ones_like(norm_factors) * 0.000001
                X_train_norm = X_train / norm_factors
                X_test_norm = X_test / norm_factors
                
                # Fit linear regression (closed-form)
                reg = LinearRegression().fit(X_train_norm, Y_train)
                
                # Compute R² on test set
                test_score = reg.score(X_test_norm, Y_test)
                
                # Get predictions for detailed R² calculation
                pred_Y = reg.predict(X_test_norm)
                
                # Calculate R² per coordinate (x, y)
                target_mean = Y_test.mean(axis=0)
                s_res = np.sum(np.square(Y_test - pred_Y), axis=0)
                s_tot = np.sum(np.square(Y_test - target_mean), axis=0)
                r_squared_per_coord = 1 - np.divide(s_res, s_tot)
                
                # Bootstrap confidence intervals for R²
                n_bootstrap = 1000
                r_squared_bootstrap = []
                
                np.random.seed(42)  # Reproducible results
                n_samples = len(Y_test)
                
                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    X_boot = X_test_norm[indices]
                    Y_boot = Y_test[indices]
                    
                    # Compute R² on bootstrap sample
                    pred_boot = reg.predict(X_boot)
                    target_mean_boot = Y_boot.mean(axis=0)
                    s_res_boot = np.sum(np.square(Y_boot - pred_boot), axis=0)
                    s_tot_boot = np.sum(np.square(Y_boot - target_mean_boot), axis=0)
                    r_squared_boot = 1 - np.divide(s_res_boot, s_tot_boot)
                    r_squared_bootstrap.append(np.mean(r_squared_boot))  # Average over x,y
                
                r_squared_bootstrap = np.array(r_squared_bootstrap)
                
                # Calculate 99% confidence intervals using percentiles
                ci_lower = np.percentile(r_squared_bootstrap, (1 - confidence) / 2 * 100)
                ci_upper = np.percentile(r_squared_bootstrap, (1 + confidence) / 2 * 100)
                
                mean_r2 = np.mean(r_squared_per_coord)
                std_r2 = np.std(r_squared_bootstrap)
                
                results[f'layer_{layer}'][f'timestep_{timestep}_{condition}'] = {
                    'r_squared_mean': mean_r2,
                    'r_squared_std': std_r2,
                    'r_squared_ci_lower': ci_lower,
                    'r_squared_ci_upper': ci_upper,
                    'r_squared_x': r_squared_per_coord[0],
                    'r_squared_y': r_squared_per_coord[1],
                    'test_score': test_score,
                    'confidence': confidence,
                    'n_bootstrap': n_bootstrap,
                    'n_test_samples': len(Y_test)
                }
                
                print(f"    R² = {mean_r2:.6f} (x: {r_squared_per_coord[0]:.6f}, y: {r_squared_per_coord[1]:.6f})")
                print(f"    {confidence*100}% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                print(f"    Overall test score: {test_score:.6f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate temporal contrastive models')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained temporal contrastive model')
    parser.add_argument('--dataset_path', type=str, 
                        default='/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5',
                        help='Path to test dataset')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu), auto-detect if not specified')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for evaluation')
    parser.add_argument('--confidence', type=float, default=0.99,
                        help='Confidence level for intervals (0.99 = 99%)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save results (optional)')
    
    args = parser.parse_args()
    
    # Set device using existing infrastructure
    if args.device is None:
        device = functions.get_device()
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load datasets using existing infrastructure
    print(f"Loading datasets from {args.dataset_path}")
    train_set = H5dataset('train', args.dataset_path, device=device, use_color=False)
    test_set = H5dataset('test', args.dataset_path, device=device, use_color=False)
    
    # Load trained model using existing RNN infrastructure
    print(f"Loading temporal contrastive model from {args.model_path}")
    
    # Initialize model with same parameters as training
    # (You'll need to adjust these parameters based on your training setup)
    net = RNN.State(
        activation_func=torch.nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-4,
        input_size=128*128,
        hidden_size=2048,
        title="temporal_contrastive_eval",
        device=device,
        use_fixation=True,
        seed=42,
        use_conv=False,
        warp_imgs=False,
        use_resNet=False,
        time_steps_img=6,
        time_steps_cords=3,
        mnist=False
    )
    
    # Load temporal contrastive model checkpoint
    print(f"Loading temporal contrastive model checkpoint...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Extract the base RNN model state dict from the temporal contrastive wrapper
    if 'model' in checkpoint:
        # The checkpoint contains the full RNNWithTemporalContrastive state
        base_model_state = checkpoint['model']
        print(f"Loading base RNN model from temporal contrastive checkpoint")
        net.model.load_state_dict(base_model_state)
    elif 'model_state_dict' in checkpoint:
        # Standard checkpoint format
        model_state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        net.model.load_state_dict(model_state_dict)
    else:
        # Try to load directly
        net.model.load_state_dict(checkpoint)
    
    net.model.eval()
    print(f"Model loaded successfully!")
    
    print("=" * 80)
    print("TEMPORAL CONTRASTIVE MODEL EVALUATION")
    print("=" * 80)
    
    # 1. Evaluate L1 preactivation loss (energy efficiency)
    preactivation_results = evaluate_preactivation_loss(
        net, test_set, args.batch_size, args.confidence
    )
    
    # 2. Evaluate XY decoding performance using closed-form approach
    decoding_results = evaluate_xy_decoding(
        net, train_set, test_set, args.confidence
    )
    
    # Combine results
    all_results = {
        'preactivation_loss': preactivation_results,
        'xy_decoding': decoding_results,
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
        'batch_size': args.batch_size,
        'confidence': args.confidence,
        'device': device
    }
    
    # Save results if requested
    if args.save_results:
        import pickle
        with open(args.save_results, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nResults saved to {args.save_results}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()