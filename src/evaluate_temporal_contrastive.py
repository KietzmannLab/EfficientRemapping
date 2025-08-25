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


def evaluate_energy_efficiency(net, validation_set, confidence=0.99):
    """
    Evaluate energy efficiency using the EXACT methodology from analyseModels.py.
    Uses plot.compare_random_fixations() to compare model vs shuffled fixations.
    
    Args:
        net: Trained RNN model
        validation_set: Validation dataset (same as paper)
        confidence: Confidence level for intervals
        
    Returns:
        dict with energy efficiency results and confidence intervals
    """
    print("Evaluating energy efficiency (following analyseModels.py methodology)...")
    print("Using plot.compare_random_fixations() - same as paper...")
    
    import plot
    
    # Set the same seed as analyseModels.py
    torch.manual_seed(2553)
    np.random.seed(2553)
    
    # Use the exact same call as analyseModels.py:223
    losses, losses_random_fix, fixations = plot.compare_random_fixations(
        net, 
        validation_set, 
        loss_fn='l1_all',  # Same loss function as paper
        use_conv=net.model.use_conv,
        warp_imgs=True,  # Same as WARP_IMGS=True in analyseModels.py
        use_resNet=False, # Same as USE_RES_NET=False in analyseModels.py
        feature_size=128*128,  # Same as INPUT_SIZE in analyseModels.py
        mnist=False,
        return_fixations=True
    )
    
    # Convert to numpy arrays for analysis
    model_losses = np.array(losses).flatten()
    random_losses = np.array(losses_random_fix).flatten()
    
    # Calculate statistics with confidence intervals
    mean_loss = np.mean(model_losses)
    mean_random_loss = np.mean(random_losses)
    std_loss = np.std(model_losses)
    std_random_loss = np.std(random_losses)
    sem_loss = scipy.stats.sem(model_losses)
    sem_random_loss = scipy.stats.sem(random_losses)
    
    # 99% Confidence intervals
    ci_lower, ci_upper = scipy.stats.norm.interval(
        confidence=confidence, 
        loc=mean_loss, 
        scale=sem_loss
    )
    ci_lower_random, ci_upper_random = scipy.stats.norm.interval(
        confidence=confidence, 
        loc=mean_random_loss, 
        scale=sem_random_loss
    )
    
    # Statistical significance test (same as paper)
    t_stat, p_value = scipy.stats.ttest_ind(model_losses, random_losses, alternative='less')
    
    results = {
        'model_loss_mean': mean_loss,
        'model_loss_std': std_loss,
        'model_loss_sem': sem_loss,
        'model_loss_ci_lower': ci_lower,
        'model_loss_ci_upper': ci_upper,
        'random_loss_mean': mean_random_loss,
        'random_loss_std': std_random_loss,
        'random_loss_sem': sem_random_loss,
        'random_loss_ci_lower': ci_lower_random,
        'random_loss_ci_upper': ci_upper_random,
        't_statistic': t_stat,
        'p_value': p_value,
        'confidence': confidence,
        'n_samples': len(model_losses),
        'raw_model_losses': model_losses,
        'raw_random_losses': random_losses
    }
    
    print(f"\nEnergy Efficiency Results:")
    print(f"Model loss: {mean_loss:.6f} ± {sem_loss:.6f}")
    print(f"{confidence*100}% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"Random fixations loss: {mean_random_loss:.6f} ± {sem_random_loss:.6f}")
    print(f"{confidence*100}% CI: [{ci_lower_random:.6f}, {ci_upper_random:.6f}]")
    print(f"T-test (model < random): t={t_stat:.4f}, p={p_value:.6f}")
    print(f"Model is {'significantly' if p_value < 0.01 else 'not significantly'} more efficient")
    
    return results


def evaluate_xy_decoding(net, train_set, validation_set, confidence=0.99):
    """
    Evaluate XY position decoding using EXACT methodology from analyseModels.py.
    Uses ClosedFormDecoding.regressionCoordinates() - same as paper.
    
    Args:
        net: Trained RNN model 
        train_set: Training dataset for fitting decoder
        validation_set: Validation dataset for evaluation (same as paper)
        confidence: Confidence level for intervals
        
    Returns:
        dict with decoding performance statistics and confidence intervals
    """
    print("\nEvaluating XY position decoding using analyseModels.py methodology...")
    print("Using ClosedFormDecoding.regressionCoordinates() - same as paper...")
    
    import ClosedFormDecoding
    
    # Set the same seed as analyseModels.py
    torch.manual_seed(2553)
    np.random.seed(2553)
    
    results = {}
    
    # Test the exact same modes as analyseModels.py:207-209
    modes_to_test = [
        ('global', 'Global position decoding'),
        ('prev_relative', 'Previous relative position decoding'), 
        ('next_relative', 'Next relative position decoding')
    ]
    
    for mode, description in modes_to_test:
        print(f"\n{description} (mode='{mode}')")
        
        # Use the exact same call as analyseModels.py:207-209
        pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(
            net, 
            train_set, 
            validation_set, 
            layer=[1, 2],  # Same layers as paper
            mode=mode, 
            timestep=None  # Same as paper - uses all timesteps
        )
        
        # The function prints results internally, but we also collect them
        results[mode] = {
            'test_score': test_score,
            'reg_weights': reg_weights,
            'pred_cells': pred_cells,
            'description': description
        }
        
        print(f"Test R² score: {test_score:.6f}")
        print(f"Number of predictive cells: {len(pred_cells) if pred_cells is not None else 'N/A'}")
    
    # Bootstrap confidence intervals for the main global decoding
    print(f"\nComputing bootstrap confidence intervals for global decoding...")
    
    # Run multiple bootstrap samples for confidence intervals
    n_bootstrap = 100  # Reduced for speed, increase for more precision
    global_scores = []
    
    for bootstrap_idx in range(n_bootstrap):
        # Set different seed for each bootstrap sample
        torch.manual_seed(2553 + bootstrap_idx)
        np.random.seed(2553 + bootstrap_idx)
        
        try:
            _, _, score = ClosedFormDecoding.regressionCoordinates(
                net, train_set, validation_set, 
                layer=[1, 2], mode='global', timestep=None
            )
            global_scores.append(score)
        except:
            # Skip failed bootstrap samples
            continue
        
        if (bootstrap_idx + 1) % 25 == 0:
            print(f"Bootstrap sample {bootstrap_idx + 1}/{n_bootstrap} completed")
    
    if len(global_scores) > 0:
        global_scores = np.array(global_scores)
        mean_score = np.mean(global_scores)
        std_score = np.std(global_scores)
        sem_score = scipy.stats.sem(global_scores)
        
        # 99% confidence interval using percentiles (more robust for small samples)
        ci_lower = np.percentile(global_scores, (1 - confidence) / 2 * 100)
        ci_upper = np.percentile(global_scores, (1 + confidence) / 2 * 100)
        
        results['global']['bootstrap_mean'] = mean_score
        results['global']['bootstrap_std'] = std_score
        results['global']['bootstrap_sem'] = sem_score
        results['global']['bootstrap_ci_lower'] = ci_lower
        results['global']['bootstrap_ci_upper'] = ci_upper
        results['global']['bootstrap_samples'] = len(global_scores)
        results['global']['confidence'] = confidence
        
        print(f"\nGlobal Decoding Bootstrap Results:")
        print(f"Mean R² = {mean_score:.6f} ± {sem_score:.6f}")
        print(f"{confidence*100}% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"Bootstrap samples: {len(global_scores)}")
    else:
        print("Warning: No successful bootstrap samples for confidence intervals")
    
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
    
    # Load datasets using existing infrastructure (same as analyseModels.py)
    print(f"Loading datasets from {args.dataset_path}")
    train_set = H5dataset('train', args.dataset_path, device=device, use_color=False)
    validation_set = H5dataset('val', args.dataset_path, device=device, use_color=False)  # Use val split like paper
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
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Standard ModelState format (new format)
        model_state_dict = checkpoint['model_state_dict']
        print(f"Loading model using standard ModelState format from epoch {checkpoint.get('epochs', 'unknown')}")
        net.model.load_state_dict(model_state_dict)
        
        # Also load optimizer and results if available for compatibility
        if hasattr(net, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            net.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'results' in checkpoint:
            net.results = checkpoint['results']
        if 'epochs' in checkpoint:
            net.epochs = checkpoint['epochs']
            
    elif 'model' in checkpoint:
        # Legacy temporal contrastive format (old format)
        base_model_state = checkpoint['model']
        print(f"Loading base RNN model from legacy temporal contrastive checkpoint")
        net.model.load_state_dict(base_model_state)
    else:
        # Try to load directly as raw state dict
        print(f"Attempting to load as raw state dict")
        net.model.load_state_dict(checkpoint)
    
    net.model.eval()
    print(f"Model loaded successfully!")
    
    print("=" * 80)
    print("TEMPORAL CONTRASTIVE MODEL EVALUATION")
    print("=" * 80)
    
    # 1. Evaluate energy efficiency using EXACT analyseModels.py methodology
    energy_results = evaluate_energy_efficiency(
        net, validation_set, args.confidence
    )
    
    # 2. Evaluate XY decoding using EXACT analyseModels.py methodology  
    decoding_results = evaluate_xy_decoding(
        net, train_set, validation_set, args.confidence
    )
    
    # Combine results
    all_results = {
        'energy_efficiency': energy_results,
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