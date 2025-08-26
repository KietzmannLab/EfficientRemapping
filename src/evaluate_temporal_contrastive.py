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
import scipy
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
    model_losses = np.array(losses)  # Shape: (batch_size, fixations=7, timesteps=6)
    random_losses = np.array(losses_random_fix)  # Shape: (batch_size, fixations=7, timesteps=6)
    
    print(f"Original loss shapes - Model: {model_losses.shape}, Random: {random_losses.shape}")
    
    # Following plotResults.py methodology: only use rnn_iter == 0 (first timestep)
    # This matches plotResults.py line 137-138
    model_losses_t0 = model_losses[:, :, 0]  # Only timestep 0
    random_losses_t0 = random_losses[:, :, 0]  # Only timestep 0
    
    print(f"Timestep 0 shapes - Model: {model_losses_t0.shape}, Random: {random_losses_t0.shape}")
    
    # Flatten for statistical analysis (matching paper methodology)
    model_losses_flat = model_losses_t0.flatten()
    random_losses_flat = random_losses_t0.flatten()
    
    print(f"Final flattened shapes - Model: {model_losses_flat.shape}, Random: {random_losses_flat.shape}")
    print(f"Using only timestep 0 (rnn_iter=0) as in plotResults.py")
    
    # Calculate statistics with confidence intervals (using timestep 0 only)
    mean_loss = np.mean(model_losses_flat)
    mean_random_loss = np.mean(random_losses_flat)
    std_loss = np.std(model_losses_flat)
    std_random_loss = np.std(random_losses_flat)
    sem_loss = scipy.stats.sem(model_losses_flat)
    sem_random_loss = scipy.stats.sem(random_losses_flat)
    
    # 99% Confidence intervals using t-distribution (more robust)
    alpha = 1 - confidence
    dof = len(model_losses_flat) - 1
    t_critical = scipy.stats.t.ppf(1 - alpha/2, dof)
    
    ci_lower = mean_loss - t_critical * sem_loss
    ci_upper = mean_loss + t_critical * sem_loss
    
    dof_random = len(random_losses_flat) - 1
    t_critical_random = scipy.stats.t.ppf(1 - alpha/2, dof_random)
    ci_lower_random = mean_random_loss - t_critical_random * sem_random_loss
    ci_upper_random = mean_random_loss + t_critical_random * sem_random_loss
    
    # Statistical significance test (same as paper, using timestep 0 only)
    t_stat, p_value = scipy.stats.ttest_ind(model_losses_flat, random_losses_flat, alternative='less')
    
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
        'n_samples': len(model_losses_flat),
        'raw_model_losses': model_losses_flat,
        'raw_random_losses': random_losses_flat,
        'full_model_losses': model_losses,  # Keep full data for CSV generation
        'full_random_losses': random_losses
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
    import os
    
    # Create necessary directories that ClosedFormDecoding expects
    os.makedirs('EmergentPredictiveCoding/Results/Fig2_mscoco', exist_ok=True)
    
    # Set the same seed as analyseModels.py
    torch.manual_seed(2553)
    np.random.seed(2553)
    
    results = {}
    
    # Only test global position decoding (as requested - cleaner output)
    modes_to_test = [
        ('global', 'Global position decoding')
    ]
    
    # First, evaluate untrained model as baseline (same as analyseModels.py:207)
    print("\n" + "="*50)
    print("UNTRAINED MODEL BASELINE")
    print("="*50)
    
    # Create untrained model with same architecture as temporal contrastive model
    print("Creating untrained baseline model...")
    untrained_net = RNN.State(
        activation_func=torch.nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-4,
        input_size=128*128,
        hidden_size=2048,
        title="untrained_baseline",
        device=net.device,
        use_fixation=True,
        seed=42,
        use_conv=False,
        warp_imgs=False,
        use_resNet=False,
        time_steps_img=6,
        time_steps_cords=3,
        mnist=False
    )
    
    results['untrained'] = {}
    
    for mode, description in modes_to_test:
        print(f"\n[UNTRAINED] {description} (mode='{mode}')")
        
        # Reset seed for consistency
        torch.manual_seed(2553)
        np.random.seed(2553)
        
        # Capture output to extract R² values cleanly
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(
            untrained_net, 
            train_set, 
            validation_set, 
            layer=[1, 2],
            mode=mode, 
            timestep=None
        )
        
        # Capture and parse output
        captured_output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Extract individual R² values from captured output
        r_squared_line = [line for line in captured_output.split('\n') if 'R squared:' in line]
        if r_squared_line:
            # Parse the R² values - they're in format: "R squared: [[x_r2 y_r2]]"
            r_squared_str = r_squared_line[0].split('R squared: ')[1]
            # Use regex to extract numbers safely instead of eval()
            import re
            # Extract numbers from the string using regex
            numbers = re.findall(r'[0-9]+\.?[0-9]*', r_squared_str)
            if len(numbers) >= 2:
                r2_x, r2_y = float(numbers[0]), float(numbers[1])
            else:
                r2_x, r2_y = None, None
        else:
            r2_x, r2_y = None, None
        
        results['untrained'][mode] = {
            'test_score': test_score,
            'reg_weights': reg_weights,
            'pred_cells': pred_cells,
            'r2_x': r2_x,
            'r2_y': r2_y,
            'description': f"[UNTRAINED] {description}"
        }
        
        print(f"[UNTRAINED] X-coordinate R²: {r2_x:.6f}")
        print(f"[UNTRAINED] Y-coordinate R²: {r2_y:.6f}")
        print(f"[UNTRAINED] Overall test score: {test_score:.6f}")
    
    # Now evaluate trained temporal contrastive model
    print("\n" + "="*50)
    print("TEMPORAL CONTRASTIVE MODEL")
    print("="*50)
    
    results['trained'] = {}
    
    for mode, description in modes_to_test:
        print(f"\n[TRAINED] {description} (mode='{mode}')")
        
        # Reset seed for consistency
        torch.manual_seed(2553)
        np.random.seed(2553)
        
        # Capture output to extract R² values cleanly
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Use the exact same call as analyseModels.py:207-209
        pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(
            net, 
            train_set, 
            validation_set, 
            layer=[1, 2],  # Same layers as paper
            mode=mode, 
            timestep=None  # Same as paper - uses all timesteps
        )
        
        # Capture and parse output
        captured_output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Extract individual R² values from captured output
        r_squared_line = [line for line in captured_output.split('\n') if 'R squared:' in line]
        if r_squared_line:
            # Parse the R² values - they're in format: "R squared: [[x_r2 y_r2]]"
            r_squared_str = r_squared_line[0].split('R squared: ')[1]
            # Use regex to extract numbers safely instead of eval()
            import re
            # Extract numbers from the string using regex
            numbers = re.findall(r'[0-9]+\.?[0-9]*', r_squared_str)
            if len(numbers) >= 2:
                r2_x, r2_y = float(numbers[0]), float(numbers[1])
            else:
                r2_x, r2_y = None, None
        else:
            r2_x, r2_y = None, None
        
        # The function prints results internally, but we also collect them
        results['trained'][mode] = {
            'test_score': test_score,
            'reg_weights': reg_weights,
            'pred_cells': pred_cells,
            'r2_x': r2_x,
            'r2_y': r2_y,
            'description': f"[TRAINED] {description}"
        }
        
        print(f"[TRAINED] X-coordinate R²: {r2_x:.6f}")
        print(f"[TRAINED] Y-coordinate R²: {r2_y:.6f}")
        print(f"[TRAINED] Overall test score: {test_score:.6f}")
        
        # Compare trained vs untrained
        untrained_score = results['untrained'][mode]['test_score']
        improvement = test_score - untrained_score
        improvement_pct = (improvement / abs(untrained_score)) * 100 if untrained_score != 0 else float('inf')
        
        print(f"[COMPARISON] Improvement over untrained: {improvement:+.6f} ({improvement_pct:+.2f}%)")
        print(f"[COMPARISON] {'BETTER' if improvement > 0 else 'WORSE'} than untrained baseline")
    
    # Bootstrap confidence intervals for global decoding comparison
    print("\n" + "="*50)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*50)
    
    n_bootstrap = 50  # Reduced for speed
    
    # Bootstrap for untrained model
    print(f"\nComputing bootstrap CI for UNTRAINED global decoding...")
    untrained_global_scores = []
    
    for bootstrap_idx in range(n_bootstrap):
        torch.manual_seed(2553 + bootstrap_idx)
        np.random.seed(2553 + bootstrap_idx)
        
        try:
            _, _, score = ClosedFormDecoding.regressionCoordinates(
                untrained_net, train_set, validation_set, 
                layer=[1, 2], mode='global', timestep=None
            )
            untrained_global_scores.append(score)
        except:
            continue
    
    # Bootstrap for trained model 
    print(f"Computing bootstrap CI for TRAINED global decoding...")
    trained_global_scores = []
    
    for bootstrap_idx in range(n_bootstrap):
        torch.manual_seed(2553 + bootstrap_idx)
        np.random.seed(2553 + bootstrap_idx)
        
        try:
            _, _, score = ClosedFormDecoding.regressionCoordinates(
                net, train_set, validation_set, 
                layer=[1, 2], mode='global', timestep=None
            )
            trained_global_scores.append(score)
        except:
            continue
        
        if (bootstrap_idx + 1) % 25 == 0:
            print(f"Bootstrap sample {bootstrap_idx + 1}/{n_bootstrap} completed")
    
    # Analyze bootstrap results
    if len(untrained_global_scores) > 0 and len(trained_global_scores) > 0:
        untrained_scores = np.array(untrained_global_scores)
        trained_scores = np.array(trained_global_scores)
        
        # Untrained stats
        untrained_mean = np.mean(untrained_scores)
        untrained_sem = scipy.stats.sem(untrained_scores)
        untrained_ci_lower = np.percentile(untrained_scores, (1 - confidence) / 2 * 100)
        untrained_ci_upper = np.percentile(untrained_scores, (1 + confidence) / 2 * 100)
        
        # Trained stats  
        trained_mean = np.mean(trained_scores)
        trained_sem = scipy.stats.sem(trained_scores)
        trained_ci_lower = np.percentile(trained_scores, (1 - confidence) / 2 * 100)
        trained_ci_upper = np.percentile(trained_scores, (1 + confidence) / 2 * 100)
        
        # Statistical comparison
        t_stat, p_value = scipy.stats.ttest_ind(trained_scores, untrained_scores, alternative='greater')
        
        # Store results
        results['bootstrap_comparison'] = {
            'untrained_mean': untrained_mean,
            'untrained_sem': untrained_sem,
            'untrained_ci_lower': untrained_ci_lower,
            'untrained_ci_upper': untrained_ci_upper,
            'trained_mean': trained_mean,
            'trained_sem': trained_sem, 
            'trained_ci_lower': trained_ci_lower,
            'trained_ci_upper': trained_ci_upper,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence': confidence,
            'n_bootstrap': n_bootstrap
        }
        
        print(f"\nBootstrap Global Decoding Comparison:")
        print(f"UNTRAINED: R² = {untrained_mean:.6f} ± {untrained_sem:.6f}")
        print(f"           {confidence*100}% CI: [{untrained_ci_lower:.6f}, {untrained_ci_upper:.6f}]")
        print(f"TRAINED:   R² = {trained_mean:.6f} ± {trained_sem:.6f}")  
        print(f"           {confidence*100}% CI: [{trained_ci_lower:.6f}, {trained_ci_upper:.6f}]")
        print(f"IMPROVEMENT: {trained_mean - untrained_mean:+.6f}")
        print(f"T-test (trained > untrained): t={t_stat:.4f}, p={p_value:.6f}")
        print(f"Training {'significantly' if p_value < 0.01 else 'not significantly'} improves decoding")
        print(f"Bootstrap samples: {len(trained_scores)} (trained), {len(untrained_scores)} (untrained)")
    else:
        print("Warning: Insufficient bootstrap samples for comparison")
    
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
    print("\n" + "="*50)
    print("ENERGY EFFICIENCY EVALUATION")
    print("="*50)
    
    # First evaluate untrained baseline
    print("\nCreating untrained baseline model for energy efficiency...")
    untrained_net = RNN.State(
        activation_func=torch.nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-4,
        input_size=128*128,
        hidden_size=2048,
        title="untrained_baseline_energy",
        device=net.device,
        use_fixation=True,
        seed=42,
        use_conv=False,
        warp_imgs=False,
        use_resNet=False,
        time_steps_img=6,
        time_steps_cords=3,
        mnist=False
    )
    
    print("\n[UNTRAINED] Energy efficiency evaluation:")
    untrained_energy_results = evaluate_energy_efficiency(
        untrained_net, validation_set, args.confidence
    )
    
    print("\n[TRAINED] Energy efficiency evaluation:")
    trained_energy_results = evaluate_energy_efficiency(
        net, validation_set, args.confidence
    )
    
    # Compare trained vs untrained energy efficiency
    trained_loss = trained_energy_results['model_loss_mean']
    untrained_loss = untrained_energy_results['model_loss_mean']
    energy_improvement = untrained_loss - trained_loss  # Lower loss is better
    energy_improvement_pct = (energy_improvement / abs(untrained_loss)) * 100 if untrained_loss != 0 else 0
    
    print(f"\n[ENERGY EFFICIENCY COMPARISON]")
    print(f"UNTRAINED: {untrained_loss:.6f} ± {untrained_energy_results['model_loss_sem']:.6f}")
    print(f"TRAINED:   {trained_loss:.6f} ± {trained_energy_results['model_loss_sem']:.6f}")
    print(f"IMPROVEMENT: {energy_improvement:+.6f} ({energy_improvement_pct:+.2f}%)")
    print(f"Training {'IMPROVES' if energy_improvement > 0 else 'WORSENS'} energy efficiency")
    
    # Combine energy results
    energy_results = {
        'untrained': untrained_energy_results,
        'trained': trained_energy_results,
        'comparison': {
            'improvement': energy_improvement,
            'improvement_pct': energy_improvement_pct,
            'untrained_loss': untrained_loss,
            'trained_loss': trained_loss
        }
    }
    
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
    
    # Generate losses.csv file in the same format as analyseModels.py
    print("\nGenerating detailed losses.csv file for plotResults.py compatibility...")
    
    # Extract the detailed loss arrays from the energy efficiency evaluation
    # Use full losses (all timesteps) for CSV generation, but note that plotResults.py will filter to timestep 0
    model_losses_detailed = energy_results['full_model_losses']  # Shape: (batch_size, fixations=7, timesteps=6)
    random_losses_detailed = energy_results['full_random_losses']  # Shape: (batch_size, fixations=7, timesteps=6)
    
    # Create the same structure as analyseModels.py (line 300-310)
    # For temporal contrastive, we'll create a simplified version with just model vs random
    import pandas as pd
    
    # Reshape losses to match expected format (batch_size, fixation*timestep)
    # The losses should be (N_samples, 7_fixations * 6_timesteps) = (N, 42)
    model_losses_reshaped = model_losses_detailed.reshape(-1, 42) if len(model_losses_detailed.shape) > 1 else model_losses_detailed.reshape(-1, 1)
    random_losses_reshaped = random_losses_detailed.reshape(-1, 42) if len(random_losses_detailed.shape) > 1 else random_losses_detailed.reshape(-1, 1)
    
    # Create column names similar to analyseModels.py format
    fixation_cols = []
    timestep_cols = []
    for fix in range(7):  # 7 fixations
        for ts in range(6):  # 6 timesteps
            fixation_cols.append(fix)
            timestep_cols.append(ts)
    
    # Create multi-level column structure
    model_columns = []
    for i in range(len(fixation_cols)):
        model_columns.append(('Temporal Contrastive', str(fixation_cols[i]), str(timestep_cols[i])))
        
    random_columns = []
    for i in range(len(fixation_cols)):
        random_columns.append(('Shuffled Temporal Contrastive', str(fixation_cols[i]), str(timestep_cols[i])))
    
    # Combine data
    combined_data = np.concatenate([model_losses_reshaped, random_losses_reshaped], axis=1)
    all_columns = model_columns + random_columns
    
    # Create DataFrame
    df = pd.DataFrame(combined_data, columns=pd.MultiIndex.from_tuples(all_columns))
    
    # Save in the same location as analyseModels.py expects
    losses_csv_path = 'EmergentPredictiveCoding/Results/Fig2_mscoco/temporal_contrastive_losses.csv'
    import os
    os.makedirs(os.path.dirname(losses_csv_path), exist_ok=True)
    df.to_csv(losses_csv_path)
    
    print(f"Detailed losses saved to: {losses_csv_path}")
    print(f"CSV shape: {df.shape}")
    print(f"Compatible with plotResults.py for detailed analysis")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()