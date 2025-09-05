#!/usr/bin/env python3
"""
Analysis script for investigating allocentric coding units spatial organization.
Analyzes 2D tuning profiles and clustering patterns of units that encode xy coordinates.

Usage:
    python analyze_allocentric_organization.py --model_path <path_to_model> --dataset_path <path_to_dataset>

Example:
    python analyze_allocentric_organization.py \
        --model_path "/share/klab/psulewski/tnortmann/efficient-remapping/EmergentPredictiveCoding/models/patterns_rev/mscoco_deepgaze3/mscoco_netl1_all_0_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_ReLU_nonCords_new_moreRL_" \
        --dataset_path /share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import os
import pickle
import json

# Set publication quality style
sns.set_context('talk')
plt.rcParams['svg.fonttype'] = 'none'  # Make text editable in SVG

# Import existing infrastructure
import RNN
import functions
from H5dataset import H5dataset
import ClosedFormDecoding


def lesion_specific_units(verbose=True):
    """
    Lesion specific predefined units based on your provided list.
    
    Args:
        verbose: Print debug information
        
    Returns:
        lesioned_units_dict: Dictionary with layer-specific unit lists
        all_lesioned_units: Combined array of all units to be lesioned
        layer_assignments: Dictionary mapping units to layers
    """
    # Predefined lesioned units (matching your specification)
    lesioned_units = {
        1: [363, 262, 1319, 578, 208, 266, 1840, 262],
        2: [1900, 787, 488, 839, 107, 820, 981, 1900, 3, 1510, 403, 2033, 251, 1345]
    }
    
    if verbose:
        print("Using predefined lesioned units:")
        print(f"Layer 1 units: {lesioned_units[1]}")
        print(f"Layer 2 units: {lesioned_units[2]}")
    
    # Create weights-like structure for layer assignment
    all_units = []
    layer_assignments = {}
    
    # Process Layer 1 units (< 2048)
    for unit in lesioned_units[1]:
        all_units.append(unit)
        layer_assignments[unit] = 1
    
    # Process Layer 2 units (add 2048 for global indexing)
    layer2_global_units = [unit + 2048 for unit in lesioned_units[2]]
    for orig_unit, global_unit in zip(lesioned_units[2], layer2_global_units):
        all_units.append(global_unit)
        layer_assignments[global_unit] = 2
    
    all_lesioned_units = np.array(all_units)
    
    if verbose:
        print(f"Layer 1 units (< 2048): {lesioned_units[1]}")
        print(f"Layer 2 units (original): {lesioned_units[2]}")
        print(f"Layer 2 units (global): {layer2_global_units}")
        print(f"Total units to lesion: {len(all_lesioned_units)}")
        print(f"All lesioned unit indices: {all_lesioned_units}")
    
    return lesioned_units, all_lesioned_units, layer_assignments


def select_top_regression_units_separate_xy(reg_weights, percentage=0.05, verbose=True):
    """
    Select top units based on regression coefficient strength for X and Y coordinates separately.
    
    Args:
        reg_weights: Regression coefficients from ClosedFormDecoding (2, n_units) - x and y coefficients
        percentage: Percentage of top units to select for each coordinate (default: 0.05 for 0.05%)
        verbose: Print debug information
        
    Returns:
        x_units: Array of unit indices with strongest X-coordinate coefficients
        y_units: Array of unit indices with strongest Y-coordinate coefficients
        combined_units: Array of unique units from both X and Y selections
        x_strengths: X-coordinate regression strengths for all units
        y_strengths: Y-coordinate regression strengths for all units
    """
    if verbose:
        print(f"Selecting top {percentage}% units for X and Y coordinates separately")
        print(f"Regression weights shape: {reg_weights.shape}")
    
    # Get absolute coefficients for each coordinate
    x_coeffs = np.abs(reg_weights[0])  # X-coordinate coefficients  
    y_coeffs = np.abs(reg_weights[1])  # Y-coordinate coefficients
    
    # Calculate number of units to select for each coordinate
    n_units = len(x_coeffs)
    n_select = max(1, int(n_units * (percentage / 100)))  # Ensure at least 1 unit
    
    # Select top units for X coordinate
    x_selected_indices = np.argsort(x_coeffs)[-n_select:][::-1]  # Descending order
    x_selected_strengths = x_coeffs[x_selected_indices]
    
    # Select top units for Y coordinate  
    y_selected_indices = np.argsort(y_coeffs)[-n_select:][::-1]  # Descending order
    y_selected_strengths = y_coeffs[y_selected_indices]
    
    # Combine unique units
    combined_units = np.unique(np.concatenate([x_selected_indices, y_selected_indices]))
    
    # Calculate overlap
    overlap_units = np.intersect1d(x_selected_indices, y_selected_indices)
    
    if verbose:
        print(f"Total units available: {n_units}")
        print(f"Units selected per coordinate: {n_select} ({n_select/n_units*100:.3f}%)")
        print(f"X-coordinate units: {x_selected_indices}")
        print(f"Y-coordinate units: {y_selected_indices}")
        print(f"Overlapping units: {len(overlap_units)} units: {overlap_units}")
        print(f"Total unique units: {len(combined_units)} units")
        print(f"X-strength range: {x_selected_strengths.min():.4f} - {x_selected_strengths.max():.4f}")
        print(f"Y-strength range: {y_selected_strengths.min():.4f} - {y_selected_strengths.max():.4f}")
    
    # Verification
    actual_percentage = (n_select / n_units) * 100
    if verbose:
        print(f"Verification: Selected {n_select}/{n_units} per coordinate = {actual_percentage:.3f}%")
        print(f"Target: {percentage}%, Actual: {actual_percentage:.3f}%, Difference: {abs(percentage - actual_percentage):.3f}%")
    
    return x_selected_indices, y_selected_indices, combined_units, x_coeffs, y_coeffs


def process_lesion_map_like_rnn(pred_cells, verbose=True):
    """
    Process lesion map exactly like RNN.py setLesionMap() function.
    
    Args:
        pred_cells: (2, n_units) array from regressionCoordinates
        verbose: Print debug information
        
    Returns:
        lesion_map: List of 4 arrays matching RNN.py format
        allocentric_units: Combined unique unit indices
    """
    # Step 1: Take top 11 units (exact RNN.py replica)
    lesion_map = pred_cells[:, -11:]
    if verbose:
        print(f"Lesion map shape after [:, -11:]: {lesion_map.shape}")
    
    # Step 2: Create 4-array structure (exact RNN.py replica)
    lesion_map_list = [lesion_map.copy()[0], lesion_map.copy()[1], 
                       lesion_map.copy()[0], lesion_map.copy()[1]]
    
    # Step 3: Process for 2-layer model (exact RNN.py replica)
    # Assume 2048 units per layer based on RNN.py logic
    for idx, elem in enumerate(lesion_map_list):
        if idx > 1:  # Layer 2 processing
            elem_filtered = elem[elem >= 2048]
            lesion_map_list[idx] = elem_filtered - 2048
        else:  # Layer 1 processing  
            lesion_map_list[idx] = elem[elem < 2048]
        
        if verbose:
            print(f"Lesion map[{idx}] (layer {1 if idx <= 1 else 2}): {lesion_map_list[idx]}")
    
    # Get combined unique units for analysis
    allocentric_units = np.unique(np.concatenate([lesion_map[0], lesion_map[1]]))
    
    return lesion_map_list, allocentric_units


def load_or_store_xy_units(net, train_set, validation_set, cache_path, force_recompute=False):
    """
    Load or compute and store identified xy units to avoid repeated regression.
    
    Args:
        net: Trained RNN model
        train_set: Training dataset
        validation_set: Validation dataset
        cache_path: Path to save/load cached xy units
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        allocentric_units: Array of unit indices that encode xy coordinates
        reg_weights: Regression weights for decoding
        test_score: Decoding performance score
    """
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached xy units from {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        # Only use cache if it contains the predefined specific units method
        selection_method = cache_data.get('selection_method', 'unknown')
        if selection_method == 'predefined_specific_units':
            print("Cache contains predefined specific lesioned units")
            if 'model_info' in cache_data:
                info = cache_data['model_info']
                print(f"  - Layer 1 units: {info.get('n_lesioned_layer1', 'unknown')}")
                print(f"  - Layer 2 units: {info.get('n_lesioned_layer2', 'unknown')}")
                print(f"  - Total units: {info.get('n_all_lesioned', 'unknown')}")
            # Return cached results
            all_lesioned_units = cache_data['all_lesioned_units']
            return all_lesioned_units, cache_data['reg_weights'], cache_data['test_score']
        else:
            print(f"Cache contains old selection method ({selection_method}) - ignoring and recomputing")
            print("Will use predefined specific lesioned units")
    
    print("Identifying allocentric coding units (this may take a while)...")
    
    # Use existing methodology to identify xy units
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(
        net, train_set, validation_set, layer=[1, 2], mode='global', timestep=None
    )
    
    # Use predefined specific lesioned units
    lesioned_units_dict, all_lesioned_units, layer_assignments = lesion_specific_units(verbose=True)
    
    # Separate units by layer for consistency with analysis structure
    lesioned_layer1 = np.array(lesioned_units_dict[1])  # Layer 1 units
    lesioned_layer2 = np.array([unit + 2048 for unit in lesioned_units_dict[2]])  # Layer 2 units (global indices)
    
    print(f"Predefined lesioned units:")
    print(f"  - Layer 1: {len(lesioned_layer1)} units")
    print(f"  - Layer 2: {len(lesioned_units_dict[2])} units (original indices)")
    print(f"  - Total unique units: {len(all_lesioned_units)} units")
    print(f"X-coordinate decoding R²: {test_score}")
    
    # Cache the results
    print(f"Caching predefined lesioned units to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {
        'lesioned_units_dict': lesioned_units_dict,     # Original layer-specific units
        'lesioned_units_layer1': lesioned_layer1,       # Layer 1 lesioned units
        'lesioned_units_layer2': lesioned_layer2,       # Layer 2 lesioned units (global indices)  
        'all_lesioned_units': all_lesioned_units,       # Combined lesioned units
        'layer_assignments': layer_assignments,         # Unit to layer mapping
        'reg_weights': reg_weights,
        'test_score': test_score,
        'pred_cells': pred_cells,
        'selection_method': 'predefined_specific_units',
        'model_info': {
            'model_title': net.title if hasattr(net, 'title') else 'unknown',
            'n_lesioned_layer1': len(lesioned_layer1),
            'n_lesioned_layer2': len(lesioned_units_dict[2]),
            'n_all_lesioned': len(all_lesioned_units),
            'decoding_score': test_score,
            'layer1_units': lesioned_units_dict[1],
            'layer2_units': lesioned_units_dict[2]
        }
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return all_lesioned_units, reg_weights, test_score


def load_or_compute_activations(net, dataset, cache_path, force_recompute=False):
    """
    Load cached activations or compute them if not available.
    
    Args:
        net: Trained RNN model
        dataset: Dataset to analyze
        cache_path: Path to save/load cached activations
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        activations: (n_samples, n_units) all unit activations
        xy_coords: (n_samples, 2) corresponding x,y coordinates
    """
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached activations from {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['activations'], cache_data['xy_coords']
    
    print("Computing activations from model (this may take a while)...")
    
    # Use existing infrastructure to get activations and coordinates  
    activations, fixations = ClosedFormDecoding.getFeatures(
        net, dataset, layer_idx=[1, 2], timestep=None
    )
    
    xy_coords = fixations[:, :2]  # x,y coordinates
    
    print(f"Computed activations shape: {activations.shape}, coords shape: {xy_coords.shape}")
    
    # Cache the results
    print(f"Caching activations to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {
        'activations': activations,
        'xy_coords': xy_coords,
        'model_info': {
            'model_title': net.title if hasattr(net, 'title') else 'unknown',
            'dataset_split': getattr(dataset, 'split', 'unknown'),
            'n_samples': len(activations),
            'n_units': activations.shape[1]
        }
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return activations, xy_coords


def get_rnn_lesioned_units_both_layers(lesion_map_list, verbose=True):
    """
    Extract ALL units that actually get lesioned in RNN.py (both Layer 1 and Layer 2).
    This constrains analysis to the exact same units that are lesioned in experiments.
    
    Args:
        lesion_map_list: 4-element list from process_lesion_map_like_rnn
        verbose: Print debug information
        
    Returns:
        lesioned_units_layer1: Units lesioned in Layer 1 (< 2048)
        lesioned_units_layer2: Units lesioned in Layer 2 (>= 2048, original indices)
        all_lesioned_units: Combined unique units from both layers
    """
    # Layer 1 units (indices 0 and 1 in lesion_map_list)
    x_units_layer1 = lesion_map_list[0]  # X-predictive units < 2048
    y_units_layer1 = lesion_map_list[1]  # Y-predictive units < 2048
    
    # Layer 2 units (indices 2 and 3 in lesion_map_list) 
    # Note: these were adjusted by -2048, so add back to get original indices
    x_units_layer2 = lesion_map_list[2] + 2048 if len(lesion_map_list[2]) > 0 else np.array([])
    y_units_layer2 = lesion_map_list[3] + 2048 if len(lesion_map_list[3]) > 0 else np.array([])
    
    lesioned_units_layer1 = np.unique(np.concatenate([x_units_layer1, y_units_layer1]))
    lesioned_units_layer2 = np.unique(np.concatenate([x_units_layer2, y_units_layer2])) if len(x_units_layer2) > 0 or len(y_units_layer2) > 0 else np.array([])
    
    # Combine all lesioned units
    if len(lesioned_units_layer2) > 0:
        all_lesioned_units = np.unique(np.concatenate([lesioned_units_layer1, lesioned_units_layer2]))
    else:
        all_lesioned_units = lesioned_units_layer1
    
    if verbose:
        print(f"RNN-lesioned units in Layer 1 (< 2048): {len(lesioned_units_layer1)} units")
        print(f"  X-units Layer 1: {x_units_layer1}")
        print(f"  Y-units Layer 1: {y_units_layer1}")
        print(f"RNN-lesioned units in Layer 2 (>= 2048): {len(lesioned_units_layer2)} units")
        print(f"  X-units Layer 2: {x_units_layer2}")
        print(f"  Y-units Layer 2: {y_units_layer2}")
        print(f"Total unique lesioned units: {len(all_lesioned_units)}")
    
    return lesioned_units_layer1, lesioned_units_layer2, all_lesioned_units


def analyze_continuous_xy_tuning(net, dataset, lesioned_units, output_dir='./results'):
    """
    Extract continuous 2D tuning curves for RNN-lesioned allocentric units only.
    
    Args:
        net: Trained RNN model
        dataset: Dataset to analyze  
        lesioned_units: Exact units that get lesioned in RNN experiments
        output_dir: Directory for caching activations
        
    Returns:
        unit_activations: (n_samples, n_lesioned_units) activations
        xy_coords: (n_samples, 2) corresponding x,y coordinates
    """
    print(f"Extracting tuning data for {len(lesioned_units)} RNN-lesioned units...")
    
    # Create cache path for activations
    cache_filename = f"activations_cache_{getattr(dataset, 'split', 'unknown')}.pkl"
    cache_path = os.path.join(output_dir, cache_filename)
    
    # Load or compute full activations
    activations, xy_coords = load_or_compute_activations(net, dataset, cache_path)
    
    # Focus ONLY on RNN-lesioned units
    unit_activations = activations[:, lesioned_units]
    
    print(f"RNN-lesioned units data shape: activations {unit_activations.shape}, coords {xy_coords.shape}")
    return unit_activations, xy_coords


def plot_spatial_receptive_fields(unit_activations, xy_coords, allocentric_units, 
                                  n_units=9, output_dir='./results'):
    """
    Plot publication-quality spatial receptive fields for top units.
    """
    print(f"Creating spatial receptive field plots for {n_units} units...")
    
    # Select units to visualize
    unit_indices = np.linspace(0, len(allocentric_units)-1, n_units).astype(int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Color palette following paper style
    cmap = 'viridis'
    
    for i, unit_idx in enumerate(unit_indices):
        # High-quality scatter plot with better styling
        scatter = axes[i].scatter(xy_coords[:, 0], xy_coords[:, 1], 
                                c=unit_activations[:, unit_idx], 
                                cmap=cmap, alpha=0.7, s=2, edgecolors='none')
        
        axes[i].set_xlabel('X coordinate', fontweight='bold')
        axes[i].set_ylabel('Y coordinate', fontweight='bold')
        axes[i].set_title(f'Unit {allocentric_units[unit_idx]}', 
                         fontweight='bold', fontsize=14)
        
        # Style colorbar
        cbar = plt.colorbar(scatter, ax=axes[i], shrink=0.8)
        cbar.set_label('Activation', fontweight='bold', rotation=270, labelpad=15)
        
        # Clean up axes
        sns.despine(ax=axes[i])
        axes[i].tick_params(axis='both', which='major', labelsize=10)
    
    fig.suptitle('Spatial Receptive Fields of Allocentric Units', 
                fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'spatial_receptive_fields.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'spatial_receptive_fields.svg'), 
                bbox_inches='tight')
    
    return fig


def perform_cluster_lesions(net, dataset, allocentric_units, unit_activations, xy_coords, cluster_labels, output_dir='./results'):
    """
    Perform targeted in-silico lesions of unit clusters and analyze systematic errors.
    
    Args:
        net: Trained RNN model
        dataset: Dataset to test lesions on
        allocentric_units: Unit indices of allocentric units
        unit_activations: Original activations before lesioning
        xy_coords: Ground truth coordinates
        cluster_labels: Cluster assignment for each unit
        output_dir: Directory to save results
        
    Returns:
        lesion_results: Dictionary containing lesion analysis results
    """
    print("Performing targeted in-silico lesions...")
    
    n_clusters = len(np.unique(cluster_labels))
    lesion_results = {}
    
    # Get baseline decoding performance (all units active)
    baseline_pred = decode_position_from_activations(unit_activations, xy_coords)
    baseline_error = compute_position_errors(xy_coords, baseline_pred)
    
    print(f"Baseline decoding error: {np.mean(baseline_error):.4f}")
    
    # Test lesioning each cluster
    cluster_errors = {}
    
    for cluster_id in range(n_clusters):
        print(f"Testing lesion of cluster {cluster_id}...")
        
        # Create lesioned activations (set cluster units to zero)
        cluster_mask = cluster_labels == cluster_id
        lesioned_activations = unit_activations.copy()
        lesioned_activations[:, cluster_mask] = 0
        
        # Decode positions with lesioned network
        lesioned_pred = decode_position_from_activations(lesioned_activations, xy_coords)
        lesion_error = compute_position_errors(xy_coords, lesioned_pred)
        
        # Compute error vectors for systematic analysis
        error_vectors = lesioned_pred - xy_coords
        
        # Analyze systematic errors
        error_analysis = analyze_systematic_errors(error_vectors)
        
        cluster_errors[cluster_id] = {
            'mean_error': np.mean(lesion_error),
            'error_increase': np.mean(lesion_error) - np.mean(baseline_error),
            'error_vectors': error_vectors,
            'systematic_analysis': error_analysis,
            'lesioned_units': np.sum(cluster_mask)
        }
        
        print(f"  Cluster {cluster_id}: {cluster_errors[cluster_id]['lesioned_units']} units lesioned, "
              f"error increase: {cluster_errors[cluster_id]['error_increase']:.4f}")
    
    # Create polar plots of systematic errors with scatter plots
    fig_polar_count, fig_polar_scatter = create_error_polar_plots_with_scatter(cluster_errors, output_dir)
    
    # Save lesion results
    lesion_summary = []
    for cluster_id, results in cluster_errors.items():
        lesion_summary.append({
            'cluster_id': cluster_id,
            'lesioned_units': results['lesioned_units'],
            'mean_error': results['mean_error'],
            'error_increase': results['error_increase'],
            'preferred_error_direction': results['systematic_analysis']['preferred_direction'],
            'error_concentration': results['systematic_analysis']['concentration'],
            'horizontal_bias': results['systematic_analysis']['horizontal_bias'],
            'vertical_bias': results['systematic_analysis']['vertical_bias']
        })
    
    lesion_df = pd.DataFrame(lesion_summary)
    lesion_df.to_csv(os.path.join(output_dir, 'cluster_lesion_analysis.csv'), index=False)
    
    lesion_results = {
        'baseline_error': baseline_error,
        'cluster_errors': cluster_errors,
        'lesion_summary': lesion_df
    }
    
    return lesion_results


def decode_position_from_activations(activations, true_coords):
    """
    Simple linear decoder to predict positions from activations.
    """
    from sklearn.linear_model import Ridge
    
    # Split data for training decoder
    n_samples = len(activations)
    train_idx = np.random.choice(n_samples, size=n_samples//2, replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    # Train decoder
    decoder = Ridge(alpha=1.0)
    decoder.fit(activations[train_idx], true_coords[train_idx])
    
    # Predict on all data
    pred_coords = decoder.predict(activations)
    return pred_coords


def compute_position_errors(true_coords, pred_coords):
    """
    Compute Euclidean distance errors between true and predicted coordinates.
    """
    return np.sqrt(np.sum((true_coords - pred_coords)**2, axis=1))


def analyze_systematic_errors(error_vectors):
    """
    Analyze systematic biases in error vectors.
    
    Args:
        error_vectors: (n_samples, 2) array of error vectors [dx, dy]
        
    Returns:
        analysis: Dictionary containing systematic error analysis
    """
    dx, dy = error_vectors[:, 0], error_vectors[:, 1]
    
    # Convert to polar coordinates
    error_magnitudes = np.sqrt(dx**2 + dy**2)
    error_angles = np.arctan2(dy, dx)
    
    # Compute circular statistics for error direction
    mean_direction = np.arctan2(np.mean(np.sin(error_angles)), np.mean(np.cos(error_angles)))
    
    # Concentration parameter (higher = more systematic)
    concentration = np.sqrt(np.mean(np.cos(error_angles))**2 + np.mean(np.sin(error_angles))**2)
    
    # Bias analysis
    horizontal_bias = np.mean(dx)  # Positive = rightward bias
    vertical_bias = np.mean(dy)    # Positive = upward bias
    
    analysis = {
        'preferred_direction': mean_direction,
        'concentration': concentration,
        'horizontal_bias': horizontal_bias,
        'vertical_bias': vertical_bias,
        'mean_error_magnitude': np.mean(error_magnitudes)
    }
    
    return analysis


def create_error_polar_plots_with_scatter(cluster_errors, output_dir):
    """
    Create publication-quality polar plots with error count histograms and amplitude scatter plots.
    Following plot_supplementary_results.py styling.
    """
    n_clusters = len(cluster_errors)
    n_cols = min(4, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    # Publication colors following plot_supplementary_results.py palette
    full_color = '#27ae60'  # Emerald green
    cluster_colors = sns.color_palette('Set2', n_clusters)
    
    # Create two figures: count histograms and amplitude scatter
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), 
                               subplot_kw=dict(projection='polar'))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), 
                               subplot_kw=dict(projection='polar'))
    
    if n_clusters == 1:
        axes1 = [axes1]
        axes2 = [axes2]
    elif n_rows == 1:
        axes1 = axes1.flatten() if n_clusters > 1 else [axes1]
        axes2 = axes2.flatten() if n_clusters > 1 else [axes2]
    else:
        axes1 = axes1.flatten()
        axes2 = axes2.flatten()
    
    for i, (cluster_id, results) in enumerate(cluster_errors.items()):
        if i >= len(axes1):
            break
            
        error_vectors = results['error_vectors']
        dx, dy = error_vectors[:, 0], error_vectors[:, 1]
        
        # Convert to polar coordinates
        error_angles = np.arctan2(dy, dx)
        error_magnitudes = np.sqrt(dx**2 + dy**2)
        
        # Plot 1: Error count histogram (publication style)
        angle_bins = np.linspace(-np.pi, np.pi, 25)
        hist_counts, _ = np.histogram(error_angles, bins=angle_bins)
        bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
        bin_width = np.diff(angle_bins)[0]
        
        axes1[i].bar(bin_centers, hist_counts, width=bin_width,
                    color=cluster_colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Plot 2: Amplitude-weighted scatter plot
        # Subsample for performance and visibility if too many points
        max_points = 1000
        if len(error_angles) > max_points:
            indices = np.random.choice(len(error_angles), max_points, replace=False)
            plot_angles = error_angles[indices]
            plot_magnitudes = error_magnitudes[indices]
        else:
            plot_angles = error_angles
            plot_magnitudes = error_magnitudes
        
        # Create scatter plot with magnitude as color
        scatter = axes2[i].scatter(plot_angles, plot_magnitudes, 
                                  c=plot_magnitudes, cmap='viridis', 
                                  alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        # Add systematic bias arrows (following paper style)
        sys_analysis = results['systematic_analysis']
        if sys_analysis['concentration'] > 0.1:
            # Arrow for count plot
            arrow_length1 = max(hist_counts) * 0.7 if max(hist_counts) > 0 else 1
            axes1[i].arrow(sys_analysis['preferred_direction'], 0, 0, arrow_length1,
                          head_width=0.15, head_length=arrow_length1*0.12, 
                          fc='red', ec='red', linewidth=2.5, alpha=0.9)
            
            # Arrow for scatter plot
            arrow_length2 = np.percentile(plot_magnitudes, 90) if len(plot_magnitudes) > 0 else 1
            axes2[i].arrow(sys_analysis['preferred_direction'], 0, 0, arrow_length2,
                          head_width=0.15, head_length=arrow_length2*0.12,
                          fc='red', ec='red', linewidth=2.5, alpha=0.9)
        
        # Publication styling (following plot_supplementary_results.py)
        for ax_idx, (ax, title_suffix) in enumerate([(axes1[i], 'Error Count'), 
                                                      (axes2[i], 'Error Amplitude')]):
            ax.set_title(f'Cluster {cluster_id}\n{title_suffix}\n({results["lesioned_units"]} units)', 
                        fontweight='bold', fontsize=14, pad=25)
            ax.grid(True, alpha=0.4, linewidth=0.8)
            ax.set_theta_zero_location('E')  # 0° at East (rightward)
            ax.set_theta_direction(1)  # Counterclockwise
            
            # Clean tick labels
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            # Add colorbar for scatter plot
            if ax_idx == 1 and len(plot_magnitudes) > 0:  # Scatter plot
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
                cbar.set_label('Error Magnitude', fontweight='bold', 
                              rotation=270, labelpad=20, fontsize=12)
                cbar.ax.tick_params(labelsize=10)
    
    # Hide unused subplots
    for i in range(n_clusters, len(axes1)):
        axes1[i].set_visible(False)
        axes2[i].set_visible(False)
    
    # Style figures following plot_supplementary_results.py
    for fig, name_suffix, plot_type in [(fig1, 'count', 'Error Count Distribution'), 
                                        (fig2, 'scatter', 'Error Amplitude Scatter')]:
        fig.suptitle(f'Systematic Error Analysis: {plot_type}', 
                    fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])  # Leave space for suptitle
        
        # Save figures (following plot_supplementary_results.py format)
        fig.savefig(os.path.join(output_dir, f'lesion_error_polar_{name_suffix}.png'), 
                   dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, f'lesion_error_polar_{name_suffix}.svg'), 
                   bbox_inches='tight')
    
    return fig1, fig2


def visualize_2d_tuning_curves(unit_activations, xy_coords, allocentric_units, 
                               unit_indices=None, grid_resolution=100, 
                               output_dir='./results'):
    """
    Create 2D heatmaps showing activation as function of xy position.
    
    Args:
        unit_activations: (n_samples, n_units) unit activations
        xy_coords: (n_samples, 2) x,y coordinates  
        allocentric_units: Original unit indices for labeling
        unit_indices: Which units to visualize (default: first 16)
        grid_resolution: Resolution for interpolated grid
        output_dir: Directory to save figures
        
    Returns:
        fig: Matplotlib figure with tuning curve heatmaps
    """
    if unit_indices is None:
        unit_indices = range(min(16, unit_activations.shape[1]))
    
    print(f"Creating 2D tuning curve visualizations for {len(unit_indices)} units...")
    
    # Create interpolated grid for smooth visualization
    x_min, x_max = xy_coords[:, 0].min(), xy_coords[:, 0].max()
    y_min, y_max = xy_coords[:, 1].min(), xy_coords[:, 1].max()
    
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Create figure
    n_cols = 4
    n_rows = int(np.ceil(len(unit_indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, unit_idx in enumerate(unit_indices):
        # Interpolate activation values onto regular grid
        try:
            zi = griddata(xy_coords, unit_activations[:, unit_idx], 
                         (xi_grid, yi_grid), method='cubic', fill_value=0)
            
            im = axes[i].contourf(xi_grid, yi_grid, zi, levels=50, cmap='viridis')
            axes[i].set_xlabel('X coordinate')
            axes[i].set_ylabel('Y coordinate')
            axes[i].set_title(f'Unit {allocentric_units[unit_idx]}')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
            
        except Exception as e:
            print(f"Warning: Could not interpolate unit {unit_idx}: {e}")
            axes[i].scatter(xy_coords[:, 0], xy_coords[:, 1], 
                          c=unit_activations[:, unit_idx], cmap='viridis', alpha=0.6)
            axes[i].set_xlabel('X coordinate')
            axes[i].set_ylabel('Y coordinate')
            axes[i].set_title(f'Unit {allocentric_units[unit_idx]} (scatter)')
    
    # Hide unused subplots
    for i in range(len(unit_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'allocentric_tuning_curves.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'allocentric_tuning_curves.svg'), 
                bbox_inches='tight')
    
    return fig


def simple_clustering_analysis(unit_activations, xy_coords, allocentric_units, output_dir='./results'):
    """
    Publication-quality clustering analysis based on spatial tuning properties.
    """
    print("Performing clustering analysis...")
    
    # Extract spatial features for clustering
    features = []
    for unit_idx in range(unit_activations.shape[1]):
        activations = unit_activations[:, unit_idx]
        
        # Spatial features: center of mass and correlations
        com_x = np.average(xy_coords[:, 0], weights=activations + 1e-10)
        com_y = np.average(xy_coords[:, 1], weights=activations + 1e-10)
        
        x_corr = np.corrcoef(xy_coords[:, 0], activations)[0, 1] if len(np.unique(activations)) > 1 else 0
        y_corr = np.corrcoef(xy_coords[:, 1], activations)[0, 1] if len(np.unique(activations)) > 1 else 0
        
        x_corr = 0 if np.isnan(x_corr) else x_corr
        y_corr = 0 if np.isnan(y_corr) else y_corr
        
        features.append([com_x, com_y, x_corr, y_corr])
    
    features_array = np.array(features)
    features_array = np.nan_to_num(features_array)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(StandardScaler().fit_transform(features_array))
    
    # Optimal k-means clustering with silhouette score
    silhouette_scores = []
    k_range = range(2, min(6, len(allocentric_units) // 2 + 1))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)
        if len(set(labels)) > 1:
            score = silhouette_score(features_pca, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    if silhouette_scores:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_pca)
    else:
        optimal_k = 2
        cluster_labels = np.zeros(len(allocentric_units))
    
    # Publication-quality visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color palette following paper style
    cluster_colors = sns.color_palette('Set2', optimal_k)
    
    # PCA plot with clusters
    for cluster_id in range(optimal_k):
        mask = cluster_labels == cluster_id
        ax1.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                   c=[cluster_colors[cluster_id]], alpha=0.8, s=80, 
                   edgecolors='black', linewidth=1, 
                   label=f'Cluster {cluster_id} (n={np.sum(mask)})')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)', 
                  fontweight='bold', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)', 
                  fontweight='bold', fontsize=12)
    ax1.set_title('Unit Clustering in PCA Space', fontweight='bold', fontsize=14)
    ax1.legend(frameon=False, fontsize=10)
    ax1.grid(True, alpha=0.3)
    sns.despine(ax=ax1)
    
    # Feature space (x vs y correlation)
    for cluster_id in range(optimal_k):
        mask = cluster_labels == cluster_id
        ax2.scatter(features_array[mask, 2], features_array[mask, 3], 
                   c=[cluster_colors[cluster_id]], alpha=0.8, s=80,
                   edgecolors='black', linewidth=1,
                   label=f'Cluster {cluster_id} (n={np.sum(mask)})')
    
    ax2.set_xlabel('X-coordinate Correlation', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Y-coordinate Correlation', fontweight='bold', fontsize=12)
    ax2.set_title('Spatial Correlation Preferences', fontweight='bold', fontsize=14)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.legend(frameon=False, fontsize=10)
    ax2.grid(True, alpha=0.3)
    sns.despine(ax=ax2)
    
    fig.suptitle(f'Allocentric Unit Clustering Analysis (k={optimal_k})', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save figures
    fig.savefig(os.path.join(output_dir, 'clustering_analysis.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'clustering_analysis.svg'), 
                bbox_inches='tight')
    
    print(f"Found {optimal_k} optimal clusters")
    return cluster_labels, optimal_k, fig


def visualize_clusters_by_tuning(unit_activations, xy_coords, allocentric_units, 
                                cluster_labels, output_dir='./results'):
    """
    Visualize representative tuning curves for each cluster in publication quality.
    
    Args:
        unit_activations: (n_samples, n_units) unit activations
        xy_coords: (n_samples, 2) x,y coordinates
        allocentric_units: Original unit indices
        cluster_labels: Cluster assignment for each unit
        output_dir: Directory to save results
        
    Returns:
        fig: Matplotlib figure showing cluster representatives
    """
    print("Visualizing cluster representatives...")
    
    n_clusters = len(np.unique(cluster_labels))
    
    # Find representative unit for each cluster (most similar to centroid)
    cluster_representatives = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_units = np.where(cluster_mask)[0]
        
        if len(cluster_units) > 0:
            cluster_activations = unit_activations[:, cluster_mask]
            cluster_centroid = np.mean(cluster_activations, axis=1)
            
            similarities = []
            for unit_idx in cluster_units:
                similarity = np.corrcoef(unit_activations[:, unit_idx], cluster_centroid)[0, 1]
                similarities.append(similarity if not np.isnan(similarity) else 0)
            
            representative_idx = cluster_units[np.argmax(similarities)]
            cluster_representatives.append(representative_idx)
        else:
            cluster_representatives.append(0)
    
    # Create publication-quality visualization
    n_cols = min(4, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_clusters == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_clusters > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Color palette for consistency
    cluster_colors = sns.color_palette('Set2', n_clusters)
    
    # Create high-resolution interpolation grid
    x_min, x_max = xy_coords[:, 0].min(), xy_coords[:, 0].max()
    y_min, y_max = xy_coords[:, 1].min(), xy_coords[:, 1].max()
    xi = np.linspace(x_min, x_max, 80)
    yi = np.linspace(y_min, y_max, 80)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    for cluster_id, rep_idx in enumerate(cluster_representatives):
        if cluster_id < len(axes):
            try:
                # High-quality interpolation
                zi = griddata(xy_coords, unit_activations[:, rep_idx], 
                             (xi_grid, yi_grid), method='cubic', fill_value=0)
                
                # Smooth contour plot with better styling
                im = axes[cluster_id].contourf(xi_grid, yi_grid, zi, levels=40, 
                                             cmap='viridis', alpha=0.9)
                
                # Add contour lines for better definition
                axes[cluster_id].contour(xi_grid, yi_grid, zi, levels=10, 
                                       colors='black', alpha=0.3, linewidths=0.5)
                
            except Exception as e:
                print(f"Interpolation failed for cluster {cluster_id}: {e}")
                # High-quality scatter fallback
                im = axes[cluster_id].scatter(xy_coords[:, 0], xy_coords[:, 1], 
                                            c=unit_activations[:, rep_idx], 
                                            cmap='viridis', alpha=0.8, s=3, 
                                            edgecolors='none')
            
            # Styling
            axes[cluster_id].set_xlabel('X coordinate', fontweight='bold', fontsize=12)
            axes[cluster_id].set_ylabel('Y coordinate', fontweight='bold', fontsize=12)
            
            cluster_size = np.sum(cluster_labels == cluster_id)
            axes[cluster_id].set_title(f'Cluster {cluster_id} Representative\n'
                                     f'Unit {allocentric_units[rep_idx]} (n={cluster_size})', 
                                     fontweight='bold', fontsize=13, 
                                     color=cluster_colors[cluster_id])
            
            # Style colorbar
            cbar = plt.colorbar(im, ax=axes[cluster_id], shrink=0.8)
            cbar.set_label('Activation', fontweight='bold', rotation=270, labelpad=15)
            
            # Clean axes
            sns.despine(ax=axes[cluster_id])
            axes[cluster_id].tick_params(axis='both', which='major', labelsize=10)
    
    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Cluster Representative Spatial Tuning Curves', 
                fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save figures
    fig.savefig(os.path.join(output_dir, 'cluster_representatives.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'cluster_representatives.svg'), 
                bbox_inches='tight')
    
    return fig


def analyze_allocentric_spatial_organization(trained_net, train_set, validation_set, 
                                           output_dir='./results', force_recompute=False):
    """
    Streamlined analysis focusing on clustering and spatial receptive fields with lesion analysis.
    
    Args:
        trained_net: Trained RNN model
        train_set: Training dataset  
        validation_set: Validation dataset
        output_dir: Directory to save all results
        force_recompute: Force recomputation of cached data
        
    Returns:
        results: Dictionary containing analysis results
    """
    print("="*60)
    print("ALLOCENTRIC ANALYSIS: PREDEFINED SPECIFIC LESIONED UNITS")
    print("="*60)
    
    # 1. Load or identify predefined lesioned units (with caching)
    print("\n1. Loading/identifying predefined specific lesioned units...")
    xy_units_cache_path = os.path.join(output_dir, "cached_xy_units.pkl")
    lesioned_units, reg_weights, test_score = load_or_store_xy_units(
        trained_net, train_set, validation_set, xy_units_cache_path, force_recompute
    )
    
    # 2. Extract continuous tuning data for predefined lesioned units
    print("\n2. Extracting 2D tuning profiles for predefined lesioned units...")
    unit_activations, xy_coords = analyze_continuous_xy_tuning(
        trained_net, validation_set, lesioned_units, output_dir=output_dir
    )
    
    # 3. Spatial receptive fields visualization (predefined lesioned units)
    print("\n3. Plotting spatial receptive fields...")
    fig_receptive = plot_spatial_receptive_fields(
        unit_activations, xy_coords, lesioned_units, n_units=len(lesioned_units), output_dir=output_dir
    )
    
    # 4. Clustering analysis (predefined lesioned units)
    print("\n4. Clustering analysis...")
    cluster_labels, optimal_k, fig_clustering = simple_clustering_analysis(
        unit_activations, xy_coords, lesioned_units, output_dir=output_dir
    )
    
    # 5. Visualize cluster representatives (predefined lesioned units)
    print("\n5. Visualizing cluster representatives...")
    fig_clusters = visualize_clusters_by_tuning(
        unit_activations, xy_coords, lesioned_units, cluster_labels, output_dir=output_dir
    )
    
    # 6. Perform targeted cluster lesions with polar scatter plots
    print("\n6. Performing targeted cluster lesions...")
    lesion_results = perform_cluster_lesions(
        trained_net, validation_set, lesioned_units, unit_activations, 
        xy_coords, cluster_labels, output_dir=output_dir
    )
    
    # 7. Save comprehensive results
    print("\n7. Saving results...")
    
    # Main results DataFrame
    results_df = pd.DataFrame({
        'unit_index': lesioned_units,
        'cluster': cluster_labels
    })
    
    results_df.to_csv(os.path.join(output_dir, 'lesioned_units_analysis.csv'), index=False)
    
    # Cluster summary with lesion effects
    cluster_summary = []
    for cluster_id in range(optimal_k):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        lesion_data = lesion_results['cluster_errors'][cluster_id]
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'lesion_error_increase': lesion_data['error_increase'],
            'preferred_error_direction': lesion_data['systematic_analysis']['preferred_direction'],
            'error_concentration': lesion_data['systematic_analysis']['concentration'],
            'horizontal_bias': lesion_data['systematic_analysis']['horizontal_bias'],
            'vertical_bias': lesion_data['systematic_analysis']['vertical_bias']
        })
    
    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_df.to_csv(os.path.join(output_dir, 'cluster_lesion_summary.csv'), index=False)
    
    # Compile all results
    results = {
        'lesioned_units': lesioned_units,
        'activations': unit_activations,
        'coordinates': xy_coords,
        'clusters': cluster_labels,
        'n_clusters': optimal_k,
        'decoding_score': test_score,
        'regression_weights': reg_weights,
        'lesion_results': lesion_results,
        'results_df': results_df,
        'cluster_summary': cluster_summary_df
    }
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"- Analyzed {len(lesioned_units)} predefined specific lesioned units")
    print(f"- Identified {optimal_k} distinct clusters")
    print(f"- Decoding performance: R² = {test_score:.4f}")
    
    # Print cluster summary with lesion effects
    print("\nCluster Lesion Summary:")
    for _, row in cluster_summary_df.iterrows():
        print(f"  Cluster {row['cluster_id']}: {row['size']} units, "
              f"lesion error increase: {row['lesion_error_increase']:.4f}, "
              f"systematic bias: ({row['horizontal_bias']:.3f}, {row['vertical_bias']:.3f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze allocentric coding unit spatial organization')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Full path to trained model (e.g. patterns_rev/mscoco_deepgaze3/mscoco_netl1_all_0_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_ReLU_nonCords_new_moreRL_)')
    
    parser.add_argument('--dataset_path', type=str, 
                        default='/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='/share/klab/psulewski/psulewski/EfficientRemapping/allocentric_analysis',
                        help='Directory to save results (default: alongside temporal contrastive results)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu), auto-detect if not specified')
    parser.add_argument('--model_instance', type=int, default=0,
                        help='Model instance to load (default: 0)')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force recomputation of cached activations')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = functions.get_device()
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Load datasets
    print(f"Loading datasets from {args.dataset_path}")
    train_set = H5dataset('train', args.dataset_path, device=device, use_color=False)
    validation_set = H5dataset('val', args.dataset_path, device=device, use_color=False)
    
    # Load trained model
    print(f"Loading energy-efficient model from {args.model_path}")
    print(args.model_path)
    model_title = args.model_path.rstrip("_").split("/")[-1]
    
    # Initialize and load model (parameters matching your energy-efficient model)
    net = RNN.State(
        activation_func=torch.nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-4,
        input_size=128*128,
        hidden_size=2048,
        title=args.model_path,
        device=device,
        use_fixation=True,
        use_conv=False,
        use_lstm=False,
        warp_imgs=True,
        use_resNet=False,
        time_steps_img=6,
        time_steps_cords=3,
        mnist=False,
        twolayer=True,  # 2-layer model as per your train_models.py
        dropout=0
    )
    
    # Load model weights
    try:
        net.load(twolayers=True)
        net.model.eval()
        print("Energy-efficient model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the model path and ensure the model files exist.")
        return
    
    # Override cached activation loading if requested
    if args.force_recompute:
        print("Force recompute flag set - will recompute all activations")
    
    # Run complete analysis
    try:
        results = analyze_allocentric_spatial_organization(
            net, train_set, validation_set, output_dir=args.output_dir, 
            force_recompute=args.force_recompute
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("- spatial_receptive_fields.png/svg: Publication-quality spatial receptive fields")
        print("- clustering_analysis.png/svg: PCA and correlation-based clustering analysis")
        print("- cluster_representatives.png/svg: Representative tuning curves for each cluster")
        print("- lesion_error_polar_count.png/svg: Error count histograms for cluster lesions")
        print("- lesion_error_polar_scatter.png/svg: Error amplitude polar scatter plots")
        print("- lesioned_units_analysis.csv: Predefined lesioned units and cluster assignments")
        print("- cluster_lesion_summary.csv: Summary of lesion effects per cluster")
        print("- cluster_lesion_analysis.csv: Detailed lesion analysis results")
        print("- cached_xy_units.pkl: Cached predefined lesioned units")
        print("- activations_cache_*.pkl: Cached activations for future use")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()