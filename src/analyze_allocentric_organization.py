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
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import os
import pickle
import json

# Import existing infrastructure
import RNN
import functions
from H5dataset import H5dataset
import ClosedFormDecoding


def load_or_store_xy_units(net, train_set, validation_set, cache_path, force_recompute=False, n_top_units=11):
    """
    Load or compute and store identified xy units to avoid repeated regression.
    
    Args:
        net: Trained RNN model
        train_set: Training dataset
        validation_set: Validation dataset
        cache_path: Path to save/load cached xy units
        force_recompute: Force recomputation even if cache exists
        n_top_units: Number of top units to select for each coordinate (default: 11)
        
    Returns:
        allocentric_units: Array of unit indices that encode xy coordinates
        reg_weights: Regression weights for decoding
        test_score: Decoding performance score
    """
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached xy units from {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(cache_data.keys())
        
        # If pred_cells exists in cache, use it to select top units by usefulness
        if 'pred_cells' in cache_data:
            pred_cells = cache_data['pred_cells']
            print(f"Selecting top {n_top_units} units each for x and y coordinates from cached regression weights")
            
            # Get top units for xy decoding (same logic as computation case)
            n_top_select = min(n_top_units, pred_cells.shape[1] // 2)
            top_x_units = pred_cells[0, -n_top_select:]  # Last n_top_select units are the most useful
            top_y_units = pred_cells[1, -n_top_select:]
            allocentric_units = np.unique(np.concatenate([top_x_units, top_y_units]))
            
            print(f"Selected {len(allocentric_units)} allocentric units from cache (top {n_top_select} for x, top {n_top_select} for y)")
        else:
            # Fallback to cached allocentric_units if pred_cells not available
            print("Warning: pred_cells not found in cache, using cached allocentric_units")
            allocentric_units = cache_data['allocentric_units']
 
        return allocentric_units, cache_data['reg_weights'], cache_data['test_score']
    
    print("Identifying allocentric coding units (this may take a while)...")
    
    # Use existing methodology to identify xy units
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(
        net, train_set, validation_set, layer=[1, 2], mode='global', timestep=None
    )
    
    # Get top units for xy decoding (combine x and y predictive units)
    n_top_select = min(n_top_units, pred_cells.shape[1] // 2)
    top_x_units = pred_cells[0, -n_top_select:]
    top_y_units = pred_cells[1, -n_top_select:]
    allocentric_units = np.unique(np.concatenate([top_x_units, top_y_units]))
    
    print(f"Identified {len(allocentric_units)} top allocentric units")
    print(f"X-coordinate decoding R²: {test_score}")
    
    # Cache the results
    print(f"Caching xy units to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {
        'allocentric_units': allocentric_units,
        'reg_weights': reg_weights,
        'test_score': test_score,
        'pred_cells': pred_cells,
        'model_info': {
            'model_title': net.title if hasattr(net, 'title') else 'unknown',
            'n_units': len(allocentric_units),
            'decoding_score': test_score
        }
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return allocentric_units, reg_weights, test_score


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


def analyze_continuous_xy_tuning(net, dataset, allocentric_units, output_dir='./results'):
    """
    Extract continuous 2D tuning curves for allocentric units.
    
    Args:
        net: Trained RNN model
        dataset: Dataset to analyze  
        allocentric_units: Indices of units that encode xy coordinates
        output_dir: Directory for caching activations
        
    Returns:
        unit_activations: (n_samples, n_units) activations of allocentric units
        xy_coords: (n_samples, 2) corresponding x,y coordinates
    """
    print(f"Extracting tuning data for {len(allocentric_units)} allocentric units...")
    
    # Create cache path for activations
    cache_filename = f"activations_cache_{getattr(dataset, 'split', 'unknown')}.pkl"
    cache_path = os.path.join(output_dir, cache_filename)
    
    # Load or compute full activations
    activations, xy_coords = load_or_compute_activations(net, dataset, cache_path)
    print("avtivations shape:", activations.shape)
    print(allocentric_units)
    # Focus on identified allocentric units
    unit_activations = activations[:, allocentric_units]
    
    print(f"Selected allocentric units data shape: activations {unit_activations.shape}, coords {xy_coords.shape}")
    return unit_activations, xy_coords


# Commented out - not needed for the essential analysis
# def plot_spatial_receptive_fields(...)
# This function was creating individual unit heatmaps but we focus on cluster analysis instead


def perform_cluster_lesions(net, dataset, allocentric_units, unit_activations, xy_coords, cluster_labels, output_dir='./results', subset_ratio=0.2):
    """
    Perform targeted in-silico lesions of unit clusters using proper network lesion maps and analyze systematic errors.
    
    Args:
        net: Trained RNN model
        dataset: Dataset to test lesions on
        allocentric_units: Unit indices of allocentric units
        unit_activations: Original activations before lesioning
        xy_coords: Ground truth coordinates
        cluster_labels: Cluster assignment for each unit
        output_dir: Directory to save results
        subset_ratio: Fraction of data to use for lesion testing (default: 0.2 to save time)
        
    Returns:
        lesion_results: Dictionary containing lesion analysis results
    """
    print("Performing targeted in-silico lesions with proper network lesion maps...")
    
    n_clusters = len(np.unique(cluster_labels))
    lesion_results = {}
    
    # Sample subset of data for faster lesion testing
    # Use the actual dataset length, not the cached activations length
    n_samples = len(dataset)
    subset_size = int(n_samples * subset_ratio)
    subset_indices = np.random.choice(n_samples, size=subset_size, replace=False)
    
    print(f"Using subset of {subset_size}/{n_samples} samples ({subset_ratio*100:.1f}%) for lesion testing")
    
    # Get baseline performance on subset using actual network forward passes
    print("Computing baseline performance...")
    
    # Ensure no lesion is active for baseline
    clearLesionMap(net.model)
    
    # Get baseline activations through proper forward passes
    subset_activations, subset_coords = get_lesioned_activations_simple(net, dataset, max_samples=subset_size)
    
    # Focus on allocentric units for decoding
    baseline_allocentric_activations = subset_activations[:, allocentric_units]
    
    # Get baseline decoding performance
    baseline_pred = decode_position_from_activations(baseline_allocentric_activations, subset_coords)
    baseline_error = compute_position_errors(subset_coords, baseline_pred)
    
    print(f"Baseline decoding error: {np.mean(baseline_error):.4f}")
    
    # Test lesioning each cluster
    cluster_errors = {}
    
    for cluster_id in range(n_clusters):
        print(f"Testing lesion of cluster {cluster_id}...")
        
        # Create lesion map for this cluster
        cluster_mask = cluster_labels == cluster_id
        lesioned_unit_indices = allocentric_units[cluster_mask]
        
        # Apply lesion to network using our simple cluster lesion function
        setClusterLesionMap(net.model, lesioned_unit_indices)
        
        # Get lesioned activations through proper forward passes
        lesioned_activations, lesioned_coords = get_lesioned_activations_simple(net, dataset, max_samples=subset_size)
        
        # Focus on allocentric units for decoding (excluding lesioned ones for fairness)
        remaining_allocentric_mask = ~cluster_mask
        if np.sum(remaining_allocentric_mask) == 0:
            print(f"  Warning: Cluster {cluster_id} contains all units, skipping...")
            continue
            
        remaining_allocentric_units = allocentric_units[remaining_allocentric_mask]
        lesioned_allocentric_activations = lesioned_activations[:, remaining_allocentric_units]
        
        # Decode positions with lesioned network
        lesioned_pred = decode_position_from_activations(lesioned_allocentric_activations, lesioned_coords)
        lesion_error = compute_position_errors(lesioned_coords, lesioned_pred)
        
        # Compute error vectors for systematic analysis
        error_vectors = lesioned_pred - lesioned_coords
        
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
    
    # Clear lesion map when done
    clearLesionMap(net.model)
    
    # Create polar plots of systematic errors
    create_error_polar_plots(cluster_errors, output_dir)
    
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
        'lesion_summary': lesion_df,
        'subset_size': subset_size,
        'subset_ratio': subset_ratio
    }
    
    return lesion_results


def get_lesioned_activations_simple(net, dataset, max_samples=500):
    """
    Get activations from network with current lesion map applied, respecting scene structure.
    
    Args:
        net: RNN model (with lesion map potentially applied) 
        dataset: Dataset to extract from
        max_samples: Maximum number of samples to process
        
    Returns:
        activations: (n_samples, n_units) activations
        coordinates: (n_samples, 2) x,y coordinates
    """
    # Use ClosedFormDecoding with full dataset
    activations, fixations = ClosedFormDecoding.getFeatures(
        net, dataset, layer_idx=[1, 2], timestep=None
    )
    
    # The data structure is: scenes -> 7 fixations per scene -> 6 timesteps per fixation
    # So each scene contributes 7 * 6 = 42 samples
    # Truncate at scene boundaries to maintain data structure integrity
    if len(activations) > max_samples:
        samples_per_scene = 42  # 7 fixations * 6 timesteps
        max_scenes = max_samples // samples_per_scene
        max_samples_aligned = max_scenes * samples_per_scene
        
        if max_samples_aligned > 0:
            activations = activations[:max_samples_aligned]
            fixations = fixations[:max_samples_aligned]
        else:
            # If max_samples is very small, just take first scene
            activations = activations[:samples_per_scene]
            fixations = fixations[:samples_per_scene]
    
    coordinates = fixations[:, :2]  # x,y coordinates
    return activations, coordinates


def setClusterLesionMap(model, lesioned_unit_indices):
    """
    Set lesion map for specific unit indices using the proper RNN lesion format.
    
    Args:
        model: The RNN model
        lesioned_unit_indices: Array of unit indices to lesion
    """
    # Convert to the format expected by RNN lesioning code
    # The RNN expects lesion_map to be a list of 4 arrays:
    # [layer1_coord1, layer1_coord2, layer2_coord1, layer2_coord2]
    
    # Filter indices based on layer (assuming 2-layer model with 2048 units per layer)
    layer1_indices = lesioned_unit_indices[lesioned_unit_indices < 2048]
    layer2_indices = lesioned_unit_indices[lesioned_unit_indices >= 2048] - 2048
    
    model.lesion = True
    model.lesion_map = [
        layer1_indices,  # layer 1, coordinate 1
        layer1_indices,  # layer 1, coordinate 2  
        layer2_indices,  # layer 2, coordinate 1
        layer2_indices   # layer 2, coordinate 2
    ]


def clearLesionMap(model):
    """
    Clear any active lesions from the model.
    
    Args:
        model: The RNN model
    """
    model.lesion = False
    model.lesion_map = None


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


def create_error_polar_plots(cluster_errors, output_dir):
    """
    Create polar plots showing systematic error directions for each cluster lesion.
    """
    n_clusters = len(cluster_errors)
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    for i, (cluster_id, results) in enumerate(cluster_errors.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='polar')
        
        error_vectors = results['error_vectors']
        dx, dy = error_vectors[:, 0], error_vectors[:, 1]
        
        # Convert to polar coordinates
        error_angles = np.arctan2(dy, dx)
        error_magnitudes = np.sqrt(dx**2 + dy**2)
        
        # Create histogram of error directions
        angle_bins = np.linspace(-np.pi, np.pi, 25)
        hist, bin_edges = np.histogram(error_angles, bins=angle_bins)
        
        # Plot as polar histogram
        theta = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(theta, hist, width=np.diff(bin_edges)[0], alpha=0.7)
        
        # # Add systematic bias vector
        # sys_analysis = results['systematic_analysis']
        # if sys_analysis['concentration'] > 0.1:  # Only show if concentrated
        #     ax.arrow(sys_analysis['preferred_direction'], 0, 0, max(hist) * 0.8,
        #             head_width=0.2, head_length=max(hist)*0.1, fc='red', ec='red', linewidth=2)
        
        ax.set_title(f'Cluster {cluster_id} Lesion\nError Direction Distribution\n'
                    f'({results["lesioned_units"]} units)', pad=20)
        ax.set_ylabel('Error Count', labelpad=30)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(output_dir, 'lesion_error_polar_plots.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'lesion_error_polar_plots.svg'), 
                bbox_inches='tight')
    
    return fig


# Commented out - not essential for the core analysis
# def visualize_2d_tuning_curves(...)
# This function was creating detailed tuning curves but we focus on cluster representatives instead


def simple_clustering_analysis(unit_activations, xy_coords, allocentric_units, output_dir='./results'):
    """
    Clustering analysis based on actual unit activation patterns.
    """
    print("Performing clustering analysis based on unit activations...")
    
    # Use the actual unit activations for clustering
    # Transpose so each row is a unit and each column is a spatial sample
    unit_patterns = unit_activations.T  # (n_units, n_samples)
    
    # Standardize activations for each unit
    scaler = StandardScaler()
    unit_patterns_scaled = scaler.fit_transform(unit_patterns)
    
    # Apply PCA for dimensionality reduction and visualization
    pca = PCA(n_components=min(10, unit_patterns_scaled.shape[1]))  
    unit_patterns_pca = pca.fit_transform(unit_patterns_scaled)
    
    # Use first 2 PCs for visualization
    features_pca = unit_patterns_pca[:, :2]
    
    # K-means clustering on the activation patterns
    silhouette_scores = []
    k_range = range(2, min(8, len(allocentric_units) // 2))
    
    if len(k_range) == 0:
        k_range = [2]
    
    for k in k_range:
        if k <= len(allocentric_units):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(unit_patterns_pca)
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(unit_patterns_pca, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        else:
            silhouette_scores.append(0)
    
    if silhouette_scores and max(silhouette_scores) > 0:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(unit_patterns_pca)
    else:
        optimal_k = 2
        cluster_labels = np.zeros(len(allocentric_units), dtype=int)
        if len(allocentric_units) > 1:
            cluster_labels[len(allocentric_units)//2:] = 1
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA plot with clusters  
    scatter = ax1.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.7)
    ax1.set_xlabel(f'PC1 [{pca.explained_variance_ratio_[0]:.1%} var]')
    ax1.set_ylabel(f'PC2 [{pca.explained_variance_ratio_[1]:.1%} var]')
    ax1.set_title('Unit activation clustering (PCA space)')
    
    # Show cumulative variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumvar)+1), cumvar, 'o-')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('PCA Variance Explained')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(output_dir, 'activation_clustering.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'activation_clustering.svg'), 
                bbox_inches='tight')
    
    print(f"Found {optimal_k} clusters based on activation patterns")
    print(f"PCA explains {cumvar[1]:.1%} variance in first 2 components")
    
    return cluster_labels, optimal_k, fig


def visualize_clusters_by_tuning(unit_activations, xy_coords, allocentric_units, 
                                cluster_labels, output_dir='./results'):
    """
    Visualize representative tuning curves for each cluster as 2D heatmaps.
    
    Args:
        unit_activations: (n_samples, n_units) unit activations
        xy_coords: (n_samples, 2) x,y coordinates
        allocentric_units: Original unit indices
        cluster_labels: Cluster assignment for each unit
        output_dir: Directory to save results
        
    Returns:
        fig: Matplotlib figure showing cluster representatives
    """
    print("Visualizing cluster representatives as 2D heatmaps...")
    
    # Set style to match supplementary results
    import seaborn as sns
    sns.set_context('talk')
    
    n_clusters = len(np.unique(cluster_labels))
    
    # For each cluster, find the unit closest to cluster center
    cluster_representatives = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_units = np.where(cluster_mask)[0]
        
        if len(cluster_units) > 0:
            # Find unit closest to cluster centroid (in feature space)
            cluster_activations = unit_activations[:, cluster_mask]
            cluster_centroid = np.mean(cluster_activations, axis=1)
            
            # Find unit most similar to centroid
            similarities = []
            for unit_idx in cluster_units:
                similarity = np.corrcoef(unit_activations[:, unit_idx], cluster_centroid)[0, 1]
                similarities.append(similarity if not np.isnan(similarity) else 0)
            
            representative_idx = cluster_units[np.argmax(similarities)]
            cluster_representatives.append(representative_idx)
        else:
            cluster_representatives.append(0)  # Fallback
    
    # Create visualization with proper aspect ratio
    n_cols = min(4, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_clusters == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    #subsample for better visualization
    if len(xy_coords) > 4000:
        subsample_indices = np.random.choice(len(xy_coords), size=2000, replace=False)
        xy_coords = xy_coords[subsample_indices]
        unit_activations = unit_activations[subsample_indices]
        
    
    for cluster_id, rep_idx in enumerate(cluster_representatives):
        if cluster_id < len(axes):
            
            # normalize activations for better visualization to 0-1 range
            act_min = np.min(unit_activations[:, rep_idx])
            act_max = np.max(unit_activations[:, rep_idx])
            norm_activations = (unit_activations[:, rep_idx] - act_min) / (act_max - act_min + 1e-6)
            
          
            # Fallback to scatter plot with same styling
            scatter = axes[cluster_id].scatter(xy_coords[:, 0], xy_coords[:, 1], 
                                    c=norm_activations, cmap='magma', alpha=0.8,s= 100*(norm_activations), )
                                    # size norm
                                
                                    
            axes[cluster_id].set_xlabel('x [coordinate]')
            axes[cluster_id].set_ylabel('y [coordinate]')
            cluster_size = np.sum(cluster_labels == cluster_id)
            axes[cluster_id].set_title(f'cluster {cluster_id} (n={cluster_size})\nunit [{allocentric_units[rep_idx]}]', 
                                        fontweight='bold')
            
            plt.colorbar(scatter, ax=axes[cluster_id], shrink=0.8)
    
    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)
    
    # Apply styling consistent with supplementary results
    for ax in axes[:n_clusters]:
        sns.despine(ax=ax)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    # Make sure text is editable in SVG
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(os.path.join(output_dir, 'cluster_representatives.pdf'), 
                dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'cluster_representatives.svg'), 
                bbox_inches='tight')
    
    return fig


def analyze_allocentric_spatial_organization(trained_net, train_set, validation_set, 
                                           output_dir='./results', force_recompute=False, n_top_units=11):
    """
    Streamlined analysis focusing on clustering and spatial receptive fields with lesion analysis.
    
    Args:
        trained_net: Trained RNN model
        train_set: Training dataset  
        validation_set: Validation dataset
        output_dir: Directory to save all results
        force_recompute: Force recomputation of cached data
        n_top_units: Number of top units to select for each coordinate (default: 11)
        
    Returns:
        results: Dictionary containing analysis results
    """
    print("="*60)
    print("STREAMLINED ALLOCENTRIC ANALYSIS WITH LESION STUDIES")
    print("="*60)
    
    # 1. Load or identify allocentric units (with caching)
    print("\n1. Loading/identifying allocentric coding units...")
    xy_units_cache_path = os.path.join(output_dir, "cached_xy_units.pkl")
    allocentric_units, reg_weights, test_score = load_or_store_xy_units(
        trained_net, train_set, validation_set, xy_units_cache_path, force_recompute, n_top_units
    )
    
    # Using heatmap visualization allows us to work with all identified allocentric units
    print(f"Using {len(allocentric_units)} allocentric units for analysis")
        
    
    # 2. Extract continuous tuning data (with caching)
    print("\n2. Extracting 2D tuning profiles...")
    unit_activations, xy_coords = analyze_continuous_xy_tuning(
        trained_net, validation_set, allocentric_units, output_dir=output_dir
    )
    
    # 3. Clustering analysis
    print("\n3. Clustering analysis...")
    cluster_labels, optimal_k, fig_clustering = simple_clustering_analysis(
        unit_activations, xy_coords, allocentric_units, output_dir=output_dir
    )
    
    # 4. Visualize cluster representatives as 2D heatmaps
    print("\n4. Visualizing cluster representative heatmaps...")
    fig_clusters = visualize_clusters_by_tuning(
        unit_activations, xy_coords, allocentric_units, cluster_labels, output_dir=output_dir
    )
    
    # 5. Perform targeted cluster lesions with proper lesion maps
    print("\n5. Performing targeted cluster lesions...")
    lesion_results = perform_cluster_lesions(
        trained_net, validation_set, allocentric_units, unit_activations, 
        xy_coords, cluster_labels, output_dir=output_dir, subset_ratio=1)
    
    
    # 7. Save comprehensive results
    print("\n7. Saving results...")
    
    # Main results DataFrame
    results_df = pd.DataFrame({
        'unit_index': allocentric_units,
        'cluster': cluster_labels
    })
    
    results_df.to_csv(os.path.join(output_dir, 'allocentric_units_analysis.csv'), index=False)
    
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
        'allocentric_units': allocentric_units,
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
    print(f"- Found {len(allocentric_units)} allocentric units (cached for future use)")
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
    parser.add_argument('--n_top_units', type=int, default=11,
                        help='Number of top units to select for each coordinate (default: 11)')
    
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
            force_recompute=args.force_recompute, n_top_units=args.n_top_units
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("- simple_clustering.png/svg: PCA and correlation-based clustering analysis")
        print("- cluster_representatives.png/svg: 2D heatmaps of representative units for each cluster")
        print("- lesion_error_polar_plots.png/svg: Systematic error analysis from proper cluster lesions")
        print("- allocentric_units_analysis.csv: Unit indices and cluster assignments")
        print("- cluster_lesion_summary.csv: Summary of lesion effects per cluster")
        print("- cluster_lesion_analysis.csv: Detailed lesion analysis results")
        print("- cached_xy_units.pkl: Cached allocentric unit identifications")
        print("- activations_cache_*.pkl: Cached activations for future use")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
    