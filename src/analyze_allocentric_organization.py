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
        return cache_data['allocentric_units'], cache_data['reg_weights'], cache_data['test_score']
    
    print("Identifying allocentric coding units (this may take a while)...")
    
    # Use existing methodology to identify xy units
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(
        net, train_set, validation_set, layer=[1, 2], mode='global', timestep=None
    )
    
    # Get top units for xy decoding (combine x and y predictive units)
    n_top_units = min(50, pred_cells.shape[1] // 2)
    top_x_units = pred_cells[0, -n_top_units:]
    top_y_units = pred_cells[1, -n_top_units:]
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
    
    # Focus on identified allocentric units
    unit_activations = activations[:, allocentric_units]
    
    print(f"Selected allocentric units data shape: activations {unit_activations.shape}, coords {xy_coords.shape}")
    return unit_activations, xy_coords


def plot_spatial_receptive_fields(unit_activations, xy_coords, allocentric_units, 
                                  n_units=9, output_dir='./results'):
    """
    Plot simple spatial receptive fields for top units - basic visualization first.
    """
    print(f"Creating spatial receptive field plots for {n_units} units...")
    
    # Select units to visualize
    unit_indices = np.linspace(0, len(allocentric_units)-1, n_units).astype(int)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, unit_idx in enumerate(unit_indices):
        # Simple scatter plot - let the data speak first
        scatter = axes[i].scatter(xy_coords[:, 0], xy_coords[:, 1], 
                                c=unit_activations[:, unit_idx], 
                                cmap='viridis', alpha=0.6, s=1)
        
        axes[i].set_xlabel('X coordinate')
        axes[i].set_ylabel('Y coordinate')
        axes[i].set_title(f'Unit {allocentric_units[unit_idx]}')
        plt.colorbar(scatter, ax=axes[i], shrink=0.8)
    
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
        
        # Add systematic bias vector
        sys_analysis = results['systematic_analysis']
        if sys_analysis['concentration'] > 0.1:  # Only show if concentrated
            ax.arrow(sys_analysis['preferred_direction'], 0, 0, max(hist) * 0.8,
                    head_width=0.2, head_length=max(hist)*0.1, fc='red', ec='red', linewidth=2)
        
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
    Simple clustering analysis based on spatial tuning properties.
    """
    print("Performing simple clustering analysis...")
    
    # Just use basic spatial features for clustering
    features = []
    for unit_idx in range(unit_activations.shape[1]):
        activations = unit_activations[:, unit_idx]
        
        # Simple features: center of mass and correlations
        com_x = np.average(xy_coords[:, 0], weights=activations + 1e-10)
        com_y = np.average(xy_coords[:, 1], weights=activations + 1e-10)
        
        x_corr = np.corrcoef(xy_coords[:, 0], activations)[0, 1] if len(np.unique(activations)) > 1 else 0
        y_corr = np.corrcoef(xy_coords[:, 1], activations)[0, 1] if len(np.unique(activations)) > 1 else 0
        
        x_corr = 0 if np.isnan(x_corr) else x_corr
        y_corr = 0 if np.isnan(y_corr) else y_corr
        
        features.append([com_x, com_y, x_corr, y_corr])
    
    features_array = np.array(features)
    features_array = np.nan_to_num(features_array)
    
    # Simple PCA visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(StandardScaler().fit_transform(features_array))
    
    # Simple k-means clustering
    silhouette_scores = []
    k_range = range(2, min(8, len(allocentric_units) // 3))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)
        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
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
    
    # Simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA plot with clusters
    scatter = ax1.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.7)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax1.set_title('Unit Clustering (PCA space)')
    
    # Feature space (x vs y correlation)
    scatter2 = ax2.scatter(features_array[:, 2], features_array[:, 3], 
                          c=cluster_labels, cmap='tab10', alpha=0.7)
    ax2.set_xlabel('X Correlation')
    ax2.set_ylabel('Y Correlation')
    ax2.set_title('X vs Y Correlation Preferences')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(output_dir, 'simple_clustering.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'simple_clustering.svg'), 
                bbox_inches='tight')
    
    print(f"Found {optimal_k} clusters")
    return cluster_labels, optimal_k, fig


def visualize_clusters_by_tuning(unit_activations, xy_coords, allocentric_units, 
                                cluster_labels, output_dir='./results'):
    """
    Visualize representative tuning curves for each cluster.
    
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
    
    # Create visualization
    fig, axes = plt.subplots(2, int(np.ceil(n_clusters/2)), figsize=(16, 8))
    if n_clusters <= 2:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    # Create interpolation grid
    x_min, x_max = xy_coords[:, 0].min(), xy_coords[:, 0].max()
    y_min, y_max = xy_coords[:, 1].min(), xy_coords[:, 1].max()
    xi = np.linspace(x_min, x_max, 50)
    yi = np.linspace(y_min, y_max, 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    for cluster_id, rep_idx in enumerate(cluster_representatives):
        if cluster_id < len(axes):
            try:
                zi = griddata(xy_coords, unit_activations[:, rep_idx], 
                             (xi_grid, yi_grid), method='cubic', fill_value=0)
                
                im = axes[cluster_id].contourf(xi_grid, yi_grid, zi, levels=30, cmap='viridis')
                axes[cluster_id].set_xlabel('X coordinate')
                axes[cluster_id].set_ylabel('Y coordinate')
                
                cluster_size = np.sum(cluster_labels == cluster_id)
                axes[cluster_id].set_title(f'Cluster {cluster_id} (n={cluster_size})\nUnit {allocentric_units[rep_idx]}')
                plt.colorbar(im, ax=axes[cluster_id], shrink=0.8)
                
            except:
                # Fallback to scatter plot
                axes[cluster_id].scatter(xy_coords[:, 0], xy_coords[:, 1], 
                                       c=unit_activations[:, rep_idx], cmap='viridis', alpha=0.6)
                axes[cluster_id].set_xlabel('X coordinate')
                axes[cluster_id].set_ylabel('Y coordinate')
                cluster_size = np.sum(cluster_labels == cluster_id)
                axes[cluster_id].set_title(f'Cluster {cluster_id} (n={cluster_size})\nUnit {allocentric_units[rep_idx]}')
    
    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
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
    print("STREAMLINED ALLOCENTRIC ANALYSIS WITH LESION STUDIES")
    print("="*60)
    
    # 1. Load or identify allocentric units (with caching)
    print("\n1. Loading/identifying allocentric coding units...")
    xy_units_cache_path = os.path.join(output_dir, "cached_xy_units.pkl")
    allocentric_units, reg_weights, test_score = load_or_store_xy_units(
        trained_net, train_set, validation_set, xy_units_cache_path, force_recompute
    )
    
    # 2. Extract continuous tuning data (with caching)
    print("\n2. Extracting 2D tuning profiles...")
    unit_activations, xy_coords = analyze_continuous_xy_tuning(
        trained_net, validation_set, allocentric_units, output_dir=output_dir
    )
    
    # 3. Spatial receptive fields visualization
    print("\n3. Plotting spatial receptive fields...")
    fig_receptive = plot_spatial_receptive_fields(
        unit_activations, xy_coords, allocentric_units, n_units=16, output_dir=output_dir
    )
    
    # 4. Clustering analysis
    print("\n4. Clustering analysis...")
    cluster_labels, optimal_k, fig_clustering = simple_clustering_analysis(
        unit_activations, xy_coords, allocentric_units, output_dir=output_dir
    )
    
    # 5. Visualize cluster representatives
    print("\n5. Visualizing cluster representatives...")
    fig_clusters = visualize_clusters_by_tuning(
        unit_activations, xy_coords, allocentric_units, cluster_labels, output_dir=output_dir
    )
    
    # 6. Perform targeted cluster lesions
    print("\n6. Performing targeted cluster lesions...")
    lesion_results = perform_cluster_lesions(
        trained_net, validation_set, allocentric_units, unit_activations, 
        xy_coords, cluster_labels, output_dir=output_dir
    )
    
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
        print("- spatial_receptive_fields.png/svg: Basic scatter plots of unit responses")
        print("- simple_clustering.png/svg: PCA and correlation-based clustering")
        print("- cluster_representatives.png/svg: Representative units for each cluster")
        print("- lesion_error_polar_plots.png/svg: Systematic error analysis from cluster lesions")
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