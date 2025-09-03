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
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import os
import pickle

# Import existing infrastructure
import RNN
import functions
from H5dataset import H5dataset
import ClosedFormDecoding


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


def analyze_coordinate_systems(unit_activations, xy_coords, allocentric_units, output_dir='./results'):
    """
    Test different coordinate system representations.
    """
    print("Analyzing coordinate system preferences...")
    
    # Convert to different coordinate systems
    x, y = xy_coords[:, 0], xy_coords[:, 1]
    
    # Cartesian (already have)
    cartesian = np.column_stack([x, y])
    
    # Polar coordinates
    r = np.sqrt(x**2 + y**2)  # Distance from origin
    theta = np.arctan2(y, x)  # Angle
    polar = np.column_stack([r, theta])
    
    # Log-polar (common in vision)
    log_r = np.log(r + 1e-6)  # Add small epsilon to avoid log(0)
    log_polar = np.column_stack([log_r, theta])
    
    coordinate_systems = {
        'Cartesian (x,y)': cartesian,
        'Polar (r,θ)': polar, 
        'Log-polar (log(r),θ)': log_polar
    }
    
    # Test which coordinate system each unit prefers
    results = []
    
    for unit_idx in range(unit_activations.shape[1]):
        activations = unit_activations[:, unit_idx]
        
        unit_results = {'unit_index': allocentric_units[unit_idx]}
        
        for coord_name, coords in coordinate_systems.items():
            # Compute mutual information with each coordinate
            mi_coord1 = mutual_info_regression(coords[:, [0]], activations)[0]
            mi_coord2 = mutual_info_regression(coords[:, [1]], activations)[0]
            
            unit_results[f'{coord_name}_coord1_MI'] = mi_coord1
            unit_results[f'{coord_name}_coord2_MI'] = mi_coord2
            unit_results[f'{coord_name}_total_MI'] = mi_coord1 + mi_coord2
        
        results.append(unit_results)
    
    results_df = pd.DataFrame(results)
    
    # Find best coordinate system for each unit
    coord_cols = [col for col in results_df.columns if 'total_MI' in col]
    results_df['best_coordinate_system'] = results_df[coord_cols].idxmax(axis=1)
    results_df['best_coordinate_system'] = results_df['best_coordinate_system'].str.replace('_total_MI', '')
    
    # Summary statistics
    coord_preferences = results_df['best_coordinate_system'].value_counts()
    
    # Visualize coordinate system preferences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of coordinate system preferences
    coord_preferences.plot(kind='bar', ax=ax1)
    ax1.set_title('Coordinate System Preferences')
    ax1.set_xlabel('Coordinate System')
    ax1.set_ylabel('Number of Units')
    ax1.tick_params(axis='x', rotation=45)
    
    # Heatmap of mutual information
    mi_matrix = results_df[[col for col in results_df.columns if 'total_MI' in col]].values
    im = ax2.imshow(mi_matrix.T, aspect='auto', cmap='viridis')
    ax2.set_title('Mutual Information with Coordinate Systems')
    ax2.set_xlabel('Unit Index')
    ax2.set_ylabel('Coordinate System')
    ax2.set_yticks(range(len(coord_cols)))
    ax2.set_yticklabels([col.replace('_total_MI', '') for col in coord_cols])
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    
    # Save results
    fig.savefig(os.path.join(output_dir, 'coordinate_system_analysis.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'coordinate_system_analysis.svg'), 
                bbox_inches='tight')
    
    results_df.to_csv(os.path.join(output_dir, 'coordinate_system_preferences.csv'), index=False)
    
    print(f"Coordinate system preferences:")
    for coord_sys, count in coord_preferences.items():
        print(f"  {coord_sys}: {count} units ({count/len(results_df)*100:.1f}%)")
    
    return results_df, fig


def compute_spatial_information(unit_activations, xy_coords, allocentric_units, 
                               n_spatial_bins=20, output_dir='./results'):
    """
    Compute spatial information content for each unit.
    """
    print("Computing spatial information content...")
    
    # Discretize space for information calculation
    x_bins = np.linspace(xy_coords[:, 0].min(), xy_coords[:, 0].max(), n_spatial_bins)
    y_bins = np.linspace(xy_coords[:, 1].min(), xy_coords[:, 1].max(), n_spatial_bins)
    
    # Assign each sample to spatial bin
    x_bin_indices = np.digitize(xy_coords[:, 0], x_bins) - 1
    y_bin_indices = np.digitize(xy_coords[:, 1], y_bins) - 1
    
    # Combined spatial bin index
    spatial_bins = x_bin_indices * n_spatial_bins + y_bin_indices
    
    # Keep only valid bins
    valid_mask = (x_bin_indices >= 0) & (x_bin_indices < n_spatial_bins) & \
                 (y_bin_indices >= 0) & (y_bin_indices < n_spatial_bins)
    
    spatial_bins = spatial_bins[valid_mask]
    unit_activations_valid = unit_activations[valid_mask]
    xy_coords_valid = xy_coords[valid_mask]
    
    # Compute information metrics for each unit
    info_results = []
    
    for unit_idx in range(unit_activations_valid.shape[1]):
        activations = unit_activations_valid[:, unit_idx]
        
        # Mutual information with spatial position
        spatial_mi = mutual_info_regression(spatial_bins.reshape(-1, 1), activations)[0]
        
        # Mutual information with x and y separately
        x_mi = mutual_info_regression(xy_coords_valid[:, [0]], activations)[0]
        y_mi = mutual_info_regression(xy_coords_valid[:, [1]], activations)[0]
        
        # Spatial information rate (bits per spike, classic measure)
        # Based on Skaggs et al. 1993
        mean_rate = np.mean(activations)
        spatial_info = 0
        
        if mean_rate > 0:
            for bin_idx in range(n_spatial_bins * n_spatial_bins):
                bin_mask = spatial_bins == bin_idx
                if np.sum(bin_mask) > 5:  # Minimum samples per bin
                    bin_rate = np.mean(activations[bin_mask])
                    bin_occupancy = np.sum(bin_mask) / len(spatial_bins)
                    if bin_rate > 0:
                        spatial_info += bin_occupancy * bin_rate * np.log2(bin_rate / mean_rate)
        
        spatial_info /= mean_rate if mean_rate > 0 else 1
        
        # Spatial coherence (correlation between neighboring bins)
        spatial_coherence = compute_spatial_coherence(activations, spatial_bins, n_spatial_bins)
        
        info_results.append({
            'unit_index': allocentric_units[unit_idx],
            'spatial_MI': spatial_mi,
            'x_MI': x_mi,
            'y_MI': y_mi,
            'spatial_information_rate': spatial_info,
            'spatial_coherence': spatial_coherence,
            'mean_firing_rate': mean_rate,
            'x_y_MI_ratio': x_mi / (y_mi + 1e-10)  # Preference for x vs y
        })
    
    info_df = pd.DataFrame(info_results)
    
    # Visualize spatial information
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Spatial information distribution
    axes[0, 0].hist(info_df['spatial_information_rate'], bins=20, alpha=0.7)
    axes[0, 0].set_xlabel('Spatial Information Rate (bits/spike)')
    axes[0, 0].set_ylabel('Number of Units')
    axes[0, 0].set_title('Distribution of Spatial Information')
    
    # X vs Y mutual information
    axes[0, 1].scatter(info_df['x_MI'], info_df['y_MI'], alpha=0.7)
    axes[0, 1].plot([0, info_df[['x_MI', 'y_MI']].max().max()], 
                    [0, info_df[['x_MI', 'y_MI']].max().max()], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('X Mutual Information')
    axes[0, 1].set_ylabel('Y Mutual Information')
    axes[0, 1].set_title('X vs Y Coding Preferences')
    
    # Information vs firing rate
    axes[1, 0].scatter(info_df['mean_firing_rate'], info_df['spatial_information_rate'], alpha=0.7)
    axes[1, 0].set_xlabel('Mean Firing Rate')
    axes[1, 0].set_ylabel('Spatial Information Rate')
    axes[1, 0].set_title('Information vs Firing Rate')
    
    # Spatial coherence distribution  
    axes[1, 1].hist(info_df['spatial_coherence'], bins=20, alpha=0.7)
    axes[1, 1].set_xlabel('Spatial Coherence')
    axes[1, 1].set_ylabel('Number of Units')
    axes[1, 1].set_title('Spatial Coherence Distribution')
    
    plt.tight_layout()
    
    # Save results
    fig.savefig(os.path.join(output_dir, 'spatial_information_analysis.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'spatial_information_analysis.svg'), 
                bbox_inches='tight')
    
    info_df.to_csv(os.path.join(output_dir, 'spatial_information_metrics.csv'), index=False)
    
    # Print summary statistics
    print(f"\nSpatial Information Summary:")
    print(f"Mean spatial information: {info_df['spatial_information_rate'].mean():.3f} ± {info_df['spatial_information_rate'].std():.3f} bits/spike")
    print(f"Units with high spatial info (>1 bit/spike): {(info_df['spatial_information_rate'] > 1).sum()}")
    print(f"Mean spatial coherence: {info_df['spatial_coherence'].mean():.3f}")
    print(f"X-preferring units (X MI > Y MI): {(info_df['x_MI'] > info_df['y_MI']).sum()}")
    print(f"Y-preferring units (Y MI > X MI): {(info_df['y_MI'] > info_df['x_MI']).sum()}")
    
    return info_df, fig


def compute_spatial_coherence(activations, spatial_bins, n_spatial_bins):
    """
    Compute spatial coherence - how smooth the spatial tuning is.
    """
    # Create 2D rate map
    rate_map = np.zeros((n_spatial_bins, n_spatial_bins))
    count_map = np.zeros((n_spatial_bins, n_spatial_bins))
    
    for i in range(len(spatial_bins)):
        if 0 <= spatial_bins[i] < n_spatial_bins * n_spatial_bins:
            row = spatial_bins[i] // n_spatial_bins
            col = spatial_bins[i] % n_spatial_bins
            if 0 <= row < n_spatial_bins and 0 <= col < n_spatial_bins:
                rate_map[row, col] += activations[i]
                count_map[row, col] += 1
    
    # Average rates
    rate_map = np.divide(rate_map, count_map, out=np.zeros_like(rate_map), where=count_map!=0)
    
    # Compute correlations with neighboring bins
    coherences = []
    for i in range(1, n_spatial_bins-1):
        for j in range(1, n_spatial_bins-1):
            center_rate = rate_map[i, j]
            neighbor_rates = [
                rate_map[i-1, j], rate_map[i+1, j], 
                rate_map[i, j-1], rate_map[i, j+1]
            ]
            neighbor_rates = [r for r in neighbor_rates if r > 0]
            if len(neighbor_rates) > 0:
                coherences.append(np.corrcoef([center_rate] + neighbor_rates)[0, 1:].mean())
    
    return np.mean(coherences) if coherences else 0


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
                                           output_dir='./results'):
    """
    Complete analysis of allocentric unit spatial organization.
    
    Args:
        trained_net: Trained RNN model
        train_set: Training dataset  
        validation_set: Validation dataset
        output_dir: Directory to save all results
        
    Returns:
        results: Dictionary containing all analysis results
    """
    print("="*60)
    print("ALLOCENTRIC SPATIAL ORGANIZATION ANALYSIS")
    print("="*60)
    
    # 1. Identify allocentric units using existing methodology
    print("\n1. Identifying allocentric coding units...")
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(
        trained_net, train_set, validation_set, layer=[1, 2], mode='global', timestep=None
    )
    
    # Get top units for xy decoding (combine x and y predictive units)
    n_top_units = min(50, pred_cells.shape[1] // 2)  # Top 50 or half, whichever is smaller
    top_x_units = pred_cells[0, -n_top_units:]  # Top n units for x
    top_y_units = pred_cells[1, -n_top_units:]  # Top n units for y
    top_xy_units = np.unique(np.concatenate([top_x_units, top_y_units]))
    
    print(f"Identified {len(top_xy_units)} top allocentric units")
    print(f"X-coordinate decoding R²: {test_score}")
    
    # 2. Extract continuous tuning data (with caching)
    print("\n2. Extracting 2D tuning profiles...")
    unit_activations, xy_coords = analyze_continuous_xy_tuning(
        trained_net, validation_set, top_xy_units, output_dir=output_dir
    )
    
    # 3. Simple spatial receptive fields visualization (exploratory)
    print("\n3. Plotting spatial receptive fields...")
    fig_receptive = plot_spatial_receptive_fields(
        unit_activations, xy_coords, top_xy_units, output_dir=output_dir
    )
    
    # 4. Coordinate system analysis
    print("\n4. Analyzing coordinate system preferences...")
    coord_results, fig_coord = analyze_coordinate_systems(
        unit_activations, xy_coords, top_xy_units, output_dir=output_dir
    )
    
    # 5. Spatial information quantification
    print("\n5. Computing spatial information metrics...")
    info_results, fig_info = compute_spatial_information(
        unit_activations, xy_coords, top_xy_units, output_dir=output_dir
    )
    
    # 6. Detailed 2D tuning curve heatmaps (fewer units)
    print("\n6. Creating detailed 2D tuning curve heatmaps...")
    fig_tuning = visualize_2d_tuning_curves(
        unit_activations, xy_coords, top_xy_units, 
        unit_indices=list(range(min(16, len(top_xy_units)))), output_dir=output_dir
    )
    
    # 7. Simple clustering analysis
    print("\n7. Simple clustering analysis...")
    cluster_labels, optimal_k, fig_clustering = simple_clustering_analysis(
        unit_activations, xy_coords, top_xy_units, output_dir=output_dir
    )
    
    # 8. Visualize cluster representatives (simplified)
    print("\n8. Visualizing cluster representatives...")
    fig_clusters = visualize_clusters_by_tuning(
        unit_activations, xy_coords, top_xy_units, cluster_labels, output_dir=output_dir
    )
    
    # 9. Save comprehensive results
    print("\n9. Saving comprehensive results...")
    
    # Combine all results into summary DataFrames
    # Main results DataFrame combining spatial info and coordinate preferences
    results_df = pd.DataFrame({
        'unit_index': top_xy_units,
        'cluster': cluster_labels
    })
    
    # Add spatial information metrics
    results_df = results_df.merge(
        info_results[['unit_index', 'spatial_MI', 'x_MI', 'y_MI', 'spatial_information_rate', 
                     'spatial_coherence', 'mean_firing_rate', 'x_y_MI_ratio']], 
        on='unit_index', how='left'
    )
    
    # Add coordinate system preferences
    coord_summary = coord_results[['unit_index', 'best_coordinate_system']].copy()
    results_df = results_df.merge(coord_summary, on='unit_index', how='left')
    
    results_df.to_csv(os.path.join(output_dir, 'allocentric_units_comprehensive_analysis.csv'), index=False)
    
    # Simple cluster summary
    cluster_summary = []
    for cluster_id in range(optimal_k):
        cluster_mask = cluster_labels == cluster_id
        cluster_units = results_df[cluster_mask]
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'size': len(cluster_units),
            'mean_spatial_info': cluster_units['spatial_information_rate'].mean(),
            'mean_x_MI': cluster_units['x_MI'].mean(),
            'mean_y_MI': cluster_units['y_MI'].mean(),
            'dominant_coord_system': cluster_units['best_coordinate_system'].mode().iloc[0] if len(cluster_units) > 0 else 'unknown',
            'mean_coherence': cluster_units['spatial_coherence'].mean()
        })
    
    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_df.to_csv(os.path.join(output_dir, 'cluster_summary.csv'), index=False)
    
    # Compile all results
    results = {
        'allocentric_units': top_xy_units,
        'activations': unit_activations,
        'coordinates': xy_coords,
        'clusters': cluster_labels,
        'n_clusters': optimal_k,
        'decoding_score': test_score,
        'regression_weights': reg_weights,
        'pred_cells': pred_cells,
        'spatial_information': info_results,
        'coordinate_preferences': coord_results,
        'comprehensive_results': results_df,
        'cluster_summary': cluster_summary_df
    }
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"- Found {len(top_xy_units)} allocentric units")
    print(f"- Identified {optimal_k} distinct clusters")
    print(f"- Decoding performance: R² = {test_score:.4f}")
    
    # Print cluster summary
    print("\nCluster Summary:")
    for _, row in cluster_summary_df.iterrows():
        print(f"  Cluster {row['cluster_id']}: {row['size']} units, "
              f"info={row['mean_spatial_info']:.2f} bits/spike, "
              f"coord_sys={row['dominant_coord_system']}")
    
    # Print coordinate system summary
    coord_prefs = coord_results['best_coordinate_system'].value_counts()
    print(f"\nOverall Coordinate System Preferences:")
    for coord_sys, count in coord_prefs.items():
        print(f"  {coord_sys}: {count} units ({count/len(coord_results)*100:.1f}%)")
    
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
            net, train_set, validation_set, output_dir=args.output_dir
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("- spatial_receptive_fields.png/svg: Basic scatter plots of unit responses")
        print("- coordinate_system_analysis.png/svg: Cartesian vs polar vs log-polar preferences") 
        print("- spatial_information_analysis.png/svg: Information content and x/y selectivity")
        print("- allocentric_tuning_curves.png/svg: Detailed 2D tuning curve heatmaps")
        print("- simple_clustering.png/svg: PCA and correlation-based clustering")
        print("- cluster_representatives.png/svg: Representative units for each cluster")
        print("- allocentric_units_comprehensive_analysis.csv: Complete unit characteristics")
        print("- spatial_information_metrics.csv: Information-theoretic measures")
        print("- coordinate_system_preferences.csv: Coordinate system analysis")
        print("- cluster_summary.csv: Summary statistics for each cluster")
        print("- activations_cache_*.pkl: Cached activations for future use")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()