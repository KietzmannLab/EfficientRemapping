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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
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


def characterize_tuning_profiles(unit_activations, xy_coords):
    """
    Characterize each unit's 2D tuning with interpretable features.
    
    Args:
        unit_activations: (n_samples, n_units) unit activations
        xy_coords: (n_samples, 2) x,y coordinates
        
    Returns:
        features: (n_units, n_features) array of tuning characteristics
        feature_names: List of feature names
    """
    print(f"Characterizing tuning profiles for {unit_activations.shape[1]} units...")
    
    features = []
    feature_names = [
        'max_activation', 'mean_activation', 'std_activation',
        'center_of_mass_x', 'center_of_mass_y', 
        'spatial_spread_x', 'spatial_spread_y',
        'x_correlation', 'y_correlation',
        'quadrant_1', 'quadrant_2', 'quadrant_3', 'quadrant_4',
        'x_gradient_strength', 'y_gradient_strength',
        'spatial_selectivity_index'
    ]
    
    for unit_idx in range(unit_activations.shape[1]):
        activations = unit_activations[:, unit_idx]
        
        # Basic statistics
        max_activation = np.max(activations)
        mean_activation = np.mean(activations)
        std_activation = np.std(activations)
        
        # Avoid division by zero for center of mass calculation
        if np.sum(activations) == 0:
            activations = activations + 1e-10
            
        # Spatial properties - Center of mass
        com_x = np.average(xy_coords[:, 0], weights=activations)
        com_y = np.average(xy_coords[:, 1], weights=activations)
        
        # Spatial spread (weighted standard deviation)
        spread_x = np.sqrt(np.average((xy_coords[:, 0] - com_x)**2, weights=activations))
        spread_y = np.sqrt(np.average((xy_coords[:, 1] - com_y)**2, weights=activations))
        
        # Correlation with spatial coordinates
        x_corr = np.corrcoef(xy_coords[:, 0], activations)[0, 1] if len(np.unique(activations)) > 1 else 0
        y_corr = np.corrcoef(xy_coords[:, 1], activations)[0, 1] if len(np.unique(activations)) > 1 else 0
        
        # Replace NaN correlations with 0
        x_corr = 0 if np.isnan(x_corr) else x_corr
        y_corr = 0 if np.isnan(y_corr) else y_corr
        
        # Quadrant preferences
        q1_mask = (xy_coords[:, 0] > com_x) & (xy_coords[:, 1] > com_y)
        q2_mask = (xy_coords[:, 0] < com_x) & (xy_coords[:, 1] > com_y)
        q3_mask = (xy_coords[:, 0] < com_x) & (xy_coords[:, 1] < com_y)
        q4_mask = (xy_coords[:, 0] > com_x) & (xy_coords[:, 1] < com_y)
        
        q1 = np.mean(activations[q1_mask]) if np.any(q1_mask) else 0
        q2 = np.mean(activations[q2_mask]) if np.any(q2_mask) else 0
        q3 = np.mean(activations[q3_mask]) if np.any(q3_mask) else 0
        q4 = np.mean(activations[q4_mask]) if np.any(q4_mask) else 0
        
        # Gradient strength (linear relationship with coordinates)
        x_gradient_strength = abs(x_corr)
        y_gradient_strength = abs(y_corr)
        
        # Spatial selectivity index (how concentrated the tuning is)
        spatial_selectivity = (max_activation - mean_activation) / (max_activation + mean_activation + 1e-10)
        
        features.append([
            max_activation, mean_activation, std_activation,
            com_x, com_y, spread_x, spread_y,
            x_corr, y_corr, q1, q2, q3, q4,
            x_gradient_strength, y_gradient_strength,
            spatial_selectivity
        ])
    
    features_array = np.array(features)
    
    # Handle any remaining NaN values
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Feature matrix shape: {features_array.shape}")
    return features_array, feature_names


def cluster_by_tuning_properties(tuning_features, feature_names, output_dir='./results'):
    """
    Cluster units based on their 2D tuning characteristics.
    
    Args:
        tuning_features: (n_units, n_features) array of tuning characteristics
        feature_names: List of feature names
        output_dir: Directory to save results
        
    Returns:
        cluster_labels: Cluster assignment for each unit
        optimal_k: Optimal number of clusters
        scaler: Fitted StandardScaler
        pca: Fitted PCA model
    """
    print("Clustering units by tuning properties...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(tuning_features)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=min(10, features_scaled.shape[1]))
    features_pca = pca.fit_transform(features_scaled)
    
    # Find optimal number of clusters
    k_range = range(2, min(10, len(tuning_features) // 2))
    silhouette_scores = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)
        score = silhouette_score(features_pca, labels)
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_pca)
    
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Best silhouette score: {max(silhouette_scores):.3f}")
    
    # Visualize clustering results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Silhouette scores
    axes[0,0].plot(k_range, silhouette_scores, 'bo-')
    axes[0,0].axvline(optimal_k, color='r', linestyle='--', alpha=0.7)
    axes[0,0].set_xlabel('Number of clusters')
    axes[0,0].set_ylabel('Silhouette score')
    axes[0,0].set_title('Cluster validation')
    axes[0,0].grid(True, alpha=0.3)
    
    # PCA visualization
    scatter = axes[0,1].scatter(features_pca[:, 0], features_pca[:, 1], 
                               c=cluster_labels, cmap='tab10', alpha=0.7)
    axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0,1].set_title('Clusters in PCA space')
    plt.colorbar(scatter, ax=axes[0,1])
    
    # Feature importance (PCA loadings)
    pc1_loadings = pca.components_[0]
    pc2_loadings = pca.components_[1]
    
    axes[1,0].barh(range(len(feature_names)), pc1_loadings)
    axes[1,0].set_yticks(range(len(feature_names)))
    axes[1,0].set_yticklabels(feature_names, fontsize=8)
    axes[1,0].set_xlabel('PC1 loading')
    axes[1,0].set_title('Feature contributions to PC1')
    
    axes[1,1].barh(range(len(feature_names)), pc2_loadings)
    axes[1,1].set_yticks(range(len(feature_names)))
    axes[1,1].set_yticklabels(feature_names, fontsize=8)
    axes[1,1].set_xlabel('PC2 loading')
    axes[1,1].set_title('Feature contributions to PC2')
    
    plt.tight_layout()
    
    # Save clustering analysis
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'clustering_analysis.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'clustering_analysis.svg'), 
                bbox_inches='tight')
    
    return cluster_labels, optimal_k, scaler, pca


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
    
    # 3. Visualize 2D tuning curves
    print("\n3. Creating 2D tuning curve visualizations...")
    fig_tuning = visualize_2d_tuning_curves(
        unit_activations, xy_coords, top_xy_units, output_dir=output_dir
    )
    
    # 4. Characterize and cluster tuning profiles
    print("\n4. Characterizing tuning profiles...")
    tuning_features, feature_names = characterize_tuning_profiles(
        unit_activations, xy_coords
    )
    
    print("\n5. Clustering analysis...")
    cluster_labels, optimal_k, scaler, pca = cluster_by_tuning_properties(
        tuning_features, feature_names, output_dir=output_dir
    )
    
    # 6. Visualize cluster representatives
    print("\n6. Visualizing cluster representatives...")
    fig_clusters = visualize_clusters_by_tuning(
        unit_activations, xy_coords, top_xy_units, cluster_labels, output_dir=output_dir
    )
    
    # 7. Save detailed results
    print("\n7. Saving detailed results...")
    
    # Create summary DataFrame
    results_df = pd.DataFrame({
        'unit_index': top_xy_units,
        'cluster': cluster_labels,
        **{name: tuning_features[:, i] for i, name in enumerate(feature_names)}
    })
    
    results_df.to_csv(os.path.join(output_dir, 'allocentric_units_analysis.csv'), index=False)
    
    # Save cluster summary
    cluster_summary = []
    for cluster_id in range(optimal_k):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_features = tuning_features[cluster_mask]
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'mean_x_corr': np.mean(cluster_features[:, feature_names.index('x_correlation')]),
            'mean_y_corr': np.mean(cluster_features[:, feature_names.index('y_correlation')]),
            'mean_selectivity': np.mean(cluster_features[:, feature_names.index('spatial_selectivity_index')]),
            'mean_spread_x': np.mean(cluster_features[:, feature_names.index('spatial_spread_x')]),
            'mean_spread_y': np.mean(cluster_features[:, feature_names.index('spatial_spread_y')])
        })
    
    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_df.to_csv(os.path.join(output_dir, 'cluster_summary.csv'), index=False)
    
    # Compile all results
    results = {
        'allocentric_units': top_xy_units,
        'activations': unit_activations,
        'coordinates': xy_coords,
        'tuning_features': tuning_features,
        'feature_names': feature_names,
        'clusters': cluster_labels,
        'n_clusters': optimal_k,
        'decoding_score': test_score,
        'regression_weights': reg_weights,
        'pred_cells': pred_cells,
        'scaler': scaler,
        'pca': pca,
        'results_df': results_df,
        'cluster_summary': cluster_summary_df
    }
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"- Found {len(top_xy_units)} allocentric units")
    print(f"- Identified {optimal_k} distinct clusters")
    print(f"- Decoding performance: R² = {test_score:.4f}")
    
    # Print cluster summary
    print("\nCluster Summary:")
    print(cluster_summary_df.round(3))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze allocentric coding unit spatial organization')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Full path to trained model (e.g. /share/klab/psulewski/tnortmann/efficient-remapping/EmergentPredictiveCoding/models/patterns_rev/mscoco_deepgaze3/mscoco_netl1_all_0_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_ReLU_nonCords_new_moreRL_)')
    parser.add_argument('--dataset_path', type=str, 
                        default='/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='/share/klab/psulewski/psulewski/EfficientRemapping/models/temporal_contrastive/allocentric_analysis',
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
        net.load(args.model_instance, twolayers=True)
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
        print("- allocentric_tuning_curves.png/svg: 2D tuning curve heatmaps")
        print("- clustering_analysis.png/svg: Clustering validation and PCA")  
        print("- cluster_representatives.png/svg: Representative units for each cluster")
        print("- allocentric_units_analysis.csv: Detailed unit characteristics")
        print("- cluster_summary.csv: Summary statistics for each cluster")
        print("- activations_cache_*.pkl: Cached activations for future use")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()