import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def reduce_dataset_stratified_clustering_fixed(X, y, target_samples=20000, n_bins=50, random_state=42):
    """
    Versión sin RandomState para evitar warnings de Pylint
    """
    
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()
    n_samples = len(y_arr)
    
    print(f"Dataset original: {n_samples} muestras")
    print(f"Objetivo: {target_samples} muestras")
    
    # 1. Estratificación por target
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_arr, quantiles)
    bin_indices = np.digitize(y_arr, bin_edges[1:-1], right=True)
    
    # 2. Asignación proporcional por bin
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    allocation = np.zeros(n_bins, dtype=int)
    
    total_samples = bin_counts.sum()
    for bin_idx in range(n_bins):
        if bin_counts[bin_idx] > 0:
            allocation[bin_idx] = max(1, int(bin_counts[bin_idx] / total_samples * target_samples))
    
    # Ajuste fino
    current_total = allocation.sum()
    diff = target_samples - current_total
    
    if diff > 0:
        sorted_bins = np.argsort(-bin_counts)
        for i in range(diff):
            bin_idx = sorted_bins[i % len(sorted_bins)]
            allocation[bin_idx] += 1
    elif diff < 0:
        sorted_bins = np.argsort(-allocation)
        for i in range(-diff):
            bin_idx = sorted_bins[i % len(sorted_bins)]
            if allocation[bin_idx] > 1:
                allocation[bin_idx] -= 1
    
    print(f"Asignación por bins: {allocation.sum()} muestras")
    
    # 3. Clustering por bin
    representatives_idx = []
    
    for bin_idx in range(n_bins):
        k_bin = allocation[bin_idx]
        if k_bin == 0:
            continue
            
        idx_in_bin = np.where(bin_indices == bin_idx)[0]
        
        if len(idx_in_bin) <= k_bin:
            representatives_idx.extend(idx_in_bin)
            continue
        
        X_bin = X_arr[idx_in_bin]
        
        # Generar random_state de manera alternativa
        bin_seed = (random_state + bin_idx) % 100000
        
        kmeans = KMeans(n_clusters=k_bin, 
                       random_state=bin_seed,
                       n_init=10,
                       max_iter=300)
        
        cluster_labels = kmeans.fit_predict(X_bin)
        cluster_centers = kmeans.cluster_centers_
        
        for cluster_id in range(k_bin):
            cluster_mask = (cluster_labels == cluster_id)
            if not np.any(cluster_mask):
                continue
                
            cluster_points_idx = idx_in_bin[cluster_mask]
            cluster_points = X_arr[cluster_points_idx]
            
            distances = np.linalg.norm(cluster_points - cluster_centers[cluster_id], axis=1)
            closest_idx = cluster_points_idx[np.argmin(distances)]
            representatives_idx.append(closest_idx)
    
    # 4. Dataset final
    representatives_idx = np.unique(representatives_idx)
    X_reduced = X_arr[representatives_idx]
    y_reduced = y_arr[representatives_idx]
    
    print(f"Dataset reducido: {len(X_reduced)} muestras ({len(X_reduced)/len(X_arr)*100:.2f}%)")
    
    return X_reduced, y_reduced