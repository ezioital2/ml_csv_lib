import numpy as np
import pandas as pd
from cuml.cluster import KMeans as cuKMeans
import jax.numpy as jnp
from jax import jit, vmap
import numba
from concurrent.futures import ThreadPoolExecutor

def reduce_dataset_stratified_clustering_jax(X, y, target_samples=20000, n_bins=50, random_state=42, n_jobs=-1):
    """
    Versión optimizada con JAX para grandes datasets (1.8M+ muestras)
    """
    
    # Convertir a arrays numpy
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    
    y_arr = np.asarray(y).ravel()
    n_samples = len(y_arr)
    
    print(f"Dataset original: {n_samples:,} muestras")
    print(f"Objetivo: {target_samples:,} muestras")
    
    # 1. Estratificación por target - optimizada con JAX
    start_time = pd.Timestamp.now()
    
    # Usar JAX para cálculo de percentiles más rápido
    y_jax = jnp.array(y_arr)
    quantiles = jnp.linspace(0, 100, n_bins + 1)
    bin_edges = jnp.percentile(y_jax, quantiles).block_until_ready()
    bin_edges = np.array(bin_edges)
    
    # Digitize optimizado con numba
    @numba.jit(nopython=True)
    def fast_digitize(data, edges):
        indices = np.zeros(len(data), dtype=np.int32)
        for i in range(len(data)):
            for j in range(len(edges)-1):
                if edges[j] <= data[i] < edges[j+1]:
                    indices[i] = j
                    break
            else:
                indices[i] = len(edges) - 2
        return indices
    
    bin_indices = fast_digitize(y_arr, bin_edges)
    
    print(f"Estratificación completada: {(pd.Timestamp.now() - start_time).total_seconds():.2f}s")
    
    # 2. Asignación proporcional por bin - optimizada
    start_time = pd.Timestamp.now()
    
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    valid_bins = np.where(bin_counts > 0)[0]
    
    # Asignación vectorizada
    total_samples = bin_counts.sum()
    allocation = np.zeros(n_bins, dtype=int)
    
    allocation[valid_bins] = np.maximum(1, (bin_counts[valid_bins] / total_samples * target_samples).astype(int))
    
    # Ajuste fino optimizado
    current_total = allocation.sum()
    diff = target_samples - current_total
    
    if diff != 0:
        adjustment_bins = valid_bins if diff > 0 else np.where(allocation > 1)[0]
        if len(adjustment_bins) > 0:
            # Usar JAX para cálculos de ajuste
            weights_jax = jnp.array(bin_counts[adjustment_bins].astype(float))
            weights_jax = weights_jax / jnp.sum(weights_jax)
            
            if diff > 0:
                adjustments = (np.array(weights_jax) * diff).astype(int)
                remainder = diff - adjustments.sum()
                if remainder > 0:
                    largest_indices = jnp.argsort(-weights_jax)[:remainder].block_until_ready()
                    adjustments[largest_indices] += 1
            else:
                adjustments = -(np.array(weights_jax) * -diff).astype(int)
                remainder = -diff + adjustments.sum()
                if remainder > 0:
                    allocation_jax = jnp.array(allocation[adjustment_bins])
                    largest_indices = jnp.argsort(-allocation_jax)[:remainder].block_until_ready()
                    adjustments[largest_indices] -= 1
            
            allocation[adjustment_bins] += adjustments
    
    print(f"Asignación por bins: {allocation.sum()} muestras ({(pd.Timestamp.now() - start_time).total_seconds():.2f}s)")
    
    # 3. Clustering por bin - optimizado con JAX para cálculo de distancias
    start_time = pd.Timestamp.now()
    
    representatives_idx = []
    X_jax = jnp.array(X_arr)  # Convertir a JAX array una vez
    
    # Función JAX optimizada para encontrar el punto más cercano al centroide
    @jit
    def find_closest_point_jax(points, center):
        distances = jnp.linalg.norm(points - center, axis=1)
        return jnp.argmin(distances)
    
    # Vectorizar la función para múltiples clusters
    find_closest_points_vectorized = jit(vmap(find_closest_point_jax, in_axes=(0, 0)))
    
    def process_bin(bin_idx):
        k_bin = allocation[bin_idx]
        if k_bin == 0:
            return []
            
        idx_in_bin = np.where(bin_indices == bin_idx)[0]
        
        if len(idx_in_bin) <= k_bin:
            return idx_in_bin.tolist()
        
        X_bin = X_arr[idx_in_bin]
        
        # KMeans con parámetros optimizados
        bin_seed = (random_state + bin_idx) % 100000
        
        kmeans = cuKMeans(
            n_clusters=min(k_bin, len(idx_in_bin)),
            random_state=bin_seed,
            n_init=3,
            max_iter=100,
            algorithm='elkan'
        )
        
        try:
            cluster_labels = kmeans.fit_predict(X_bin)
            cluster_centers = kmeans.cluster_centers_
            
            bin_representatives = []
            
            # Procesar clusters en lotes usando JAX
            X_bin_jax = X_jax[idx_in_bin]  # Usar el array JAX precomputado
            
            for cluster_id in range(k_bin):
                cluster_mask = (cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                    
                cluster_points_idx = idx_in_bin[cluster_mask]
                cluster_points_jax = X_bin_jax[cluster_mask]
                center_jax = jnp.array(cluster_centers[cluster_id])
                
                # Usar JAX para encontrar el punto más cercano (mucho más rápido)
                closest_idx_jax = find_closest_point_jax(cluster_points_jax, center_jax)
                closest_idx = cluster_points_idx[closest_idx_jax]
                bin_representatives.append(closest_idx)
            
            return bin_representatives
        
        except Exception as e:
            print(f"Error en bin {bin_idx}: {e}")
            # Fallback: selección aleatoria
            rng = np.random.default_rng(bin_seed)
            return rng.choice(idx_in_bin, size=k_bin, replace=False, axis=0).tolist()
    
    # Ejecutar en paralelo
    if n_jobs != 1:
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            futures = [executor.submit(process_bin, bin_idx) for bin_idx in valid_bins]
            for future in futures:
                representatives_idx.extend(future.result())
    else:
        # Ejecución secuencial (útil para debugging)
        for bin_idx in valid_bins:
            representatives_idx.extend(process_bin(bin_idx))
    
    # 4. Dataset final
    representatives_idx = np.unique(representatives_idx)
    X_reduced = X_arr[representatives_idx]
    y_reduced = y_arr[representatives_idx]
    
    print(f"Dataset reducido: {len(X_reduced):,} muestras ({len(X_reduced)/len(X_arr)*100:.2f}%)")
    print(f"Tiempo total de clustering: {(pd.Timestamp.now() - start_time).total_seconds():.2f}s")
    
    return X_reduced, y_reduced

# Instalación requerida:
# pip install jax jaxlib
# Para GPU: pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html