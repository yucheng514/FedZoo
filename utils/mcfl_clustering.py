import numpy as np
import torch

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    AgglomerativeClustering = None


def _kmeans_numpy(points, num_clusters, seed=42, max_iters=50):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=num_clusters, replace=False)
    centers = points[indices].copy()

    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(max_iters):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break

        labels = new_labels
        for cluster_id in range(num_clusters):
            cluster_points = points[labels == cluster_id]
            if len(cluster_points) == 0:
                centers[cluster_id] = points[rng.integers(0, len(points))]
            else:
                centers[cluster_id] = cluster_points.mean(axis=0)

    return labels


def kmeans_cluster(embeddings, num_clusters, seed=42, device=None):
    """K-means clustering with GPU support.

    Args:
        embeddings: numpy array or torch tensor
        num_clusters: number of clusters
        seed: random seed
        device: 'cpu', 'cuda', torch.device, or None (auto-detect from torch tensor if passed)

    Returns:
        numpy array of cluster labels
    """
    # Auto-detect device from tensor if embeddings is already on GPU
    if isinstance(embeddings, torch.Tensor):
        if device is None:
            device = embeddings.device
        # Convert to string if it's a torch.device object
        device_str = str(device) if not isinstance(device, str) else device
        # Use PyTorch version for GPU tensors
        if "cuda" in device_str or embeddings.device.type == "cuda":
            return _kmeans_torch(embeddings, num_clusters, seed=seed, device=embeddings.device)

    # Fallback for numpy arrays or CPU
    if device is None:
        device = "cpu"
    else:
        device = str(device) if not isinstance(device, str) else device

    if isinstance(embeddings, torch.Tensor):
        # Try PyTorch version even for CPU (faster convergence for large datasets)
        return _kmeans_torch(embeddings, num_clusters, seed=seed, device=device)

    # Original numpy path for compatibility
    points = np.asarray(embeddings, dtype=np.float32)
    n_points = len(points)

    if n_points == 0:
        return np.array([], dtype=np.int64)
    if n_points < num_clusters:
        return np.arange(n_points) % num_clusters

    if not np.isfinite(points).all():
        points = np.nan_to_num(points, nan=0.0, posinf=1e6, neginf=-1e6)

    if np.allclose(points, points[0], atol=1e-8, rtol=1e-5):
        return np.arange(n_points) % num_clusters

    if KMeans is not None:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
        return kmeans.fit_predict(points)

    return _kmeans_numpy(points, num_clusters=num_clusters, seed=seed)


def _kmeans_torch(points_tensor, num_clusters, seed=42, max_iters=50, device="cpu"):
    """PyTorch KMeans implementation that runs on any device (CPU/GPU)."""
    if not isinstance(points_tensor, torch.Tensor):
        points_tensor = torch.as_tensor(points_tensor, dtype=torch.float32, device=device)
    else:
        points_tensor = points_tensor.to(device)

    n_points = points_tensor.shape[0]

    # Initialize centers
    torch.manual_seed(seed)
    indices = torch.randperm(n_points, device=device)[:num_clusters]
    centers = points_tensor[indices].clone()

    labels = torch.zeros(n_points, dtype=torch.long, device=device)

    for iteration in range(max_iters):
        # Calculate distances
        distances = torch.cdist(points_tensor, centers)  # (n_points, num_clusters)
        new_labels = distances.argmin(dim=1)

        if torch.equal(new_labels, labels):
            break
        labels = new_labels

        # Update centers
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            if mask.sum() == 0:
                new_idx = torch.randint(n_points, (1,), device=device).item()
                centers[cluster_id] = points_tensor[new_idx].clone()
            else:
                centers[cluster_id] = points_tensor[mask].mean(dim=0)

    return labels.cpu().numpy()



def _sanitize_points(embeddings):
    points = np.asarray(embeddings, dtype=np.float32)
    n_points = len(points)

    if n_points == 0:
        return points, n_points

    if not np.isfinite(points).all():
        points = np.nan_to_num(points, nan=0.0, posinf=1e6, neginf=-1e6)

    return points, n_points


def _cosine_distance_matrix(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = points / norms
    similarity = normalized @ normalized.T
    similarity = np.clip(similarity, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def _agglomerative_numpy(points, num_clusters):
    distance = _cosine_distance_matrix(points)
    clusters = [[idx] for idx in range(len(points))]

    while len(clusters) > num_clusters:
        best_i, best_j = 0, 1
        best_score = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pair_dist = distance[np.ix_(clusters[i], clusters[j])]
                score = float(pair_dist.mean())
                if score < best_score:
                    best_score = score
                    best_i, best_j = i, j

        clusters[best_i].extend(clusters[best_j])
        del clusters[best_j]

    labels = np.zeros(len(points), dtype=np.int64)
    for cluster_id, indices in enumerate(clusters):
        labels[indices] = cluster_id
    return labels


def agglomerative_cluster(embeddings, num_clusters):
    points, n_points = _sanitize_points(embeddings)

    if n_points == 0:
        return np.array([], dtype=np.int64)
    if n_points < num_clusters:
        return np.arange(n_points) % num_clusters
    if np.allclose(points, points[0], atol=1e-8, rtol=1e-5):
        return np.arange(n_points) % num_clusters

    distance = _cosine_distance_matrix(points)

    if AgglomerativeClustering is not None:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                affinity="precomputed",
                linkage="average",
            )
        return clustering.fit_predict(distance)

    return _agglomerative_numpy(points, num_clusters=num_clusters)


def align_clusters(old_labels, new_labels):
    """Aligns new cluster labels to best match the old cluster labels.
    
    Args:
        old_labels: list or array of old cluster assignments
        new_labels: list or array of new cluster assignments
        
    Returns:
        A list of aligned new cluster assignments
    """
    if old_labels is None or len(old_labels) == 0:
        return new_labels
        
    old_labels = np.asarray(old_labels)
    new_labels = np.asarray(new_labels)
    
    unique_new = np.unique(new_labels)
    unique_old = np.unique(old_labels)
    
    # Cost matrix: overlap between new cluster i and old cluster j
    # We want to maximize overlap, which is equivalent to minimizing negative overlap.
    cost_matrix = np.zeros((len(unique_new), len(unique_old)))
    
    for i, n_label in enumerate(unique_new):
        for j, o_label in enumerate(unique_old):
            overlap = np.sum((new_labels == n_label) & (old_labels == o_label))
            cost_matrix[i, j] = -overlap
            
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ImportError:
        # Fallback to greedy assignment if scipy is not available
        row_ind, col_ind = [], []
        cost_copy = cost_matrix.copy()
        for _ in range(min(len(unique_new), len(unique_old))):
            r, c = np.unravel_index(cost_copy.argmin(), cost_copy.shape)
            if cost_copy[r, c] == 0:
                break # no more overlap
            row_ind.append(r)
            col_ind.append(c)
            cost_copy[r, :] = float('inf')
            cost_copy[:, c] = float('inf')
            
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[unique_new[r]] = unique_old[c]
        
    # Map any unassigned new clusters to available old labels, or keep original if none available
    used_old_labels = set(mapping.values())
    available_old_labels = list(set(unique_old) - used_old_labels)
    
    for n_label in unique_new:
        if n_label not in mapping:
            if available_old_labels:
                mapping[n_label] = available_old_labels.pop(0)
            else:
                mapping[n_label] = n_label
                
    aligned_labels = np.array([mapping[l] for l in new_labels])
    return aligned_labels
