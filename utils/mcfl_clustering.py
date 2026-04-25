import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


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


def kmeans_cluster(embeddings, num_clusters, seed=42):
    points = np.asarray(embeddings, dtype=np.float32)
    n_points = len(points)

    if n_points == 0:
        return np.array([], dtype=np.int64)
    if n_points < num_clusters:
        return np.arange(n_points) % num_clusters

    if KMeans is not None:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
        return kmeans.fit_predict(points)

    return _kmeans_numpy(points, num_clusters=num_clusters, seed=seed)
