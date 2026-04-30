import numpy as np

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


def kmeans_cluster(embeddings, num_clusters, seed=42):
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
