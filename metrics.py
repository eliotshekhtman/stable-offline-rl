# Tasks:
# - Measure distance from rollout samples to an offline training distribution.

import numpy as np
from scipy.spatial import cKDTree


EPS = 1e-12


def knn_distances(tree: cKDTree, queries: np.ndarray, k: int = 5) -> np.ndarray:
    """Return each query's mean Euclidean distance to its k nearest reference points."""
    neighbor_count = min(k, tree.n)
    distances, _ = tree.query(queries, k=neighbor_count)
    if neighbor_count == 1:
        distances = distances[:, None]
    return np.asarray(distances, dtype=np.float32).mean(axis=1)
