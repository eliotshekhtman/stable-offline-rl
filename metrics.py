# Tasks:
# - Phase-align trajectory pairs for the global stability metric.
# - Fit empirical C and rho bounds to collections of state-distance curves.
# - Measure distance from rollout samples to an offline training distribution.

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree


EPS = 1e-12
BOUND_MARGIN = 1e-5


def align_trajectory_pair(
    first: np.ndarray,
    second: np.ndarray,
    max_offset: int,
) -> tuple[int, np.ndarray]:
    """Choose a phase offset using second-half error, then return full-overlap distances."""
    best_offset = 0
    best_score = np.inf

    # This phase-alignment heuristic is an intentional, revisitable design
    # choice for periodic and goal-tracking trajectories.
    for offset in range(-max_offset, max_offset + 1):
        first_start = max(0, -offset)
        second_start = max(0, offset)
        overlap = min(len(first) - first_start, len(second) - second_start)
        if overlap <= 0:
            continue

        first_indices = np.arange(first_start, first_start + overlap)
        second_indices = np.arange(second_start, second_start + overlap)
        score_mask = (first_indices >= len(first) // 2) & (second_indices >= len(second) // 2)
        if not np.any(score_mask):
            continue

        differences = first[first_indices[score_mask]] - second[second_indices[score_mask]]
        score = float(np.linalg.norm(differences, axis=1).mean())
        if score < best_score:
            best_score = score
            best_offset = offset

    first_start = max(0, -best_offset)
    second_start = max(0, best_offset)
    overlap = min(len(first) - first_start, len(second) - second_start)
    differences = first[first_start : first_start + overlap] - second[second_start : second_start + overlap]
    return best_offset, np.linalg.norm(differences, axis=1).astype(np.float32)


def fit_empirical_bound(distance_curves: list[np.ndarray]) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit the tightest mean-log-slack exponential upper bound to normalized distances."""
    if not distance_curves:
        raise ValueError("At least one distance curve is required")

    max_length = max(len(curve) for curve in distance_curves)
    normalized = np.full((len(distance_curves), max_length), np.nan, dtype=np.float64)
    for index, curve in enumerate(distance_curves):
        curve = np.asarray(curve, dtype=np.float64)
        normalized[index, : len(curve)] = curve / max(float(curve[0]), EPS)

    envelope = np.nanmax(normalized, axis=0)
    support = np.sum(np.isfinite(normalized), axis=0).astype(np.int64)
    if np.all(envelope <= EPS):
        return 1.0, 0.0, envelope.astype(np.float32), support

    timesteps = np.arange(max_length, dtype=np.float64)
    positive = envelope > EPS
    fit_times = timesteps[positive]
    log_envelope = np.log(envelope[positive]) + np.log1p(BOUND_MARGIN)
    result = linprog(
        c=[1.0, fit_times.mean()],
        A_ub=np.column_stack((-np.ones(len(fit_times)), -fit_times)),
        b_ub=-log_envelope,
        bounds=[(0.0, None), (None, None)],
        method="highs",
    )
    log_c, log_rho = result.x
    c = float(np.exp(log_c))
    rho = float(np.exp(log_rho))
    return c, rho, envelope.astype(np.float32), support


def knn_distances(reference: np.ndarray, queries: np.ndarray, k: int = 5) -> np.ndarray:
    """Return each query's mean Euclidean distance to its k nearest reference points."""
    neighbor_count = min(k, len(reference))
    distances, _ = cKDTree(reference).query(queries, k=neighbor_count)
    if neighbor_count == 1:
        distances = distances[:, None]
    return np.asarray(distances, dtype=np.float32).mean(axis=1)
