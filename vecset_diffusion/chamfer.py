from functools import partial
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh


def one_sided_chamfer(points1, points2):
    """Calculate (arg)min_j \|points1[i] - points2[j]\| for each i."""
    points2_kd_tree = KDTree(points2)
    distances12, closest_indices2 = points2_kd_tree.query(points1)
    return distances12, closest_indices2

def normal_angle_deg(normals1, normals2):
    dot_products = np.einsum('...i, ...i -> ...', normalize(normals1), normalize(normals2))
    return np.arccos(np.clip(dot_products, -1, 1)) * 180 / np.pi

def unoriented_normal_angle_deg(normals1, normals2):
    dot_products = np.einsum('...i, ...i -> ...', normalize(normals1), normalize(normals2))
    return np.arccos(np.abs(np.clip(dot_products, -1, 1))) * 180 / np.pi

def normalize(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)

def compute_shape_metrics(dist_fn, metric_name, points1, normals1, points2, normals2):
    if (len(points1) == 0) or (len(points2) == 0):
        return {
            f'{metric_name}_distance': None,
            f'{metric_name}_square_distance': None,
            f'{metric_name}_normal_angle': None,
            f'{metric_name}_unoriented_normal_angle': None,
        }

    distances12, closest_indices2 = one_sided_chamfer(points1, points2)
    distances21, closest_indices1 = one_sided_chamfer(points2, points1)
    distance_metric = dist_fn([dist_fn(distances12), dist_fn(distances21)])
    distance_square_metric = dist_fn([dist_fn(distances12 ** 2), dist_fn(distances21 ** 2)])

    normal_angle12 = normal_angle_deg(normals1, normals2[closest_indices2])
    normal_angle21 = normal_angle_deg(normals2, normals1[closest_indices1])
    normal_angle_metric = dist_fn([dist_fn(normal_angle12), dist_fn(normal_angle21)])

    # compute normal metric again with the normals of one of the meshes flipped, and select whichever is smallest
    normal_angle12 = normal_angle_deg(normals1, -normals2[closest_indices2])
    normal_angle21 = normal_angle_deg(-normals2, normals1[closest_indices1])
    normal_angle_metric_flipped = dist_fn([dist_fn(normal_angle12), dist_fn(normal_angle21)])

    normal_angle_metric = min(normal_angle_metric, normal_angle_metric_flipped)

    unoriented_normal_angle12 = unoriented_normal_angle_deg(normals1, normals2[closest_indices2])
    unoriented_normal_angle21 = unoriented_normal_angle_deg(normals2, normals1[closest_indices1])
    unoriented_normal_angle_metric = dist_fn([dist_fn(unoriented_normal_angle12), dist_fn(unoriented_normal_angle21)])

    return {
        f'{metric_name}_distance': distance_metric,
        f'{metric_name}_square_distance': distance_square_metric,
        f'{metric_name}_normal_angle': normal_angle_metric,
        f'{metric_name}_unoriented_normal_angle': unoriented_normal_angle_metric,
    }

compute_chamfer = partial(compute_shape_metrics, np.mean, 'chamfer')
compute_hausdorff = partial(compute_shape_metrics, np.max, 'hausdorff')

def compute_f1_score(points1, points2, threshold=0.02):
    """
    Compute F1 score between two point clouds at a given distance threshold.

    F1 = 2 * (precision * recall) / (precision + recall)
    where:
    - precision = fraction of points1 within threshold distance to points2
    - recall = fraction of points2 within threshold distance to points1

    Args:
        points1: numpy array of shape (N, 3)
        points2: numpy array of shape (M, 3)
        threshold: distance threshold for considering points as matching

    Returns:
        Dictionary with precision, recall, and f1_score
    """
    if (len(points1) == 0) or (len(points2) == 0):
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
        }

    # Compute one-sided distances
    distances12, _ = one_sided_chamfer(points1, points2)
    distances21, _ = one_sided_chamfer(points2, points1)

    # Precision: fraction of points1 within threshold
    precision = np.mean(distances12 < threshold)

    # Recall: fraction of points2 within threshold
    recall = np.mean(distances21 < threshold)

    # F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        f'precision_{threshold}': float(precision),
        f'recall_{threshold}': float(recall),
        f'f1_score_{threshold}': float(f1_score),
    }