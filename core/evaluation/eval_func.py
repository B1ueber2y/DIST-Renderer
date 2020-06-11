import os, sys
import numpy as np
from scipy.spatial import cKDTree as KDTree

def compute_chamfer_distance(points_1, points_2, use_square_dist=True):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    """
    # one direction
    points_1_kd_tree = KDTree(points_1)
    one_distances, one_vertex_ids = points_1_kd_tree.query(points_2)
    if use_square_dist:
        dist_chamfer_2to1 = np.mean(np.square(one_distances))
    else:
        dist_chamfer_2to1 = np.mean(one_distances)

    # other direction
    points_2_kd_tree = KDTree(points_2)
    two_distances, two_vertex_ids = points_2_kd_tree.query(points_1)
    if use_square_dist:
        dist_chamfer_1to2 = np.mean(np.square(two_distances))
    else:
        dist_chamfer_1to2 = np.mean(two_distances)
    return dist_chamfer_2to1 + dist_chamfer_1to2

def compute_chamfer_distance_separate(points_1, points_2):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    """
    # one direction
    points_1_kd_tree = KDTree(points_1)
    one_distances, one_vertex_ids = points_1_kd_tree.query(points_2)
    dist_chamfer_2to1 = np.mean(np.square(one_distances))

    # other direction
    points_2_kd_tree = KDTree(points_2)
    two_distances, two_vertex_ids = points_2_kd_tree.query(points_1)
    dist_chamfer_1to2 = np.mean(np.square(two_distances))

    return dist_chamfer_2to1, dist_chamfer_1to2

