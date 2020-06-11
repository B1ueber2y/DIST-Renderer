import os
import numpy as np
import cv2
import pickle
import copy

def get_vis_depth(depth, mask=None, dmin=None, dmax=None, bg_color=255):
    depth_vis = copy.deepcopy(depth)
    if mask is not None:
        depth_vis[mask == 0] = 0
        depth_vis[depth_vis > 1e5] = 0 # handle lidar data
    else:
        bg_flag = (depth_vis > 1e5)
        depth_vis[bg_flag] = 0.0
    if (dmin is None) and (dmax is None):
        max_, min_ = depth_vis.max(), depth_vis.min()
    elif (dmin is not None) and (dmax is not None):
        max_, min_ = dmax, dmin
    else:
        raise ValueError('dmin and dmax should all be set.')
    bg_flag = (depth_vis == 0)
    depth_vis = np.clip(depth_vis, min_, max_)
    depth_vis = 255.0 * (depth_vis - min_) / (max_ - min_ + 1e-12)
    depth_vis[bg_flag] = bg_color
    depth_vis = depth_vis.astype(np.uint8)
    return depth_vis

def visualize_depth(fname, depth, mask=None, dmin=None, dmax=None):
    depth_vis = get_vis_depth(depth, mask=mask, dmin=dmin, dmax=dmax)
    cv2.imwrite(fname, depth_vis)

def get_vis_normal(normal, mask=None):
    normal_vis = copy.deepcopy(normal)
    norm = np.tile((normal_vis ** 2).sum(2)[:,:,None], (1,1,3))
    normal_vis = normal_vis / (norm + 1e-12)
    normal_vis[:,:,0] = normal_vis[:,:,0] * (1)
    normal_vis[:,:,1] = normal_vis[:,:,1] * (1)
    normal_vis[:,:,2] = normal_vis[:,:,2] * (-1)

    if mask is None:
        bg_flag = (normal_vis == 0)
    else:
        bg_flag = (mask == 0)
    normal_vis_new = copy.deepcopy(normal_vis)
    normal_vis_new[:,:,0] = 255.0 * (0.5 * normal_vis[:,:,0] + 0.5)
    normal_vis_new[:,:,1] = 255.0 * (0.5 * normal_vis[:,:,1] + 0.5)
    normal_vis_new[:,:,2] = 255.0 * (0.5 * normal_vis[:,:,2] + 0.5)
    normal_vis = normal_vis_new
    normal_vis[bg_flag] = 255
    normal_vis = normal_vis.astype(np.uint8)
    return normal_vis

def visualize_normal(fname, normal, mask=None):
    normal_vis = get_vis_normal(normal, mask=mask)
    normal_vis = cv2.cvtColor(normal_vis, cv2.COLOR_BGR2RGB)
    cv2.imwrite(fname, normal_vis)

def get_vis_mask(mask):
    mask_vis = mask.astype(np.float32) * 255.0
    mask_vis = 255 - mask_vis
    mask_vis = mask_vis.astype(np.uint8)
    return mask_vis

def visualize_mask(fname, mask):
    mask_vis = get_vis_mask(mask)
    cv2.imwrite(fname, mask_vis)

def visualize_pkl(prefix, pkl_fname):
    with open(pkl_fname, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    depth, normal, mask = data['depth'], data['normal'], data['valid_mask']
    depth = depth * mask
    normal = normal * np.tile(mask[:,:,None], (1,1,3))
    visualize_depth(prefix + '_depth.png', depth)
    visualize_mask(prefix + '_mask.png', mask)
    visualize_normal(prefix + '_normal.png', normal)

def generate_depth_map(xyz, img_hw):
    x, y, Z = xyz[0,:], xyz[1,:], xyz[2,:]
    eps = 1e-12
    X = x / (Z + eps)
    Y = y / (Z + eps)

    h, w = img_hw
    depth_map = np.ones((h, w)) * 1e11
    for i in range(X.shape[0]):
        x, y, z = X[i], Y[i], Z[i]
        x_, y_ = np.round(x).astype(int), np.round(y).astype(int)
        if x_ >= w or x_ < 0 or y_ >= h or y_ < 0:
            continue
        if z < depth_map[y_, x_]:
            depth_map[y_, x_] = z
    depth_map[depth_map == 1e11] = 0.0
    return depth_map

def project_points(points, projection, img_hw=(480, 480)):
    N = points.shape[0]
    ones = np.ones(N)
    points_homo = np.concatenate([points, ones[:,None]], 1).transpose(1,0)
    xyz = np.dot(projection, points_homo)
    depth_map = generate_depth_map(xyz, img_hw)
    return depth_map

def visualize_points(fname, points, projection, img_hw=(480, 480)):
    depth_map = project_points(points, projection, img_hw=img_hw)
    visualize_depth(fname, depth_map)
    return depth_map

def transform_points_search_axis(points, axis, sign):
    import copy
    points_new = copy.deepcopy(points)
    points_new[:,0] = sign[0] * points[:, axis[0]]
    points_new[:,1] = sign[1] * points[:, axis[1]]
    points_new[:,2] = sign[2] * points[:, axis[2]]
    return points_new

