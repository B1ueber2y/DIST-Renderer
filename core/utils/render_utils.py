import trimesh
import numpy as np
import cv2
import copy
import pickle
import torch
import pdb

def depth2normal(depth, f_pix_x, f_pix_y=None):
    '''
    To compute a normal map from the depth map
    Input:
    - depth:		torch.Tensor (H, W)
    - f_pix_x:		K[0, 0]
    - f_pix_y:		K[1, 1]
    Return:
    - normal:		torch.Tensor (H, W, 3)
    '''
    if f_pix_y is None:
        f_pix_y = f_pix_x
    h, w = depth.shape
    eps = 1e-12

    bg_flag = (depth > 1e5) | (depth == 0)
    depth[bg_flag] = 0.0

    depth_left, depth_right, depth_up, depth_down = torch.zeros(h, w), torch.zeros(h, w), torch.zeros(h, w), torch.zeros(h, w)
    if depth.get_device() != -1:
        device_id = depth.get_device()
        depth_left, depth_right, depth_up, depth_down = depth_left.to(device_id), depth_right.to(device_id), depth_up.to(device_id), depth_down.to(device_id)
    depth_left[:, 1:w-1] = depth[:, :w-2].clone()
    depth_right[:, 1:w-1] = depth[:, 2:].clone()
    depth_up[1:h-1, :] = depth[:h-2, :].clone()
    depth_down[1:h-1, :] = depth[2:, :].clone()

    dzdx = (depth_right - depth_left) * f_pix_x / 2.0
    dzdy = (depth_down - depth_up) * f_pix_y / 2.0

    normal = torch.stack([dzdx, dzdy, -torch.ones_like(dzdx)]).permute(1, 2, 0)
    normal_length = torch.norm(normal, p=2, dim=2)
    normal = normal / (normal_length + 1e-12)[:,:,None]
    normal[bg_flag] = 0.0
    return normal

def quad2rotation(quad):
    '''
    input: torch.Tensor (4)
    '''
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:,0], quad[:,1], quad[:,2], quad[:,3]

    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:,0,0] = 1 - 2 * (qj ** 2 + qk ** 2)
    rot_mat[:,0,1] = 2 * (qi * qj - qk * qr)
    rot_mat[:,0,2] = 2 * (qi * qk + qj * qr)
    rot_mat[:,1,0] = 2 * (qi * qj + qk * qr)
    rot_mat[:,1,1] = 1 - 2 * (qi ** 2 + qk ** 2)
    rot_mat[:,1,2] = 2 * (qj * qk - qi * qr)
    rot_mat[:,2,0] = 2 * (qi * qk - qj * qr)
    rot_mat[:,2,1] = 2 * (qj * qk + qi * qr)
    rot_mat[:,2,2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return rot_mat

def get_camera_from_tensor(inputs):
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:,:4], inputs[:,4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:,:,None]], 2)
    if N == 1:
        RT = RT[0]
    return RT

def get_tensor_from_camera(RT):
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:,:3], RT[:,3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor

def downsize_camera_intrinsic(intrinsic, factor):
    '''
    Input:
    - intrinsic		type: np.array (3,3)
    - factor		int
    '''
    img_h, img_w = int(2 * intrinsic[1,2]), int(2 * intrinsic[0,2])

    img_h_new, img_w_new = img_h / factor, img_w / factor
    if (img_h_new - round(img_h_new)) > 1e-12 or (img_w_new - round(img_w_new)) > 1e-12:
        raise ValueError('The image size {0} should be divisible by the factor {1}.'.format((img_h, img_w), factor))

    intrinsic_new = copy.deepcopy(intrinsic)
    intrinsic_new[0,:] = intrinsic[0,:] / factor
    intrinsic_new[1,:] = intrinsic[1,:] / factor
    return intrinsic_new

def sample_points_from_mesh(mesh, N=30000):
    '''
    Return:
    -- points:	np.array (N, 3)
    '''
    points = trimesh.sample.sample_surface(mesh, N)[0]
    return points

def transform_point_cloud(points):
    '''
    solve the mismatch between the point cloud coordinate and the mesh obj.
    '''
    points_new = copy.deepcopy(points)
    points_new[:,1] = -points[:,2]
    points_new[:,2] = points[:,1]
    return points_new

def read_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def save_render_output(render_output, fname):
    depth_rendered, normal_rendered, valid_mask_rendered, _ = render_output

    output = {}
    output['depth'] = depth_rendered.detach().cpu().numpy()
    output['normal'] = normal_rendered.detach().cpu().numpy()
    output['valid_mask'] = valid_mask_rendered.detach().cpu().numpy()
    save_pkl(output, fname)

def save_pkl(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    

