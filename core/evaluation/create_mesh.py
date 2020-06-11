import os, sys
import numpy as np
import time
import torch
import skimage.measure
import plyfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from decoder_utils import decode_sdf

def transform_points_tensor(samples):
    samples_new = samples.clone()
    samples_new[:,1] = samples[:,2]
    samples_new[:,2] = -samples[:,1]
    return samples_new

def get_samples(N, voxel_origin, voxel_size, transform=False):
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]
    if transform:
        samples = transform_points_tensor(samples)
    return samples

def infer_samples(decoder, latent_vec, samples, max_batch=32 ** 3):
    '''
    Inputs:
    - samples [M, 4]
    Returns:
    - sdf_values M
    '''
    samples.requires_grad = False
    num_samples = samples.shape[0]

    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset).squeeze(1).detach().cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    return sdf_values

def create_mesh(decoder, latent_vec, filename, N=256, max_batch=32 ** 3, silent=False, transform=False):
    start = time.time()
    ply_filename = filename
    decoder.eval()

    # the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    # full resolution
    samples = get_samples(N, voxel_origin, voxel_size, transform=transform)
    sdf_values = infer_samples(decoder, latent_vec, samples, max_batch=max_batch)
    sdf_values = sdf_values.reshape(N, N, N)

    # convert sample grid to ply.
    flag = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(), voxel_origin, voxel_size, ply_filename + ".ply", silent=silent
    )

    end = time.time()
    if not silent:
        print("sampling takes: %f" % (end - start))
    return flag

def tile(a, dim, n_tile):
    # clever solution from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def upsample_cubic(values, N_prev, N_new):
    assert (values.shape[0] == N_prev ** 3)
    assert (N_new % N_prev == 0)

    scale = int(N_new / N_prev)
    new_values = values.reshape(N_prev, N_prev, N_prev)
    new_values = tile(new_values, 2, scale)
    new_values = tile(new_values, 1, scale)
    new_values = tile(new_values, 0, scale)
    new_values = new_values.reshape(-1)
    return new_values

def check_valid(sdf_values, voxel_size):
    relaxation = 1.5
    # (torch.abs(sdf_values) < voxel_size) # should be 1.732 / 2N, but we do some relaxation here to set 1.5N.
    # Note that our low-res grid does not perfectly match the high-res one, since the voxel size is not perfectly doubled.
    pos_samples = sdf_values > voxel_size * relaxation
    neg_samples = -sdf_values > voxel_size * relaxation
    valid_samples = torch.abs(sdf_values) <= voxel_size * relaxation
    return pos_samples, neg_samples, valid_samples

def create_mesh_speedup(decoder, latent_vec, filename, N=256, max_batch=32 ** 3, silent=False, transform=False):
    start = time.time()
    ply_filename = filename
    decoder.eval()

    # the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    voxel_size_half = 2.0 / (N/2 - 1)

    # 1/2 resolution
    samples_half = get_samples(int(N/2), voxel_origin, voxel_size_half, transform=transform)
    sdf_values_half = infer_samples(decoder, latent_vec, samples_half, max_batch=max_batch)
    sdf_values_half_upsampled = upsample_cubic(sdf_values_half, int(N/2), N)
    pos_samples, neg_samples, valid_samples = check_valid(sdf_values_half_upsampled, voxel_size_half)

    # full resolution
    samples = get_samples(N, voxel_origin, voxel_size, transform=transform)
    samples[pos_samples, 3] = 0.1
    samples[neg_samples, 3] = -0.1
    samples[valid_samples, 3] = infer_samples(decoder, latent_vec, samples[valid_samples, :], max_batch=max_batch)
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    # convert sample grid to ply.
    flag = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(), voxel_origin, voxel_size, ply_filename + ".ply", silent=silent
    )

    end = time.time()
    if not silent:
        print("sampling takes: %f" % (end - start))
    return flag

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor, voxel_grid_origin, voxel_size, ply_filename_out, silent=False
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    if not silent:
        print('Min: {0:.3f}, Max:{1:.3f}'.format(numpy_3d_sdf_tensor.min(), numpy_3d_sdf_tensor.max()))
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        print('ValueError! level 0.0 is not within the range of [Min, Max]')
        return False

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    return True

