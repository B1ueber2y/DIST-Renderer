import numpy as np
import os, sys
import copy
import torch
import torch.utils.data
import pdb
import trimesh

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.geometry import Camera
from core.visualize.vis_utils import project_points, transform_points_search_axis
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mesh_dataset import MeshLoader

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class LoaderColor(torch.utils.data.Dataset):
    def __init__(self, synthetic_data_dir, mesh_data_dir, experiment_directory, split_file, dataset_name='syn_shapenet_ambient'):
        self.dataset_name = dataset_name
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        if self.dataset_name == 'syn_shapenet_ambient':
            from syn_dataset import SynShapenetAmbientLoader
            self.image_loader = SynShapenetAmbientLoader(synthetic_data_dir, split_file)
        else:
            raise ValueError('dataset name {0} not defined'.format(self.dataset_name))
        self.mesh_loader = MeshLoader(mesh_data_dir, experiment_directory, split_file, load_mesh=False, load_code=False)
        self.num_points = 30000

        if self.dataset_name == 'syn_shapenet_choy':
            # special matrix to transform the point cloud (choy specifically)
            self.transform_matrix = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
        else:
            self.transform_matrix = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])

    def __len__(self):
        return len(self.image_loader)

    def get_data_from_md5(self, shape_md5, idx):
        image_data = self.image_loader.get_data_from_md5(shape_md5, idx)
        mesh_data = self.mesh_loader.get_data_from_md5(shape_md5)
        return mesh_data, image_data

    def transform_points_choy(self, points):
        points_new = np.dot(self.transform_matrix, points.T).T
        return points_new

    def inv_transform_points_choy(self, points):
        points_new = np.dot(self.transform_matrix.T, points.T).T
        return points_new

    def transform_camera(self, camera, norm_param):
        '''
        transform the extrinsic parameters to align with the sdf volume.
        '''
        K = camera.intrinsic
        R, T = camera.extrinsic[:,:3], camera.extrinsic[:,3]
        offset, scale = norm_param['offset'], norm_param['scale']
        T_new = (np.dot(np.dot(R, self.transform_matrix), -offset) + T) * scale

        RT_new = np.concatenate([R, T_new[:,None]], 1)
        camera_transformed = Camera(K, RT_new)
        return camera_transformed

    def generate_grid(self, img_hw):
        h, w = img_hw[0], img_hw[1]
        x_grid = np.tile(np.arange(0, w), (h))
        y_grid = np.tile(np.arange(0, h)[:,None], (1, w)).reshape(-1)
        return x_grid, y_grid

    def unprojection(self, abs_depth, camera, img_hw=(480, 480)):
        '''
        returns: points (h*w, 3)
        '''
        # get homo 2d
        h, w = img_hw[0], img_hw[1]
        x_grid, y_grid = self.generate_grid(img_hw) # [h*w], [h*w]
        z = abs_depth.reshape(-1)
        homo_2d = np.concatenate([(x_grid * z)[:,None], (y_grid * z)[:,None], z[:,None]], 1).T # [3, h*w]

        # unproject
        K = camera.intrinsic
        K_inv = np.linalg.inv(K)
        R, T = camera.extrinsic[:, :3], camera.extrinsic[:, 3]
        points = np.dot(R.T, (np.dot(K_inv, homo_2d) - np.tile(T[:,None], (1, h*w)))) # [3, h*w]
        invalid_mask = np.logical_or((abs_depth.reshape(-1) == 0), (abs_depth.reshape(-1) > 1e10))
        # pdb.set_trace()
        # random_array = np.random.rand(points[:, invalid_mask].shape[1] * 3).reshape(3, -1) * 0.1 - 0.05
        # points[:, invalid_mask] = 1.0 + random_array
        points[:, invalid_mask] = 0.0
        valid_mask = np.logical_not(invalid_mask)
        return points.T, valid_mask

    def set_axes(self, ax):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    def plot_points_and_savefig(self, points, color='Greens', fname='points_vis.png'):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        self.set_axes(ax)
        self.plot_points(ax, points, color=color)
        plt.savefig(fname)
        plt.close()

    def plot_points(self, ax, points, color='Greens'):
        '''
        input: points (N, 3)
        '''
        ax.scatter3D(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap=color)

    def transform_depth(self, depth, norm_param):
        depth_new = copy.deepcopy(depth)
        mask = depth < 1e5
        depth_new[mask] = depth[mask] * norm_param['scale']
        return depth_new

    def to_tensor(self, array):
        return torch.from_numpy(array)

    def __getitem__(self, idx):
        # idx = 0
        image_data = self.image_loader[idx]

        instance_name = self.image_loader.get_instance_name(idx)
        mesh_data = self.mesh_loader.get_data_from_md5(instance_name)

        img, depth, camera, num = image_data
        gt_samples, norm_param = mesh_data

        # (projection) transform camera and get abs depth
        camera_transformed = self.transform_camera(camera, norm_param)
        # abs_depth = project_points(points_choy, camera_transformed.projection, img.shape[:2])

        # (unprojection)
        abs_depth = self.transform_depth(depth, norm_param)
        points_unproj, valid_mask = self.unprojection(abs_depth, camera_transformed, img.shape[:2])

        # (switch the axis back)
        points_unproj = self.inv_transform_points_choy(points_unproj)
        img = img / 255.0
        latent_code = latent_code.squeeze(0)
        img, latent_code, points_unproj, valid_mask = self.to_tensor(img).float(), self.to_tensor(latent_code).float(), self.to_tensor(points_unproj).float(), self.to_tensor(valid_mask).float()
        return img, latent_code, points_unproj, valid_mask, idx

