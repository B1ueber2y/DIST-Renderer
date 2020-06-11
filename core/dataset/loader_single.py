import numpy as np
import os, sys
import copy
import torch
import torch.utils.data
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.geometry import Camera
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mesh_dataset import MeshLoader

class LoaderSingle(torch.utils.data.Dataset):
    def __init__(self, synthetic_data_dir, mesh_data_dir, experiment_directory, split_file, dataset_name='syn_shapenet', flag_multi=False, load_mesh=False, load_code=False):
        self.dataset_name = dataset_name
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        if self.dataset_name == 'syn_shapenet':
            from syn_dataset import SynShapenetLoader
            self.image_loader = SynShapenetLoader(synthetic_data_dir, split_file=split_file)
        else:
            raise ValueError('dataset name {0} not defined'.format(self.dataset_name))
        self.mesh_loader = MeshLoader(mesh_data_dir, experiment_directory, split_file, load_mesh=load_mesh, load_code=load_code)
        self.load_mesh = load_mesh
        self.load_code = load_code

        # special matrix to transform the point cloud
        self.transform_matrix = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])

    def __len__(self):
        return len(self.image_loader)

    def get_data_from_md5(self, shape_md5, idx):
        image_data = self.image_loader.get_data_from_md5(shape_md5, idx)
        mesh_data = self.mesh_loader.get_data_from_md5(shape_md5)
        return mesh_data, image_data

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

    def transform_depth(self, depth, norm_param):
        depth_new = copy.deepcopy(depth)
        mask = depth < 1e5
        depth_new[mask] = depth[mask] * norm_param['scale']
        return depth_new

    def __getitem__(self, idx):
        image_data = self.image_loader[idx]

        instance_name = self.image_loader.get_instance_name(idx)
        mesh_data = self.mesh_loader.get_data_from_md5(instance_name)
        img, depth, normal, camera = image_data

        if self.load_code:
            if self.load_mesh:
                mesh_recon, latent_code, gt_samples, norm_param = mesh_data
            else:
                latent_code, gt_samples, norm_param = mesh_data
        else:
            gt_samples, norm_param = mesh_data

        # transform camera
        camera_transformed = self.transform_camera(camera, norm_param)
        depth_transformed = self.transform_depth(depth, norm_param)
        return instance_name, image_data, mesh_data, camera_transformed, depth_transformed

