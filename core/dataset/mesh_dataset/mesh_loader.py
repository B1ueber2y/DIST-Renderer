import torch
import torch.utils.data
import os, sys
import numpy as np
import json
import trimesh

class MeshLoader(object):
    def __init__(self, data_dir, experiment_directory, split_file, checkpoint_num=2000, mode=None, load_mesh=False, load_code=False):
        self.data_dir = data_dir
        self.experiment_directory = experiment_directory
        self.checkpoint_num = checkpoint_num
        self.load_mesh = load_mesh
        self.load_code = load_code
        if mode == None:
            self.mode = split_file[:-5].split('_')[-1]
        else:
            self.mode = mode

        with open(split_file, 'r') as f:
            split = json.load(f)

        key_list = list(split.keys())
        assert(len(key_list) == 1)
        self.dataset = key_list[0]

        data = split[self.dataset]
        key_list = list(data.keys())
        assert(len(key_list) == 1)
        self.class_name = key_list[0]

        self.instance_list = split[self.dataset][self.class_name]

        instance_list_new = []
        for instance in self.instance_list:
            fname = os.path.join(os.path.expanduser('~/data'), 'NormalizationParameters', 'ShapeNetV2', self.class_name, '{}.npz'.format(instance))
            if os.path.exists(fname):
                instance_list_new.append(instance)
            else:
                print('Instance {} does not exist!'.format(instance))
        print(self.class_name)
        self.instance_list = instance_list_new

        if self.load_code:
            if self.mode == 'train':
                latent_code_fname = os.path.join(self.experiment_directory, 'LatentCodes', '{}.pth'.format(checkpoint_num))
                npz_fname = os.path.splitext(latent_code_fname)[0] + '.npz'
                if os.path.isfile(npz_fname):
                    self.latent_codes = np.load(npz_fname)['data']
                else:
                    self.latent_codes = torch.load(latent_code_fname)['latent_codes'].detach().cpu().numpy() # [N, 1, 256]

    def __len__(self):
        return len(self.instance_list)

    def get_filenames(self, instance_name):
        if instance_name[-4:] == '.png':
            instance_name = instance_name[:-4]
        gt_samples_fname = os.path.join(self.data_dir, 'SurfaceSamples', self.dataset, self.class_name, instance_name + '.ply')
        norm_param_fname = os.path.join(self.data_dir, 'NormalizationParameters', self.dataset, self.class_name, instance_name + '.npz')

        latent_code_fname = os.path.join(self.experiment_directory, 'Reconstructions', str(self.checkpoint_num), 'Codes', self.dataset, self.class_name, instance_name + '.pth')
        reconstruction_fname = os.path.join(self.experiment_directory, 'Reconstructions', str(self.checkpoint_num), 'Meshes', self.dataset, self.class_name, instance_name + '.ply')
        return gt_samples_fname, norm_param_fname, latent_code_fname, reconstruction_fname

    def get_data_from_md5(self, shape_md5):
        return self.get_data(shape_md5)

    def get_data(self, instance_name):
        gt_samples_fname, norm_param_fname, latent_code_fname, reconstruction_fname = self.get_filenames(instance_name)
        if not os.path.exists(norm_param_fname):
            pdb.set_trace()
        normalization_params = np.load(norm_param_fname)
        if self.load_code:
            if self.mode == 'train':
                idx = self.instance_list.index(instance_name)
                latent_code = self.latent_codes[[idx]]
            elif self.mode == 'test':
                npz_fname = os.path.splitext(latent_code_fname)[0] + '.npz'
                if os.path.isfile(npz_fname):
                    latent_code = np.load(npz_fname)['data']
                else:
                    latent_code = torch.load(latent_code_fname).detach().cpu().numpy()
        gt_samples = trimesh.load(gt_samples_fname)
        if self.load_code:
            if self.load_mesh:
                mesh_reconstruction = trimesh.load(reconstruction_fname)
                return mesh_reconstruction, latent_code, gt_samples, normalization_params
            else:
                return latent_code, gt_samples, normalization_params
        else:
            return gt_samples, normalization_params

    def get_instance_name(self, idx):
        return self.instance_list[idx]

    def __getitem__(self, idx):
        instance_name = self.instance_list[idx]
        return self.get_data(instance_name)

if __name__ == '__main__':
    data_dir = os.path.expanduser('~/data')
    experiment_directory = os.path.expanduser('~/workspace/summer-2019/render_sdf/experiments/sofas-bs-32')
    split_file = os.path.expanduser('~/workspace/summer-2019/render_sdf/examples/splits/sv2_sofas_train.json')

    mesh_loader = MeshLoader(data_dir, experiment_directory, split_file, mode='train')
    mesh_recon, latent_code, gt_samples, norm_params = mesh_loader[0]
    pdb.set_trace()

