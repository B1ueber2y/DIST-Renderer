import os, sys
nowpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(nowpath)
sys.path.append(os.path.join(nowpath, '..'))
from data_choy import DataLoader_Choy
from core.mystruct import Camera, View, FnameGroup, RenderParam
from core.dataset.ShapeNetV2 import ShapeNetV2
from core.randomize.randomizer import Randomizer
from core.tools.render import Renderer
import numpy as np
from tqdm import tqdm
import pdb

class ChoyRenderer(object):
    def __init__(self, shapenet_dir, choy_dir, output_dir, resolution=(137, 137)):
        self.data_shapenet = ShapeNetV2(shapenet_dir)
        self.data_choy = DataLoader_Choy(choy_dir)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        randomizer_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core', 'randomize', 'choy_renderer.yaml')
        self.randomizer = Randomizer(config_file=randomizer_config_file)
        self.resolution = resolution

        self.transform_matrix_deepsdf = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        self.transform_matrix_choy = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
        self.transform_matrix = np.dot(self.transform_matrix_choy, self.transform_matrix_deepsdf.T)

    def transform_camera(self, camera):
        K, RT = camera.intrinsic, camera.extrinsic
        R, T = RT[:,:3], RT[:,[3]]
        R_new = np.dot(R, self.transform_matrix)
        RT_new = np.concatenate([R_new, T], 1)
        camera_new = Camera(K, RT_new)
        return camera_new

    def generate_render_param_list(self):
        render_param_list = []
        for i, data in enumerate(self.data_choy):
            basedir = data['basedir']
            class_id = data['class_id']
            instance_name = data['instance_name']
            camera_list = data['camera_list']
            view_list = data['view_list']
            render_param = self.generate_render_param_from_camera(basedir, class_id, instance_name, camera_list)
            # render_param = self.generate_render_param_from_view(basedir, class_id, instance_name, camera_list)
            render_param_list.append(render_param)
        return render_param_list

    def generate_render_param_from_camera(self, basedir, class_id, instance_name, camera_list):
        shape = self.data_shapenet.get_shape_from_instance_name(class_id, instance_name)
        view_list, lighting_list, target_cfg_list = [], [], []
        for idx, camera in enumerate(camera_list):
            # set view
            view = View()
            camera = self.transform_camera(camera)
            view.set_camera(camera)
            # set lighting
            lighting = self.randomizer.randomize_lighting(use_point_lighting=False)
            # set target filename
            # basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
            basedir = os.path.join(self.output_dir, class_id, instance_name)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            target_cfg = FnameGroup(os.path.join(basedir, '{0}_{1}_{2}.png'.format(class_id, instance_name, idx)))
            # append to list
            view_list.append(view)
            lighting_list.append(lighting)
            target_cfg_list.append(target_cfg)
        render_param = RenderParam(shape, view_list, lighting_list, target_cfg_list, resolution=self.resolution) 
        return render_param

    def generate_render_param_from_view(self, basedir, class_id, instance_name, view_list):
        shape = self.data_shapenet.get_shape_from_instance_name(class_id, instance_name)
        view_list, lighting_list, target_cfg_list = [], [], []
        for idx, view in enumerate(view_list):
            lighting = self.randomizer.randomize_lighting()
            basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
            target_cfg = FnameGroup(os.path.join(basedir, '{0}_{1}_{2}.png'.format(class_id, instance_name, idx)))
            view_list.append(view)
            lighting_list.append(lighting)
            target_cfg_list.append(target_cfg)
        render_param = RenderParam(shape, view_list, lighting_list, target_cfg_list, resolution=self.resolution) 
        return render_param

if __name__ == '__main__':
    nowpath = os.path.dirname(os.path.abspath(__file__))
    shapenet_dir = os.path.join(nowpath, '../data/ShapeNetCore.v2')
    choy_dir = os.path.expanduser('~/data/ShapeNetRendering')
    blender_dir = os.path.join(nowpath, '../install/blender-2.71-linux-glibc211-x86_64/blender')
    output_dir = os.path.expanduser('~/data/ambient-choy-ShapeNetV2')

    choy_renderer = ChoyRenderer(shapenet_dir, choy_dir, output_dir)
    render_param_list = choy_renderer.generate_render_param_list()
    RP = Renderer(blender_dir, num_worker=8)
    RP.render_all(render_param_list, cam_lens=35, render_depth=True, render_normal=False, render_albedo=False)

