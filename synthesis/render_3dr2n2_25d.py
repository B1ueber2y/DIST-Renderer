'''
randomize my own dataset for 2d training with respect to the same data of choy.
'''
import os, sys
import yaml
nowpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(nowpath))
from common.dataset import ShapeNetV2, Sun2012pascal
from common.geometry import View, Camera
sys.path.append(nowpath)
from preprocess_choy.data_choy import DataLoader_Choy
from src.mystruct import FnameGroup, RenderParam
from src.randomize.randomizer import Randomizer
from src.tools.render import Renderer
from src.tools.crop import Cropper
from src.tools.composite import Compositor
from src.param_decomposer import AllParams, ParamDecomposer
import numpy as np

MIN_DIST, MAX_DIST = 1.0, 3.0

class OursRenderer25D(object):
    def __init__(self, shapenet_dir, choy_dir, output_dir, randomizer, resolution=(256, 256)):
        self.data_shapenet = ShapeNetV2(shapenet_dir)
        self.data_choy = DataLoader_Choy(choy_dir)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.randomizer = randomizer
        self.resolution = resolution

    def generate_all_param_list(self):
        all_param_list = []
        for i, data in enumerate(self.data_choy):
            basedir = data['basedir']
            class_id = data['class_id']
            instance_name = data['instance_name']
            camera_list = data['camera_list']
            view_list = data['view_list']
            all_param = self.generate_all_params(basedir, class_id, instance_name, camera_list)
            all_param_list.append(all_param)
        return all_param_list

    def generate_all_params(self, basedir, class_id, instance_name, camera_list):
        shape = self.data_shapenet.get_shape_from_instance_name(class_id, instance_name)
        view_list, lighting_list, fname_list, truncparam_list, cropbg_param_list = [], [], [], [], []
        for idx, camera in enumerate(camera_list):
            # set view
            view = self.randomizer.randomize_view(min_dist=MIN_DIST, max_dist=MAX_DIST)
            # set lighting
            lighting = self.randomizer.randomize_lighting()
            # set truncparam and cropbg_param
            truncparam = self.randomizer.randomize_truncparam()
            cropbg_param = self.randomizer.randomize_cropbg_param()
            # set target filename
            fname = os.path.join(class_id, instance_name, '{0}_{1}_{2}.png'.format(class_id, instance_name, idx))
            # append to list
            view_list.append(view)
            lighting_list.append(lighting)
            truncparam_list.append(truncparam)
            cropbg_param_list.append(cropbg_param)
            fname_list.append(fname)
        all_params = AllParams(shape, view_list, lighting_list, truncparam_list, cropbg_param_list, fname_list, resolution=self.resolution) 
        return all_params

def initialize(cfg):
    dataset_sun2012pascal = Sun2012pascal(os.path.join(nowpath, cfg['sun2012pascal_dir']))
    bg_image_list = dataset_sun2012pascal.get_image_list()
    randomizer = Randomizer(os.path.join(nowpath, cfg['path_to_view_file']))
    renderer = Renderer(os.path.join(nowpath, cfg['blender_dir']), num_worker = cfg['num_worker_rendering'])
    cropper = Cropper(num_worker = cfg['num_worker_cropping'])
    compositor = Compositor(num_worker = cfg['num_worker_compositing'])
    return bg_image_list, randomizer, renderer, cropper, compositor

if __name__ == '__main__':
    # SPECIFY YOUR OWN PATH HERE
    choy_dir = os.path.expanduser('~/data/ShapeNetRendering')
    # TODO
    #output_dir = os.path.expanduser('~/data/Ours-ShapeNetV2-25D')
    output_dir = os.path.expanduser('~/data/test')

    # load global config
    with open('cfg_global.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    nowpath = os.path.dirname(os.path.abspath(__file__))
    shapenet_dir = os.path.join(nowpath, cfg['shapenet_dir'])
    blender_dir = os.path.join(nowpath, cfg['blender_dir'])

    bg_image_list, randomizer, renderer, cropper, compositor = initialize(cfg)
    ours_renderer = OursRenderer25D(shapenet_dir, choy_dir, output_dir, randomizer)
    all_params_list = ours_renderer.generate_all_param_list()

    # process
    param_decomposer = ParamDecomposer(output_dir)
    render_param_list, crop_param_list, composite_param_list = param_decomposer.decompose_param(all_params_list)
    renderer.render_all(render_param_list)
    cropper.crop_all(crop_param_list)
    compositor.composite_all(composite_param_list, bg_image_list)

