import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import namedtuple
from src.mystruct import FnameGroup
from src.mystruct import RenderParam, CropParam, CompositeParam
import random

# struct render param
'''
struct render param:
- shape			type: Shape
- view_cfg		type: list of View
- light_cfg		type: list of Lighting
- target_cfg		type: list of FnameGroup
- truncparam_cfg	type: list of truncparam ([t1, t2, t3, t4])
- cropbg_param_cfg	type: list of crpbg_param ([c1, c2])
- fname_cfg		type: list of fname (png) without prefix
- resolution		type: tuple (reso_y, reso_x)
'''
AllParams = namedtuple("AllParams", "shape view_cfg light_cfg truncparam_cfg cropbg_param_cfg fname_cfg resolution")

# overwrite all params
class AllParams(object):
    def __init__(self, shape, view_cfg, light_cfg, truncparam_cfg, cropbg_param_cfg, fname_cfg, resolution=None):
        self.shape = shape
        self.view_cfg = view_cfg
        self.light_cfg = light_cfg
        self.truncparam_cfg = truncparam_cfg
        self.cropbg_param_cfg = cropbg_param_cfg
        self.fname_cfg = fname_cfg
        self.resolution = resolution

class ParamDecomposer(object):
    def __init__(self, folder, render_dir='rendering', crop_dir='cropping', final_dir='final'):
        self.folder = folder
        self.render_dir = os.path.join(folder, render_dir)
        if not os.path.exists(self.render_dir): os.makedirs(self.render_dir)
        self.crop_dir = os.path.join(folder, crop_dir)
        if not os.path.exists(self.crop_dir): os.makedirs(self.crop_dir)
        self.final_dir = os.path.join(folder, final_dir)
        if not os.path.exists(self.final_dir): os.makedirs(self.final_dir)

    def decompose_param(self, all_params_list):
        render_param_list, crop_param_list, composite_param_list = [], [], []
        from tqdm import tqdm
        print('processing params...')
        for all_param in tqdm(all_params_list):
            # parse render_param
            shape = all_param.shape
            view_cfg = all_param.view_cfg
            light_cfg = all_param.light_cfg
            fname_cfg = all_param.fname_cfg
            render_target_cfg = [FnameGroup(os.path.join(self.render_dir, fname)) for fname in fname_cfg]
            resolution = all_param.resolution
            if resolution == None:
                render_param = RenderParam(shape, view_cfg, light_cfg, render_target_cfg)
            else:
                render_param = RenderParam(shape, view_cfg, light_cfg, render_target_cfg, resolution=resolution)
            render_param_list.append(render_param)

            # parse crop_param
            truncparam_cfg = all_param.truncparam_cfg
            for fname, truncparam in zip(fname_cfg, truncparam_cfg):
                input_cfg = FnameGroup(os.path.join(self.render_dir, fname))
                output_cfg = FnameGroup(os.path.join(self.crop_dir, fname))
                if not os.path.exists(os.path.join(self.crop_dir, os.path.dirname(fname))):
                    os.makedirs(os.path.join(self.crop_dir, os.path.dirname(fname)))
                crop_param = CropParam(truncparam, input_cfg, output_cfg)
                crop_param_list.append(crop_param)

            # parse composite_param
            cropbg_param_cfg = all_param.cropbg_param_cfg
            for fname, cropbg_param in zip(fname_cfg, cropbg_param_cfg):
                input_cfg = FnameGroup(os.path.join(self.crop_dir, fname))
                output_cfg = FnameGroup(os.path.join(self.final_dir, fname))
                if not os.path.exists(os.path.join(self.final_dir, os.path.dirname(fname))):
                    os.makedirs(os.path.join(self.final_dir, os.path.dirname(fname)))
                composite_param = CompositeParam(cropbg_param, input_cfg, output_cfg)
                composite_param_list.append(composite_param)

        random.shuffle(crop_param_list)
        random.shuffle(composite_param_list)
        return render_param_list, crop_param_list, composite_param_list


