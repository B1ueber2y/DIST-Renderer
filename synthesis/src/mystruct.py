import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.geometry import *
from common.utils.camera_utils import *
from collections import namedtuple

# struct output target
'''
absolute path for a group of image/depth/normal/albedo and the corresponding rendering parameters.
'''
FnameGroup = namedtuple("FnameGroup", "image depth normal albedo param")

# struct render param
'''
struct render param:
- shape		type: Shape
- view_cfg	type: list of View
- light_cfg	type: list of Lighting
- target_cfg	type: list of FnameGroup
- resolution	type: tuple (reso_y, reso_x)
'''
RenderParam = namedtuple("RenderParam", "shape view_cfg light_cfg target_cfg resolution")

# struct crop param
'''
struct crop param
- truncparam	type: list of float (sized 4)
- input_cfg	type: FnameGroup
- target_cfg	type: FnameGroup
- is_crop	type: bool (whether to crop)
'''
CropParam = namedtuple("CropParam", "truncparam input_cfg target_cfg is_crop")

# struct composite param
'''
struct composite param
- cropbg_param	type: list of float (sized 2)
- input_cfg	type: FnameGroup
- target_cfg	type: FnameGroup
'''
CompositeParam = namedtuple("CompositeParam", "cropbg_param input_cfg target_cfg")

# overwrite FnameGroup with efficient initialization
class FnameGroup(object):
    def __init__(self, image, depth=-1, normal=-1, albedo=-1, param=-1):
        self.image = image
        if depth == -1:
            self.init_with_image_name(image)
        else:
            if not (type(depth) == str and type(normal) == str and type(albedo) == str and type(param) == str):
                raise AssertionError('the number of arguments should be exactly 1 or 5.')
            self.depth = depth
            self.normal = normal
            self.albedo = albedo
            self.param = param

    def init_with_image_name(self, image):
        self.depth = image[:-4] + '_depth.exr'
        self.normal = image[:-4] + '_normal.exr'
        self.albedo = image[:-4] + '_albedo.exr'
        self.param = image[:-4] + '_param.pkl'

# overwrite render param
class RenderParam(object):
    def __init__(self, shape, view_cfg, light_cfg, target_cfg, resolution=(224, 224)):
        self.shape = shape
        self.view_cfg = view_cfg
        self.light_cfg = light_cfg
        self.target_cfg = target_cfg
        self.resolution = resolution

# overwrite crop param
class CropParam(object):
    def __init__(self, truncparam, input_cfg, target_cfg, is_crop=False):
        self.truncparam = truncparam
        self.input_cfg = input_cfg
        self.target_cfg = target_cfg
        self.is_crop = is_crop

# overwrite composite param
class CompositeParam(object):
    def __init__(self, cropbg_param, input_cfg, target_cfg):
        self.cropbg_param = cropbg_param
        self.input_cfg = input_cfg
        self.target_cfg = target_cfg

        MAX_NUM = 1e6
        self.seed = self.generate_seed(MAX_NUM)

    def generate_seed(self, MAX_NUM):
        seed = np.random.randint(MAX_NUM)
        return seed


