'''
This is the main program for synthesis pipeline.

e.g. python run.py -p testcase1
'''
import os, sys
import yaml
import shutil
nowpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(nowpath))
from common.dataset import ShapeNetV2, Sun2012pascal
sys.path.append(nowpath)
from src.randomize.randomizer import Randomizer
from src.tools.render import Renderer
from src.tools.crop import Cropper
from src.tools.composite import Compositor
from src.param_decomposer import AllParams, ParamDecomposer

def initialize(cfg):
    nowpath = os.path.dirname(os.path.abspath(__file__))
    dataset_shapenet = ShapeNetV2(os.path.join(nowpath, cfg['shapenet_dir']))
    dataset_sun2012pascal = Sun2012pascal(os.path.join(nowpath, cfg['sun2012pascal_dir']))
    bg_image_list = dataset_sun2012pascal.get_image_list()

    randomizer = Randomizer(os.path.join(nowpath, cfg['path_to_view_file']))
    renderer = Renderer(os.path.join(nowpath, cfg['blender_dir']), num_worker = cfg['num_worker_rendering'])
    cropper = Cropper(num_worker = cfg['num_worker_cropping'])
    compositor = Compositor(num_worker = cfg['num_worker_compositing'])
    return dataset_shapenet, bg_image_list, randomizer, renderer, cropper, compositor

if __name__ == '__main__':
    nowpath = os.path.dirname(os.path.abspath(__file__))
    import argparse
    parser = argparse.ArgumentParser(description="Synthesis.")
    parser.add_argument('-p', '--param_filename', type=str, required=True, help='choose the file which has the function generate_params(shape_list, randomizer).')
    parser.add_argument('-o', '--folder', type=str, default='default', help='output folder')
    parser.add_argument('-n', '--class_name', type=str, default='sofa', help='class name')
    parser.add_argument('--mode', type=str, default='train', help='mode')
    args = parser.parse_args()
    sys.path.append(os.path.join(nowpath, 'paramGen'))
    paramGen_module = __import__(args.param_filename)

    with open('cfg_global.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    dataset_shapenet, bg_image_list, randomizer, renderer, cropper, compositor = initialize(cfg)

    if args.mode != 'train' and args.mode != 'test':
        raise ValueError('Mode {} should be train/test.'.format(args.mode))
    CLASS_NAME = args.class_name
    shape_list = dataset_shapenet.get_shape_list_from_name(CLASS_NAME, use_split_file=True, mode=args.mode)

    # generate all parameters
    folder, all_params_list = paramGen_module.generate_params(shape_list, randomizer)
    if args.folder != 'default':
        folder = os.path.join(nowpath, 'output', '{}'.format(args.folder))
    folder = folder + '_{0}_{1}'.format(args.class_name, args.mode)
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copyfile(os.path.join(nowpath, 'paramGen', args.param_filename + '.py'), os.path.join(folder, args.param_filename + '.py'))

    # process
    param_decomposer = ParamDecomposer(folder)
    render_param_list, crop_param_list, composite_param_list = param_decomposer.decompose_param(all_params_list)
    renderer.render_all(render_param_list)
    cropper.crop_all(crop_param_list)
    compositor.composite_all(composite_param_list, bg_image_list)

