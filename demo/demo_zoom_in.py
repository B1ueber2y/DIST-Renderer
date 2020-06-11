import numpy as np
import os, sys
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils.render_utils import *
from core.utils.decoder_utils import load_decoder
from core.visualize.visualizer import *
from core.sdfrenderer import SDFRenderer_color as SDFRenderer
from common.geometry import *

def init_info(class_name):
    if class_name == 'sofa':
        experiment_directory = ('deepsdf/experiments/sofas')
        if not os.path.exists(experiment_directory):
            raise ValueError('experiment directory not found.')
        split_file = ('deepsdf/examples/splits/sv2_sofas_train.json')
    elif class_name == 'lamp':
        experiment_directory = ('deepsdf/experiments/lamps')
        if not os.path.exists(experiment_directory):
            raise ValueError('experiment directory not found.')
        split_file = ('deepsdf/examples/splits/sv2_lamps_train.json')
    elif class_name == 'plane':
        experiment_directory = ('deepsdf/experiments/planes')
        if not os.path.exists(experiment_directory):
            raise ValueError('experiment directory not found.')
        split_file = ('deepsdf/examples/splits/sv2_planes_train.json')
    else:
        raise NotImplementedError
    experiment_directory_color = os.path.join(experiment_directory, 'Color')
    return split_file, experiment_directory, experiment_directory_color

def resize_intrinsic(intrinsic, scale):
    intrinsic[:2] = intrinsic[:2] * scale
    return intrinsic

def generate_camera_list(sdf_renderer):
    NUMBER = 100
    MIN_DIST, MAX_DIST = 1.0, 3.0
    camera_list = []
    K = sdf_renderer.get_intrinsic()
    for idx in range(NUMBER + 1):
        azimuth_deg = 120
        dist = MAX_DIST + (MIN_DIST - MAX_DIST) * idx / NUMBER
        view = View(azimuth_deg, 25, 0, dist)
        RT = view.get_extrinsic()
        camera_list.append(Camera(K, RT))
    return camera_list

def main(args):
    # initialize workspace
    split_file, experiment_directory, experiment_directory_color = init_info(args.class_name)
    instance_num = args.instance_id

    # manually set a initial camera intrinsic parameters
    img_h, img_w = 137, 137
    intrinsic = np.array([[150., 0., 68.5], [0., 150., 68.5], [0., 0., 1.]])

    # load codes
    shape_code_fname = os.path.join(experiment_directory, 'LatentCodes', '{}.pth'.format(args.checkpoint))
    shape_code = torch.load(shape_code_fname)['latent_codes'][instance_num].detach().cpu()
    color_code_fname = os.path.join(experiment_directory, 'Color', 'LatentCodes', '{}.pth'.format(args.checkpoint_color))
    color_code = torch.load(color_code_fname)['latent_codes'][instance_num].detach().cpu()
    shape_code = shape_code.float().cuda()
    color_code = color_code.float().cuda()

    # load decoders
    decoder = load_decoder(experiment_directory, args.checkpoint)
    decoder = decoder.module.cuda()

    decoder_color = load_decoder(experiment_directory, args.checkpoint_color, color_size=args.color_size, experiment_directory_color=experiment_directory_color)
    decoder_color = decoder_color.module.cuda()

    # generate camera views
    img_hw = (img_h, img_w)
    print('Image size: {0}.'. format(img_hw))

    # [Note] To render high-quality consistent visualization, we set ray marching ratio back to 1.0 here. If speed up is needed, aggressive strategy can still be used.
    output_shape = args.resolution
    sdf_renderer = SDFRenderer(decoder, decoder_color, resize_intrinsic(intrinsic, output_shape / img_h), march_step=100, buffer_size=1, threshold=args.threshold, ray_marching_ratio=1.0)
    camera_list = generate_camera_list(sdf_renderer)

    # visualization settings
    dist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    prefix = os.path.join(dist_path, args.vis_folder, '{0}_{1}'.format(args.class_name, instance_num), 'vis')

    basedir = os.path.dirname(prefix)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    else:
        import shutil
        shutil.rmtree(basedir)
        os.makedirs(basedir)
    for idx, camera in enumerate(tqdm(camera_list)):
        prefix_idx = prefix + '_{0:04d}'.format(idx)
        demo_color_save_render_output(prefix_idx, sdf_renderer, shape_code, color_code, camera)

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Color training pipeline."
    )
    arg_parser.add_argument('-g', '--gpu', default='0', help='gpu id.')
    arg_parser.add_argument('--color_size', type=int, default=256, help='the size of color code.')
    arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="2000",
        help='The checkpoint weights to use. This can be a number indicated an epoch or "latest" '
        + "for the latest weights (this is the default)",
    )
    arg_parser.add_argument("--checkpoint_color", "-cc", dest="checkpoint_color", default="999",
        help='The checkpoint weights to use. This can be a number indicated an epoch or "latest" '
        + "for the latest weights (this is the default)",
    )
    arg_parser.add_argument('--resolution', type=float, default=1024, help='resolution.')
    arg_parser.add_argument('--profile', action='store_true', help='renderer profiling.')
    arg_parser.add_argument('--class_name', type=str, default='plane', help='class name.')

    arg_parser.add_argument('-i', '--instance_id', type=int, default=257, help='instance id.')
    arg_parser.add_argument('--vis_folder', type=str, default='vis/demo_zoom_in', help='folder for visualization')
    arg_parser.add_argument('--threshold', type=float, default=5e-5, help='rendering threshold')
    args = arg_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    main(args)

