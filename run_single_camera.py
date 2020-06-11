import numpy as np
import os, sys
import cv2
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.dataset import LoaderSingle
from core.utils.decoder_utils import load_decoder
from core.utils.render_utils import *
from core.visualize.vis_utils import *
from core.inv_optimizer import optimize_single_view
from core.evaluation import *
from core.visualize.visualizer import Visualizer
from core.sdfrenderer import SDFRenderer
import pickle

def init_info():
    # TODO
    # set this mesh_data_dir to the path to your NormalizationParameters and SurfaceSamples
    mesh_data_dir = os.path.expanduser('data')
    # mesh_data_dir = os.path.expanduser('~/data')

    basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(basedir, 'data')
    model_dir = os.path.join(basedir, 'deepsdf')
    experiment_directory = os.path.join(model_dir, 'experiments/sofas')
    split_file = os.path.join(model_dir, 'examples/splits/sv2_sofas_test.json')
    synthetic_data_dir = os.path.join(data_dir, 'demo_singleview_syn')
    return mesh_data_dir, experiment_directory, split_file, synthetic_data_dir

def train(args):
    # initialize output_dir
    if not os.path.exists(args.vis_folder):
        os.makedirs(args.vis_folder)

    #########################################################################
    # load data
    #########################################################################
    mesh_data_dir, experiment_directory, split_file, synthetic_data_dir = init_info()
    upper_loader = LoaderSingle(synthetic_data_dir, mesh_data_dir, experiment_directory, split_file)
    shape_md5, image_data, mesh_data, camera, depth = upper_loader[6]
    img, _, normal, _ = image_data
    gt_samples, norm_params = mesh_data

    points_gt = np.array(gt_samples.vertices)
    points_gt = (points_gt + norm_params['offset']) * norm_params['scale']

    gt_pack = {}
    gt_pack['depth'] = torch.from_numpy(depth).cuda()
    gt_pack['normal'] = torch.from_numpy(normal).cuda()
    gt_pack['silhouette'] = torch.from_numpy(depth < 1e5).type(torch.uint8).cuda()

    # # visualize gt
    # cv2.imwrite('img.png', img)
    # visualize_depth('test0.png', depth)
    # with open('camera.pkl', 'wb') as f:
    #     pickle.dump(camera, f)

    #########################################################################
    # prepare tensor
    #########################################################################
    decoder = load_decoder(experiment_directory, args.checkpoint)
    decoder = decoder.module.cuda()
    evaluator = Evaluator(decoder)
    latent_code_dir = os.path.join(synthetic_data_dir, 'latent_codes', '{0}.pth'.format(shape_md5))
    latent_code = torch.load(latent_code_dir)
    latent_tensor = latent_code[0].detach().cpu()
    latent_size = latent_code.shape[-1]
    latent_tensor = latent_tensor.float().cuda()
    latent_tensor.requires_grad = False

    K, RT = camera.intrinsic, camera.extrinsic
    camera_tensor = get_tensor_from_camera(RT)
    std_cam = 1e-1
    if args.use_gt_camera:
        camera_tensor = camera_tensor
    else:
        camera_tensor = camera_tensor + torch.ones(camera_tensor.shape[0]).normal_(mean=0, std=std_cam)
    from torch.autograd import Variable
    camera_tensor = Variable(camera_tensor.cuda(), requires_grad=True)
    optimizer_camera = torch.optim.Adam([camera_tensor], lr=args.lr)

    #########################################################################
    # optimization
    #########################################################################
    weight_dict = {}
    weight_dict['w_depth'] = 10.0
    weight_dict['w_normal'] = 10.0
    weight_dict['w_mask_gt'] = 1.0
    weight_dict['w_mask_out'] = 1.0
    weight_dict['w_l2reg'] = 1.0

    img_h, img_w = img.shape[0], img.shape[1]
    img_hw = (img_h, img_w)
    print('Image size: {0}.'. format(img_hw))
    if args.visualize:
        visualizer = Visualizer(img_hw)
    else:
        visualizer = None

    # initialize renderer
    sdf_renderer = SDFRenderer(decoder, camera.intrinsic, img_hw=img_hw, march_step=100, buffer_size=3, threshold=args.threshold, ray_marching_ratio=args.ratio, use_depth2normal=args.use_depth2normal)
    renderer_list = [sdf_renderer]

    if args.oracle:
        num_iters = 1
    else:
        num_iters = args.num_iters

    # optimization start
    camera_tensor, optimizer_latent = optimize_single_view(renderer_list, evaluator, optimizer_camera, latent_tensor, camera_tensor, gt_pack, weight_dict, optimizer_type="camera", num_iters=num_iters, points_gt=points_gt, test_step=args.test_step, profile=args.profile, visualizer=visualizer, ray_marching_type=args.method, vis_folder=args.vis_folder)

# Main
if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Use differentiable renderer to optimize camera extrinsics from 2D observations."
    )
    arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="2000",
        help='The checkpoint weights to use. This can be a number indicated an epoch or "latest" '
        + "for the latest weights (this is the default)",
    )

    # optimization setting
    arg_parser.add_argument('--gpu', '-g', default='0', help='gpu id.')
    arg_parser.add_argument('--lr', type=float, default=5e-3, help='learning rate.')
    arg_parser.add_argument('--test_step', '-t', type=int, default=2000, help='test step.')
    arg_parser.add_argument('--num_iters', type=int, default=400, help='number of iterations.')
    arg_parser.add_argument('--vis_folder', type=str, default='vis/demo_singleview_camera', help='folder for visualization')
    arg_parser.add_argument('--profile', action='store_true', help='renderer profiling.')
    arg_parser.add_argument('--use_gt_camera', action='store_true', help='gt camera flag.')
    arg_parser.add_argument('--oracle', action='store_true', help='oracle rendering feedforward')

    arg_parser.add_argument('--ratio', type=float, default=1.5, help='test step.')
    arg_parser.add_argument('--method', type=str, default='pyramid_recursive', help='ray marching implementation.')
    arg_parser.add_argument('--visualize', action='store_true', help='visualization flag.')
    arg_parser.add_argument('--threshold', type=float, default=5e-5, help='threshold')
    arg_parser.add_argument('--use_depth2normal', action='store_true', help='use normal converted from depth')
    args = arg_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.oracle:
        args.visualize = True
    train(args)

