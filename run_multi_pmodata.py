import numpy as np
import os, sys
import cv2
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.dataset import LoaderMultiPMO
from core.utils.render_utils import *
from core.utils.decoder_utils import load_decoder
from core.visualize.visualizer import print_loss_pack_color, Visualizer
from core.visualize.vis_utils import *
from core.evaluation import *
from core.sdfrenderer import SDFRenderer_warp
from core.inv_optimizer import optimize_multi_view

LR = 1e-2
THRESHOLD = 5e-5


class_ID = ['02691156', '03001627', '02958343']
class_type = ['planes', 'chairs', 'cars']

def main():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Color training pipeline."
    )
    arg_parser.add_argument('-g', '--gpu', default='0', help='gpu id.')
    arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="2000",
        help='The checkpoint weights to use. This can be a number indicated an epoch or "latest" '
        + "for the latest weights (this is the default)",
    )
    arg_parser.add_argument('--test_step', '-t', type=int, default=5, help='test step.')
    arg_parser.add_argument('--visualize', action='store_true', help='visualization flag.')
    arg_parser.add_argument('--data_path', default='data/demo_multiview_syn/', help='path to PMO dataset.')
    arg_parser.add_argument('--obj_name', default='cars', help='deepsdf class model for experiments. (support "planes", "chairs", "cars"')
    arg_parser.add_argument('--scale', type=float, default=0.5, help='scale the size of input image, 224x224 -> 112x112.')
    arg_parser.add_argument('--focal', type=float, default=None, help='resize the image and change focal length, try 2')
    arg_parser.add_argument('--full', action='store_true', help='run over all PMO data, otherwise run demo')

    args = arg_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    ################################
    num_sample_points = 10000
    num_views_per_round = 8
    ################################

    # load data
    ind = class_type.index(args.obj_name)
    class_id = class_ID[ind]
    exp_dir = os.path.join('deepsdf/experiments/', args.obj_name)

    upper_loader = LoaderMultiPMO(args.data_path, class_id, scale=args.scale, num_points=num_sample_points, focal=args.focal)
    if args.full:
        total_num_instance = 50 # consider 50 instances in total
        out_dir = os.path.join('vis/multiview_syn/',  args.obj_name)
    else:
        total_num_instance = len(upper_loader) # demo data
        out_dir = os.path.join('vis/demo_multiview_syn/',  args.obj_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cf_dist1_total = 0.0
    cf_dist2_total = 0.0

    for instance_num in range(total_num_instance):
        vis_dir = os.path.join(out_dir, '{}'.format(instance_num))
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        instance_name, imgs, masks, cameras, points_gt = upper_loader[instance_num]

        # RANDOMLY initialize shape code
        latent_size = 256
        std_ = 0.1
        shape_code = torch.ones(1, latent_size).normal_(mean=0, std=std_)
        shape_code = shape_code.float().cuda()
        shape_code.requires_grad = True

        decoder = load_decoder(exp_dir, args.checkpoint)
        decoder = decoder.module.cuda()
        optimizer_latent = torch.optim.Adam([shape_code], lr=LR)

        img_h, img_w = imgs[0].shape[0], imgs[0].shape[1]
        img_hw = (img_h, img_w)
        print('Image size: {0}.'. format(img_hw))
        sdf_renderer = SDFRenderer_warp(decoder, cameras[0].intrinsic, march_step=100, buffer_size=1, threshold=THRESHOLD)
        evaluator = Evaluator(decoder)
        visualizer = Visualizer(img_hw)

        weight_list = {}
        weight_list['color'] = 5.0
        weight_list['l2reg'] = 1.0

        shape_code, optimizer_latent = optimize_multi_view(sdf_renderer, evaluator, shape_code, optimizer_latent, imgs, cameras, weight_list, num_views_per_round=num_views_per_round, num_iters=20, num_sample_points=num_sample_points, visualizer=visualizer, points_gt=points_gt, vis_dir=vis_dir, vis_flag=args.visualize, full_flag=args.full)

        if args.full:
            # final evaluation
            points_tmp = evaluator.latent_vec_to_points(shape_code, num_points=num_sample_points, silent=True)
            dist1, dist2 = evaluator.compute_chamfer_distance(points_gt, points_tmp, separate=True)

            cf_dist1_total += dist1 * 1000
            cf_dist2_total += dist2 * 1000

    if args.full:
        print('Final Average Chamfer Loss: ', cf_dist1_total / total_num_instance, cf_dist2_total / total_num_instance)
    print('Finished. check results {}'.format(out_dir))

if __name__ == '__main__':
    main()


