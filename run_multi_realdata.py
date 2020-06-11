import numpy as np
import os, sys
import cv2
import torch
from tqdm import tqdm
import easydict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.dataset import LoaderMultiReal
from core.utils.render_utils import *
from core.utils.decoder_utils import load_decoder
from core.utils.train_utils import params_to_mtrx
from core.visualize.visualizer import print_loss_pack_color, Visualizer
from core.visualize.vis_utils import *
from core.evaluation import *
from core.sdfrenderer import SDFRenderer_warp
from core.inv_optimizer import optimize_multi_view

LR = 1e-2
THRESHOLD = 5e-5

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
    arg_parser.add_argument('--test_step', '-t', type=int, default=1, help='test step.')
    arg_parser.add_argument('--data_path', default='data/demo_multiview_real/', help='path to the dataset.') #TODO need to change the default
    arg_parser.add_argument('--obj_name', default='chairs', help='deepsdf class model for experiments. (currently only support "chairs" ')
    arg_parser.add_argument('--visualize', action='store_true', help='visualization flag.')
    arg_parser.add_argument('--scale', type=float, default=0.2, help='scale the size of input image.')
    

    args = arg_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    ################################
    num_views_per_round = 6
    num_epoch = 5
    sep_dist = 1
    refine_sim = True
    ini_mean_shape = False
    ################################

    # load data
    exp_dir = os.path.join('deepsdf/experiments/', args.obj_name)
    out_dir = 'vis/demo_multiview_real'

    
    upper_loader = LoaderMultiReal(args.data_path, scale=args.scale, refine_sim=refine_sim)

    for instance_num in range(len(upper_loader)):
        # load data
        instance_name, imgs, _, cameras, sim_mtrx_ini = upper_loader[instance_num]

        vis_dir = os.path.join(out_dir, '{}'.format(instance_num))
        os.makedirs(vis_dir, exist_ok=True)
        total_num_img = len(imgs)

        # RANDOMLY initialize shape code
        latent_size = 256
        std_ = 0.1
        if ini_mean_shape:
            shape_code = torch.zeros(1, latent_size)
        else:
            shape_code = torch.ones(1, latent_size).normal_(mean=0, std=std_)

        shape_code = shape_code.float().cuda()
        shape_code.requires_grad = True

        if refine_sim:
            sim3 = easydict.EasyDict({
                "scale": torch.tensor(0.,requires_grad=True, device="cuda"),
                "rot": torch.tensor([0.,0.,0.],requires_grad=True, device="cuda"),
                "trans": torch.tensor([0.,0.,0.],requires_grad=True, device="cuda"),
            })
            optim_list = [{ "params": [v for k,v in sim3.items()], "lr": LR },
                          { "params": [shape_code], "lr": LR }]
            optimizer_latent = torch.optim.Adam(optim_list)
        else:
            optimizer_latent = torch.optim.Adam([shape_code] , lr=LR)
            sim3 = None

        decoder = load_decoder(exp_dir, args.checkpoint)
        decoder = decoder.module.cuda()
        img_h, img_w = imgs[0].shape[0], imgs[0].shape[1]
        img_hw = (img_h, img_w)
        print('Image size: {0}.'. format(img_hw))
        sdf_renderer = SDFRenderer_warp(decoder, cameras[0].intrinsic, march_step=200, buffer_size=1, threshold=THRESHOLD, transform_matrix=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
        evaluator = Evaluator(decoder)
        visualizer = Visualizer(img_hw)

        weight_list = {}
        weight_list['color'] = 5.0
        weight_list['l2reg'] = 1.0

        shape_code, optimizer_latent = optimize_multi_view(sdf_renderer, evaluator, shape_code, optimizer_latent, imgs, cameras, weight_list, num_views_per_round=num_views_per_round, num_iters=num_epoch, sep_dist=sep_dist, test_step=args.test_step, visualizer=visualizer, points_gt=None, sim3=sim3, sim3_init=sim_mtrx_ini, vis_flag=args.visualize, vis_dir=vis_dir)
    print('Finished. check results {}'.format(out_dir))

if __name__ == '__main__':
    main()

