import os, sys
import torch
from tqdm import trange
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loss_multi import compute_loss_color_warp
from core.utils.train_utils import params_to_mtrx

'''
This function performs inverse optimization over either shape code (optionally with a similarity transformations) on posed multi-view images.

Required parameters:
- sdfrenderer:          A differentiable sdf renderer,
- evaluator:            An instance of class <"Evaluator">
- shape_code:           class <torch.Tensor>
- shape_optimizer:      Ann instance of PyTorch optimizer, either for shape code or camera extrinsics, which is optional for the key "optimizer_type".
- images                List of image tensors, each has type class <torch.Tensor>.
- cameras               List of cameras, each has type class <common.geometry.Camera>.
- weight_dict:          A dictionary of weights. ['color', 'l2reg']

Optional parameters:
- num_views_per_round:  number of used views for each round.
- num_iters:            number of iterations.
- num_sample_points:    number of sampled points (10k for PMO evaluation)
- sep_dist:             sampling strategy (set sep_dist > 1 to skip frames and improve efficientcy).
- test_step:            evaluation interval.
- points_gt:            groundtruth shapes for evaluation.
- sim3:                 input similarity transform. If not None, the shape code is jointly optimized with sim3.
- full_flag:            To evaluate and update the best model.

Returns:
The optimized shape_code and the optimizer are returned.
'''

def optimize_multi_view(renderer, evaluator, shape_code, shape_optimizer, images, cameras, weight_list,
                        num_views_per_round=8, num_iters=20, num_sample_points=30000, sep_dist=1, test_step=5, points_gt=None, sim3=None, sim3_init=None, visualizer=None, vis_dir=None, vis_flag=None,  full_flag=True):
    counter = 0
    best_chamfer = 100
    best_epoch = 0

    # save initialized shape
    fname = os.path.join(vis_dir, 'mesh_initial.ply')
    evaluator.latent_vec_to_points(shape_code, num_points=num_sample_points, fname=fname, silent=True)

    # optimization start
    num_images = len(images)
    rot_freq = num_images / num_views_per_round
    for epoch in trange(num_iters):
        for idx in range(0, int(np.ceil(rot_freq)), sep_dist):
            loss_total = 0.0
            shape_optimizer.zero_grad()

            if sim3 is not None:
                sim_mtrx = params_to_mtrx(sim3).clone()
                sim_mtrx[:, 3] = torch.matmul(sim_mtrx[:3, :3].clone(), sim3_init[:, 3]) + sim_mtrx[:, 3].clone()
                sim_mtrx[:3, :3] = torch.matmul(sim_mtrx[:3, :3].clone(), sim3_init[:3, :3])
                sim3_scale = torch.norm(sim_mtrx[:3, :3].clone())/np.sqrt(3)
            else:
                sim_mtrx = None
                sim3_scale = None

            for i in range(0, num_views_per_round):
                if visualizer is not None:
                    visualizer.reset_data()
                idx1, idx2 = idx + int(np.floor(i * rot_freq)), idx + int(np.floor(i * rot_freq)) + sep_dist
                if idx2 >= num_images:
                    idx1 = num_images - 1
                    idx2 = idx1 - sep_dist
                loss, loss_pack = compute_loss_color_warp(renderer,
                                        shape_code,
                                        images,
                                        cameras,
                                        idx1, idx2,
                                        weight_list,
                                        sim3=sim_mtrx, sim3_scale=sim3_scale,
                                        visualizer=visualizer)
                loss_total += loss

            counter += 1
            loss_total.backward()
            shape_optimizer.step()
            loss_total = loss_total.detach()

        if vis_flag:
            loss_color = loss_pack['color']
            loss_l2reg = loss_pack['l2reg']
            print('[{0}] loss_color: {1:.4f}, loss_l2reg: {2:.4f}\n'.format(epoch, loss_color, loss_l2reg))
            visualizer.show_all_data_color_warp(os.path.join(vis_dir, 'vis_{}.png'.format(epoch)))

        if (epoch % test_step == 0):
            fname = os.path.join(vis_dir, 'mesh_{}.ply'.format(epoch))
            points_pred = evaluator.latent_vec_to_points(shape_code, num_points=num_sample_points, fname=fname, silent=True)

            if points_pred is None:
                print('The current latent code does not correspond to a valid shape.')
            elif (points_gt is not None) and full_flag: # evaluation and update the best model.
                dist1, dist2 = evaluator.compute_chamfer_distance(points_gt, points_pred, separate=True)
                dist = dist1 + dist2
                if dist * 1000 < best_chamfer:
                    best_chamfer = dist * 1000
                    best_epoch = epoch
                    fname = os.path.join(vis_dir, 'mesh_best.ply')
                    evaluator.latent_vec_to_points(shape_code, num_points=num_sample_points, fname=fname, silent=True)

                print('CHAMFER DISTANCE: {0:.3f} & {1:.3f} at epoch {2}'.format(dist1 * 1000, dist2 * 1000, epoch))
                print('BEST SUM CHAMFER DISTANCE: {0:.3f} at epoch {1}'.format(best_chamfer, best_epoch))
    return shape_code, shape_optimizer



