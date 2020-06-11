import os, sys
import torch
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.utils.loss_utils import *

def compute_loss_color_warp(sdf_renderer, shape_code, images, cameras, idx1, idx2, weight_list, sim3=None, sim3_scale=None, visualizer=None):
    camera_1 = cameras[idx1]
    camera_2 = cameras[idx2]
    R1, T1 = camera_1.extrinsic[:,:3], camera_1.extrinsic[:,3]
    R2, T2 = camera_2.extrinsic[:,:3], camera_2.extrinsic[:,3]
    R1, T1 = torch.from_numpy(R1).float().cuda(), torch.from_numpy(T1).float().cuda()
    R2, T2 = torch.from_numpy(R2).float().cuda(), torch.from_numpy(T2).float().cuda()

    if sim3 is not None:
        T1 = torch.matmul(R1, sim3[:, 3]) + T1
        R1 = torch.matmul(R1, sim3[:3, :3])
        R1 = R1 / sim3_scale
        T1 = T1 / sim3_scale
        T2 = torch.matmul(R2, sim3[:, 3]) + T2
        R2 = torch.matmul(R2, sim3[:3, :3])
        R2 = R2 / sim3_scale
        T2 = T2 / sim3_scale

    view1 = images[idx1]
    view2 = images[idx2]

    render_output = sdf_renderer.render_warp(shape_code, R1, T1, R2, T2, view1, view2, no_grad_normal=True)
    loss_color, color_valid_1, color_valid_2, valid_mask1, valid_mask2, min_sdf_sample1, min_sdf_sample2, normal1, depth1_rendered = render_output

    if visualizer is not None:
        visualizer.reset_data()
        visualizer.add_data('color_gt-1', view1.detach().cpu().numpy())
        visualizer.add_data('color_gt-2', view2.detach().cpu().numpy())
        visualizer.add_data('color_valid-1', color_valid_1.detach().cpu().numpy())
        visualizer.add_data('color_valid-2', color_valid_2.detach().cpu().numpy())
        visualizer.add_data('color_valid_loss', torch.abs(color_valid_1-color_valid_2).detach().cpu().numpy())

    loss_l2reg = torch.mean(shape_code.pow(2))

    # total loss
    w_color = weight_list['color']
    w_l2reg = weight_list['l2reg']
    loss = w_color * loss_color + w_l2reg * loss_l2reg

    loss_pack = {}
    loss_pack['color'] = loss_color.detach().cpu().numpy()
    loss_pack['l2reg'] = loss_l2reg.detach().cpu().numpy()
    return loss, loss_pack

