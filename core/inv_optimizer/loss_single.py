import os, sys
import torch
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.utils.loss_utils import *
import copy

def compute_all_loss(sdf_renderer, latent_tensor, extrinsic, gt_pack, threshold=5e-5, profile=False, visualizer=None, ray_marching_type='pyramid_recursive', grad_settings=None):
    # update the task to compute
    if grad_settings is None:
        grad_settings = {"depth": True, "normal": True, "silhouette": True}
    for key in list(grad_settings.keys()):
        if key in gt_pack.keys():
            grad_settings[key] = grad_settings[key] and (gt_pack[key] is not None)
        else:
            grad_settings[key] = False

    render_output = sdf_renderer.render(latent_tensor, extrinsic[:,:3], extrinsic[:,3], profile=profile, sample_index_type='min_abs', ray_marching_type=ray_marching_type, no_grad_depth=(not grad_settings["depth"]), no_grad_normal=(not grad_settings["normal"]))
    depth_rendered, normal_rendered, valid_mask_rendered, min_sdf_sample = render_output

    ratio = gt_pack[list(gt_pack.keys())[0]].shape[0] / depth_rendered.shape[0]
    gt_pack = copy.deepcopy(gt_pack)
    for key in gt_pack.keys():
        gt_pack[key] = downsize_img_tensor(gt_pack[key], ratio)

    loss_pack = {}
    if visualizer is not None:
        visualizer.reset_data()

    # compute silhouette loss
    if "silhouette" in gt_pack.keys():
        loss_pack['mask_gt'], loss_pack['mask_out'], visualizer = compute_loss_mask(min_sdf_sample, valid_mask_rendered, gt_pack["silhouette"], threshold=threshold, visualizer=visualizer)
        if not grad_settings['silhouette']:
            loss_pack['mask_gt'], loss_pack['mask_out'] = loss_pack['mask_gt'].detach(), loss_pack['mask_out'].detach()
    else:
        loss_pack['mask_gt'], loss_pack['mask_out'] = 0.0, 0.0

    # compute depth loss
    if "depth" in gt_pack.keys():
        loss_pack['depth'], visualizer = compute_loss_depth(depth_rendered, valid_mask_rendered, gt_pack["depth"], gt_pack["silhouette"], visualizer=visualizer)
        if not grad_settings['depth']:
            loss_pack['depth'] = loss_pack['depth'].detach()
    else:
        loss_pack['depth'] = 0.0

    # compute normal loss
    if "normal" in gt_pack.keys():
        loss_pack['normal'], visualizer = compute_loss_normal(normal_rendered, valid_mask_rendered, gt_pack["normal"], gt_pack["silhouette"], visualizer=visualizer)
        if not grad_settings['normal']:
            loss_pack['normal'] = loss_pack['normal'].detach()
    else:
        loss_pack['normal'] = 0.0

    # compute l2reg loss
    loss_pack['l2reg'] = torch.mean(latent_tensor.pow(2))
    if visualizer is not None:
        visualizer.add_loss_from_pack(loss_pack)
    return loss_pack, visualizer


