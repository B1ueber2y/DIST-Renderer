import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pytorch_ssim import loss_ssim

def grid_sample_on_img(img, xy):
    '''
    use xy to grid sample on image using bilinear interpolation.
    Inputs:
    img: 	[B, C, Hin, Win]
    xy : 	[B, 2, Hout, Wout]
    Outputs:
    output: 	[B, C, Hout, Wout]
    '''
    B, C, H, W = img.size()
    vgrid = xy.clone()
    vgrid[:,0,:,:] = 2.0 * vgrid[:,0,:,:].clone() / max(W - 1, 1) - 1.0
    vgrid[:,1,:,:] = 2.0 * vgrid[:,1,:,:].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0,2,3,1)

    output = F.grid_sample(img, vgrid)
    return output

def downsize_img_tensor(img, factor):
    '''
    downsize a tensor (H, W) / (H, W, 3)
    '''
    img_h, img_w = img.shape[0], img.shape[1]
    img_h_new, img_w_new = img_h / factor, img_w / factor
    if (img_h_new - round(img_h_new)) > 1e-12 or (img_w_new - round(img_w_new)) > 1e-12:
        raise ValueError('The image size {0} should be divisible by the factor {1}.'.format((img_h, img_w), factor))
    new_img_shape = (int(img_h_new), int(img_w_new))
    if len(img.shape) == 3:
        img = img.permute(2,0,1)
    elif len(img.shape) == 2:
        img = img[None,:,:]
    else:
        raise NotImplementedError

    if img.type() == 'torch.cuda.ByteTensor':
        is_byte = True
        img = img.float()
    else:
        is_byte = False

    with torch.no_grad():
        img_new = (F.adaptive_avg_pool2d(img, new_img_shape)).data
    if is_byte:
        img_new = img_new.to(torch.uint8)
    if img_new.shape[0] == 1:
        img_new = img_new[0]
    else:
        img_new = img_new.permute(1,2,0)
    return img_new

def compute_loss_mask(min_sdf_sample, valid_mask, valid_mask_gt, threshold=5e-5, visualizer=None, name=['mask_output', 'mask_gt', 'loss_mask_gt', 'loss_mask_out'], handle_first_query_corner_case=False):
    '''
    compute mask loss
    Input:
    - min_sdf_sample:	torch.Tensor (H, W)
    - valid_mask:	torch.Tensor (H, W)
    - valid_mask_gt:	torch.Tensor (H, W)
    '''
    if visualizer is not None:
        visualizer.add_data(name[0], valid_mask.detach().cpu().numpy())
        visualizer.add_data(name[1], valid_mask_gt.detach().cpu().numpy())

    # compute mask loss (gt \ out)
    false_mask_gt = valid_mask_gt & (~(valid_mask & valid_mask_gt))

    if handle_first_query_corner_case:
        # To handle a corner case: first query negative
        mask_first_query_negative = (min_sdf_sample < threshold) & (~valid_mask)
        min_sdf_sample[mask_first_query_negative] = min_sdf_sample[mask_first_query_negative].clone() * (-1) + 2.0 * threshold + 0.1 # this is a bias to help model faster escape from the corner case.

    if torch.nonzero(false_mask_gt).shape[0] != 0:
        min_sdf_sample_gt = min_sdf_sample[false_mask_gt]
        loss_mask_gt = torch.max(min_sdf_sample_gt - threshold, torch.zeros_like(min_sdf_sample_gt))
        if visualizer is not None:
            visualizer.add_data(name[2], loss_mask_gt.detach().cpu().numpy(), false_mask_gt.detach().cpu().numpy())
    else:
        loss_mask_gt = torch.zeros_like(false_mask_gt).float()
        if visualizer is not None:
            visualizer.add_data(name[2], loss_mask_gt.detach().cpu().numpy())
    loss_mask_gt = loss_mask_gt.mean()

    # compute mask loss (out \ gt)
    false_mask_out = valid_mask & (~(valid_mask & valid_mask_gt))
    if torch.nonzero(false_mask_out).shape[0] != 0:
        min_sdf_sample_out = min_sdf_sample[false_mask_out]
        loss_mask_out = torch.max(-min_sdf_sample_out + threshold, torch.zeros_like(min_sdf_sample_out))
        loss_mask_out = loss_mask_out.mean()
        if visualizer is not None:
            visualizer.add_data(name[3], loss_mask_out.detach().cpu().numpy(), false_mask_out.detach().cpu().numpy())
    else:
        loss_mask_out = torch.zeros_like(false_mask_out).float()
        if visualizer is not None:
            visualizer.add_data(name[3], loss_mask_out.detach().cpu().numpy())
    loss_mask_out = loss_mask_out.mean()
    return loss_mask_gt, loss_mask_out, visualizer

def compute_loss_depth(depth_output, valid_mask, depth_gt, valid_mask_gt, visualizer=None):
    '''
    compute loss between depths
    Input:
    - depth_output:	torch.Tensor (H, W)
    - valid_mask:	torch.Tensor (H, W)
    - depth_gt:		torch.Tensor (H, W)
    - valid_mask_gt:	torch.Tensor (H, W)
    '''
    if visualizer is not None:
        visualizer.add_data('depth_output', depth_output.detach().cpu().numpy())
        visualizer.add_data('depth_gt', depth_gt.detach().cpu().numpy())

    # compute depth loss
    valid_depth_mask = (depth_gt > 0) & (depth_gt < 1e5) # handle lidar data (sparse depth).

    valid_mask_overlap = valid_mask & valid_mask_gt
    valid_mask_overlap = valid_mask_overlap & valid_depth_mask
    if torch.nonzero(valid_mask_overlap).shape[0] != 0:
        loss_depth = depth_output[valid_mask_overlap] - depth_gt[valid_mask_overlap]
        if visualizer is not None:
            visualizer.add_data('loss_depth', loss_depth.detach().cpu().numpy(), valid_mask_overlap.detach().cpu().numpy())
    else:
        loss_depth = torch.zeros_like(valid_mask_overlap).float()
        if visualizer is not None:
            visualizer.add_data('loss_depth', loss_depth.detach().cpu().numpy())
    loss_depth = torch.abs(loss_depth).mean()
    return loss_depth, visualizer

def normalize_vectors(x, dim=0):
    norm = torch.norm(x, p=2, dim=dim)[:,None].repeat(1,3)
    eps = 1e-12
    x = x.div(norm + eps)
    return x

def compute_loss_normal(normal_output, valid_mask, normal_gt, valid_mask_gt, visualizer=None):
    '''
    compute loss between normals
    Input:
    - normal_output:	torch.Tensor (H, W, 3)
    - valid_mask:	torch.Tensor (H, W)
    - normal_gt:	torch.Tensor (H, W, 3)
    - valid_mask_gt:	torch.Tensor (H, W)
    '''
    if visualizer is not None:
        visualizer.add_data('normal_output', normal_output.detach().cpu().numpy())
        visualizer.add_data('normal_gt', normal_gt.detach().cpu().numpy())

    # compute normal loss
    normal_length = torch.norm(normal_output, p=2, dim=2)
    valid_normal_mask = (normal_length != 0) # handle depth2normal boundary

    valid_mask_overlap = valid_mask & valid_mask_gt
    valid_mask_overlap = valid_mask_overlap & valid_normal_mask
    if torch.nonzero(valid_mask_overlap).shape[0] != 0:
        normal_output_masked = normal_output[valid_mask_overlap]
        normal_output_masked = normalize_vectors(normal_output_masked, dim=1)
        normal_gt_masked = normal_gt[valid_mask_overlap]
        normal_gt_masked = normalize_vectors(normal_gt_masked, dim=1)
        loss_normal = - (normal_output_masked * normal_gt_masked).sum(1)
        if visualizer is not None:
            visualizer.add_data('loss_normal', loss_normal.detach().cpu().numpy(), valid_mask_overlap.detach().cpu().numpy())
    else:
        loss_normal = torch.zeros_like(valid_mask_overlap).float()
        if visualizer is not None:
            visualizer.add_data('loss_normal', loss_normal.detach().cpu().numpy())
    loss_normal = loss_normal.mean()
    return loss_normal, visualizer

def compute_loss_color(color_output, valid_mask, color_gt, valid_mask_gt, visualizer=None, name=['color_output', 'color_gt', 'loss_color'], use_ssim=False):
    '''
    compute loss between colors
    Input:
    - color_output:    torch.Tensor (H, W, 3)
    - valid_mask:   torch.Tensor (H, W)
    - color_gt:    torch.Tensor (H, W, 3)
    - valid_mask_gt:    torch.Tensor (H, W)
    '''
    if visualizer is not None:
        visualizer.add_data(name[0], color_output.detach().cpu().numpy())
        visualizer.add_data(name[1], color_gt.detach().cpu().numpy())

    # compute rgb loss
    valid_mask_overlap = valid_mask & valid_mask_gt

    if torch.nonzero(valid_mask_overlap).shape[0] != 0:
        loss_color = color_output[valid_mask_overlap] - color_gt[valid_mask_overlap]
        # valid_mask_overlap_repeat = valid_mask_overlap[:, :,None].expand(valid_mask_overlap.shape[0],valid_mask_overlap.shape[1],3)
        loss_color_vis = torch.mean(torch.abs(loss_color), 1)
        if visualizer is not None:
            visualizer.add_data(name[2], loss_color_vis.detach().cpu().numpy(), valid_mask_overlap.detach().cpu().numpy())
    else:
        loss_color_vis = torch.zeros_like(valid_mask_overlap).float()
        if visualizer is not None:
            visualizer.add_data(name[2], loss_color_vis.detach().cpu().numpy())
    loss_color = torch.mean(torch.abs(loss_color))

    if use_ssim:
        color_output[~valid_mask_overlap] = 0.0
        color_gt[~valid_mask_overlap] = 0.0

        loss_color_ssim = loss_ssim(color_gt.permute(2,0,1)[None,:,:,:], color_output.permute(2,0,1)[None,:,:,:])
        loss_color_ssim = loss_color_ssim[0]

        return loss_color, loss_color_ssim, visualizer
    else:
        return loss_color, visualizer

