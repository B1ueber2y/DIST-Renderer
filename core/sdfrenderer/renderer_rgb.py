import os, sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from renderer import SDFRenderer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.utils.decoder_utils import decode_sdf, decode_sdf_gradient, decode_color
from core.visualize.profiler import Profiler
import copy
import time

class SDFRenderer_color(SDFRenderer):
    def __init__(self, decoder, decoder_color, intrinsic, img_hw=None, march_step=50, buffer_size=5, ray_marching_ratio=1.5, max_sample_dist=0.2, threshold=5e-5, use_gpu=True, is_eval=True):
        super(SDFRenderer_color, self).__init__(decoder, intrinsic, img_hw=img_hw, march_step=march_step, buffer_size=buffer_size, ray_marching_ratio=ray_marching_ratio, max_sample_dist=max_sample_dist, threshold=threshold, use_gpu=use_gpu, is_eval=is_eval)

        self.decoder_color = decoder_color
        if is_eval:
            self.decoder_color = self.decoder_color.eval()

    def render_color(self, latent_color, latent, cam_pos, cam_rays, Zdepth, valid_mask, no_grad=False):
        h, w = self.img_hw
        color = torch.zeros_like(self.imgmap_init)[None,:,:].repeat(3, 1, 1) # (H, W, 3)
        color.requires_grad = True

        valid_cam_rays = cam_rays[:, valid_mask]
        valid_zdepth = Zdepth[valid_mask]
        if valid_zdepth.shape[0] == 0:
            return color.reshape(3, -1) # (H*W, 3)

        points = self.generate_point_samples(cam_pos, valid_cam_rays, valid_zdepth, has_zdepth_grad=False)

        color_out = decode_color(self.decoder_color, latent_color, latent, points.transpose(1,0), no_grad=no_grad) # (N, 3)
        color = self.copy_index(color.permute(1,2,0), valid_mask, color_out) # (H, W, 3)
        if no_grad:
            color = color.detach()
        return color

    def compute_shading_maps(self, R, T, lighting_locations, Zdepth, Znormal, valid_mask):
        '''
        computation of multiple shading maps with respect to multiple lighting locations.
        Input:
        - lighting_locations		type torch.Tensor (M, 3)
        Returns:
        - shading_maps			type torch.Tensor (M, H*W)
        '''
        # initialization
        cam_pos = self.get_camera_location(R, T)
        cam_rays = self.get_camera_rays(R)
        valid_cam_rays = cam_rays[:, valid_mask]
        valid_zdepth = Zdepth[valid_mask]
        valid_znormal = Znormal[valid_mask, :]

        # compute directions
        points = self.generate_point_samples(cam_pos, valid_cam_rays, valid_zdepth, inv_transform=False, has_zdepth_grad=False).transpose(1,0)
        directions = (lighting_locations[:,None,:] - points[None,:,:]).permute(0,2,1) # (M, 3, N)
        directions_norm = torch.norm(directions, p=2, dim=1)
        directions = directions / directions_norm[:,None,:].repeat(1,3,1)
        Zdirections = torch.bmm(R.unsqueeze(0), directions).permute(0,2,1) # (M, N, 3)

        # inner product
        valid_shading_maps = Zdirections * valid_znormal[None,:,:].repeat(Zdirections.shape[0], 1, 1) # (M, N, 3)
        valid_shading_maps = valid_shading_maps.sum(2) # (M, N)

        # finalize
        shading_maps = torch.zeros_like(Zdepth.unsqueeze(0).repeat(valid_shading_maps.shape[0], 1))
        shading_maps[:, valid_mask] = valid_shading_maps
        return shading_maps

    def render(self, latent_color, latent, R, T, clamp_dist=0.1, profile=False, no_grad=False, lighting_locations=None, lighting_energies=None):
        '''
        differentiable rendering.
        Input:
        - latent_color     type torch.Tensor (1, latent_size)
        - latent	       type torch.Tensor (1, latent_size)
        - R	               type torch.Tensor (3,3)
        - T		           type torch.Tensor (3)
        Return:
        - color            type torch.Tensor (H, W, 3) - rendered rgb image
        - Zdepth		   type torch.Tensor (H, W) - rendered depth
        - Znormal		   type torch.Tensor (H, W, 3) - rendered normal
        - valid_mask	   type torch.Tensor (H, W) - rendered silhoutte
        - min_sdf_sample   type torch.Tensor (H, W) - minimum_depth_sample
        '''
        profiler = Profiler(silent = not profile)
        profiler.report_process('\nprepare camera time')

        # render depth
        Zdepth, valid_mask, min_sdf_sample = self.render_depth(latent, R, T, clamp_dist=clamp_dist, profile=profile, no_grad=no_grad) # (H*W), (H*W), (H*W)
        profiler.report_process('render depth time')

        # render normal
        normal = self.render_normal(latent, R, T, Zdepth, valid_mask, clamp_dist=clamp_dist, no_grad=no_grad) # (3, H*W)
        Znormal = torch.matmul(R, normal) # (3, H*W)
        Znormal[0,:] = Znormal[0,:].clone() * (-1) # transformed the direction to align with rendering engine (left-hand sided).
        profiler.report_process('render normal time')

        # render rgb image
        cam_pos = self.get_camera_location(R, T)
        cam_rays = self.get_camera_rays(R)
        color = self.render_color(latent_color, latent, cam_pos, cam_rays, Zdepth, valid_mask, no_grad=no_grad)
        profiler.report_process('render color time')

        # finalize and reshape
        h, w = self.img_hw
        depth = torch.ones_like(Zdepth) * 1e11
        depth[valid_mask] = Zdepth[valid_mask].clone() * self.calib_map[valid_mask]
        depth = depth.reshape(h, w)
        Znormal = Znormal.reshape(3, h, w).permute(1,2,0)
        valid_mask = valid_mask.reshape(h, w).type(torch.uint8)
        if len(min_sdf_sample.shape) == 1:
            min_sdf_sample = min_sdf_sample.reshape(h, w)
        profiler.report_process('finalization time')
        profiler.report_all('total time')
        if profile:
            pdb.set_trace()
        if lighting_locations is None:
            return depth, Znormal, color, valid_mask, min_sdf_sample
        else:
            if lighting_energies is None:
                lighting_energies = torch.ones_like(lighting_locations[:, 0])
            shading_maps = self.compute_shading_maps(R, T, lighting_locations, Zdepth.reshape(-1), Znormal.reshape(-1, 3), valid_mask.reshape(-1))
            shading = shading_maps * lighting_energies[:,None].repeat(1, shading_maps.shape[1])
            shading = shading.sum(0).reshape(h, w)
            color = color * shading[:,:,None].repeat(1,1,3)
            return depth, Znormal, color, valid_mask, min_sdf_sample

if __name__ == '__main__':
    pass

