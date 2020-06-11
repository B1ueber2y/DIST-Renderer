import os, sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from renderer import SDFRenderer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.utils.decoder_utils import decode_sdf, decode_sdf_gradient, decode_color
from core.visualize.profiler import Profiler

class SDFRenderer_deepsdf(SDFRenderer):
    def __init__(self, decoder, intrinsic, img_hw=None, march_step=50, buffer_size=5, ray_marching_ratio=1.5, max_sample_dist=0.2, threshold=5e-5, use_gpu=True, is_eval=True):
        super(SDFRenderer_deepsdf, self).__init__(decoder, intrinsic, img_hw=img_hw, march_step=march_step, buffer_size=buffer_size, ray_marching_ratio=ray_marching_ratio, max_sample_dist=max_sample_dist, threshold=threshold, use_gpu=use_gpu, is_eval=is_eval)

    def get_samples(self, latent, RT, depth, normal, clamp_dist=0.1, eta=0.01, use_rand=True):
        R, T = RT[:,:3], RT[:,3]
        cam_pos = self.get_camera_location(R, T)
        cam_rays = self.get_camera_rays(R)

        depth = depth.reshape(-1)
        normal = normal.reshape(-1, 3)
        valid_mask = (depth < 1e5) & (depth > 0)

        valid_depth, valid_normal = depth[valid_mask], normal[valid_mask, :]
        valid_zdepth = valid_depth / self.calib_map[valid_mask]

        points = self.generate_point_samples(cam_pos, cam_rays[:, valid_mask], valid_zdepth, has_zdepth_grad=False)
        points = points.transpose(1,0)

        if use_rand:
            eta_map = torch.rand_like(valid_depth) * eta
        else:
            eta_map = torch.ones_like(valid_depth) * eta
        valid_normal_inv = self.inv_transform_points(valid_normal.transpose(1,0)).transpose(1,0)
        offset = valid_normal_inv * eta_map.unsqueeze(-1)

        points_pos = points + offset
        samples_pos = decode_sdf(self.decoder, latent, points_pos, clamp_dist=clamp_dist).squeeze(-1)
        samples_pos = samples_pos - eta_map

        points_neg = points - offset
        samples_neg = decode_sdf(self.decoder, latent, points_neg, clamp_dist=clamp_dist).squeeze(-1)
        samples_neg = samples_neg + eta_map
        return samples_pos, samples_neg

    def get_freespace_samples(self, latent, RT, depth, clamp_dist=0.1, number=1):
        R, T = RT[:,:3], RT[:,3]
        cam_pos = self.get_camera_location(R, T)
        cam_rays = self.get_camera_rays(R)

        depth = depth.reshape(-1)
        valid_mask = (depth < 1e5) & (depth > 0)
        valid_depth = depth[valid_mask]
        valid_zdepth = valid_depth / self.calib_map[valid_mask]

        samples = []
        for idx in range(number):
            ratio_sample = torch.rand_like(valid_zdepth) * 1.0
            input_zdepth = valid_zdepth * ratio_sample
            points = self.generate_point_samples(cam_pos, cam_rays[:, valid_mask], input_zdepth, has_zdepth_grad=False)
            points = points.transpose(1,0)
            sample = decode_sdf(self.decoder, latent, points, clamp_dist=clamp_dist).squeeze(-1)
            samples.append(sample)
        samples = torch.cat(samples, 0)
        return samples

if __name__ == '__main__':
    pass

