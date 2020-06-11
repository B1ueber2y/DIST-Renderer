import os, sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.utils.decoder_utils import decode_sdf, decode_sdf_gradient
from core.visualize.profiler import Profiler
from core.utils.render_utils import depth2normal
import copy
import time

class SDFRenderer(object):
    def __init__(self, decoder, intrinsic, img_hw=None, transform_matrix=None, march_step=50, buffer_size=5, ray_marching_ratio=1.5, use_depth2normal=False, max_sample_dist=0.2, radius=1.0, threshold=5e-5, scale_list=[4, 2, 1], march_step_list=[3, 3, -1], use_gpu=True, is_eval=True):
        self.decoder = decoder
        self.device = next(self.decoder.parameters()).get_device()
        if is_eval:
            self.decoder.eval()
        self.march_step = march_step
        self.buffer_size = buffer_size
        self.max_sample_dist = max_sample_dist
        self.ray_marching_ratio = ray_marching_ratio
        self.use_depth2normal=use_depth2normal
        self.radius = radius
        self.threshold = threshold
        self.scale_list = scale_list
        self.march_step_list = march_step_list
        if type(intrinsic) == torch.Tensor:
            intrinsic = intrinsic.detach().cpu().numpy()
        self.intrinsic = intrinsic

        if img_hw is None:
            img_h, img_w = int(intrinsic[1,2] * 2), int(intrinsic[0,2] * 2)
            self.img_hw = (img_h, img_w)
        else:
            self.img_hw = img_hw

        self.homo_2d = self.init_grid_homo_2d(self.img_hw)
        self.K, self.K_inv = self.init_intrinsic(intrinsic)
        self.homo_calib = torch.matmul(self.K_inv, self.homo_2d) # (3, H*W)
        self.homo_calib.requires_grad=False
        self.imgmap_init = self.init_imgmap(self.img_hw)
        self.imgmap_init.requires_grad=False

        if transform_matrix is None:
            self.transform_matrix = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        else:
            self.transform_matrix = transform_matrix
        self.transform_matrix = torch.from_numpy(self.transform_matrix).float()

        if use_gpu:
            if torch.cuda.device_count() == 0:
                raise ValueError('No GPU device found.')
            self.homo_2d = self.homo_2d.to(self.device)
            self.homo_calib = self.homo_calib.to(self.device) # (3, H*W)
            self.imgmap_init = self.imgmap_init.to(self.device) # (H*W)
            self.transform_matrix = self.transform_matrix.to(self.device) # (3,3)
            self.K, self.K_inv = self.K.to(self.device), self.K_inv.to(self.device)

        self.calib_map = self.normalize_vectors(self.homo_calib)[2,:]

    def get_intrinsic(self):
        return self.intrinsic

    def get_threshold(self):
        return self.threshold

    def get_img_hw(self):
        return self.img_hw

    def visualize_calib_map(self, fname='calib_map_vis.png'):
        import cv2
        data = self.calib_map.detach().cpu().numpy()
        min_, max_ = data.min(), data.max()
        data = (data - min_) / (max_ - min_)
        data = (data * 255.0).reshape(self.img_hw[0], self.img_hw[1]).astype(np.uint8)
        cv2.imwrite(fname, data)

    def apply_3Dsim(self, points,sim3_mtrx,inv=False):
        sR,t = sim3_mtrx[:,:3],sim3_mtrx[:,3]
        points = (points-t)@sR.inverse().t() if inv else \
                 points@sR.t()+t
        return points

    def transform_points(self, points):
        '''
        transformation for point coordinates.
        Input:
        - points	type: torch.Tensor (3, H*W)
        Return:
        - points_new	type: torch.Tensor (3, H*W)
        '''
        if self.transform_matrix.shape[1] == 4:
            # sR, t = self.transform_matrix[:,:3], self.transform_matrix[:,3]
            # points_new = sR @ points + t[:, None]
            points_new = self.apply_3Dsim(points.t(), self.transform_matrix).t()
        else:
            points_new = torch.matmul(self.transform_matrix, points)
        return points_new

    def inv_transform_points(self, points):
        '''
        inverse transformation for point coordinates.
        Input:
        - points	type: torch.Tensor (3, H*W)
        Return:
        - points_new	type: torch.Tensor (3, H*W)
        '''
        if self.transform_matrix.shape[1] == 4:
            # sR, t = self.transform_matrix[:,:3], self.transform_matrix[:,3]
            # points_new = sR.inverse() @ (points-t[:, None])
            #pdb.set_trace()
            #points = np.array([0.419, 1.837, 2.495])
            #points = torch.from_numpy(points)
            #points = points[:, None]

            pdb.set_trace()
            points_new = self.apply_3Dsim(points.t(), self.transform_matrix, inv=True).t()
        else:
            points_new = torch.matmul(self.transform_matrix.transpose(1,0), points)
        return points_new

    def get_meshgrid(self, img_hw):
        '''
        To get meshgrid:
        Input:
        - img_hw	(h, w)
        Return:
        - grid_map	type: torch.Tensor (H, W, 2)
        '''
        h, w = img_hw
        Y, X = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_map = torch.cat([X[:,:,None], Y[:,:,None]], 2) # (h, w, 2)
        grid_map = grid_map.float()
        return grid_map

    def get_homo_2d_from_xy(self, xy):
        '''
        get homo 2d from xy
        Input:
        - xy		type: torch.Tensor (H, W, 2)
        Return:
        - homo		type: torch.Tensor (H, W, 3)
        '''
        H, W = xy.shape[0], xy.shape[1]
        homo_ones = torch.ones(H, W, 1)
        if xy.get_device() != -1:
            homo_ones = homo_ones.to(xy.get_device())
        homo_2d = torch.cat([xy, homo_ones], 2)
        return homo_2d

    def get_homo_2d(self, img_hw):
        xy = self.get_meshgrid(img_hw)
        homo_2d = self.get_homo_2d_from_xy(xy)
        return homo_2d

    def init_grid_homo_2d(self, img_hw):
        homo_2d = self.get_homo_2d(img_hw)
        homo_2d = homo_2d.reshape(-1, 3).transpose(1,0) # (3, H*W)
        return homo_2d

    def init_intrinsic(self, intrinsic):
        K = torch.from_numpy(intrinsic).float()
        K_inv = torch.from_numpy(np.linalg.inv(intrinsic)).float()
        return K, K_inv

    def init_imgmap(self, img_hw):
        h, w = img_hw
        imgmap_init = torch.zeros(h, w)
        return imgmap_init

    def normalize_vectors(self, x):
        '''
        normalize the vector by the first dim
        '''
        norm = torch.norm(x, p=2, dim=0).expand_as(x)
        eps = 1e-12
        x = x.div(norm + eps)
        return x

    def get_camera_location(self, R, T):
        '''
        Input:
        - R	type: torch.Tensor (3,3)
        - T	type: torch.Tensor (3)
        '''
        pos = torch.matmul(-R.transpose(1,0), T[:,None]) # (3,1)
        pos = pos.squeeze(1) # (3)
        return pos

    def get_camera_rays(self, R, homo=None):
        '''
        Input:
        - R	type: torch.Tensor (3,3)
        - T	type: torch.Tensor (3)
        '''
        if homo is None:
            homo = self.homo_calib
        rays = torch.matmul(R.transpose(1,0), homo) # (3, H*W)
        rays = self.normalize_vectors(rays) # (3, H*W)
        return rays

    def generate_point_samples(self, cam_pos, cam_rays, Zdepth, inv_transform=True, has_zdepth_grad=False):
        '''
        Input:
        - cam_pos	type torch.Tensor (3)
        - cam_ays	type torch.Tensor (3, N)
        - Zdepth	type torch.Tensor (N)
        Return:
        - points	type torch.Tensor (3, N)
        '''
        if not has_zdepth_grad:
            Zdepth = Zdepth.detach()
        N = Zdepth.shape[0]
        if N == 0:
            raise ValueError('No valid depth.')
        cam_pos_pad = cam_pos[:,None].repeat(1,N) # (3, N)
        Zdepth_pad = Zdepth[None,:].repeat(3,1) # (3, N)
        points = cam_rays * Zdepth_pad + cam_pos_pad # (3, N)
        if inv_transform:
            points = self.inv_transform_points(points)
        if not points.requires_grad:
            points.requires_grad=True
        return points

    def get_distance_from_origin(self, cam_pos, cam_rays):
        '''
        get_distance_from_origin
        Input:
        - cam_pos	type torch.FloatTensor (3)
        - cam_rays	type torch.FloatTensor (3, H*W)
        '''
        N = cam_rays.shape[1]
        cam_pos_pad = cam_pos[:,None].expand_as(cam_rays) # (3, N)

        p, q = cam_pos_pad, cam_rays # (3, N), (3, N)
        ptq = (p * q).sum(0) # (N)
        dist = p - ptq[None,:].repeat(3,1) * q # (3, N)
        dist = torch.norm(dist, p=2, dim=0) # (N)
        return dist

    def get_maxbound_zdepth_from_dist(self, dist):
        '''
        Input:
        - dist		type torch.FloatTensor (N)
        '''
        with torch.no_grad():
            value = self.radius ** 2 - dist ** 2
            valid_mask = (value >= 0)

            maxbound_zdepth = torch.zeros_like(dist)
            maxbound_zdepth[valid_mask] = 2 * torch.sqrt(value[valid_mask])
        return maxbound_zdepth

    def get_intersections_with_unit_spheres(self, cam_pos, cam_rays):
        '''
        get_intersections_with_unit_sphere
        Input:
        - cam_pos	type torch.FloatTensor (3)
        - cam_rays	type torch.FloatTensor (3, H*W)
        '''
        with torch.no_grad():
            dist = self.get_distance_from_origin(cam_pos, cam_rays)
            valid_mask = (dist <= self.radius)
            maxbound_marching_zdepth = self.get_maxbound_zdepth_from_dist(dist) # (H*W)

            cam_pos_dist = torch.sqrt((cam_pos ** 2).sum())
            if torch.nonzero((cam_pos_dist < self.radius).unsqueeze(0)).shape[0] != 0:
                init_zdepth = torch.zeros_like(dist)
            else:
                init_zdepth_valid = torch.sqrt(cam_pos_dist ** 2 - dist[valid_mask] ** 2) - maxbound_marching_zdepth[valid_mask] / 2.0 # (N)
                init_zdepth = torch.ones_like(dist) * init_zdepth_valid.max() # (H*W)
                init_zdepth = self.copy_index(init_zdepth, valid_mask, init_zdepth_valid)
        return init_zdepth, valid_mask

    def get_maxbound_zdepth(self, cam_pos, valid_cam_rays):
        with torch.no_grad():
            init_zdepth, _ = self.get_intersections_with_unit_spheres(cam_pos, valid_cam_rays) # (N)

            dist = self.get_distance_from_origin(cam_pos, valid_cam_rays) # (N)
            maxbound_marching_zdepth = self.get_maxbound_zdepth_from_dist(dist) # (N)
            max_zdepth = init_zdepth + maxbound_marching_zdepth # (N)
        return max_zdepth

    def copy_index(self, inputs, mask, src):
        '''
        out-of-place copy index.
        Input:
        - inputs:	torch.Tensor (H*W) / (H, W) / (H, W, k)
        - mask:		torch.Tensor (H*W)
        - src:		torch.Tensor (N) / (N, k)
        '''
        inputs_shape = inputs.shape
        if len(inputs_shape) <= 2:
            inputs, mask = inputs.reshape(-1), mask.reshape(-1)
        elif len(inputs_shape) == 3:
            inputs, mask = inputs.reshape(-1, inputs_shape[-1]), mask.reshape(-1)
        else:
            raise NotImplementedError
        index = torch.nonzero(mask).reshape(-1).long()
        outputs = inputs.index_copy(0, index, src)
        outputs = outputs.reshape(inputs_shape)
        return outputs

    def get_index_from_sdf_list(self, sdf_list, index_size, index_type='min', clamp_dist=0.1):
        '''
        get index with certain method.
        Input:
        - sdf_list:		type: torch.Tensor (self.march_step, N)
        Return:
        - sdf:			type: torch.Tensor (N, index_size)
        - index:		type: torch.Tensor (N, index_size). Note: the first dimension (index[0]) is always the min index.
        '''
        if index_type == 'min':
            sdf, index = torch.topk(-sdf_list.transpose(1,0), index_size, dim=1)
            sdf = -sdf
        elif index_type == 'min_abs':
            sdf_list_new = torch.abs(sdf_list)
            _, index = torch.topk(-sdf_list_new.transpose(1,0), index_size, dim=1)
            sdf = self.collect_data_from_index(sdf_list, index)
        elif index_type == 'max_neg':
            sdf_list_new = sdf_list.clone()
            sdf_list_pos = (sdf_list_new >= 0)
            sdf_list_new[sdf_list_pos] = sdf_list_new[sdf_list_pos].clone() * (-1) - 2
            sdf, index = torch.topk(sdf_list_new.transpose(1,0), index_size, dim=1) # (N, index_size)
            sdf_pos = (sdf <= -2)
            sdf[sdf_pos] = sdf[sdf_pos].clone() * (-1) - 2
        elif index_type == 'last_valid':
            march_step, N = sdf_list.shape[0], sdf_list.shape[1]
            valid = (torch.abs(sdf_list) < clamp_dist)
            idx_list = torch.arange(0, march_step)[:,None].repeat(1,N).to(sdf_list.get_device())
            idx_list = idx_list.float() * valid.float()
            _, index = torch.topk(idx_list.transpose(1,0), index_size, dim=1) # (N, index_size)
            sdf = self.collect_data_from_index(sdf_list, index)[0].transpose(1,0)
        elif index_type == 'last':
            march_step, N = sdf_list.shape[0], sdf_list.shape[1]
            sdf = sdf_list[-index_size:, :].transpose(1,0)
            index = torch.arange(march_step - index_size, march_step)[None,:].repeat(N, 1)
            index = index.to(sdf.get_device())
        else:
            raise NotImplementedError
        return sdf, index

    def collect_data_from_index(self, data, index):
        '''
        Input:
        - data:		type: torch.Tensor (self.march_step, N) / (self.march_step, N, k)
        - index:	type: torch.Tensor (N, index_size)
        Return:
        - data_sampled:	type: torch.Tensor (index_size, N) / (index_size, N, k)
        '''
        index_size = index.shape[1]
        count_index = torch.arange(index.shape[0]).repeat(index_size).to(index.get_device())
        point_index = index.transpose(1,0).reshape(-1) * data.shape[1] + count_index

        if len(data.shape) == 3:
            data_shape = data.shape
            data_sampled = data.reshape(-1, data_shape[-1])[point_index].reshape(index_size, -1, data_shape[-1]).clone() # (index_size, N, 3)
        elif len(data.shape) == 2:
            data_sampled = data.reshape(-1)[point_index].reshape(index_size, -1).clone() # (index_size, N)
        else:
            raise NotImplementedError
        return data_sampled

    def sample_points_uniform(self, points, cam_rays, num_samples=None):
        '''
        Input:
        points:		type: torch.Tensor (N, 3)
        cam_rays:	type: torch.Tensor (3, N)
        Return:
        points_sampled:	type: torch.Tensor (num_samples, N, 3)
        '''
        if num_samples == None:
            num_samples = self.buffer_size
        N = points.shape[0]
        points = points[None,:,:].repeat(num_samples, 1, 1) # (num_samples, N, 3)
        cam_rays = cam_rays.transpose(1, 0)[None,:,:].repeat(num_samples, 1, 1) # (num_samples, N, 3)
        delta_depth = torch.linspace(0, -self.max_sample_dist, num_samples).to(points.get_device()) # (num_samples)
        delta_depth = delta_depth[:,None,None].repeat(1, N, 3) # (num_samples, N, 3)
        points_sampled = delta_depth * cam_rays + points # (num_smaples, N, 3)
        return points_sampled

    def get_min_sdf_sample(self, sdf_list, points_list, latent, index_type='min_abs', clamp_dist=0.1, profile=False, no_grad=False):
        profiler = Profiler(silent = not profile)
        _, index = self.get_index_from_sdf_list(sdf_list, 1, index_type=index_type)
        points = self.collect_data_from_index(points_list, index)[0] # (N, 3)
        min_sdf_sample = decode_sdf(self.decoder, latent, points, clamp_dist=None, no_grad=no_grad).squeeze(-1)
        profiler.report_process('[DEPTH] [SAMPLING] sample min sdf time\t')
        if no_grad:
            min_sdf_sample = min_sdf_sample.detach()
        return min_sdf_sample

    def get_sample_on_marching_zdepth_along_ray(self, marching_zdepth_list, sdf_list, points_list, cam_rays, latent, index_type='min_abs', use_uniform_sample=False, clamp_dist=0.1, profile=False, no_grad=False):
        # initialization
        profiler = Profiler(silent = not profile)

        # collect points
        if use_uniform_sample:
            sdf_selected, index_selected = self.get_index_from_sdf_list(sdf_list, 1, index_type=index_type, clamp_dist=clamp_dist)
            points = self.collect_data_from_index(points_list, index_selected)[0] # (N, 3)
            points_sampled = self.sample_points_uniform(points, cam_rays)
        else:
            sdf_selected, index_selected = self.get_index_from_sdf_list(sdf_list, self.buffer_size, index_type=index_type, clamp_dist=clamp_dist)
            points_sampled = self.collect_data_from_index(points_list, index_selected)
        profiler.report_process('[DEPTH] [SAMPLING] collect points time\t')

        # generate new marching zdepth
        marching_zdepth = self.collect_data_from_index(marching_zdepth_list, index_selected[:,[0]])[0] # (N)
        marching_zdepth = marching_zdepth + (1 - self.ray_marching_ratio) * torch.clamp(sdf_selected[0,:], -clamp_dist, clamp_dist) # (N)

        if no_grad:
            marching_zdepth_final = marching_zdepth
        else:
            marching_zdepth_new = marching_zdepth
            for i in range(self.buffer_size):
                sdf = decode_sdf(self.decoder, latent, points_sampled[i], clamp_dist=clamp_dist, no_grad=no_grad).squeeze(-1)
                marching_zdepth_new = marching_zdepth_new - sdf.detach() * self.ray_marching_ratio
                marching_zdepth_new = marching_zdepth_new + sdf * self.ray_marching_ratio
            profiler.report_process('[DEPTH] [SAMPLING] re-ray marching time')
            marching_zdepth_final = marching_zdepth_new
        return marching_zdepth_final

    def ray_marching_trivial_non_parallel(self, cam_pos, cam_rays, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=0.1, no_grad=False, use_transform=True):
        valid_cam_rays = cam_rays[:, valid_mask]
        init_zdepth = init_zdepth[valid_mask]

        if march_step is None:
            march_step = self.march_step

        marching_zdepth = torch.zeros_like(init_zdepth)
        marching_zdepth_list, sdf_list, points_list = [], [], []

        for j in range(valid_cam_rays.shape[1]):
            marching_zdepth_list_per_ray, sdf_list_per_ray, points_list_per_ray = [], [], []
            marching_zdepth_per_ray = marching_zdepth[[j]]
            for i in range(march_step):
                # get corresponding sdf value
                points = self.generate_point_samples(cam_pos, valid_cam_rays[:,[j]], init_zdepth[[j]] + marching_zdepth_per_ray, inv_transform=use_transform)
                sdf = decode_sdf(self.decoder, latent, points.transpose(1,0), clamp_dist=None, no_grad=no_grad).squeeze(-1)
                points_list_per_ray.append(points.transpose(1,0)[None,:])

                # clamp sdf from below if the flag is invalid, which means that it has not meet any sdf < 0
                sdf = sdf.detach()
                sdf_list_per_ray.append(sdf[None,:])
                sdf_marching = torch.clamp(sdf, -clamp_dist, clamp_dist)

                # aggressive ray marching
                marching_zdepth_per_ray = marching_zdepth_per_ray + sdf_marching * self.ray_marching_ratio
                marching_zdepth_list_per_ray.append(marching_zdepth_per_ray[None,:])

            # concat ray marching info
            marching_zdepth_list_per_ray = torch.cat(marching_zdepth_list_per_ray, 0) # (self.march_step, N)
            sdf_list_per_ray = torch.cat(sdf_list_per_ray, 0)
            points_list_per_ray = torch.cat(points_list_per_ray, 0)
            marching_zdepth_list.append(marching_zdepth_list_per_ray)
            sdf_list.append(sdf_list_per_ray)
            points_list.append(points_list_per_ray)
        marching_zdepth_list = torch.cat(marching_zdepth_list, 1)
        sdf_list = torch.cat(sdf_list, 1)
        points_list = torch.cat(points_list, 1)

        # get valid mask
        maxbound_zdepth = self.get_maxbound_zdepth(cam_pos, valid_cam_rays)
        valid_mask_max_marching_zdepth = (marching_zdepth_list[-1] + init_zdepth < maxbound_zdepth)
        min_sdf, _ = torch.abs(sdf_list).min(0)
        valid_mask_ray_marching = (min_sdf <= self.threshold)

        # get corner case: the first query is lower than threshold.
        valid_mask_first_query = sdf_list[0] > self.threshold
        valid_mask_render = valid_mask_max_marching_zdepth & valid_mask_ray_marching & valid_mask_first_query # (N)
        return sdf_list, marching_zdepth_list, points_list, valid_mask_render

    def ray_marching_trivial(self, cam_pos, cam_rays, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=0.1, no_grad=False, use_transform=True):
        valid_cam_rays = cam_rays[:, valid_mask]
        init_zdepth = init_zdepth[valid_mask]

        if march_step is None:
            march_step = self.march_step

        marching_zdepth = torch.zeros_like(init_zdepth)
        marching_zdepth_list, sdf_list, points_list = [], [], []
        for i in range(march_step):
            # get corresponding sdf value
            points = self.generate_point_samples(cam_pos, valid_cam_rays, init_zdepth + marching_zdepth, inv_transform=use_transform)
            sdf = decode_sdf(self.decoder, latent, points.transpose(1,0), clamp_dist=None, no_grad=no_grad).squeeze(-1)
            points_list.append(points.transpose(1,0)[None,:])

            # clamp sdf from below if the flag is invalid, which means that it has not meet any sdf < 0
            sdf = sdf.detach()
            sdf_list.append(sdf[None,:])
            sdf_marching = torch.clamp(sdf, -clamp_dist, clamp_dist)

            # aggressive ray marching
            marching_zdepth = marching_zdepth + sdf_marching * self.ray_marching_ratio
            marching_zdepth_list.append(marching_zdepth[None,:])

        # concat ray marching info
        marching_zdepth_list = torch.cat(marching_zdepth_list, 0) # (self.march_step, N)
        sdf_list = torch.cat(sdf_list, 0)
        points_list = torch.cat(points_list, 0)

        # get valid mask
        maxbound_zdepth = self.get_maxbound_zdepth(cam_pos, valid_cam_rays)
        valid_mask_max_marching_zdepth = (marching_zdepth_list[-1] + init_zdepth < maxbound_zdepth)
        min_sdf, _ = torch.abs(sdf_list).min(0)
        valid_mask_ray_marching = (min_sdf <= self.threshold)

        # get corner case: the first query is lower than threshold.
        valid_mask_first_query = sdf_list[0] > self.threshold
        valid_mask_render = valid_mask_max_marching_zdepth & valid_mask_ray_marching & valid_mask_first_query # (N)
        return sdf_list, marching_zdepth_list, points_list, valid_mask_render

    def ray_marching_recursive(self, cam_pos, cam_rays, init_zdepth, valid_mask, latent, march_step=None, stop_threshold=None, clamp_dist=0.1, no_grad=False, use_transform=True, use_first_query_check=True):
        if stop_threshold is None:
            stop_threshold = self.threshold
        valid_cam_rays = cam_rays[:, valid_mask]
        init_zdepth = init_zdepth[valid_mask]

        if march_step is None:
            march_step = self.march_step

        maxbound_zdepth = self.get_maxbound_zdepth(cam_pos, valid_cam_rays)

        marching_zdepth_list, sdf_list, points_list = [], [], []
        marching_zdepth = torch.zeros_like(init_zdepth)

        valid_mask_max_marching_zdepth = (marching_zdepth + init_zdepth < maxbound_zdepth)
        unfinished_mask = valid_mask_max_marching_zdepth # (N)
        for i in range(march_step):
            # get unfinished
            cam_rays_now = valid_cam_rays[:, unfinished_mask] # (3, K)
            init_zdepth_now = init_zdepth[unfinished_mask] # (K)
            marching_zdepth_now = marching_zdepth[unfinished_mask] # (K)

            # get corresponding sdf value
            points_now = self.generate_point_samples(cam_pos, cam_rays_now, init_zdepth_now + marching_zdepth_now, inv_transform=use_transform) # (3, K)
            if no_grad:
                points_now = points_now.detach()
            sdf_now = decode_sdf(self.decoder, latent, points_now.transpose(1,0), clamp_dist=None, no_grad=no_grad).squeeze(-1) # (K)
            points = torch.zeros_like(marching_zdepth)[:,None].repeat(1,3)
            points[unfinished_mask,:] = points_now.transpose(1,0)
            if no_grad:
                points = points.detach()
            points_list.append(points[None,:])

            # clamp sdf from below if the flag is invalid, which means that it has not meet any sdf < 0
            sdf = torch.zeros_like(marching_zdepth)
            sdf[unfinished_mask] = sdf_now.detach()
            sdf_marching = torch.clamp(sdf, -clamp_dist, clamp_dist)

            # aggressive ray marching
            marching_zdepth = marching_zdepth + sdf_marching * self.ray_marching_ratio
            marching_zdepth_list.append(marching_zdepth[None,:])

            # update sdf list
            sdf[~unfinished_mask] = 1.0
            sdf_list.append(sdf[None,:])

            # update unfinised mask
            valid_mask_max_marching_zdepth = (marching_zdepth + init_zdepth < maxbound_zdepth)
            unstop_mask = torch.abs(sdf) >= stop_threshold
            unfinished_mask = unfinished_mask & valid_mask_max_marching_zdepth & unstop_mask
            if torch.nonzero(unfinished_mask).shape[0] == 0:
                while(len(marching_zdepth_list) < self.buffer_size):
                    marching_zdepth_list.append(marching_zdepth[None,:])
                    sdf_list.append(sdf[None,:])
                    points_list.append(points[None,:])
                break
        # concat ray marching info
        marching_zdepth_list = torch.cat(marching_zdepth_list, 0) # (self.march_step, N)
        sdf_list = torch.cat(sdf_list, 0)
        points_list = torch.cat(points_list, 0)

        # get valid mask
        valid_mask_max_marching_zdepth = (marching_zdepth_list[-1] + init_zdepth < maxbound_zdepth)
        min_sdf, _ = torch.abs(sdf_list).min(0)
        valid_mask_ray_marching = (min_sdf <= self.threshold)

        # get corner case: the first query is lower than threshold.
        valid_mask_render = valid_mask_max_marching_zdepth & valid_mask_ray_marching # (N)
        if use_first_query_check:
            valid_mask_first_query = sdf_list[0] > self.threshold
            valid_mask_render = valid_mask_render & valid_mask_first_query
        return sdf_list, marching_zdepth_list, points_list, valid_mask_render

    def index_sample(self, basemap, indexmap):
        '''
        To use indexmap to index basemap.
        Inputs:
        - basemap		type: torch.Tensor (H', W', C)
        - indexmap		type: torch.Tensor (H, W, 2)
        Returns:
        - newmap		type: torch.Tensor (H, W, C)
        '''
        h, w, c = basemap.shape[0], basemap.shape[1], basemap.shape[2]
        h_index, w_index = indexmap.shape[0], indexmap.shape[1]

        index = indexmap.reshape(-1, 2)
        index = (index[:,0] + index[:,1] * w).type(torch.long)

        newmap = basemap.reshape(-1, c)[index]
        newmap = newmap.reshape(h_index, w_index, c)
        return newmap

    def get_downscaled_grid_map(self, grid_map, scale=2.0):
        '''
        Inputs:
        - grid_map		type: torch.Tensor (H, W, 2)
        Returns:
        - new_grid_map		type: torch.Tensor (H/scale, W/scale, 2)
        - index_map		type: torch.Tensor (H, W, 2)
        '''
        h, w = grid_map.shape[0], grid_map.shape[1]
        stride = grid_map[0,1,0] - grid_map[0,0,0]
        new_h, new_w = np.ceil(h / scale), np.ceil(w / scale)

        new_grid_map = self.get_meshgrid((new_h, new_w))
        new_grid_map = (scale * stride) * new_grid_map + ((scale * stride) - 1) / 2
        new_grid_map = new_grid_map.to(grid_map.get_device())

        if stride == 1:
            grid_map_meshgrid = grid_map
        else:
            grid_map_meshgrid = self.get_meshgrid((h,w))
            grid_map_meshgrid = grid_map_meshgrid.to(grid_map.get_device())
        index_map = torch.ceil((grid_map_meshgrid + 1) / scale) - 1

        if (index_map[:,:,0] + index_map[:,:,1] * new_grid_map.shape[1]).max().detach().cpu().numpy() > new_grid_map.reshape(-1,2).shape[0]:
            raise ValueError('Error! Index map out of bound.')
        return new_grid_map, index_map

    def get_rays_from_grid_map(self, grid_map, R):
        homo_2d = self.get_homo_2d_from_xy(grid_map) # (H', W', 3)
        homo_calib = torch.matmul(self.K_inv, homo_2d.reshape(-1,3).transpose(1,0)) # (3, H'*W')
        rays = self.get_camera_rays(R, homo=homo_calib)
        calib_map = self.normalize_vectors(homo_calib)[2,:]
        return rays, calib_map

    def get_downscaled_camera_rays(self, grid_map, R, scale=2.0):
        '''
        To get downscaled camera rays along with related infos
        Inputs:
        - grid_map		type: torch.Tensor (H, W, 2)
        - R			type: torch.Tensor (3, 3)
        Returns:
        - new_grid_map		type: torch.Tensor (H', W', 2)
        - new_rays		type: torch.Tensor (3, H'*W')
        - index_map		type: torch.Tensor (H*W, 2) (easy index-based upsampling is available simply with new_rays[:, index_map])
        - recalib_map		type: torch.Tensor (H*W)
        '''
        # get downsampled grid map and corresponding index map
        new_grid_map, index_map = self.get_downscaled_grid_map(grid_map, scale=scale)
        new_img_hw = new_grid_map.shape[:2]

        # get downsampled camera rays
        new_rays, new_calib_map = self.get_rays_from_grid_map(new_grid_map, R)

        # get corresponding index
        h, w = new_grid_map.shape[0], new_grid_map.shape[1]
        index_map = index_map.reshape(-1, 2)
        index_map = (index_map[:,0] + index_map[:,1] * w).type(torch.long)

        # upsample downsampled camera rays and compute angles
        rays, calib_map = self.get_rays_from_grid_map(grid_map, R)
        new_calib_map_upsampled = new_calib_map[index_map]
        recalib_map = new_calib_map_upsampled / calib_map
        return new_grid_map, new_rays, index_map, recalib_map

    def maxpool_valid_mask_with_index(self, valid_mask, index_map):
        '''
        to max pooling a binary mask (0/1 float tensor)
        Inputs:
        - valid_mask:		type: torch.Tensor (H*W)
        - index_map:		type: torch.Tensor (H*W) with max entries (H'*W' - 1)
        Returns:
        - new_valid_mask:	type: torch.Tensor (H'*W')
        '''
        from torch_scatter import scatter_max
        with torch.no_grad():
            new_valid_mask, _ = scatter_max(valid_mask, index_map)
        return new_valid_mask

    def upsample_zdepth_and_recalib(self, zdepth_lowres, index_map, recalib_map):
        zdepth_highres = zdepth_lowres[:, index_map]
        zdepth_highres = zdepth_highres * recalib_map
        return zdepth_highres

    def unmap_tensor_with_valid_mask(self, tensor, valid_mask, fill_value=0.):
        '''
        Inputs:
        - tensor: 		type: torch.Tensor (C, N) / (C, N, 3)
        - valid_mask:		type: torch.Tensor (H*W) with N valid entries
        Returns:
        - output:		type: torch.Tensor (C, H*W) / (C, H*W, 3)
        '''
        C, N = tensor.shape[0], tensor.shape[1]
        N_new = valid_mask.shape[0]

        if len(tensor.shape) == 2:
            if fill_value == 0:
                output = torch.zeros(C, N_new).to(tensor.get_device())
            else:
                output = (torch.ones(C, N_new) * fill_value).to(tensor.get_device())
            output[:, valid_mask] = tensor
        else:
            M = tensor.shape[2]
            if fill_value == 0:
                output = torch.zeros(C, N_new, M).to(tensor.get_device())
            else:
                output = (torch.ones(C, N_new, M) * fill_value).to(tensor.get_device())
            output[:, valid_mask, :] = tensor
        return output

    def ray_marching_pyramid_recursive(self, cam_pos, R, valid_mask, latent, scale_list=None, march_step_list=None, march_step=None, stop_threshold=None, clamp_dist=0.1, no_grad=False, use_transform=True, split_type='raydepth'):
        if stop_threshold is None:
            stop_threshold = self.threshold

        # initialization
        if march_step is None:
            march_step = self.march_step
        if scale_list is None:
            scale_list = copy.deepcopy(self.scale_list)
        if march_step_list is None:
            march_step_list = copy.deepcopy(self.march_step_list)
        if march_step_list[-1] == -1:
            march_step_list[-1] = march_step - sum(march_step_list[:-1])
        assert (scale_list[-1] == 1)

        # get pyramid rays, index maps, recalib maps and downscaled valid masks.
        grid_map_list, rays_list, index_map_list, recalib_map_list, img_hw_list = [], [], [], [], []
        valid_mask_list = []
        scale_list_rev, march_step_list_rev = scale_list[::-1], march_step_list[::-1]
        for idx, scale in enumerate(scale_list_rev):
            if idx == 0: # the original scale
                grid_map_now = self.homo_2d.reshape(-1, self.img_hw[0], self.img_hw[1])[:2].permute(1,2,0) # [h,w,2]
                rays_now = self.get_camera_rays(R)
                index_map_now, recalib_map_now = None, None
                valid_mask_now = valid_mask
            else:
                grid_map_now, rays_now, index_map_now, recalib_map_now = self.get_downscaled_camera_rays(grid_map_list[idx-1], R, scale / scale_list_rev[idx-1])
                if split_type == 'raydepth':
                    recalib_map_now = torch.ones_like(recalib_map_now)
                valid_mask_now = self.maxpool_valid_mask_with_index(valid_mask_list[idx-1], index_map_now)
            grid_map_list.append(grid_map_now)
            img_hw_list.append(grid_map_now.shape[:2])

            rays_list.append(rays_now)
            index_map_list.append(index_map_now)
            recalib_map_list.append(recalib_map_now)
            valid_mask_list.append(valid_mask_now)

        # get init zdepth
        init_zdepth_lowres, _ = self.get_intersections_with_unit_spheres(cam_pos, rays_list[-1])
        init_zdepth_original, _ = self.get_intersections_with_unit_spheres(cam_pos, rays_list[0])

        # pyramid recursive ray marching
        sdf_list, zdepth_list, points_list = None, None, None
        num_scales = len(rays_list)
        for idx in range(num_scales - 1, -1, -1):
            cam_rays_now = rays_list[idx]
            valid_mask_now = valid_mask_list[idx]

            index_map_now = index_map_list[idx]
            recalib_map_now = recalib_map_list[idx]

            march_step_now = march_step_list_rev[idx]
            if idx == num_scales - 1: # first (large) scale: initialization
                init_zdepth_now = init_zdepth_lowres
            else:
                init_zdepth_now = zdepth_list[-1]

            # single-scale ray marching
            if idx != 0: # first (large) scale: initialization
                sdf_list_now, marching_zdepth_list_now, points_list_now, _ = self.ray_marching_trivial(cam_pos, cam_rays_now, init_zdepth_now, valid_mask_now, latent, march_step=march_step_now, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)

                # unmap
                sdf_list_now = self.unmap_tensor_with_valid_mask(sdf_list_now, valid_mask_now, fill_value=1.0)
                points_list_now = self.unmap_tensor_with_valid_mask(points_list_now, valid_mask_now)
                marching_zdepth_list_now = self.unmap_tensor_with_valid_mask(marching_zdepth_list_now, valid_mask_now)
                zdepth_list_now = marching_zdepth_list_now + init_zdepth_now

                if idx != num_scales - 1: # not first iteration
                    sdf_list_now = torch.cat([sdf_list, sdf_list_now], 0)
                    points_list_now = torch.cat([points_list, points_list_now], 0)
                    zdepth_list_now = torch.cat([zdepth_list, zdepth_list_now], 0)

                # upsample (and recalib)
                sdf_list_now = sdf_list_now[:, index_map_now]
                points_list_now = points_list_now[:, index_map_now, :]
                zdepth_list_now = self.upsample_zdepth_and_recalib(zdepth_list_now, index_map_now, recalib_map_now)

                # update global info
                sdf_list, zdepth_list, points_list = sdf_list_now, zdepth_list_now, points_list_now

            else: # i.e. idx == 0: final (original) scale: recursive ray marching.
                sdf_list_now, marching_zdepth_list_now, points_list_now, valid_mask_render = self.ray_marching_recursive(cam_pos, cam_rays_now, init_zdepth_now, valid_mask_now, latent, march_step=march_step_now, clamp_dist=clamp_dist, no_grad=no_grad, use_first_query_check=False, use_transform=use_transform)

                # map down global info to (N)
                sdf_list = torch.cat([sdf_list[:, valid_mask], sdf_list_now], 0)
                points_list = torch.cat([points_list[:, valid_mask, :], points_list_now], 0)
                zdepth_list_now = marching_zdepth_list_now + init_zdepth_now[valid_mask]
                zdepth_list = torch.cat([zdepth_list[:, valid_mask], zdepth_list_now], 0)

        # get the corresponding marching zdepth
        marching_zdepth_list = zdepth_list - init_zdepth_original[valid_mask][None,:]
        return sdf_list, marching_zdepth_list, points_list, valid_mask_render

    def ray_marching(self, cam_pos, R, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=0.1, no_grad=False, use_transform=True, ray_marching_type='recursive', split_type='raydepth'):
        '''
        ray marching function
        Input:
        - init_zdepth			type: torch.Tensor (H*W)
        - valid_mask			type: torch.Tensor (H*W) with N valid entries
        - split_type                    ['depth', 'raydepth'], which is the spliting strategy for pyramid recursive marching
        Return:
        - sdf_list			type: torch.Tensor (march_step, N)
        - marching_zdepth_list		type: torch.Tensor (march_step, N)
        - points_list			type: torch.Tensor (march_step, N, 3)
        - valid_mask_render		type: torch.Tensor (N)
        '''
        if not (split_type in ['depth', 'raydepth']):
            raise NotImplementedError
        if ray_marching_type == 'trivial_non_parallel':
            cam_rays = self.get_camera_rays(R)
            return self.ray_marching_trivial_non_parallel(cam_pos, cam_rays, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)
        elif ray_marching_type == 'trivial':
            cam_rays = self.get_camera_rays(R)
            return self.ray_marching_trivial(cam_pos, cam_rays, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)
        elif ray_marching_type == 'recursive':
            cam_rays = self.get_camera_rays(R)
            return self.ray_marching_recursive(cam_pos, cam_rays, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform)
        elif ray_marching_type == 'pyramid_recursive':
            return self.ray_marching_pyramid_recursive(cam_pos, R, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad, use_transform=use_transform, split_type=split_type)
        else:
            raise ValueError('Error! Invalid type of ray marching: {}.'.format(ray_marching_type))

    def render_depth(self, latent, R, T, clamp_dist=0.1, sample_index_type='min_abs', profile=False, no_grad=False, no_grad_depth=False, no_grad_mask=False, no_grad_camera=False, ray_marching_type='recursive', use_transform=True):
        if no_grad:
            no_grad_depth, no_grad_mask, no_grad_camera = True, True, True

        cam_pos = self.get_camera_location(R, T)
        cam_rays = self.get_camera_rays(R)
        dist = self.get_distance_from_origin(cam_pos, cam_rays)

        profiler = Profiler(silent = not profile)
        # initialization on the unit sphere
        h, w = self.img_hw
        init_zdepth, valid_mask = self.get_intersections_with_unit_spheres(cam_pos, cam_rays)
        profiler.report_process('[DEPTH] initialization time')

        # ray marching
        sdf_list, marching_zdepth_list, points_list, valid_mask_render = self.ray_marching(cam_pos, R, init_zdepth, valid_mask, latent, clamp_dist=clamp_dist, no_grad=no_grad_camera, ray_marching_type=ray_marching_type, use_transform=use_transform)
        profiler.report_process('[DEPTH] ray marching time')

        # get differnetiable samples
        min_sdf_sample = self.get_min_sdf_sample(sdf_list, points_list, latent, index_type='min_abs', clamp_dist=clamp_dist, profile=profile, no_grad=no_grad_mask)
        marching_zdepth = self.get_sample_on_marching_zdepth_along_ray(marching_zdepth_list, sdf_list, points_list, cam_rays[:, valid_mask], latent, use_uniform_sample=False, index_type=sample_index_type, clamp_dist=clamp_dist, profile=profile, no_grad=no_grad_depth)
        profiler.report_process('[DEPTH] re-sampling time')

        # generate output
        min_sdf_sample_new = torch.zeros_like(valid_mask).float() # (H, W)
        min_sdf_sample_new.requires_grad = True
        min_sdf_sample_new = self.copy_index(min_sdf_sample_new, valid_mask, min_sdf_sample)
        min_sdf_sample_new = self.copy_index(min_sdf_sample_new, ~valid_mask, dist[~valid_mask] + self.threshold - self.radius) # help handle camera gradient

        ## get zdepth
        Zdepth = torch.ones_like(self.imgmap_init) * 1e11 # (H, W)
        Zdepth.requires_grad = True
        src_zdepth = init_zdepth[valid_mask] + marching_zdepth # (N)
        Zdepth = self.copy_index(Zdepth, valid_mask, src_zdepth)
        Zdepth = Zdepth.reshape(-1) # (H*W)

        ## update valid_mask
        valid_mask = valid_mask.clone()
        valid_mask[valid_mask] = valid_mask_render
        profiler.report_process('[DEPTH] finalize time\t')
        if no_grad_depth:
            Zdepth = Zdepth.detach()
        return Zdepth, valid_mask, min_sdf_sample_new # (H*W), (H*W), (H*W)

    def render_normal(self, latent, R, T, Zdepth, valid_mask, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False, normalize=True, use_transform=True):
        cam_pos = self.get_camera_location(R, T)
        cam_rays = self.get_camera_rays(R)

        h, w = self.img_hw
        Znormal = torch.zeros_like(self.imgmap_init)[None,:,:].repeat(3, 1, 1) # (3, H, W)
        Znormal.requires_grad = True

        # initialization
        valid_cam_rays = cam_rays[:, valid_mask]
        valid_zdepth = Zdepth[valid_mask]
        if valid_zdepth.shape[0] == 0:
            return Znormal.reshape(3, -1) # (3, H*W)

        # compute normal
        points = self.generate_point_samples(cam_pos, valid_cam_rays, valid_zdepth, has_zdepth_grad=False, inv_transform=use_transform)
        gradient = decode_sdf_gradient(self.decoder, latent, points.transpose(1,0), clamp_dist=clamp_dist, no_grad=no_grad, MAX_POINTS=MAX_POINTS) # (N, 3)
        gradient = gradient.transpose(1,0) # (3, N)
        if normalize:
            valid_normal_untransformed = self.normalize_vectors(gradient) # (3, N)
        else:
            valid_normal_untransformed = gradient
        valid_normal = self.transform_points(valid_normal_untransformed)

        # generate output
        Znormal = self.copy_index(Znormal.permute(1,2,0), valid_mask, valid_normal.transpose(1,0)) # (H, W, 3)
        Znormal = Znormal.reshape(-1, 3).transpose(1,0)

        if no_grad:
            Znormal = Znormal.detach()
        return Znormal # (3, H*W)

    def forward_sampling(self, latent, R, T, Zdepth, valid_mask, clamp_dist=0.1, num_forward_sampling=1, no_grad=False, use_transform=True):
        '''
        To sample forward along the ray (sampling inside)
        This function should be used when the latent space is not pretrained.
        Returns:
        - inside_samples		torch.Tensor (H*W, num_forward_sampling) (sdf + offset, should be negative)
        '''
        assert (num_forward_sampling > 0)
        cam_pos = self.get_camera_location(R, T)
        cam_rays = self.get_camera_rays(R)

        # initialization
        h, w = self.img_hw
        inside_samples = torch.zeros_like(self.imgmap_init).reshape(-1)[:,None].repeat(1, num_forward_sampling) # (3, H, W)
        valid_cam_rays = cam_rays[:, valid_mask]
        valid_zdepth = Zdepth[valid_mask]
        if valid_zdepth.shape[0] == 0:
            return inside_samples # (H*W, 3)

        grid_list = 0.5 * clamp_dist * (torch.arange(num_forward_sampling).float() + 1) / num_forward_sampling
        if cam_pos.get_device() != -1:
            grid_list.to(cam_pos.get_device())
        inside_samples_list = []
        for idx in range(num_forward_sampling):
            grid = grid_list[idx]
            points = self.generate_point_samples(cam_pos, valid_cam_rays, valid_zdepth + grid, has_zdepth_grad=False, inv_transform=use_transform)
            sdf = decode_sdf(self.decoder, latent, points.transpose(1,0), clamp_dist=None, no_grad=no_grad).squeeze(-1)
            inside_samples_list.append(sdf[:,None] + grid)
        inside_samples[valid_mask] = torch.cat(inside_samples_list, 1)
        return inside_samples

    def render(self, latent, R, T, clamp_dist=0.1, sample_index_type='min_abs', profile=False, no_grad=False, no_grad_depth=False, no_grad_normal=False, no_grad_mask=False, no_grad_camera=False, normalize_normal=True, use_transform=True, ray_marching_type='pyramid_recursive', num_forward_sampling=0):
        '''
        differentiable rendering.
        Input:
        - latent	type torch.Tensor (1, latent_size)
        - R		type torch.Tensor (3,3)
        - T		type torch.Tensor (3)
        Return:
        - Zdepth		type torch.Tensor (H, W) - rendered depth
        - Znormal		type torch.Tensor (H, W, 3) - rendered normal
        - valid_mask		type torch.Tensor (H, W) - rendered silhoutte
        - min_sdf_sample	type torch.Tensor (H, W) - minimum_depth_sample
        '''
        if no_grad:
            no_grad_depth, no_grad_normal, no_grad_mask, no_grad_camera = True, True, True, True

        profiler = Profiler(silent = not profile)
        h, w = self.img_hw
        profiler.report_process('\ninitialization time')

        # render depth
        Zdepth, valid_mask, min_abs_query = self.render_depth(latent, R, T, clamp_dist=clamp_dist, sample_index_type=sample_index_type, profile=profile, no_grad=no_grad, no_grad_depth=no_grad_depth, no_grad_mask=no_grad_mask, no_grad_camera=no_grad_camera, ray_marching_type=ray_marching_type, use_transform=use_transform) # (H*W), (H*W), (H*W)
        profiler.report_process('render depth time')

        depth = torch.ones_like(Zdepth) * 1e11
        depth[valid_mask] = Zdepth[valid_mask].clone() * self.calib_map[valid_mask]
        depth = depth.reshape(h, w)

        # render normal
        if self.use_depth2normal:
            f_x_pix = self.K.detach().cpu().numpy()[0,0]
            f_y_pix = self.K.detach().cpu().numpy()[1,1]
            normal = depth2normal(depth, f_x_pix, f_y_pix)
        else:
            normal = self.render_normal(latent, R, T, Zdepth, valid_mask, clamp_dist=clamp_dist, no_grad=no_grad_normal, normalize=normalize_normal, use_transform=use_transform) # (3, H*W)
            normal = torch.matmul(R, normal) # (3, H*W)
            normal[0,:] = normal[0,:].clone() * (-1) # transformed the direction to align with rendering engine (left-hand sided).
            normal = normal.reshape(3, h, w).permute(1,2,0)
        profiler.report_process('render normal time')

        # (optional) forward sampling inside the surface
        if num_forward_sampling != 0:
            inside_samples = self.forward_sampling(latent, R, T, Zdepth, valid_mask, clamp_dist=clamp_dist, num_forward_sampling=num_forward_sampling, use_transform=use_transform) # (H*W, k)
            inside_samples = inside_samples.reshape(h, w, num_forward_sampling)
            profiler.report_process('forward sampling time')

        # reshape mask and return
        binary_mask = valid_mask.reshape(h, w).type(torch.uint8)
        min_abs_query = min_abs_query.reshape(h, w)
        profiler.report_process('finalization time')
        profiler.report_all('total time')
        if profile:
            pdb.set_trace()
        if num_forward_sampling == 0:
            return depth, normal, binary_mask, min_abs_query
        else:
            return depth, normal, binary_mask, min_abs_query, inside_samples

if __name__ == '__main__':
    pass


