import os, sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from renderer import SDFRenderer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.utils.decoder_utils import decode_sdf, decode_sdf_gradient
from core.visualize.profiler import Profiler
from core.utils.loss_utils import compute_loss_mask, compute_loss_color
from core.utils.loss_utils import grid_sample_on_img

class SDFRenderer_warp(SDFRenderer):
    def __init__(self, decoder, intrinsic, img_hw=None, march_step=50, buffer_size=5, ray_marching_ratio=1.5, max_sample_dist=0.2, threshold=5e-5, use_gpu=True, is_eval=True, transform_matrix=None):
        super(SDFRenderer_warp, self).__init__(decoder, intrinsic, img_hw=img_hw, transform_matrix=transform_matrix, march_step=march_step, buffer_size=buffer_size, ray_marching_ratio=ray_marching_ratio, max_sample_dist=max_sample_dist, threshold=threshold, use_gpu=use_gpu, is_eval=is_eval)
        self.counter = 0

    def get_valid_points(self, render_out1, render_out2, R1, T1, R2, T2, thres_depth, gt_mask=None):
        Zdepth1, valid_mask1, min_sdf_sample1 = render_out1
        Zdepth2, valid_mask2, min_sdf_sample2 = render_out2

        # acquire points from view 1
        cam_pos1 = self.get_camera_location(R1, T1)
        cam_rays1 = self.get_camera_rays(R1)

        cam_rays1 = cam_rays1[:, valid_mask1]
        Zdepth1 = Zdepth1[valid_mask1]
        points_view1 = self.generate_point_samples(cam_pos1, cam_rays1, Zdepth1, inv_transform=False, has_zdepth_grad=True)

        # project 3D points acquired from view 1, to view 2
        xyz_proj = torch.matmul(self.K, torch.matmul(R2, points_view1) + T2[:, None])
        xy_proj = xyz_proj[:2, :] / xyz_proj[2,:]
        xy_proj = xy_proj[None, :, :, None]

        # eliminate the points projected outside the mask in view 2
        # if gt_mask is None:
        #     valid_mask_index = self.valid_points_mask(xy_proj, valid_mask2)
        # else:
        #     valid_mask_index = self.valid_points_mask(xy_proj, gt_mask)

        #####################################################
        # Now no need to eliminate the points projected outside the mask
        valid_mask_index = torch.ones(xyz_proj.shape[1]).byte().cuda()
        #####################################################

        xy_proj = xy_proj[:, :, valid_mask_index, :]

        # eliminate the points with invalid depth
        depth2_proj = xyz_proj[2,:]
        depth2_proj = depth2_proj[valid_mask_index]
        valid_depth_index = self.valid_points_depth(xy_proj, Zdepth2, depth2_proj, thres_depth)
        xy_proj = xy_proj[:, :, valid_depth_index, :]

        return xy_proj, valid_mask_index, valid_depth_index

    def valid_points_mask(self, xy_proj, valid_mask2):
        h, w = self.img_hw
        valid_mask2_reshape = valid_mask2.reshape(h, w).float().to(self.device)
        valid_mask2_reshape = valid_mask2_reshape[None, None, :, :]
        valid_mask_index = torch.squeeze(grid_sample_on_img(valid_mask2_reshape, xy_proj)>0.5)
        return valid_mask_index

    def valid_points_depth(self, xy_proj, Zdepth2, depth2_proj, thres_depth):
        h, w = self.img_hw
        depth2_sample = Zdepth2 * self.calib_map
        depth2_sample = depth2_sample.reshape(h, w)
        depth2_sample = depth2_sample[None, None, :, :]
        depth2_sample = torch.squeeze(grid_sample_on_img(depth2_sample, xy_proj))

        depth2_error = ((depth2_proj - depth2_sample)**2)
        valid_depth_index = depth2_error < thres_depth
        return valid_depth_index

    def compute_loss_color(self, img1, img2, xy_proj, valid_mask1, valid_mask_index, valid_depth_index):
        h, w = self.img_hw
        img1_valid = img1.reshape(h*w, 3).to(self.device)
        img1_valid = img1_valid[valid_mask1]
        img1_valid = img1_valid[valid_mask_index]
        img1_valid = img1_valid[valid_depth_index]

        img2_ = img2.permute(2,0,1).to(self.device)
        img2_ = img2_[None, :]
        img2_sample = torch.squeeze(grid_sample_on_img(img2_, xy_proj)).permute(1, 0)

        loss_color = torch.mean(torch.abs(img1_valid - img2_sample))

        # final valid mask in view 1
        mask_final_1 = torch.zeros(valid_depth_index.shape[0], dtype=torch.uint8)
        mask_final_1[valid_depth_index] = 1
        mask_final_2 = torch.zeros(valid_mask_index.shape[0], dtype=torch.uint8)
        mask_final_2[valid_mask_index] = mask_final_1
        mask_final = torch.zeros(valid_mask1.shape[0], dtype=torch.uint8)
        mask_final[valid_mask1] = mask_final_2
        mask_final = mask_final.reshape(h, w)

        # visualize the valid part in view 1 and from warped view 2
        color_valid_1, color_valid_2 = torch.zeros_like(img1).to(self.device), torch.zeros_like(img1).to(self.device)
        color_valid_1[mask_final] = img1_valid
        color_valid_2[mask_final] = img2_sample

        return loss_color, color_valid_1, color_valid_2

    def render_warp(self, latent, R1, T1, R2, T2, img1, img2, clamp_dist=0.1, profile=False, no_grad_normal=False, thres_depth=0.001):
        h, w = self.img_hw
        profiler = Profiler(silent = not profile)

        # render depth for both views
        render_out1 = self.render_depth(latent, R1, T1, clamp_dist=clamp_dist, profile=profile) # (H*W), (H*W), (H*W)
        render_out2 = self.render_depth(latent, R2, T2, clamp_dist=clamp_dist, profile=profile, no_grad_depth=True) # (H*W), (H*W), (H*W)
        profiler.report_process('render depth time')

        Zdepth1, valid_mask1, min_sdf_sample1 = render_out1
        Zdepth2, valid_mask2, min_sdf_sample2 = render_out2

        if len(min_sdf_sample1.shape) == 1:
            min_sdf_sample1 = min_sdf_sample1.reshape(h, w)
            min_sdf_sample2 = min_sdf_sample2.reshape(h, w)

        if torch.nonzero(valid_mask1).shape[0] == 0: # no valid mask at all
            loss_color = torch.from_numpy(np.array(0)).float().to(self.device)
            loss_color.requires_grad = True
            color_valid_1, color_valid_2 = torch.zeros_like(img1).to(self.device), torch.zeros_like(img1).to(self.device)
        else:
            # acquire valid points back in view 2
            xy_proj, valid_mask_index, valid_depth_index = self.get_valid_points(render_out1, render_out2, R1, T1, R2, T2, thres_depth)
            # calculate the loss of the corresponding colors
            loss_color, color_valid_1, color_valid_2 = self.compute_loss_color(img1, img2, xy_proj, valid_mask1, valid_mask_index, valid_depth_index)


        # acquire surface normal to visualize
        normal1 = self.render_normal(latent, R1, T1, Zdepth1, valid_mask1, no_grad=no_grad_normal, clamp_dist=clamp_dist, MAX_POINTS=100000) # (3, H*W)
        Znormal1 = torch.matmul(R1, normal1) # (3, H*W)
        Znormal1[0,:] = Znormal1[0,:] * (-1) # transformed the direction to align with rendering engine (left-hand sided).
        Znormal1 = Znormal1.reshape(3, h, w).permute(1,2,0)

        # acquire rendered depth to visualize
        Zdepth1[~valid_mask1] = 0
        Zdepth1[valid_mask1] = Zdepth1[valid_mask1] * self.calib_map[valid_mask1]
        depth1_rendered = Zdepth1.reshape(h, w)

        valid_mask1 = valid_mask1.reshape(h, w).type(torch.uint8)
        valid_mask2 = valid_mask2.reshape(h, w).type(torch.uint8)

        return loss_color, color_valid_1, color_valid_2, valid_mask1, valid_mask2, min_sdf_sample1, min_sdf_sample2, Znormal1, depth1_rendered

    def visualize_error(self, depth2_error, xyz_proj, valid_mask_index, valid_mask1, valid_mask2, img1, img2, color_valid_1, color_valid_2):
        h, w = self.img_hw
        depth2_error[depth2_error>10e+5]=0.0
        depth_error_unindexed = torch.zeros(xyz_proj.shape[1], dtype=torch.float).to(self.device)
        depth_error_unindexed[valid_mask_index] = depth2_error
        depth_error_img = torch.zeros_like(valid_mask1).float().to(self.device)
        depth_error_img[valid_mask1] = depth_error_unindexed
        depth_error_img = depth_error_img.reshape(h, w).detach().cpu().numpy()

        mask_inters_pre = torch.zeros(xyz_proj.shape[1], dtype=torch.float).to(self.device)
        mask_inters_pre[valid_mask_index] = 1

        mask_inters = torch.zeros_like(valid_mask1).float().to(self.device)
        mask_inters[valid_mask1] = mask_inters_pre

        max_ = np.max(depth_error_img)
        min_ = np.min(depth_error_img)
        depth_error_img = 255*(depth_error_img - min_) / (max_ - min_)

        import matplotlib.pyplot as plt
        import cv2
        fig, axs = plt.subplots(1, 8, figsize=(24,3))
        axs[0].imshow(cv2.cvtColor(img1.detach().cpu().numpy(),cv2.COLOR_BGR2RGB))
        axs[1].imshow(cv2.cvtColor(img2.detach().cpu().numpy(),cv2.COLOR_BGR2RGB))
        axs[2].imshow(valid_mask1.detach().cpu().numpy().reshape(h, w))
        axs[3].imshow(valid_mask2.detach().cpu().numpy().reshape(h, w))
        axs[4].imshow(mask_inters.detach().cpu().numpy().reshape(h, w))
        axs[5].imshow(depth_error_img, cmap = 'plasma')
        axs[6].imshow(cv2.cvtColor(color_valid_1.detach().cpu().numpy(),cv2.COLOR_BGR2RGB))
        axs[7].imshow(cv2.cvtColor(color_valid_2.detach().cpu().numpy(),cv2.COLOR_BGR2RGB))
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        axs[3].axis('off')
        axs[4].axis('off')
        axs[5].axis('off')
        axs[6].axis('off')
        axs[7].axis('off')

        axs[0].set_title('image 1')
        axs[1].set_title('image 2')
        axs[2].set_title('mask 1')
        axs[3].set_title('mask 2')
        axs[4].set_title('valid mask 1 in image 2')
        axs[5].set_title('depth error')
        axs[6].set_title('final valid view 1')
        axs[7].set_title('warped value from view 2 ')
        fig.savefig('vis_results/occlusion/occlusion_{}.png'.format(self.counter))
        self.counter += 1
        plt.close('all')



    def render_warp_gt(self, latent, R1, T1, R2, T2, depth1, depth2, points_view1, points_view2, valid_mask1, valid_mask2, img1, img2, clamp_dist=0.1, profile=False, no_grad=False):
        h, w = self.img_hw

        # project 3D points acquired from view 1, to view 2
        points_view1 = points_view1[:, valid_mask1]
        xyz_proj = torch.matmul(self.K, torch.matmul(R2, points_view1) + T2[:, None])
        xy_proj = xyz_proj[:2, :] / xyz_proj[2,:]

        # eliminate the points projected outside the mask in view 2
        xy_proj = xy_proj[None, :, :, None]
        valid_mask2_reshape = valid_mask2.clone().reshape(h, w).float().to(self.device)
        valid_mask2_reshape = valid_mask2_reshape[None, None, :, :]
        valid_mask_index = torch.squeeze(grid_sample_on_img(valid_mask2_reshape, xy_proj)>0.5)
        xy_proj = xy_proj[:, :, valid_mask_index, :]

        # eliminate the points with invalid depth
        depth2_sample = depth2[None, None, :, :]
        depth2_sample = torch.squeeze(grid_sample_on_img(depth2_sample, xy_proj))
        depth2_proj = xyz_proj[2,:]
        depth2_proj = depth2_proj[valid_mask_index]
        depth2_error = ((depth2_proj - depth2_sample)**2)
        valid_depth_index = depth2_error < 0.001
        xy_proj = xy_proj[:, :, valid_depth_index, :]

        # calculate the loss of the corresponding color
        img1_valid = img1.reshape(h*w, 3).to(self.device)
        img1_valid = img1_valid[valid_mask1]
        img1_valid = img1_valid[valid_mask_index]
        img1_valid = img1_valid[valid_depth_index]
        img2_ = img2.permute(2,0,1).to(self.device())
        img2_ = img2_[None, :]
        img2_sample = torch.squeeze(grid_sample_on_img(img2_, xy_proj)).permute(1, 0)

        # final valid mask in view 1
        mask_final_1 = torch.zeros(valid_depth_index.shape[0], dtype=torch.uint8)
        mask_final_1[valid_depth_index] = 1
        mask_final_2 = torch.zeros(valid_mask_index.shape[0], dtype=torch.uint8)
        mask_final_2[valid_mask_index] = mask_final_1
        mask_final = torch.zeros(valid_mask1.shape[0], dtype=torch.uint8)
        mask_final[valid_mask1] = mask_final_2
        mask_final = mask_final.reshape(h, w)

        color_valid_1, color_valid_2 = torch.zeros_like(img1).to(self.device), torch.zeros_like(img1).to(self.device)
        color_valid_1[mask_final] = img1_valid
        color_valid_2[mask_final] = img2_sample

        self.visualize_error(depth2_error,
                             xyz_proj,
                             valid_mask_index,
                             valid_mask1,
                             valid_mask2,
                             img1, img2,
                             color_valid_1,
                             color_valid_2)

        color_loss = torch.mean(torch.abs(img1_valid - img2_sample))
        return color_loss

if __name__ == '__main__':
    pass

