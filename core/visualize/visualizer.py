import os, sys
import time
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vis_utils import get_vis_depth, get_vis_mask, get_vis_normal
import copy
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as pil
import pickle

def print_loss_pack(loss_pack, name):
    loss_depth, loss_mask_gt, loss_mask_out, loss_normal, loss_l2reg = loss_pack['depth'], loss_pack['mask_gt'], loss_pack['mask_out'], loss_pack['normal'], loss_pack['l2reg']
    if len(loss_depth.shape) == 1:
        loss_mask_gt, loss_mask_out, loss_depth, loss_normal, loss_l2reg = loss_mask_gt.mean(), loss_mask_out.mean(), loss_depth.mean(), loss_normal.mean(), loss_l2reg.mean()
    print('NAME = [{0}] -- loss_depth: {1:.4f}, loss_mask_gt: {2:.4f}, loss_mask_out: {3:.4f}, loss_normal: {4:.4f}, loss_l2reg: {5:.4f}'.format(name, loss_depth.detach().cpu().numpy(), loss_mask_gt.detach().cpu().numpy(), loss_mask_out.detach().cpu().numpy(), loss_normal.detach().cpu().numpy(), loss_l2reg.detach().cpu().numpy()))

def print_loss_pack_color(loss_pack, name):
    loss_color, loss_depth, loss_mask_gt, loss_mask_out, loss_normal, loss_l2reg, loss_l2reg_c = loss_pack['color'], loss_pack['depth'], loss_pack['mask_gt'], loss_pack['mask_out'], loss_pack['normal'], loss_pack['l2reg'], loss_pack['l2reg_c']
    print('NAME = [{0}] -- loss_color: {1:.4f}, loss_depth: {2:.4f}, loss_mask_gt: {3:.4f}, loss_mask_out: {4:.4f}, loss_normal: {5:.4f}, loss_l2reg: {6:.4f}, loss_l2re_cg: {7:.4f}'.format(name, loss_color.detach().cpu().numpy(), loss_depth.detach().cpu().numpy(), loss_mask_gt.detach().cpu().numpy(), loss_mask_out.detach().cpu().numpy(), loss_normal.detach().cpu().numpy(), loss_l2reg.detach().cpu().numpy(), loss_l2reg_c.detach().cpu().numpy()))


def demo_color_save_render_output(prefix, sdf_renderer, shape_code, color_code, camera, lighting_loc=None, profile=False):
    R, T = camera.extrinsic[:,:3], camera.extrinsic[:,3]
    R, T = torch.from_numpy(R).float().cuda(), torch.from_numpy(T).float().cuda()
    R.requires_grad, T.requires_grad = False, False

    if lighting_loc is not None:
        lighting_locations = torch.from_numpy(lighting_loc).float().unsqueeze(0).cuda()
    else:
        lighting_locations = None
    render_output = sdf_renderer.render(color_code, shape_code, R, T, profile=profile, no_grad=True, lighting_locations=lighting_locations)
    depth_rendered, normal_rendered, color_rgb, valid_mask_rendered, min_sdf_sample = render_output

    data = {}
    data['depth'] = depth_rendered.detach().cpu().numpy()
    data['normal'] = normal_rendered.detach().cpu().numpy()
    data['mask'] = valid_mask_rendered.detach().cpu().numpy()
    data['color'] = color_rgb.detach().cpu().numpy()
    data['min_sdf_sample'] = min_sdf_sample.detach().cpu().numpy()
    data['latent_tensor'] = shape_code.detach().cpu().numpy()
    data['K'] = sdf_renderer.get_intrinsic()
    data['RT'] = torch.cat([R, T[:,None]], 1).detach().cpu().numpy()
    fname = prefix + '_info.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    img_hw = sdf_renderer.get_img_hw()
    visualizer = Visualizer(img_hw)
    print('Writing to prefix: {}'.format(prefix))
    visualizer.visualize_depth(prefix + '_depth.png', depth_rendered.detach().cpu().numpy(), valid_mask_rendered.detach().cpu().numpy())
    visualizer.visualize_normal(prefix + '_normal.png', normal_rendered.detach().cpu().numpy(), valid_mask_rendered.detach().cpu().numpy(), bgr2rgb=True)
    visualizer.visualize_mask(prefix + '_silhouette.png', valid_mask_rendered.detach().cpu().numpy())
    cv2.imwrite(prefix + '_rendered_rgb.png', color_rgb.detach().cpu().numpy() * 255)


class Visualizer(object):
    def __init__(self, img_hw, dmin=0.0, dmax=10.0):
        self.img_h, self.img_w = img_hw[0], img_hw[1]
        self.data = {}
        self.dmin, self.dmax = dmin, dmax

        self.loss_counter = 0
        self.loss_curve = {}
        self.loss_list = []
        self.chamfer_list = []

    def get_data(self, data_name):
        if data_name in self.data.keys():
            return self.data[data_name]
        else:
            raise ValueError('Key {0} does not exist.'.format(data_name))

    def set_data(self, data):
        self.data = data

    def reset_data(self):
        self.data = {}
        keys = ['mask_gt', 'mask_output', 'loss_mask_gt', 'loss_mask_out',
                'depth_gt', 'depth_output', 'loss_depth',
                'normal_gt', 'normal_output', 'loss_normal']
        for key in keys:
            self.data[key] = np.zeros((64, 64))

    def reset_loss_curve(self):
        self.loss_counter = 0
        self.loss_curve = {}

    def reset_all(self):
        self.reset_data()
        self.reset_loss_curve()

    def add_loss_from_pack(self, loss_pack):
        '''
        potential properties:
        ['mask_gt', 'mask_out', 'depth' 'normal', 'l2reg']
        '''
        loss_name_list = list(loss_pack.keys())
        if self.loss_curve == {}:
            for loss_name in loss_name_list:
                self.loss_curve[loss_name] = []
        for loss_name in loss_name_list:
            loss_value = loss_pack[loss_name].detach().cpu().numpy()
            self.loss_curve[loss_name].append(loss_value)
        self.loss_counter = self.loss_counter + 1

    def add_loss(self, loss):
        self.loss_list.append(loss.detach().cpu().numpy())

    def add_chamfer(self, chamfer):
        self.chamfer_list.append(chamfer)

    def add_data(self, data_name, data_src, data_mask=None):
        '''
        potential properties:
        mask: 	['mask_gt', 'mask_output', 'loss_mask_gt', 'loss_mask_out']
        depth: 	['depth_gt', 'depth_output', 'loss_depth']
        normal:	['normal_gt', 'normal_output', 'loss_normal']
        '''
        if data_mask is None:
            self.data[data_name] = data_src
        else:
            data_map = np.zeros(data_mask.shape)
            data_map[data_mask != 0] = data_src
            self.data[data_name] = data_map

    def save_depth(self, fname, depth_vis, cmap='magma', direct=False):
        if direct:
            cv2.imwrite(fname, depth_vis)
            return 0
        vmin, vmax = 0, 255
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        colormapped_im = (mapper.to_rgba(depth_vis)[:,:,:3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        im.save(fname)

    def save_mask(self, fname, mask_vis, bgr2rgb=False):
        if bgr2rgb:
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fname, mask_vis)

    def save_normal(self, fname, normal_vis, bgr2rgb=False):
        if bgr2rgb:
            normal_vis = cv2.cvtColor(normal_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fname, normal_vis)

    def save_error(self, fname, error_vis, bgr2rgb=False):
        self.save_depth(fname, error_vis, cmap='jet')

    def visualize_depth(self, fname, depth, mask=None):
        # depth_vis = get_vis_depth(depth, mask=mask, dmin=self.dmin, dmax=self.dmax)
        depth_vis = get_vis_depth(depth, mask=mask)
        # self.save_depth(fname, depth_vis)
        cv2.imwrite(fname, depth_vis)

    def visualize_normal(self, fname, normal, mask=None, bgr2rgb=False):
        normal_vis = get_vis_normal(normal, mask=mask)
        if bgr2rgb:
            normal_vis = cv2.cvtColor(normal_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fname, normal_vis)

    def visualize_mask(self, fname, mask, bgr2rgb=False):
        mask_vis = get_vis_mask(mask)
        if bgr2rgb:
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fname, mask_vis)

    def imshow(self, ax, img, title=None):
        ax.imshow(img)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)

    def imshow_bgr2rgb(self, ax, img, title=None):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)

    def show_loss_curve(self, fname):
        pass

    def show_all_data_3x4(self, fname):
        fig, axs = plt.subplots(3, 4, figsize=(30,30))

        # first row, groundtruth
        depth_gt_vis = get_vis_depth(self.data['depth_gt'], mask=self.data['mask_gt'], dmin=self.dmin, dmax=self.dmax)
        self.imshow_bgr2rgb(axs[0, 0], 255 - depth_gt_vis, title='depth gt')
        normal_gt_vis = get_vis_normal(self.data['normal_gt'], mask=self.data['mask_gt'])
        self.imshow(axs[0, 1], normal_gt_vis, title='normal gt')
        mask_gt_vis = get_vis_mask(self.data['mask_gt'])
        self.imshow_bgr2rgb(axs[0, 2], 255 - mask_gt_vis, title='mask gt')
        axs[0, 3].axis('off')

        # second row, output
        depth_output_vis = get_vis_depth(self.data['depth_output'], mask=self.data['mask_output'], dmin=self.dmin, dmax=self.dmax)
        self.imshow_bgr2rgb(axs[1, 0], 255 - depth_output_vis, title='depth output')
        normal_output_vis = get_vis_normal(self.data['normal_output'], mask=self.data['mask_output'])
        self.imshow(axs[1, 1], normal_output_vis, title='normal output')
        mask_output_vis = get_vis_mask(self.data['mask_output'])
        self.imshow_bgr2rgb(axs[1, 2], 255 - mask_output_vis, title='mask output')
        axs[1, 3].axis('off')

        # third row, loss
        valid_mask = np.logical_and(self.data['mask_gt'], self.data['mask_output'])
        loss_depth_vis = get_vis_depth(np.abs(self.data['loss_depth']), valid_mask, dmin=0.0, dmax=0.5)
        self.imshow_bgr2rgb(axs[2, 0], 255 - loss_depth_vis, title='depth loss')
        loss_normal_vis = get_vis_depth(self.data['loss_normal'], valid_mask, dmin=-1.0, dmax=0.0)
        self.imshow_bgr2rgb(axs[2, 1], 255 - loss_normal_vis, title='normal loss')
        loss_mask_gt_vis = get_vis_mask(np.abs(self.data['loss_mask_gt']) > 0)
        self.imshow_bgr2rgb(axs[2, 2], 255 - loss_mask_gt_vis, title='gt \ output')
        loss_mask_out_vis = get_vis_mask(np.abs(self.data['loss_mask_out']) > 0)
        self.imshow_bgr2rgb(axs[2, 3], 255 - loss_mask_out_vis, title='output \ gt')

        # savefig
        fig.savefig(fname)
        plt.close('all')

    def save_all_data(self, prefix):
        # groundtruth
        depth_gt_vis = get_vis_depth(self.data['depth_gt'], mask=self.data['mask_gt'], dmin=self.dmin, dmax=self.dmax)
        self.save_depth(prefix + '_depth_gt.png', depth_gt_vis, cmap='magma', direct=True)
        normal_gt_vis = get_vis_normal(self.data['normal_gt'], mask=self.data['mask_gt'])
        self.save_normal(prefix + '_normal_gt.png', normal_gt_vis, bgr2rgb=True)
        mask_gt_vis = get_vis_mask(self.data['mask_gt'])
        self.save_mask(prefix + '_mask_gt.png', mask_gt_vis)

        # output
        depth_output_vis = get_vis_depth(self.data['depth_output'], mask=self.data['mask_output'], dmin=self.dmin, dmax=self.dmax)
        self.save_depth(prefix + '_depth_output.png', depth_output_vis, cmap='magma', direct=True)
        normal_output_vis = get_vis_normal(self.data['normal_output'], mask=self.data['mask_output'])
        self.save_normal(prefix + '_normal_output.png', normal_output_vis, bgr2rgb=True)
        mask_output_vis = get_vis_mask(self.data['mask_output'])
        self.save_mask(prefix + '_mask_output.png', mask_output_vis)

        # third row, loss
        valid_mask = np.logical_and(self.data['mask_gt'], self.data['mask_output'])
        loss_depth_vis = get_vis_depth(np.abs(self.data['loss_depth']), valid_mask, dmin=0.0, dmax=0.5, bg_color=0)
        self.save_error(prefix + '_depth_loss.png', loss_depth_vis, bgr2rgb=True)
        loss_normal_vis = get_vis_depth(self.data['loss_normal'], valid_mask, dmin=-1.0, dmax=0.0, bg_color=0)
        self.save_error(prefix + '_normal_loss.png', loss_normal_vis, bgr2rgb=True)

        loss_mask_gt_vis = get_vis_depth(np.abs(self.data['loss_mask_gt']), bg_color=0)
        self.save_error(prefix + '_mask_gt_loss.png', loss_mask_gt_vis, bgr2rgb=True)
        loss_mask_out_vis = get_vis_depth(np.abs(self.data['loss_mask_out']), bg_color=0)
        self.save_error(prefix + '_mask_out_loss.png', loss_mask_out_vis, bgr2rgb=True)
        self.save_error(prefix + '_mask_loss.png', loss_mask_gt_vis + loss_mask_out_vis, bgr2rgb=True)

    def dump_all_data(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump({'data': self.data, 'loss_curve': self.loss_curve, 'loss_list': self.loss_list, 'chamfer_list': self.chamfer_list}, f)

    def show_all_data(self, fname):
        self.show_all_data_3x4(fname)
        # self.save_all_data(fname[:-4])

    def show_all_data_color(self, fname):
        fig, axs = plt.subplots(3, 4, figsize=(30,30))

        # first row, groundtruth
        depth_gt_vis = get_vis_depth(self.data['depth_gt'], mask=self.data['mask_gt'], dmin=self.dmin, dmax=self.dmax)
        self.imshow_bgr2rgb(axs[0, 0], depth_gt_vis, title='depth gt')
        normal_gt_vis = get_vis_normal(self.data['normal_gt'])
        self.imshow_bgr2rgb(axs[0, 1], normal_gt_vis, title='normal gt')
        mask_gt_vis = get_vis_mask(self.data['mask_gt'])
        self.imshow_bgr2rgb(axs[0, 2], mask_gt_vis, title='mask gt')
        self.imshow_bgr2rgb(axs[0, 3], self.data['color_gt'], title='rgb gt')

        # second row, output
        depth_output_vis = get_vis_depth(self.data['depth_output'], mask=self.data['mask_output'], dmin=self.dmin, dmax=self.dmax)
        self.imshow_bgr2rgb(axs[1, 0], depth_output_vis, title='depth output')
        normal_output_vis = get_vis_normal(self.data['normal_output'])
        self.imshow_bgr2rgb(axs[1, 1], normal_output_vis, title='normal output')
        mask_output_vis = get_vis_mask(self.data['mask_output'])
        self.imshow_bgr2rgb(axs[1, 2], mask_output_vis, title='mask output')
        self.imshow_bgr2rgb(axs[1, 3], self.data['color_output'], title='rgb output')

        # third row, loss
        valid_mask = np.logical_and(self.data['mask_gt'], self.data['mask_output'])
        loss_depth_vis = get_vis_depth(np.abs(self.data['loss_depth']), valid_mask, dmin=0.0, dmax=0.5)
        self.imshow_bgr2rgb(axs[2, 0], loss_depth_vis, title='depth loss')
        loss_normal_vis = get_vis_depth(self.data['loss_normal'], valid_mask, dmin=-1.0, dmax=0.0)
        self.imshow_bgr2rgb(axs[2, 1], loss_normal_vis, title='normal loss')
        loss_mask_gt_vis = get_vis_mask(np.abs(self.data['loss_mask_gt']) > 0)
        loss_mask_out_vis = get_vis_mask(np.abs(self.data['loss_mask_out']) > 0)
        loss_mask_gt_vis += loss_mask_out_vis
        self.imshow_bgr2rgb(axs[2, 2], loss_mask_gt_vis, title='mask loss')
        self.imshow_bgr2rgb(axs[2, 3], self.data['loss_color'], title='rgb loss')

        # savefig
        fig.savefig(fname)
        plt.close('all')

    def return_output_data_color(self):
        return self.data['color_output'], self.data['depth_output'], self.data['normal_output'], self.data['mask_output']

    def show_all_data_color_multi(self, fname, num_img=4):
        fig, axs = plt.subplots(3, 2*num_img, figsize=(8*2*num_img,25))

        for i in range(num_img):
            # first row, ground truth
            self.imshow_bgr2rgb(axs[0, 2*i], self.data['color_gt-{}'.format(i)], title='rgb gt {}'.format(i))
            mask_gt_vis = get_vis_mask(self.data['mask_gt-{}'.format(i)])
            self.imshow_bgr2rgb(axs[0, 2*i+1], mask_gt_vis, title='mask gt {}'.format(i))

            # second row, output
            self.imshow_bgr2rgb(axs[1, 2*i], self.data['color_output-{}'.format(i)], title='rgb output {}'.format(i))
            mask_output_vis = get_vis_mask(self.data['mask_output-{}'.format(i)])
            self.imshow_bgr2rgb(axs[1, 2*i+1], mask_output_vis, title='mask output {}'.format(i))

            # third row, loss
            self.imshow_bgr2rgb(axs[2, 2*i], self.data['loss_color-{}'.format(i)], title='rgb loss {}'.format(i))
            loss_mask_gt_vis = get_vis_mask(np.abs(self.data['loss_mask_gt-{}'.format(i)]) > 0)
            loss_mask_out_vis = get_vis_mask(np.abs(self.data['loss_mask_out-{}'.format(i)]) > 0)
            loss_mask_gt_vis += loss_mask_out_vis
            self.imshow_bgr2rgb(axs[2, 2*i+1], loss_mask_gt_vis, title='mask loss {}'.format(i))

        # savefig
        plt.subplots_adjust(top=0.95, right=0.99, left=0.01, bottom=0.01, wspace=0.05, hspace=0.1)
        fig.savefig(fname)
        plt.close('all')

    def show_all_data_color_warp(self, fname):
        fig, axs = plt.subplots(1, 5, figsize=(15, 3.4))
        self.imshow_bgr2rgb(axs[0], self.data['color_gt-1'], title='view 1')
        self.imshow_bgr2rgb(axs[1], self.data['color_gt-2'], title='view 2')
        self.imshow_bgr2rgb(axs[2], self.data['color_valid-1'], title='valid region in view 1')
        self.imshow_bgr2rgb(axs[3], self.data['color_valid-2'], title='warped color from view 2')
        self.imshow_bgr2rgb(axs[4], self.data['color_valid_loss'], title='color loss')

        # savefig
        plt.subplots_adjust(top=0.99, right=0.99, left=0.01, bottom=0.00, wspace=0.05, hspace=0)
        fig.savefig(fname)
        plt.close('all')

