import os, sys
import numpy as np
from tqdm import tqdm
import datetime
import cv2
import torch
import torch.utils.data
import pickle

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.mystruct import FnameGroup, CropParam
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from common.utils.io_utils import read_from_stat_file, read_exr, write_exr, get_image_file_list

class CroppingProcessor(torch.utils.data.Dataset):
    def __init__(self, crop_param_list, skip_exist=True):
        self.crop_param_list = crop_param_list
        self.skip_exist = skip_exist

    def __len__(self):
        return len(self.crop_param_list)

    def crop_gray(self, alpha, truncParam, img_hw, is_crop):
        img_h, img_w = img_hw[0], img_hw[1]
        tp = truncParam

        valid = np.where(alpha != 0)
        top, bottom = valid[0].min(), valid[0].max() + 1
        left, right = valid[1].min(), valid[1].max() + 1

        width, height = right - left, bottom - top
        dx1, dx2 = width * tp[0], width * tp[1] 
        dy1, dy2 = height * tp[2], height * tp[3] 

        if not is_crop:
            bbox = [top, bottom, left, right]
            return None, bbox

        left_new = np.clip(left + dx1, 0, img_w - 1)
        right_new = np.clip(right + dx2, 0, img_w - 1) 
        if left_new > right_new:
            left_new, right_new = left, right
        top_new = np.clip(top + dy1, 0, img_h - 1)
        bottom_new = np.clip(bottom + dy2, 0, img_h - 1)
        if top_new > bottom_new:
            top_new, bottom_new = top, bottom
        top_new, bottom_new, left_new, right_new = np.round(top_new).astype(int), np.round(bottom_new).astype(int), np.round(left_new).astype(int), np.round(right_new).astype(int)
        alpha_new = alpha[top_new:bottom_new, left_new:right_new] 
        bbox = [top_new, bottom_new, left_new, right_new]
        return alpha_new, bbox

    def crop_exr(self, bbox, fname_in, fname_out):
        top, bottom, left, right = bbox
        channels, keys = read_exr(fname_in)
        channels_new = [channel[top:bottom, left:right] for channel in channels]
        write_exr(fname_out, channels_new, keys)

    def add_crop_param(self, truncparam, fname_in, fname_out, bbox, initial_shape, is_crop=True):
        '''
        We just save the cropping parameters here.
        '''
        with open(fname_in, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        data['crop_param'] = {}
        data['crop_param']['is_crop'] = is_crop
        data['crop_param']['truncparam'] = truncparam
        data['crop_param']['initial_shape'] = initial_shape
        data['crop_param']['bbox'] = bbox
        with open(fname_out, 'wb') as f:
            pickle.dump(data, f)

    def crop(self, crop_param):
        truncparam = crop_param.truncparam
        input_cfg = crop_param.input_cfg
        target_cfg = crop_param.target_cfg
        is_crop = crop_param.is_crop
        # initialize directory
        # if not (os.path.exists(input_cfg.image) and os.path.exists(input_cfg.depth) and input_cfg.normal and os.path.exists(input_cfg.albedo) and os.path.exists(input_cfg.param)):
        #     return
        if self.skip_exist and (os.path.exists(target_cfg.image) and os.path.join(target_cfg.param)):
            return
        if os.path.getsize(input_cfg.image) == 0:
            return

        # read image
        raw = cv2.imread(input_cfg.image, cv2.IMREAD_UNCHANGED)
        if type(raw) == type(None):
            return

        raw = (raw.astype(float) / 256.0).astype(np.uint8) # 16bit -> 8bit
        img, alpha = raw[:,:,:3], raw[:,:,3]
        img_height, img_width, _ = img.shape

        alpha_new, bbox = self.crop_gray(alpha, truncparam, [img_height, img_width], is_crop=is_crop)
        if not is_crop:
            cv2.imwrite(target_cfg.image, raw)
            self.add_crop_param(truncparam, input_cfg.param, target_cfg.param, bbox, [img_height, img_width], is_crop=False)
            return

        # crop image
        top, bottom, left, right = bbox
        img_new = img[top:bottom, left:right, :]
        png_new = np.concatenate([img_new, alpha_new[:,:,None]], axis=2)

        # save image
        cv2.imwrite(target_cfg.image, png_new)

        # other properties
        # self.crop_exr(bbox, input_cfg.depth, target_cfg.depth)
        # self.crop_exr(bbox, input_cfg.normal, target_cfg.normal)
        # self.crop_exr(bbox, input_cfg.albedo, target_cfg.albedo)

        # param
        self.add_crop_param(truncparam, input_cfg.param, target_cfg.param, bbox, [img_height, img_width], is_crop=True)

    def __getitem__(self, idx):
        crop_param = self.crop_param_list[idx]
        self.crop(crop_param)
        return 0

class Cropper(object):
    def __init__(self, num_worker=0):
        self.num_worker = num_worker

    def crop_all(self, crop_param_list, skip_exist=True):
        processor = CroppingProcessor(crop_param_list, skip_exist=skip_exist)
        dataloader = torch.utils.data.DataLoader(processor, batch_size=1, shuffle=False, num_workers=self.num_worker, drop_last=False)
        print('Start cropping at time {0}...it takes for a while!!'.format(datetime.datetime.now()))
        for i, _ in enumerate(tqdm(dataloader)):
            pass

if __name__ == '__main__':
    truncparam = [0.0, 0.0, 0.0, 0.0]
    input_cfg = FnameGroup('/nas/shaoliu/Downloads/test_rendering/9cd0529b7ce926017dbe6b236c44c533_3.png')
    output_cfg = FnameGroup('/nas/shaoliu/Downloads/test.png')
    crop_param = CropParam(truncparam, input_cfg, output_cfg)

    cropper = Cropper()
    cropper.crop_all([crop_param])


