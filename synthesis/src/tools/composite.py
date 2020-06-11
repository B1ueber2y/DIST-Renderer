import os, sys
import numpy as np
from tqdm import tqdm
import datetime
import cv2
import torch
import torch.utils.data
import pickle

class CompositeProcessor(torch.utils.data.Dataset):
    def __init__(self, bg_image_list, composite_param_list, cluttered_bg_ratio, skip_exist=True):
        self.bg_image_list = bg_image_list
        self.bg_image_num = len(self.bg_image_list)
        self.composite_param_list = composite_param_list
        self.cluttered_bg_ratio = cluttered_bg_ratio
        self.skip_exist = skip_exist

    def __len__(self):
        return len(self.composite_param_list)

    def crop_bg(self, fg_hw, bg_hw, cropbg_param):
        fg_h, fg_w = fg_hw
        bg_h, bg_w = bg_hw
        if bg_h < fg_h or bg_w < fg_w:
            return None
        y_start = int(round(float(bg_h - fg_h) * cropbg_param[0]))
        x_start = int(round(float(bg_w - fg_w) * cropbg_param[1]))
        y_end, x_end = y_start + fg_h, x_start + fg_w
        return [y_start, y_end, x_start, x_end]

    def add_composite_param(self, cropbg_param, bbox, bg_hw, fname_in, fname_out, path_to_bg_image):
        '''
        We just save the compositing parameters here.
        '''
        with open(fname_in, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        data['composite_param'] = {}
        data['composite_param']['cropbg_param'] = cropbg_param
        data['composite_param']['initial_bg_shape'] = bg_hw
        data['composite_param']['bbox'] = bbox
        data['composite_param']['path_to_bg_image'] = path_to_bg_image
        with open(fname_out, 'wb') as f:
            pickle.dump(data, f)

    def composite(self, composite_param):
        cropbg_param = composite_param.cropbg_param
        input_cfg = composite_param.input_cfg
        target_cfg = composite_param.target_cfg

        # initialize seed
        np.random.seed(composite_param.seed)

        # initialize directory
        if not os.path.exists(input_cfg.image):
            return
        if self.skip_exist and (os.path.exists(target_cfg.image) and os.path.exists(target_cfg.param)):
            return
        if os.path.getsize(input_cfg.image) == 0:
            return

        # read foreground image
        raw = cv2.imread(input_cfg.image, cv2.IMREAD_UNCHANGED)
        if type(raw) == type(None):
            return
        img_fg, alpha = raw[:,:,:3], raw[:,:,3]
        fg_height, fg_width, _ = img_fg.shape

        # composite
        mask = alpha.astype(float) / 255.0
        mask = np.tile(mask[:,:,None], (1,1,3))
        if np.random.rand() > self.cluttered_bg_ratio:
            # read background image
            while True:
                idx = np.random.randint(self.bg_image_num)
                path_to_bg_image = self.bg_image_list[idx]
                img_bg = cv2.imread(path_to_bg_image)
                bg_height, bg_width, _ = img_bg.shape
                bbox = self.crop_bg((fg_height, fg_width), (bg_height, bg_width), cropbg_param)
                if bbox != None:
                    break
            top, bottom, left, right = bbox
            cropped_bg = img_bg[top:bottom, left:right, :]

            png_new = img_fg.astype(float) * mask + cropped_bg.astype(float) * (1 - mask) 
        else:
            bbox = [-1, -1, -1, -1]
            bg_height, bg_width = -1, -1
            path_to_bg_image = 'default'
            png_new = img_fg.astype(float) * mask + np.random.rand() * 255 * (1 - mask)
        png_new = png_new.astype(np.uint8)
        cv2.imwrite(target_cfg.image, png_new)

        # param
        self.add_composite_param(cropbg_param, bbox, (bg_height, bg_width), input_cfg.param, target_cfg.param, path_to_bg_image)

    def __getitem__(self, idx):
        composite_param = self.composite_param_list[idx]
        self.composite(composite_param)
        return 0

class Compositor(object):
    def __init__(self, num_worker=0, cluttered_bg_ratio=0.2):
        self.num_worker = num_worker
        self.cluttered_bg_ratio = cluttered_bg_ratio

    def composite_all(self, composite_param_list, bg_image_list, skip_exist=True):
        processor = CompositeProcessor(bg_image_list, composite_param_list, self.cluttered_bg_ratio, skip_exist=skip_exist)
        dataloader = torch.utils.data.DataLoader(processor, batch_size=1, shuffle=False, num_workers=self.num_worker, drop_last=False)
        print('Start compositing at time {0}...it takes for a while!!'.format(datetime.datetime.now()))
        for i, _ in enumerate(tqdm(dataloader)):
            pass

