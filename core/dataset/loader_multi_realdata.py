import numpy as np
import os, sys
import copy
import json
import torch
import trimesh
import torch.utils.data
import cv2
import easydict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.geometry import Camera

class LoaderMultiReal(torch.utils.data.Dataset):
    def __init__(self, data_dir, scale=1, refine_sim=False):
        self.data_dir = data_dir
        self.scale = scale

        self.folder_list = [f.name for f in os.scandir(self.data_dir) if f.is_dir()]
        self.refine_sim = refine_sim

    def transform_camera(self, R, t, sim_transform, scale):
        R_new = np.matmul(R, sim_transform[:3, :3])
        t_new = np.matmul(R, sim_transform[:, 3]) + t
        R_new = R_new/scale
        t_new = t_new/scale
        return R_new, t_new

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        instance_name = self.folder_list[idx]
        base_path = os.path.join(self.data_dir, instance_name)
        data_parsed = np.load(os.path.join(base_path, 'parsed.npz'), allow_pickle=True)
        data_all = data_parsed['frames']
        num_img = len(data_all)

        if os.path.exists(os.path.join(base_path, 'picked_frame.npz')):
            f_pick = np.load(os.path.join(base_path, 'picked_frame.npz'))
            picked_frame = f_pick['picked_frame']

        img_list = []
        mask_list = []
        camera_list = []
        sim_transform = np.load(os.path.join(base_path, 'estim_3Dsim.npy'))
        

        # estimate the scale from the provided similarity transform
        scale = np.max(np.linalg.svd(sim_transform[:3, :3])[1])

        for i in range(num_img):
            if (picked_frame is not None) and (i not in picked_frame):
                continue
            else:
                img_cur = cv2.imread(os.path.join(base_path+'/images_clean', data_all[i]['name']))/255
                if self.refine_sim is False:
                    R_cur = data_all[i]['extr'][:3, :3]
                    t_cur = data_all[i]['extr'][:, 3]
                    data_all[i]['extr'][:3, :3], data_all[i]['extr'][:, 3] = self.transform_camera(R_cur, t_cur, sim_transform, scale)
                cam_cur = Camera(data_all[i]['intr'], data_all[i]['extr'])

                mask_cur = np.zeros((img_cur.shape[0], img_cur.shape[1]))
                if self.scale != 1:
                    img_cur = cv2.resize(img_cur, None, fx=self.scale, fy=self.scale)
                    mask_cur = cv2.resize(mask_cur, None, fx=self.scale, fy=self.scale)
                    cam_cur.intrinsic[:2] = cam_cur.intrinsic[:2] * self.scale

                img_list.append(torch.from_numpy(img_cur).float())
                mask_list.append(torch.from_numpy(mask_cur).type(torch.uint8).cuda())
                camera_list.append(cam_cur)

        sim_transform = torch.from_numpy(np.load(os.path.join(base_path, 'estim_3Dsim.npy'))).float().cuda()
        return instance_name, img_list, mask_list, camera_list, sim_transform

