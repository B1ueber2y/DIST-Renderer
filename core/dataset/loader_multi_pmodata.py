'''
This file is partly adapted from the original PMO repository
[*] https://github.com/chenhsuanlin/photometric-mesh-optim
'''
import numpy as np
import os, sys
import copy
import json
import torch
import trimesh
import torch.utils.data
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.geometry import Camera

class LoaderMultiPMO(torch.utils.data.Dataset):
    def __init__(self, data_dir, class_num, scale=1, num_points=10000, focal=None):
        self.class_num = class_num
        self.sequence_path = os.path.join(data_dir, 'sequences', class_num)
        self.list_file = os.path.join(data_dir, 'lists', 'all_test.list')
        self.camera_file = os.path.join(data_dir, 'camera.npz')
        self.pointcloud_path = os.path.join(data_dir, 'customShapeNet')

        self.seq_list = []
        with open(self.list_file) as file:
            for line in file:
                c,m = line.strip().split()
                if class_num is not None and c!=class_num: continue
                self.seq_list.append((m))

        self.scale = scale
        self.num_points = num_points
        self.focal = focal

    def __len__(self):
        return len(self.seq_list)

    def sample_points_from_ply(self, num_points_all,ply_fname):
        points_str = []
        # fast but dirty data reading...
        # assume points in line are shuffled, and each line is about of length 60
        file_size = os.stat(ply_fname).st_size
        chunk_size = 60*(num_points_all+2)
        with open(ply_fname) as file:
            file.seek(np.random.randint(400,file_size-chunk_size))
            chunk = file.read(chunk_size)
            points_str = chunk.split(os.linesep)[1:num_points_all+1]
        points = [[float(n) for n in s.split()] for s in points_str]
        assert(len(points)==num_points_all)
        points = np.array(points,dtype=np.float32)[:,:3] # ignore normals
        return points

    def resize_image(self, image, scale):
        h, w = image.shape[:2]
        new = np.zeros(image.shape)
        ns_h, ns_w = int(h*scale), int(w*scale)
        if scale < 1:
            new[int(h/2 -ns_h/2):int(h/2 + ns_h/2), int(w/2-ns_w/2):int(w/2 + ns_w/2)] = cv2.resize(image, (ns_h, ns_w))
        else:
            new_img = cv2.resize(image, (ns_h, ns_w))
            h_new, w_new = new_img.shape[:2]
            new = new_img[int(h_new/2 - h/2):int(h_new/2 + h/2), int(w_new/2 - w/2):int(w_new/2 + w/2)]
        return new

    def __getitem__(self, idx):
        instance_name = self.seq_list[idx]
        seq_fname = '{}/{}.npy'.format(self.sequence_path, instance_name)
        RGBA = np.load(seq_fname)
        imgs = RGBA[...,:3]/255.0
        masks = RGBA[...,3]/255.0
        num_img = imgs.shape[0]
        cam = np.load(self.camera_file)

        # get gt point cloud
        ply_fname = "{0}/{1}/ply/{2}.points.ply".format(self.pointcloud_path, self.class_num, self.seq_list[idx])
        points_gt = self.sample_points_from_ply(self.num_points, ply_fname)

        img_list = []
        mask_list = []
        camera_list = []
        for i in range(num_img):
            mask_cur = masks[i]
            mask_cur[mask_cur<1] = 0
            mask_cur = mask_cur.astype(np.bool)
            img_cur = imgs[i]
            cam_cur = Camera(cam['intr'], cam['extr'][i])

            if self.focal is not None:
                img_cur = self.resize_image(img_cur, self.focal)
                mask_cur = self.resize_image(mask_cur.astype(np.float), self.focal)
                mask_cur[mask_cur<1] = 0
                mask_cur = mask_cur.astype(np.bool)
                cam_cur.intrinsic[0, 0] = cam_cur.intrinsic[0, 0]*self.focal
                cam_cur.intrinsic[1, 1] = cam_cur.intrinsic[1, 1]*self.focal

            if self.scale != 1:
                mask_cur = cv2.resize(mask_cur.astype(np.float), None, fx=self.scale, fy=self.scale)
                mask_cur[mask_cur<1] = 0
                mask_cur = mask_cur.astype(np.bool)
                img_cur = cv2.resize(img_cur, None, fx=self.scale, fy=self.scale)
                cam_cur.intrinsic[:2] = cam_cur.intrinsic[:2] * self.scale

            img_list.append(torch.from_numpy(img_cur).float())
            mask_list.append(torch.from_numpy(mask_cur).type(torch.uint8).cuda())
            camera_list.append(cam_cur)
        return instance_name, img_list, mask_list, camera_list, points_gt

