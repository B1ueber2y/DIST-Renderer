import os, sys
import cv2
import json
import pickle
import torch
import torch.utils.data
from tqdm import tqdm
from random import randint
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from synthesis import *
from common.geometry import *
from common.utils.io_utils import read_exr

class SynShapenetAmbientLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, split_file, load_normal=False):
        self.data_dir = data_dir
        self.home_dir = os.path.join(data_dir, '..', '..', '..')

        with open(split_file, 'r') as f:
            split = json.load(f)

        key_list = list(split.keys())
        assert(len(key_list) == 1)
        self.dataset = key_list[0]

        data = split[self.dataset]
        key_list = list(data.keys())
        assert(len(key_list) == 1)
        self.class_name = key_list[0]

        self.instance_list = split[self.dataset][self.class_name]
        self.load_normal = load_normal

    def __len__(self):
        return len(self.instance_list)

    def get_instance_name(self, idx):
        return self.instance_list[idx]

    def get_camera(self, param):
        camera = param['render_param']['camera']
        return camera

    def read_depth(self, fname_depth):
        data, keys = read_exr(fname_depth)
        depth = data[0]
        return depth

    def read_normal(self, fname_normal):
        data, keys = read_exr(fname_normal)
        channels = []
        channels.append(data[2]) # 'R' -> 'X'
        channels.append(data[1]) # 'G' -> 'Y'
        channels.append(data[0]) # 'B' -> 'Z'
        normal = np.array(channels).transpose(1,2,0) # (H, W, 3)
        return normal

    def get_data(self, instance_name):
        # idx = randint(0,23)
        idx = 0
        # load img
        fname_img = os.path.join(self.data_dir, self.class_name, instance_name, ('{0}_{1}_{2}.png'.format(self.class_name, instance_name, idx)))
        img = cv2.imread(fname_img)

        # load depth
        fname_depth = fname_img[:-4] + '_depth.exr'
        depth = self.read_depth(fname_depth)

        # load camera
        fname_param = fname_img[:-4] + '_param.pkl'
        with open(fname_param, 'rb') as f:
            param = pickle.load(f, encoding='latin1')
        camera = self.get_camera(param)

        if self.load_normal:
            fname_normal = fname_img[:-4] + '_normal.exr'
            normal = self.read_normal(fname_normal)
            return img, depth, normal, camera
        else:
            return img, depth, camera

    def __getitem__(self, idx):
        return self.get_data(self.instance_list[idx])

if __name__ == '__main__':
    folder_name = 'mytest'
    data_dir = os.path.join(basepath, 'synthesis', 'output', folder_name)
    dataset = SynShapenetLoader(data_dir)
    output = dataset[4]


