import os, sys
import cv2
import pickle
import torch
import torch.utils.data
import json
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from synthesis import *
from common.geometry import *
from common.utils.io_utils import read_exr

def get_image_file_list_mp(src_folder, flist, queue):
    image_list = []
    for fname in tqdm(flist):
        nowpath = os.path.join(src_folder, fname)
        if os.path.isfile(nowpath) and fname[-4:] == '.png':
            image_list.append(fname)
        if not os.path.isdir(nowpath):
            continue
        sub_flist = os.listdir(nowpath)
        for sub_fname in sub_flist:
            if sub_fname[-4:] != '.png':
                continue
            image_list.append(os.path.join(fname, sub_fname))
    queue.put(image_list)
    return

class SynShapenetLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, split_file=None, load_depth=True, load_normal=True, load_albedo=False, num_worker=8):
        self.data_dir = data_dir
        self.home_dir = os.path.join(data_dir, '..', '..', '..')
        self.split_file = split_file
        self.instance_list = None
        if self.split_file is not None:
            with open(self.split_file) as f:
                data = json.load(f)
            data = data[list(data.keys())[0]]
            data = data[list(data.keys())[0]]
            self.instance_list = data

        self.load_depth = load_depth
        self.load_normal = load_normal
        self.load_albedo = load_albedo
        self.num_worker = num_worker

        self.image_dir = os.path.join(self.data_dir, 'final')
        self.image_list = self.get_image_file_list(self.image_dir)

    def __len__(self):
        return len(self.image_list)

    def check_if_in_instance_list(self, instance_name):
        if self.instance_list is None:
            return True
        else:
            return (instance_name in self.instance_list)

    def split_list(self, flist, num_worker):
        split = []
        num_total = len(flist)
        num_per_process = int(np.floor(num_total / num_worker)) + 1
        start = 0
        for idx in range(num_worker):
            end = np.minimum(start + num_per_process, num_total)
            split.append(flist[start:end])
            start = end
        return split

    def get_image_file_list_mp(self, src_folder):
        '''
        get list of image files.
        '''
        import torch.multiprocessing as mp
        flist = os.listdir(src_folder)

        split = self.split_list(flist, self.num_worker)
        queue = mp.Queue()
        processes = []
        for idx in range(self.num_worker):
            p = mp.Process(target=get_image_file_list_mp, args=(src_folder, split[idx], queue))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        image_list = []
        for idx in range(self.num_worker):
            image_list_sub = queue.get()
            image_list.extend(image_list_sub)

        print('{0} training images in total.'.format(len(image_list)))
        return image_list

    def get_image_file_list(self, src_folder):
        '''
        get list of image files.
        '''
        image_list = []
        flist = os.listdir(src_folder)
        print('traversing the folder.')
        for fname in flist:
            nowpath = os.path.join(src_folder, fname)
            if os.path.isfile(nowpath):
                if fname[-4:] == '.png':
                    image_list.append(fname)
                continue
            if not os.path.isdir(nowpath):
                continue
            if not self.check_if_in_instance_list(fname):
                continue
            sub_flist = os.listdir(nowpath)
            for sub_fname in sub_flist:
                if sub_fname[-4:] != '.png':
                    continue
                image_list.append(os.path.join(fname, sub_fname))
        image_list.sort()
        print('{0} training images in total.'.format(len(image_list)))
        return image_list

    def read_depth(self, fname_depth, initial_shape):
        data, keys = read_exr(fname_depth)
        depth = data[0]

        assert(list(depth.shape[:2]) == initial_shape)
        return depth

    def read_normal(self, fname_normal, initial_shape):
        data, keys = read_exr(fname_normal)
        channels = []
        channels.append(data[2]) # 'R' -> 'X'
        channels.append(data[1]) # 'G' -> 'Y'
        channels.append(data[0]) # 'B' -> 'Z'
        normal = np.array(channels).transpose(1,2,0) # (H, W, 3)

        assert(list(normal.shape[:2]) == initial_shape)
        return normal

    def read_albedo(self, fname_albedo, initial_shape):
        data, keys = read_exr(fname_albedo)
        channels = []
        channels.append(data[1]) # 'B'
        channels.append(data[2]) # 'G'
        channels.append(data[3]) # 'R'
        albedo = np.array(channels).transpose(1,2,0) # (H, W, 3)
        albedo = (albedo * 255.0).astype(np.uint8)

        assert(list(albedo.shape[:2]) == initial_shape)
        return albedo

    def get_camera(self, param):
        camera = param['render_param']['camera']
        return camera

    def get_data_from_md5(self, shape_md5, idx):
        fname_img = '{0}_{1}.png'.format(shape_md5, idx)
        return self.get_data(fname_img)

    def get_data(self, fname_png):
        fname_img = os.path.join(self.image_dir, fname_png)
        fname_param = fname_img[:-4] + '_param.pkl'
        img = cv2.imread(fname_img)

        with open(fname_param, 'rb') as f:
            param = pickle.load(f, encoding='latin1')
        initial_shape = param['crop_param']['initial_shape']
        bbox = param['crop_param']['bbox']

        top, bottom, left, right = bbox
        if self.load_depth:
            fname_depth = fname_img[:-4] + '_depth.exr'
            fname_depth = fname_depth.replace('/final/', '/rendering/')
            depth = self.read_depth(fname_depth, initial_shape)
            if param['crop_param']['is_crop']:
                depth = depth[top:bottom, left:right]
        if self.load_normal:
            fname_normal = fname_img[:-4] + '_normal.exr'
            fname_normal = fname_normal.replace('/final/', '/rendering/')
            normal = self.read_normal(fname_normal, initial_shape)
            if param['crop_param']['is_crop']:
                normal = normal[top:bottom, left:right]
        if self.load_albedo:
            fname_albedo = fname_img[:-4] + '_albedo.exr'
            fname_albedo = fname_albedo.replace('/final/', '/rendering/')
            albedo = self.read_albedo(fname_albedo, initial_shape)
            if param['crop_param']['is_crop']:
                albedo = albedo[top:bottom, left:right]

        camera = self.get_camera(param)
        # normalize the surface normal
        length_normal = np.sqrt((normal * normal).sum(2))
        normal = normal / (length_normal[:,:,None] + 1e-12)
        if self.load_albedo:
            return img, depth, normal, camera, albedo
        else:
            return img, depth, normal, camera

    def get_instance_name_from_fname_png(self, fname_png):
        if len(fname_png.split('/')) == 1:
            instance_name = fname_png.split('_')[0]
        else:
            instance_name = fname_png.split('/')[0]
        return instance_name

    def get_instance_name(self, idx):
        fname_png = self.image_list[idx]
        return self.get_instance_name_from_fname_png(fname_png)

    def __getitem__(self, idx):
        fname_png = self.image_list[idx]
        return self.get_data(fname_png)

if __name__ == '__main__':
    folder_name = 'sdf_test_sofa_224'
    data_dir = os.path.join(basepath, 'synthesis', 'output', folder_name)
    dataset = SynShapenetLoader(data_dir, load_albedo=True)
    output = dataset[4]
    cv2.imwrite(os.path.expanduser('~/Downloads/img.png'), output[0])

