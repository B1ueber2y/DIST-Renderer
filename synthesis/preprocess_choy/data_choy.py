import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.geometry import Camera, View
from tqdm import tqdm

class DataLoader_Choy(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = self.get_data_list(self.data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_data_list(self, data_dir):
        data_list = []
        class_list = os.listdir(data_dir)
        for class_id in class_list:
            # if class_id != '04256520': # sofa
            #     continue
            nowpath = os.path.join(data_dir, class_id)
            instance_list = os.listdir(nowpath)
            print('{0}: {1}'.format(class_id, len(instance_list)))
            for instance_name in tqdm(instance_list):
                basedir = os.path.join(nowpath, instance_name, 'rendering')

                image_file = os.path.join(basedir, 'renderings.txt')
                image_list = self.read_image_file(image_file)

                camera_file = os.path.join(basedir, 'cameras.npz')
                camera_list = self.read_camera_file(camera_file)

                view_file = os.path.join(basedir, 'rendering_metadata.txt')
                view_list = self.read_meta_file(view_file)

                data = {}
                data['basedir'] = basedir
                data['class_id'] = class_id
                data['instance_name'] = instance_name
                data['camera_list'] = camera_list
                data['view_list'] = view_list
                data['image_list'] = image_list
                data_list.append(data)
        return data_list

    def read_camera_file(self, camera_file):
        cam_data = np.load(camera_file)
        cam_list = []
        for i in range(24):
            K = cam_data['camera_mat_{}'.format(i)]
            RT = cam_data['world_mat_{}'.format(i)]
            cam_list.append(Camera(K, RT))
        return cam_list

    def read_image_file(self, image_file):
        with open(image_file, 'r') as f:
             image_list = f.readlines()
        image_list = [k.strip('\n') for k in image_list]
        return image_list

    def read_meta_file(self, view_file):
        with open(view_file, 'r') as f:
            lines = f.readlines()
        view_list = []
        for line in lines:
            k = line.strip('\n').split(' ')
            k = [float(m) for m in k]
            view = View(k[0], k[1], k[2], k[3])
            view_list.append(view)
        return view_list

if __name__ == '__main__':
    data_dir = os.path.expanduser('~/data/Choy/ShapeNetRendering')
    dataset = DataLoader_Choy(data_dir)


