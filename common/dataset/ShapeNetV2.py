import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from geometry import Shape
import json
from tqdm import tqdm

def read_split_file(split_file):
    with open(split_file, 'r') as f:
        split = json.load(f)

    key_list = list(split.keys())
    assert(len(key_list) == 1)
    dataset = key_list[0]

    data = split[dataset]
    key_list = list(data.keys())
    assert(len(key_list) == 1)
    class_name = key_list[0]

    instance_list = split[dataset][class_name]
    return instance_list

def filter_shape(shape_list, split_file):
    instance_list = read_split_file(split_file)
    new_list = []
    for shape in tqdm(shape_list):
        if shape.shape_md5 in instance_list:
            new_list.append(shape)
    return new_list

class ShapeNetV2(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.shape_name_pairs = [('02691156', 'plane'),
                             ('02834778', 'bicycle'),
                             ('02858304', 'boat'),
                             ('02876657', 'bottle'),
                             ('02924116', 'bus'),
                             ('02958343', 'car'),
                             ('03001627', 'chair'),
                             ('04379243', 'table'),
                             ('03790512', 'motorbike'),
                             ('04256520', 'sofa'),
                             ('04468005', 'train'),
                             ('03211117', 'tvmonitor')]
        self.shape_class_ids = [x[0] for x in self.shape_name_pairs]
        self.shape_names = [x[1] for x in self.shape_name_pairs]

    def get_class_id_from_name(self, name):
        if not (name in self.shape_names):
            raise ValueError('class name {0} not found.'.format(name))
        idx = self.shape_names.index(name)
        return self.shape_class_ids[idx]

    def get_name_from_class_id(self, class_id):
        if not (class_id in self.shape_class_ids):
            raise ValueError('class id {0} not found.'.format(class_id))
        idx = self.shape_class_ids.index(class_id)
        return self.shape_names[idx]

    def get_split_file_name(self, name, mode='train'):
        nowpath = os.path.dirname(os.path.abspath(__file__))
        basepath = os.path.join(nowpath, '..', '..')
        split_file = os.path.join(basepath, 'examples', 'splits', 'sv2_{0}s_{1}.json'.format(name, mode))
        return split_file

    def get_shape_list_from_name(self, name, use_split_file=None, mode='train'):
        class_id = self.get_class_id_from_name(name)
        return self.get_shape_list_from_class_id(class_id, use_split_file=use_split_file, mode=mode)

    def get_shape_list_from_class_id(self, class_id, use_split_file=None, mode='train'):
        path = os.path.join(self.data_dir, class_id)
        if not os.path.exists(path):
            return []
        shape_md5_list = os.listdir(path)
        shape_list = [Shape(class_id, shape_md5, os.path.join(path, shape_md5, 'models/model_normalized.obj')) for shape_md5 in shape_md5_list]
        if use_split_file is not None:
            name = self.get_name_from_class_id(class_id)
            split_file = self.get_split_file_name(name, mode=mode)
            shape_list = filter_shape(shape_list, split_file)
        return shape_list

    def get_shape_from_instance_name(self, class_id, instance_name):
        path = os.path.join(self.data_dir, class_id)
        shape = Shape(class_id, instance_name, os.path.join(path, instance_name, 'models/model_normalized.obj'))
        return shape

