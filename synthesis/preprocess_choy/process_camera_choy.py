'''
preprocess the dataset of choy to get camera parameters.
'''
import os
from tqdm import tqdm
import pickle
import numpy as np
import torch.utils.data

nowpath = os.path.dirname(os.path.abspath(__file__))
PYTHON_FILE = os.path.join(nowpath, 'get_r2n2_cameras.py')

def process_path(path):
    processor = torch.utils.data.DataLoader(Processor(path), batch_size=16, num_workers=8, drop_last=False, shuffle=False)
    for i, _ in enumerate(tqdm(processor)):
        pass

class Processor(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.flist = os.listdir(path)
        print('{0}: {1}'.format(path, len(self.flist)))

    def __len__(self):
        return len(self.flist)

    def process_file(self, fname):
        in_file = os.path.join(self.path, fname, 'rendering', 'rendering_metadata.txt')
        output_path = os.path.join(self.path, fname, 'rendering')
        out_file = os.path.join(output_path, 'cameras.npz')
        cmd = 'python {0} {1} {2}'.format(PYTHON_FILE, in_file, out_file)
        os.system(cmd)

    def __getitem__(self, idx):
        fname = self.flist[idx]
        self.process_file(fname)
        return 0

def process_dataset(dataset_path):
    class_list = os.listdir(dataset_path)
    for class_id in class_list:
        # if class_id != '04256520': # sofa
        #     continue
        path = os.path.join(dataset_path, class_id)
        process_path(path)

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="To process the dataset of choy to get camera parameters."
    )
    arg_parser.add_argument('-i', '--data_dir', default=os.path.expanduser('~/data/Choy/ShapeNetRendering'), help='directory to choy data')
    args = arg_parser.parse_args()
    process_dataset(args.data_dir)

