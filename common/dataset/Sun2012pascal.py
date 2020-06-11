import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from geometry import Shape

class Sun2012pascal(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_image_list(self):
        nowpath = os.path.join(self.data_dir, 'JPEGImages')
        flist = os.listdir(nowpath)
        image_list = [os.path.join(nowpath, fname) for fname in flist]
        return image_list

