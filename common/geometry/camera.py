from collections import namedtuple
import numpy as np

# Camera
'''
struct camera
- projection	np.array (3,4)
- intrinsic	np.array (3,3)
- extrinsic	np.array (3,4)
'''
Camera = namedtuple("Camera", "projection intrinsic extrinsic")

# overwrite camera with methods
class Camera(object):
    def __init__(self, intrinsic, extrinsic, projection='default'):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        if projection != 'default':
            self.projection = projection
        else:
            self.projection = np.dot(self.intrinsic, self.extrinsic)

    def get_camera_location(self):
        R, T = self.extrinsic[:,:3], self.extrinsic[:,3]
        R_t = np.transpose(R, (1,0))
        location = np.dot(-R_t, T)
        return location

    def get_resolution(self):
        w = 2.0 * self.intrinsic[0,2]
        h = 2.0 * self.intrinsic[1,2]
        reso = (h, w)
        return reso

    def get_focal_lens(self, sensor_width, sensor_height):
        ax, ay = self.intrinsic[0,0], self.intrinsic[1,1]
        h, w = self.get_resolution()
        fx = ax * sensor_width / h
        fy = ay * sensor_height / h
        assert(fx == fy)
        return fx

