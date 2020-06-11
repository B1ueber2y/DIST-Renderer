import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from camera_utils import *
from collections import namedtuple

# struct view
View = namedtuple("View", "azimuth_deg elevation_deg theta_deg dist")

# overwrite View with methods
class View(object):
    def __init__(self, azimuth_deg=None, elevation_deg=None, theta_deg=None, dist=None):
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = elevation_deg
        self.theta_deg = theta_deg
        self.dist = dist

        # initialize camera position and quaternion for blender.
        self.camera_pos, self.quaternion = None, None
        if (azimuth_deg is not None) and (elevation_deg is not None) and (theta_deg is not None) and (dist is not None):
            self.init_blender()

    def print_(self):
        print('azimuth: {0:.2f}, elevation: {1:.2f}, theta: {2:.2f}, dist: {3:.2f}'.format(self.azimuth_deg, self.elevation_deg, self.theta_deg, self.dist))

    def init_blender(self):
        # get camera position
        cx, cy, cz = obj_centened_camera_pos(self.dist, self.azimuth_deg, self.elevation_deg)
        self.camera_pos = [cx, cy, cz]

        # get quaternion
        q1 = camPosToQuaternion(cx, cy, cz)
        q2 = camRotQuaternion(cx, cy, cz, self.theta_deg)
        q = quaternionProduct(q2, q1)
        self.quaternion = q

    def get_blender_quaternion_from_rotation(self, rotation_matrix):
        from mathutils import Matrix
        R_bcam2cv = Matrix(
            ((1, 0,  0),
             (0, -1, 0),
             (0, 0, -1)))

        R_world2cv = Matrix(rotation_matrix)
        R_world2bcam = R_bcam2cv * R_world2cv
        rotation = R_world2bcam.transposed()
        quaternion = rotation.to_quaternion()
        return quaternion

    def get_rotation_from_blender_quaternion(self, quaternion):
        from mathutils import Matrix
        R_bcam2cv = Matrix(
            ((1, 0,  0),
             (0, -1, 0),
             (0, 0, -1)))
        R = quaternionToRotation(quaternion)
        R = Matrix(R).transposed()
        R_new = R_bcam2cv * R
        R_new = np.array(R_new)
        return R_new

    def get_extrinsic(self):
        R = self.get_rotation_from_blender_quaternion(self.quaternion)
        T = (-1) * np.matmul(R, np.array(self.camera_pos)[:,None])
        RT = np.concatenate([R, T], 1)
        return RT

    def set_camera(self, camera):
        self.camera_pos = camera.get_camera_location().tolist()
        self.quaternion = self.get_blender_quaternion_from_rotation(camera.extrinsic[:,:3])

    def get_camera_pos(self):
        if self.camera_pos is None:
            raise ValueError('Camera not set.')
        else:
            cx, cy, cz = self.camera_pos 
        return cx, cy, cz

    def get_quaternion(self):
        if self.quaternion is None:
            raise ValueError('Camera not set.')
        else:
            q = self.quaternion
        return q


