import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from camera_utils import *
from collections import namedtuple

# struct point light
PointLight = namedtuple("PointLight", "azimuth_deg elevation_deg dist energy")

# struct lighting
'''
struct lighting
- env_energy	type: float
- points_light	type: list of PointLight
'''
Lighting = namedtuple("Lighting", "env_energy points_light")

# overwrite lighting
class Lighting(object):
    def __init__(self, env_energy, points_light):
        self.env_energy = env_energy
        self.points_light = points_light

# overwrite PointLight with methods
class PointLight(object):
    def __init__(self, azimuth_deg, elevation_deg, dist, energy):
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = elevation_deg
        self.dist = dist
        self.energy = energy

    def get_camera_pos(self):
        cx, cy, cz = obj_centened_camera_pos(self.dist, self.azimuth_deg, self.elevation_deg)
        return cx, cy, cz

