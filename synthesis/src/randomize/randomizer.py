import os, sys
import random
import numpy as np
import yaml
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from common.geometry import View, PointLight, Lighting
from common.utils.io_utils import read_from_stat_file

class Randomizer(object):
    def __init__(self, view_file=None, truncparam_file='None', config_file='default'):
        nowpath = os.path.dirname(os.path.abspath(__file__))
        if config_file == 'default':
            config_file = os.path.join(nowpath, 'default.yaml')
        if not os.path.exists(config_file):
            raise ValueError('config file {0} not found.'.format(config_file))
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        self.param = data

        self.view_random_list = None
        if view_file is not None:
            self.view_random_list = read_from_stat_file(view_file)
        # self.truncparam_random_list = read_from_stat_file(truncparam_file)

    def randomize_view_dist(self, min_dist, max_dist):
        dist = np.random.rand() * (max_dist - min_dist) + min_dist
        return dist

    def randomize_view(self, min_dist=1.0, max_dist=3.5, num=0):
        if self.view_random_list is None:
            raise ValueError('view randomization list not defined.')
        if num == 0:
            line = random.choice(self.view_random_list)
            data = [float(x) for x in line.strip('\n').split()]
            view = View(data[0], data[1], data[2], self.randomize_view_dist(min_dist, max_dist))
            return view
        else:
            lines = random.sample(self.view_random_list, num)
            data = [[float(x) for x in line.strip('\n').split()] for line in lines]
            views = [View(k[0], k[1], k[2], self.randomize_view_dist(min_dist, max_dist)) for k in data]
            return views

    def randomize_truncparam(self):
        data = []
        for idx in range(4):
            while True:
                rnd = np.random.normal(0, self.param['truncation_param_std'])
                if abs(rnd) < self.param['truncation_param_bound']:
                    break
            data.append(rnd)
        return data

    def randomize_point_light(self):
        azimuth_deg = np.random.uniform(self.param['light_azimuth_degree_lowbound'], self.param['light_azimuth_degree_highbound'])
        elevation_deg = np.random.uniform(self.param['light_elevation_degree_lowbound'], self.param['light_elevation_degree_highbound'])
        dist = np.random.uniform(self.param['light_dist_lowbound'], self.param['light_dist_highbound'])
        energy = np.random.normal(self.param['light_energy_mean'], self.param['light_energy_std'])

        data = PointLight(azimuth_deg, elevation_deg, dist, energy)
        return data

    def randomize_lighting(self, use_point_lighting=True):
        while True:
            env_energy = np.random.uniform(self.param['light_env_energy_lowbound'], self.param['light_env_energy_highbound'])
            point_light_num = random.randint(self.param['light_num_lowbound'], self.param['light_num_highbound'])
            if env_energy >= self.param['light_env_energy_lowbound_without_point'] or (use_point_lighting and point_light_num != 0):
                break
        points_light = []
        if use_point_lighting:
            for idx in range(point_light_num):
                point_light = self.randomize_point_light()
                points_light.append(point_light)
        data = Lighting(env_energy, points_light)
        return data

    def randomize_cropbg_param(self):
        data = [np.random.rand(), np.random.rand()]
        return data

    def standardize_truncparam(self):
        data = [0.0, 0.0, 0.0, 0.0]
        return data

    def standardize_cropbg_param(self):
        data = [0.5, 0.5]
        return data

