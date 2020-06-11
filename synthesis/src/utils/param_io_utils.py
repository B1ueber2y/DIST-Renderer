import os, sys
import pickle
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.mystruct import FnameGroup, RenderParam
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from common.geometry import Shape, View, PointLight, Lighting

def write_render_param(fname, render_param):
    shape, view_cfg, light_cfg, target_cfg = render_param.shape, render_param.view_cfg, render_param.light_cfg, render_param.target_cfg
    resolution = render_param.resolution
    with open(fname, 'w') as f:
        # write shape
        f.writelines(shape.class_id + '\n')
        f.writelines(shape.shape_md5 + '\n')
        f.writelines(shape.path_to_obj + '\n')
        # write resolution
        f.writelines('{0} {1}\n'.format(resolution[0], resolution[1]))
        # write number of renderings
        num = len(view_cfg)
        f.writelines('{0:d}\n'.format(num))
        # write view
        for idx in range(num):
            view = view_cfg[idx]
            f.writelines('{0:.12f} {1:.12f} {2:.12f}\n'.format(view.camera_pos[0], view.camera_pos[1], view.camera_pos[2]))
            f.writelines('{0:.12f} {1:.12f} {2:.12f} {3:.12f}\n'.format(view.quaternion[0], view.quaternion[1], view.quaternion[2], view.quaternion[3]))
        # write lighting
        for idx in range(num):
            lighting = light_cfg[idx]
            f.writelines('{0:.12f}\n'.format(lighting.env_energy))
            point_num = len(lighting.points_light)
            f.writelines('{0:d}\n'.format(point_num))
            for j in range(point_num):
                point_light = lighting.points_light[j]
                f.writelines('{0:.12f} {1:.12f} {2:.12f} {3:.12f}\n'.format(point_light.azimuth_deg, point_light.elevation_deg, point_light.dist, point_light.energy))
        # write target
        for idx in range(num):
            target = target_cfg[idx]
            f.writelines('{0} {1} {2} {3} {4}\n'.format(target.image, target.depth, target.normal, target.albedo, target.param))
        

def read_render_param(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [line.strip('\n') for line in lines]

    counter = 0
    # read shape
    class_id = lines[counter]
    shape_md5 = lines[counter + 1]
    path_to_obj = lines[counter + 2]
    shape = Shape(class_id, shape_md5, path_to_obj)
    counter = counter + 3
    # read resolution
    resolution = [int(k) for k in lines[counter].split()]
    counter = counter + 1
    # read number of renderings
    num = int(lines[counter])
    counter = counter + 1
    # read view
    view_cfg = []
    for idx in range(num):
        view = View()
        camera_pos = [float(k) for k in lines[counter].split()]
        view.camera_pos = camera_pos
        counter = counter + 1
        quaternion = [float(k) for k in lines[counter].split()]
        view.quaternion = quaternion
        counter = counter + 1
        view_cfg.append(view)
    # read lighting
    light_cfg = []
    for idx in range(num):
        env_energy = float(lines[counter])
        counter = counter + 1
        point_num = int(lines[counter])
        counter = counter + 1
        points_light = []
        for j in range(point_num):
            data = [float(k) for k in lines[counter].split()]
            counter = counter + 1
            points_light.append(PointLight(data[0], data[1], data[2], data[3]))
        lighting = Lighting(env_energy, points_light)
        light_cfg.append(lighting)
    # read target
    target_cfg = []
    for idx in range(num):
        data = lines[counter].split()
        counter = counter + 1
        target = FnameGroup(data[0], data[1], data[2], data[3], data[4])
        target_cfg.append(target)

    # integration
    render_param = RenderParam(shape, view_cfg, light_cfg, target_cfg, resolution)
    return render_param

