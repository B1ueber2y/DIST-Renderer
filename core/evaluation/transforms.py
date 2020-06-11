import os, sys
import trimesh
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from create_mesh import create_mesh, create_mesh_speedup
from decoder_utils import decode_sdf

def sample_points_from_ply_file(fname, num_points):
    mesh = trimesh.load(fname)
    points = trimesh.sample.sample_surface(mesh, num_points)[0]
    return points

def latent_vec_to_points(decoder, latent_vec, N=256, max_batch=32 ** 3, num_points=30000, silent=False, fname=None, transform=False, meshcreator_type='speedup'):
    if fname is None:
        tmpfname = os.path.join('create_trimesh_object_tmpfile')
    else:
        tmpfname = os.path.join(os.path.dirname(fname), 'create_trimesh_object_tmpfile')
    if meshcreator_type == 'original':
        flag = create_mesh(decoder, latent_vec, tmpfname, N=N, max_batch=max_batch, silent=silent, transform=transform)
    elif meshcreator_type == 'speedup':
        flag = create_mesh_speedup(decoder, latent_vec, tmpfname, N=N, max_batch=max_batch, silent=silent, transform=transform)
    else:
        raise NotImplementedError
    if not flag: 
        return None
    points = sample_points_from_ply_file(tmpfname + '.ply', num_points)
    if fname is None:
        os.remove(tmpfname + '.ply')
    else:
        os.rename(tmpfname + '.ply', fname)
    return points


