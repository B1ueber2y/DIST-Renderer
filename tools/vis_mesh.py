import numpy as np
import trimesh
import pickle
import os, sys
nowpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(nowpath, '..', '..', 'synthesis'))
from core.mystruct import *
nowpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(nowpath, '..'))
from mylib.utils.render_utils import *
from mylib.visualize.vis_utils import *
import pdb

ROTATE = trimesh.transformations.rotation_matrix(
    angle=np.radians(210.0),
    direction=[0, 1, 0],
    point=[0.,0.,0.])

def vis_mesh(mesh, rotate=ROTATE, output='test.png', show=False):
    scene = mesh.scene()
    camera = scene.graph[scene.camera.name][0]
    camera[2,3] = camera[2,3] + 2 
    scene.graph[scene.camera.name] = camera

    camera_old = scene.graph[scene.camera.name]
    camera_new = np.dot(rotate, camera_old[0])
    scene.graph[scene.camera.name] = camera_new
    if show:
        scene.show()

    png = scene.save_image(resolution=None, visible=True)
    fname_output = output
    with open(fname_output, 'wb') as f:
        f.write(png)
        f.close()

def vis_mesh_from_file(ply_file, rotate=ROTATE, output='test.png', show=False):
    mesh = trimesh.load(ply_file)
    vis_mesh(mesh, rotate=rotate, output=output, show=show)

def vis_mesh_with_camera(mesh, camera, output='test.png', transform=True, show=False, out_size=None):
    '''
    Input:
    - mesh		trimesh mesh object
    - camera		core.struct.Camera objet
    - output		output image name
    - transform		whether to transform to left handed version
    - show		whether to show the mesh
    '''
    cam_data = camera
    K = cam_data.intrinsic # (3,3)
    RT = cam_data.extrinsic # (3,4)
    if transform:
        R, T = RT[:,:3], RT[:,[3]]
        transform_matrix = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        R_trans = np.dot(R, transform_matrix)
        RT = np.concatenate([R_trans, T], 1)

    if out_size is not None:
        img_w, img_h = 2 * K[0, 2], 2 * K[1, 2]
        K[0,:] = K[0,:] * out_size / img_w
        K[1,:] = K[1,:] * out_size / img_h

    # transform mesh vertices and use identity transformation as camera matrices.
    mesh.vertices = - (np.dot(mesh.vertices, RT[:3, :3].T) + RT[:3, 3])
    mesh.vertices[:,0] = (-1) * mesh.vertices[:,0]
    scene = mesh.scene()
    scene.camera.K = K
    scene.camera_transform = np.eye(4)
    if show:
        scene.show()

    # render image
    png = scene.save_image(resolution=None, visible=True)
    with open(output, 'wb') as f:
        f.write(png)
        f.close()

def change_camera_resolution(camera, reso):
    K, RT = camera.intrinsic, camera.extrinsic
    reso_orig = K[0,2] * 2.0
    K_new = downsize_camera_intrinsic(K, reso_orig / reso)
    camera_new = Camera(K_new, RT)
    return camera_new

def vis_mesh_with_camera_from_file(ply_file, camera_file, output='test.png', transform=True, show=False, out_size=None):
    # load mesh data and camera matrices 
    mesh = trimesh.load(ply_file)
    with open(camera_file, 'rb') as f:
        cam_data = pickle.load(f)
    cam_data = change_camera_resolution(cam_data, 512)
    vis_mesh_with_camera(mesh, cam_data, output=output, transform=transform, show=show, out_size=out_size)

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="visualize a mesh with trimesh"
    )
    arg_parser.add_argument('-n', '--ply_file', required=True, help="filename")
    arg_parser.add_argument('-c', '--camera_file', default='camera.pkl', help="output filename without extension")
    arg_parser.add_argument('-o', '--output', default='test_trimesh.png', help="output filename without extension")
    arg_parser.add_argument('--show', action='store_true', help='show the mesh.')
    arg_parser.add_argument('--no_transform', action='store_true', help='disable transform.')
    arg_parser.add_argument('--out_size', type=int, default=None, help='out_size.')
    args = arg_parser.parse_args()

    if os.path.exists(args.camera_file):
        vis_mesh_with_camera_from_file(args.ply_file, args.camera_file, output=args.output, transform=not args.no_transform, show=args.show, out_size=args.out_size)
    else:
        vis_mesh_from_file(args.ply_file, output=args.output, show=args.show)

