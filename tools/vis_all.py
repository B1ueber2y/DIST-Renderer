import os, sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vis_mesh import vis_mesh_from_file, vis_mesh_with_camera_from_file

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="visualize all meshes within a folder"
    )
    arg_parser.add_argument("-i", "--input_directory", type=str, required=True, help="dir to ply files.")
    arg_parser.add_argument('-c', '--camera_file', default='default.pkl', help="output filename without extension")
    arg_parser.add_argument("-o", "--output_directory", type=str, default='vis', help="dir to output rendered images.")
    arg_parser.add_argument('--show', action='store_true', help='show the mesh.')
    arg_parser.add_argument('--no_transform', action='store_true', help='disable transform.')
    args = arg_parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    flist = os.listdir(args.input_directory)
    for i, fname in enumerate(tqdm(flist)):
        pythonfile = os.path.join(nowpath, 'vis_mesh.py')
        fpath = os.path.join(args.input_directory, fname)
        fout = os.path.join(args.output_directory, '{0}.png'.format(fname[:-4]))
        if args.camera_file == 'default.pkl':
            vis_mesh_from_file(fpath, output=fout)
        else:
            vis_mesh_with_camera_from_file(fpath, args.camera_file, output=fout, transform=not args.no_transform, show=args.show)


