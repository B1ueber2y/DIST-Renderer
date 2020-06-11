'''
scripts for rendering. must use blender-2.71.
'''
import os, sys
import numpy as np
import pickle
import datetime
import shutil
from multiprocessing.dummy import Pool
from functools import partial
from subprocess import call
import _pickle as cPickle
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.param_io_utils import write_render_param, read_render_param
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from common.geometry import Camera

class RenderingProcessor(object):
    def __init__(self, cam_lens=60.0, render_depth=True, render_normal=True, render_albedo=False):
        self.cam_lens = cam_lens
        self.render_depth = render_depth
        self.render_normal = render_normal
        self.render_albedo = render_albedo

    def init_env(self):
        import bpy
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
        
        # add passes for additionally dumping albedo and normals.
        bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
        bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR' 
        
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
        return tree

    def get_calibration_matrix_K_from_blender(self, cam):
        import bpy
        f_in_mm = cam.lens
        scene = bpy.context.scene
        resolution_x_in_px = scene.render.resolution_x
        resolution_y_in_px = scene.render.resolution_y
        scale = scene.render.resolution_percentage / 100
        sensor_width_in_mm = cam.sensor_width
        sensor_height_in_mm = cam.sensor_height
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        if (cam.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal), 
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
            s_v = resolution_y_in_px * scale / sensor_height_in_mm
        else: # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal), 
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = resolution_x_in_px * scale / sensor_width_in_mm
            s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    
    
        # Parameters of intrinsic calibration matrix K
        alpha_u = f_in_mm * s_u
        alpha_v = f_in_mm * s_v
        u_0 = resolution_x_in_px * scale / 2
        v_0 = resolution_y_in_px * scale / 2
        skew = 0 # only use rectangular pixels
    
        K = np.array(
            [[alpha_u, skew,    u_0],
             [    0  , alpha_v, v_0],
             [    0  , 0,        1 ]])
        return K

    def get_3x4_RT_matrix_from_blender(self, cam):
        # bcam stands for blender camera
        R_bcam2cv = np.array(
            [[1, 0,  0],
             [0, -1, 0],
             [0, 0, -1]])
    
        # Transpose since the rotation is object rotation, 
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam * location
        #
        # Use matrix_world instead to account for all constraints
        location, rotation = cam.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()
    
        # Convert camera location to translation vector used in coordinate changes
        # T_world2bcam = -1*R_world2bcam*cam.location
        # Use location from matrix_world to account for constraints:     
        T_world2bcam = -1 * R_world2bcam * location
    
        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
        T_world2cv = np.dot(R_bcam2cv, T_world2bcam)
    
        # put into 3x4 matrix
        RT = np.concatenate([R_world2cv, T_world2cv[:,None]], 1) 
        return RT

    def test_func(self, cam):
        from mathutils import Matrix
        # bcam stands for blender camera
        R_bcam2cv = Matrix(
            ((1, 0,  0),
             (0, -1, 0),
             (0, 0, -1)))
    
        # Transpose since the rotation is object rotation, 
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam * location
        #
        # Use matrix_world instead to account for all constraints
        location, rotation = cam.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()
    
        # Convert camera location to translation vector used in coordinate changes
        # T_world2bcam = -1*R_world2bcam*cam.location
        # Use location from matrix_world to account for constraints:     
        T_world2bcam = -1*R_world2bcam * location
    
        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = R_bcam2cv*R_world2bcam
        T_world2cv = R_bcam2cv*T_world2bcam
    
        # put into 3x4 matrix
        RT = Matrix((
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
             ))
    
        motionMat = np.zeros((4,4))
        motionMat[3][3] = 1
        motionMat[0:3,0:4] = np.asarray(RT)
        return motionMat

    def get_3x4_P_matrix_from_blender(self, cam):
        '''
        interfaces borrowed from the link below:
        [-] https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
        '''
        K = self.get_calibration_matrix_K_from_blender(cam.data)
        RT = self.get_3x4_RT_matrix_from_blender(cam)
        return np.dot(K, RT), K, RT

    def create_node(self, tree):
        # create input render layer node.
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        depth_node, normal_node, albedo_node = None, None, None
        
        if self.render_depth:
            depth_node = tree.nodes.new(type="CompositorNodeOutputFile")
            depth_node.label = 'Depth Output'
            tree.links.new(render_layers.outputs['Z'], depth_node.inputs[0])
        
        if self.render_normal:
            normal_node = tree.nodes.new(type="CompositorNodeOutputFile")
            normal_node.label = 'Normal Output'
            tree.links.new(render_layers.outputs['Normal'], normal_node.inputs[0])
        
        if self.render_albedo:
            albedo_node = tree.nodes.new(type="CompositorNodeOutputFile")
            albedo_node.label = 'Albedo Output'
            tree.links.new(render_layers.outputs['Color'], albedo_node.inputs[0])
        return depth_node, normal_node, albedo_node 

    def render(self, render_param):
        shape = render_param.shape
        view_cfg = render_param.view_cfg
        lighting_cfg = render_param.light_cfg
        target_cfg = render_param.target_cfg
        resolution = render_param.resolution
        im_y, im_x = resolution[0], resolution[1]

        # initialize blender environment
        tree = self.init_env()
        depth_node, normal_node, albedo_node = self.create_node(tree)

        # import obj file
        import bpy
        bpy.ops.import_scene.obj(filepath=shape.path_to_obj)
        bpy.context.scene.render.resolution_y = im_y
        bpy.context.scene.render.resolution_x = im_x
        bpy.context.scene.render.resolution_percentage = 100

        bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
        bpy.data.objects['Lamp'].data.energy = 0

        camObj = bpy.data.objects['Camera']
        camObj.data.lens = self.cam_lens 
        camObj.data.sensor_height = 32.0
        camObj.data.sensor_width = float(camObj.data.sensor_height) / im_y * im_x

        bpy.ops.object.select_all(action='TOGGLE')
        if 'Lamp' in list(bpy.data.objects.keys()):
            bpy.data.objects['Lamp'].select = True # remove default light
        bpy.ops.object.delete()

        for view, lighting, target in zip(view_cfg, lighting_cfg, target_cfg):
            # clear lights
            bpy.ops.object.select_by_type(type='LAMP')
            bpy.ops.object.delete(use_global=False)

            # set environment lighting
            bpy.context.scene.world.light_settings.use_environment_light = True
            bpy.context.scene.world.light_settings.environment_energy = lighting.env_energy 
            bpy.context.scene.world.light_settings.environment_color = 'PLAIN'
            bpy.context.scene.render.image_settings.file_format = 'PNG'  # set output format to .png

            # set point lights
            for point_light in lighting.points_light:
                lx, ly, lz = point_light.get_camera_pos()
                bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
                bpy.data.objects['Point'].data.energy = point_light.energy

            # set camera
            cx, cy, cz = view.get_camera_pos()
            camObj.location[0] = cx
            camObj.location[1] = cy
            camObj.location[2] = cz
            camObj.rotation_mode = 'QUATERNION'
            q = view.get_quaternion()
            camObj.rotation_quaternion[0] = q[0]
            camObj.rotation_quaternion[1] = q[1]
            camObj.rotation_quaternion[2] = q[2]
            camObj.rotation_quaternion[3] = q[3]

            # set output path
            bpy.data.scenes['Scene'].render.filepath = target.image
            if self.render_depth:
                depth_node.file_slots[0].path = target.depth
            if self.render_normal:
                normal_node.file_slots[0].path = target.normal
            if self.render_albedo:
                albedo_node.file_slots[0].path = target.albedo
            fname_param = target.param

            # render the image
            bpy.ops.render.render(write_still=True)
            P, K, RT = self.get_3x4_P_matrix_from_blender(camObj) 
            camera = Camera(K, RT, P)

            # move the depth, normal and albedo from /tmp to the correct folder
            # for fname in [depth_node.file_slots[0].path, normal_node.file_slots[0].path, albedo_node.file_slots[0].path]:
            for node in [depth_node, normal_node, albedo_node]:
                if node is not None:
                    fname = node.file_slots[0].path
                    cmd = 'mv /tmp/' + fname + '0001.exr' + ' ' + fname
                    os.system(cmd)

            # save the parameters
            params = {}
            params['shape'] = shape
            params['render_param'] = {}
            params['render_param']['view'] = view
            params['render_param']['lighting'] = lighting
            params['render_param']['target'] = target
            params['render_param']['camera'] = camera
            with open(fname_param, 'wb') as f:
                pickle.dump(params, f)

class Renderer(object):
    def __init__(self, blender_path, num_worker=0):
        self.blender_path = blender_path
        self.num_worker = num_worker
        import torch
        import torch.utils.data
        class WritingProcessor(torch.utils.data.Dataset):
            def __init__(self, render_param_list, folder):
                self.render_param_list = render_param_list
                self.folder = folder
            def __len__(self):
                return len(self.render_param_list)
            def __getitem__(self, idx):
                render_param = self.render_param_list[idx]
                fname = os.path.join(self.folder, 'render_param_{0:08d}.txt'.format(idx))
                write_render_param(fname, render_param)
                return 0
        self.writing_processor = WritingProcessor

    def render_all(self, render_param_list, tmpdir='tmp', cam_lens=60.0, render_depth=True, render_normal=True, render_albedo=False, skip_exist=False):
        # write out the render parameters to the tmp tmpdir.
        nowpath = os.path.dirname(os.path.abspath(__file__))
        tmpdir = os.path.join(nowpath, tmpdir)
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir)
        processor = self.writing_processor(render_param_list, tmpdir)
        import torch
        import torch.utils.data
        dataloader = torch.utils.data.DataLoader(processor, batch_size=64, shuffle=False, num_workers=16, drop_last=False)

        print('writing render params to temp file...')
        from tqdm import tqdm
        for i, _ in enumerate(tqdm(dataloader)):
            pass

        # generate blender command
        commands = []
        for i, render_param in enumerate(render_param_list):
            if skip_exist:
                flag = True
                for target in render_param.target_cfg:
                    flag = flag and (os.path.exists(target.image) and os.path.exists(target.param))
                if flag:
                    continue
            path_to_render_param = os.path.join(tmpdir, 'render_param_{0:08d}.txt'.format(i))
            command = '{0} {1} --background --python {2} -- {3} {4:.6f} {5} {6} {7} > /dev/null 2>&1'.format(self.blender_path, os.path.join(nowpath, 'blank.blend'), os.path.abspath(__file__), path_to_render_param, cam_lens, int(render_depth), int(render_normal), int(render_albedo))
            commands.append(command)
        print(commands[0])

        # start rendering
        print('Start rendering at time {0}...it takes a long time...'.format(datetime.datetime.now()))
        report_step = 10
        pool = Pool(self.num_worker)
        for idx, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
            if idx % report_step == 0:
                print('[%s] Rendering command %d of %d' % (datetime.datetime.now().time(), idx, len(commands)))
            if return_code != 0:
                print('Rendering command %d of %d (\"%s\") failed' % (idx, len(commands), commands[idx]))
        shutil.rmtree(tmpdir)

if __name__ == '__main__':
    path_to_render_param = sys.argv[-5]
    cam_lens = float(sys.argv[-4])
    render_depth, render_normal, render_albedo = bool(int(sys.argv[-3])), bool(int(sys.argv[-2])), bool(int(sys.argv[-1]))
    render_param = read_render_param(path_to_render_param)
    RP = RenderingProcessor(cam_lens=cam_lens, render_depth=render_depth, render_normal=render_normal, render_albedo=render_albedo)
    RP.render(render_param)


