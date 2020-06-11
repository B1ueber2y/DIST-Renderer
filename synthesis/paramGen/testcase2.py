'''
2019-08-07 00:01
Method:
20 x 5 grid over (camera x lighting)
'''
VIEW_NUM, LIGHTING_NUM = 20, 5 
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.param_decomposer import AllParams

def generate_params(shape_list, randomizer):
    nowpath = os.path.dirname(os.path.abspath(__file__))
    basepath = os.path.dirname(nowpath)
    folder = os.path.join(basepath, 'output', os.path.splitext(os.path.basename(__file__))[0])
    all_params_list = []

    shape_list = shape_list[:5] # take only five for testing.

    print('generating rendering params...')
    from tqdm import tqdm
    for shape in tqdm(shape_list):
        view_cfg, light_cfg, truncparam_cfg, cropbg_param_cfg, fname_cfg = [], [], [], [], []
        # generate cameras and lights
        camera_list, lighting_list = [], []
        for idx in range(VIEW_NUM):
            view = randomizer.randomize_view()
            truncparam = randomizer.randomize_truncparam()
            camera_list.append((view, truncparam))
        for idx in range(LIGHTING_NUM):
            lighting_list.append(randomizer.randomize_lighting())
       
        counter = 0
        for j1 in range(VIEW_NUM): # 10 cameras (views and truncparams)
            camera = camera_list[j1]
            view, truncparam = camera[0], camera[1]
            for j2 in range(LIGHTING_NUM): # 10 lighting condtions and bg
                lighting = lighting_list[j2]
                cropbg_param = randomizer.randomize_cropbg_param()

                # to append info to the list.
                view_cfg.append(view)
                light_cfg.append(lighting)
                truncparam_cfg.append(truncparam)
                cropbg_param_cfg.append(cropbg_param)
                fname = os.path.join(shape.shape_md5, shape.shape_md5 + '_{0:08d}.png'.format(counter))
                fname_cfg.append(fname)
                counter = counter + 1
        # to append all_params
        all_params = AllParams(shape, view_cfg, light_cfg, truncparam_cfg, cropbg_param_cfg, fname_cfg)
        all_params_list.append(all_params)
    return folder, all_params_list


