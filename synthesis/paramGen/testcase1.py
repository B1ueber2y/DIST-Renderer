'''
2019-08-10 01:01
Method:
randomization
'''
VIEW_NUM, LIGHTING_NUM = 20, 5 
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.param_decomposer import AllParams

MIN_DIST, MAX_DIST = 1.0, 3.5
RESOLUTION = (224, 224)

def generate_params(shape_list, randomizer):
    nowpath = os.path.dirname(os.path.abspath(__file__))
    basepath = os.path.dirname(nowpath)
    folder = os.path.join(basepath, 'output', os.path.splitext(os.path.basename(__file__))[0])
    all_params_list = []

    shape_list = shape_list[:5] # take only five for testing

    print('generating rendering params...')
    from tqdm import tqdm
    for shape in tqdm(shape_list):
        view_cfg, light_cfg, truncparam_cfg, cropbg_param_cfg, fname_cfg = [], [], [], [], []
        counter = 0
        for j1 in range(VIEW_NUM): # 20 cameras (views and truncparams)
            view = randomizer.randomize_view(min_dist=MIN_DIST, max_dist=MAX_DIST)
            truncparam = randomizer.randomize_truncparam()
            for j2 in range(LIGHTING_NUM): # 5 lighting condtions and bg
                lighting = randomizer.randomize_lighting()
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
        all_params = AllParams(shape, view_cfg, light_cfg, truncparam_cfg, cropbg_param_cfg, fname_cfg, resolution=RESOLUTION)
        all_params_list.append(all_params)
    return folder, all_params_list


