import yaml
import io
fname = 'cfg_default.yaml'

data = {}

# files
data['view_file'] = '/nas/shaoliu/workspace/RenderForCNN/data/view_distribution/chair.txt'
data['truncparam_file'] = '/nas/shaoliu/workspace/RenderForCNN/data/truncation_distribution/chair.txt'

# render_model_views
data['light_num_lowbound'] = 0
data['light_num_highbound'] = 6
data['light_dist_lowbound'] = 8
data['light_dist_highbound'] = 20
data['light_azimuth_degree_lowbound'] = 0
data['light_azimuth_degree_highbound'] = 360
data['light_elevation_degree_lowbound'] = -90
data['light_elevation_degree_highbound'] = 90
data['light_energy_mean'] = 2
data['light_energy_std'] = 2
data['light_env_energy_lowbound'] = 0
data['light_env_energy_highbound'] = 1

with io.open(fname, 'w', encoding='utf8') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

