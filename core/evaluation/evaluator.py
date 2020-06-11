import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from eval_func import *
from transforms import *

class Evaluator(object):
    def __init__(self, decoder):
        self.decoder = decoder
        self.device = next(self.decoder.parameters()).get_device()
        self.decoder.eval()

    def latent_vec_to_points(self, latent_vec, N=256, max_batch=32 ** 3, num_points=30000, silent=False, fname=None, transform=False, meshcreator_type='speedup'):
        return latent_vec_to_points(self.decoder, latent_vec=latent_vec, N=N, max_batch=max_batch, num_points=num_points, silent=silent, fname=fname, transform=transform, meshcreator_type=meshcreator_type)

    def compute_chamfer_distance(self, points_1, points_2, separate=False):
        if not separate:
            return compute_chamfer_distance(points_1, points_2)
        else:
            return compute_chamfer_distance_separate(points_1, points_2)

