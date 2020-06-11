#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

'''
Ported from DeepSDF
https://github.com/facebookresearch/DeepSDF
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def threshold_min_max(tensor, min_vec, max_vec):
    return torch.min(max_vec, torch.max(tensor, min_vec))

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        last_dim=1,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [last_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and l in self.norm_layers:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    def latent_size_regul(self, lat_vecs):
        latent_loss = lat_vecs.pow(2).mean(1) 
        return latent_loss

    # input: N x (L+3)
    def inference(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_in:
                x = torch.cat([x, input], 1)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)

            # last layer Tanh
            if l == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if l < self.num_layers - 2:
                if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, "bn" + str(l))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)
        return x

    # input: N x (L+3), N
    def forward(self, sdf_data, lat_vecs_idx, min_vec, max_vec, enforce_minmax=True):
        num_samp_per_scene = sdf_data.shape[1]
        sdf_data = sdf_data.reshape(-1, 4)
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        latent_dim = lat_vecs_idx.shape[1]
        latent_inputs = lat_vecs_idx.repeat(1, num_samp_per_scene).view(-1, latent_dim)

        inputs = torch.cat([latent_inputs, xyz], 1)
        pred_sdf = self.inference(inputs)

        if enforce_minmax:
            sdf_gt = threshold_min_max(sdf_gt, min_vec, max_vec)
            pred_sdf = threshold_min_max(
                pred_sdf, min_vec, max_vec
            )
        loss_l1 = torch.abs(pred_sdf - sdf_gt).squeeze(1)
        loss_l2_size = self.latent_size_regul(lat_vecs_idx)
        return pred_sdf, loss_l1, loss_l2_size


