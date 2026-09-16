# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from SemCity
# (https://github.com/zoomin-lee/SemCity), Copyright (c) 2024 Jumin Lee,
# licensed under the MIT License. See THIRD-PARTY-NOTICES.txt.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import TriplaneGroupResnetBlock, DecoderMLPSkipConcat


class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, z_down, padding_mode, kernel_size=(5, 5, 3), padding=(2, 2, 1)):
        super().__init__()
        self.z_down = z_down
        self.conv0 = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode)
        self.convblock1 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
        if self.z_down:
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        self.convblock2 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
        self.scene_head = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1),
                                    padding=padding, bias=True, padding_mode=padding_mode)

    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        x = self.conv0(x)

        residual_feat = x
        x = self.convblock1(x)
        x = x + residual_feat
        x = self.downsample(x)  # [b, geo_feat_channels, X//2, Y//2, Z]

        residual_feat = x
        x = self.convblock2(x)
        x = x + residual_feat

        return self.scene_head(x)


class AutoEncoderGroupSkip(nn.Module):
    """Triplane encoder + shared-MLP decoder (SemCity-based).

    Input: a (B, X, Y, Z) voxel grid with values in {0, 1..num_class-1, 255} (see
    priors.py). Each value is embedded, 255 getting its own learned embedding, the
    3D features are encoded and averaged into three axis-aligned planes (xy, xz, yz),
    and the class logits of any query point are decoded from its plane features.
    Output: (B, N, num_class) logits for the N query points.
    """

    def __init__(self, num_class=20, geo_feat_channels=16, feat_channel_up=64, mlp_hidden_channels=256,
                 mlp_hidden_layers=4, padding_mode='replicate', z_down=False, pos=True) -> None:
        super().__init__()
        self.class_num = num_class
        # num_class + 1 embeddings: classes 0..num_class-1, plus one for the 255 value.
        self.embedding = nn.Embedding(num_class + 1, geo_feat_channels)
        self.geo_encoder = Encoder(geo_feat_channels, z_down, padding_mode)

        self.norm = nn.InstanceNorm2d(geo_feat_channels)
        self.geo_feat_dim = geo_feat_channels
        self.pos = pos
        self.pos_num_freq = 6  # NeRF-style positional encoding of the query coordinates

        self.geo_convs = TriplaneGroupResnetBlock(geo_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False)
        decoder_in = feat_channel_up + (6 * self.pos_num_freq if pos else 0)
        self.geo_decoder = DecoderMLPSkipConcat(decoder_in, num_class, mlp_hidden_channels, mlp_hidden_layers)

    def encode(self, vol):
        x = vol.detach().clone()
        x[x == 255] = self.class_num  # the 255 value (unknown) uses the last embedding

        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)  # move feature dim to position 1 -> B x C x X x Y x Z

        scene_triplane = self.geo_encoder(x)

        scene_xy_feat = scene_triplane.mean(dim=4)  # flatten Z
        scene_xz_feat = scene_triplane.mean(dim=3)  # flatten Y
        scene_yz_feat = scene_triplane.mean(dim=2)  # flatten X

        scene_xy_feat = (self.norm(scene_xy_feat) * 0.5).tanh()
        scene_xz_feat = (self.norm(scene_xz_feat) * 0.5).tanh()
        scene_yz_feat = (self.norm(scene_yz_feat) * 0.5).tanh()

        return [scene_xy_feat, scene_xz_feat, scene_yz_feat]

    def sample_feature_plane2D(self, feat_map, x):
        """Sample a 2D feature plane at query coords. feat_map: [bs, C, H, W], x: [bs, N, 2]."""
        sample_coords = x.view(x.shape[0], 1, -1, 2)  # [bs, 1, N, 2] (extra height dim for grid_sample)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1), align_corners=False, padding_mode='border')  # [bs, C, 1, N]
        feat = feat[:, :, 0, :]  # [bs, C, N]
        feat = feat.transpose(1, 2)  # [bs, N, C]
        return feat

    def decode(self, feat_maps, query):
        # which coordinate pairs correspond to each triplane (XY, XZ, YZ)
        coords_list = [[0, 1], [0, 2], [1, 2]]
        geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)  # changes the feature dim from 16 to 64

        h_geo = 0
        for i in range(3):
            h_geo += self.sample_feature_plane2D(geo_feat_maps[i], query[..., coords_list[i]])  # [bs, N, C]

        if self.pos:
            PE = []
            for freq in range(self.pos_num_freq):
                PE.append(torch.sin((2. ** freq) * query))
                PE.append(torch.cos((2. ** freq) * query))
            PE = torch.cat(PE, dim=-1)  # [bs, N, 6*pos_num_freq]
            h_geo = torch.cat([h_geo, PE], dim=-1)

        h = self.geo_decoder(h_geo)  # [bs, N, num_class]
        return h

    def forward(self, vol, query):
        """The whole voxel volume is encoded but only the query points are decoded."""
        feat_map_scene = self.encode(vol)
        return self.decode(feat_map_scene, query)
