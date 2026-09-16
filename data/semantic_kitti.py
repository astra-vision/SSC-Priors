# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from SemCity
# (https://github.com/zoomin-lee/SemCity), Copyright (c) 2024 Jumin Lee,
# licensed under the MIT License. See THIRD-PARTY-NOTICES.txt.

import os
from glob import glob

import numpy as np
import torch
from torch.utils import data

import priors
from data.query import get_query

CLASS_NAMES = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign',
]


def unpack(compressed):
    """Unpack a bit-encoded SemanticKITTI voxel file (e.g. *.invalid) into one uint8 per voxel."""
    return np.unpackbits(compressed)


def ssc_frames(kitti_root, sequences, every_5th=False):
    """(seq, frame) pairs of the scans that have SSC voxel ground truth.

    The SemanticKITTI SSC benchmark evaluates every 5th frame of the validation sequence.
    """
    frames = []
    for seq in sequences:
        voxels_dir = os.path.join(kitti_root, 'dataset', 'sequences', seq, 'voxels')
        paths = sorted(glob(os.path.join(voxels_dir, '*.label')))
        if not paths:
            raise FileNotFoundError(f'No voxel ground truth found in {voxels_dir}')
        for path in paths:
            frame = os.path.splitext(os.path.basename(path))[0]
            if every_5th and int(frame) % 5 != 0:
                continue
            frames.append((seq, frame))
    return frames


def split_sequences(split, kitti_yaml):
    key = {'train': 'train', 'val': 'valid'}[split]
    return [str(s).zfill(2) for s in kitti_yaml['split'][key]]


class SemanticKITTISSC(data.Dataset):
    """SemanticKITTI semantic scene completion with semantic and visibility priors.

    Each item holds the voxel ground truth (256, 256, 32) with classes 0..19 and 255
    for invalid voxels, the network input built by `priors.build_input` (read from
    `precomputed_root` when given, else computed on the fly on CPU), and the query
    points at which the decoder is supervised or evaluated.

    Args:
        kitti_root: directory containing dataset/sequences/{seq}/{velodyne,voxels}.
        split: 'train' (sequences 00-07, 09, 10) or 'val' (sequence 08, every 5th frame).
        semantics: semantic prior, one of priors.SEMANTICS.
        visibility: visibility prior, one of priors.VISIBILITY.
        pseudo_labels_root: root of the pseudo-label trees (needed unless semantics='none').
        precomputed_root: root of the inputs written by precompute_priors.py (optional).
        augment: random flips along x and/or y (training only).
    """

    def __init__(self, kitti_root, split, semantics, visibility, pseudo_labels_root=None,
                 precomputed_root=None, augment=False):
        assert semantics in priors.SEMANTICS, semantics
        assert visibility in priors.VISIBILITY, visibility
        self.kitti_root = kitti_root
        self.split = split
        self.semantics = semantics
        self.visibility = visibility
        self.pseudo_labels_root = pseudo_labels_root
        self.precomputed_root = precomputed_root
        self.augment = augment

        kitti_yaml = priors.load_kitti_yaml()
        self.learning_map = priors.build_learning_map(kitti_yaml)
        self.learning_map_inv = kitti_yaml['learning_map_inv']
        self.frames = ssc_frames(kitti_root, split_sequences(split, kitti_yaml), every_5th=(split == 'val'))

    def __len__(self):
        return len(self.frames)

    def load_ground_truth(self, seq, frame):
        base = os.path.join(self.kitti_root, 'dataset', 'sequences', seq, 'voxels', frame)
        labels = np.fromfile(base + '.label', dtype=np.uint16)
        invalid = unpack(np.fromfile(base + '.invalid', dtype=np.uint8))
        labels = self.learning_map[labels].reshape(priors.GRID_SIZE)
        invalid = invalid.reshape(priors.GRID_SIZE)
        labels[invalid == 1] = priors.UNKNOWN
        return labels, invalid

    def load_input(self, seq, frame):
        if self.precomputed_root is not None:
            path = os.path.join(self.precomputed_root, self.semantics, self.visibility, seq, f'{frame}.npy')
            return np.load(path)
        points = priors.load_scan(self.kitti_root, seq, frame)
        labels = priors.load_point_semantics(self.semantics, points.shape[0], self.learning_map,
                                             self.pseudo_labels_root, seq, frame)
        return priors.build_input(points, labels, self.visibility)

    def __getitem__(self, index):
        seq, frame = self.frames[index]
        target, invalid = self.load_ground_truth(seq, frame)
        net_input = self.load_input(seq, frame)

        if self.augment:
            # Same draw as the original training: flip x, flip y, flip both, or no flip.
            p = torch.randint(0, 6, (1,)).item()
            flip_axes = {0: (0,), 1: (1,), 2: (0, 1)}.get(p, ())
            for axis in flip_axes:
                target = np.flip(target, axis=axis)
                invalid = np.flip(invalid, axis=axis)
                net_input = np.flip(net_input, axis=axis)
            target, invalid, net_input = target.copy(), invalid.copy(), net_input.copy()

        # Training supervises occupied voxels, a random subset of empty ones and the invalid
        # ones (ignored by the loss); evaluation queries every voxel of the grid.
        query, query_labels, query_coords = get_query(target, is_train=(self.split == 'train'))

        return {
            'target': target.astype(np.int64),
            'invalid': invalid.astype(np.int64),
            'input': net_input.astype(np.int64),
            'query': query,
            'query_labels': query_labels.long(),
            'query_coords': query_coords.long(),
            'seq': seq,
            'frame': frame,
        }
