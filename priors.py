# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Semantic and visibility priors: turn a lidar scan into the SSC network input.

The network input is a (256, 256, 32) voxel grid (0.2 m voxels) covering
x in [0, 51.2], y in [-25.6, 25.6], z in [-2, 4.4] meters in the sensor frame.
Its values depend on the two priors:

  semantic prior      each occupied voxel carries the majority class (1-19) of the
                      pseudo-labels of its points; without it, occupied voxels are 1.

  visibility prior    'none'  -> vanilla occupancy: every non-occupied voxel is 0.
                      'dense' -> free-space samples are placed every 0.2 m along each
                                 lidar ray; a voxel crossed by a ray and containing no
                                 point is observed-empty (0), and every voxel that is
                                 neither occupied nor crossed is unknown (255).

Value encoding of the returned grid (uint8):
  0      empty (visibility 'none': not occupied; 'dense': observed empty)
  1..19  occupied, with its semantic class (or 1 when there is no semantic prior)
  255    visibility 'dense': unknown / never observed
         visibility 'none' : occupied by points that have no valid class
"""

import os

import numpy as np
import torch
import yaml

SEMANTICS = ('none', 'waffleiron', 'waffleiron_tta')
VISIBILITY = ('none', 'dense')

NUM_CLASSES = 20            # 19 semantic classes + empty (0)
UNKNOWN = 255
GRID_SIZE = (256, 256, 32)
VOXEL_SIZE = 0.2
GRID_MIN = (0.0, -25.6, -2.0)
GRID_MAX = (51.2, 25.6, 4.4)
RAY_STEP = 0.2              # spacing of the free-space samples along each ray (m)

YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'semantic-kitti.yaml')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_kitti_yaml(yaml_path=YAML_PATH):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def build_learning_map(kitti_yaml):
    """Lookup table: original SemanticKITTI label id -> class 0..19, with 0 (unlabeled) -> 255.

    Class 0 is reserved for "empty": original id 0 ("unlabeled") stays 0 (this is how the
    voxel ground truth encodes empty voxels) while every other id mapped to 0 by
    `learning_map` (outlier, other-structure, ...) becomes 255 (ignored).
    """
    learning_map = kitti_yaml['learning_map']
    lut = np.zeros(max(learning_map.keys()) + 100, dtype=np.int32)
    lut[list(learning_map.keys())] = list(learning_map.values())
    lut[lut == 0] = UNKNOWN
    lut[0] = 0
    return lut


def pseudo_label_path(pseudo_labels_root, semantics, seq, frame):
    """Pseudo-labels layout: {root}/{semantics}/sequences/{seq}/predictions/{frame}.label

    This is the layout written by tools/generate_waffleiron_labels.py (and by
    WaffleIron's own eval_kitti.py), one tree per semantic source.
    """
    return os.path.join(pseudo_labels_root, semantics, 'sequences', seq, 'predictions', f'{frame}.label')


def load_scan(kitti_root, seq, frame):
    """Lidar points (N, 3) of a SemanticKITTI scan, in the sensor frame."""
    path = os.path.join(kitti_root, 'dataset', 'sequences', seq, 'velodyne', f'{frame}.bin')
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_point_semantics(semantics, num_points, learning_map, pseudo_labels_root=None, seq=None, frame=None):
    """Per-point classes (N,) in 1..19 (255 = no valid class) for the chosen semantic prior.

    semantics='none' gives every point the same class 1, i.e. plain occupancy.
    Pseudo-labels are stored with original SemanticKITTI ids (uint32, instance id in the
    upper 16 bits), like the SemanticKITTI point labels.
    """
    if semantics == 'none':
        return np.ones(num_points, dtype=np.int32)
    if pseudo_labels_root is None:
        raise ValueError(f"semantics='{semantics}' requires --pseudo_labels_root")
    path = pseudo_label_path(pseudo_labels_root, semantics, seq, frame)
    labels = np.fromfile(path, dtype=np.uint32) & 0xFFFF
    if labels.shape[0] != num_points:
        raise ValueError(f'{path}: {labels.shape[0]} labels for a scan of {num_points} points')
    labels = learning_map[labels]
    labels[labels == 0] = UNKNOWN
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------

def _voxel_indices(points):
    """Flat voxel index of every point inside the grid, and the mask of those points."""
    device = points.device
    grid_min = torch.tensor(GRID_MIN, device=device)
    grid_max = torch.tensor(GRID_MAX, device=device)
    inside = ((points >= grid_min) & (points <= grid_max)).all(dim=1)
    ijk = torch.floor((points[inside] - grid_min) / VOXEL_SIZE).long()
    # Points exactly on the max boundary fall one voxel outside the grid.
    in_grid = ((ijk >= 0) & (ijk < torch.tensor(GRID_SIZE, device=device))).all(dim=1)
    mask = inside.clone()
    mask[inside] = in_grid
    ijk = ijk[in_grid]
    flat = ijk[:, 0] * (GRID_SIZE[1] * GRID_SIZE[2]) + ijk[:, 1] * GRID_SIZE[2] + ijk[:, 2]
    return flat, mask


def _class_votes(flat_indices, labels):
    """Per-voxel histogram (X, Y, Z, 21) of labels 0..19, with 255 counted in the last bin."""
    device = labels.device
    bins = torch.full((256,), NUM_CLASSES, dtype=torch.long, device=device)
    bins[:NUM_CLASSES] = torch.arange(NUM_CLASSES, device=device)
    one_hot = torch.nn.functional.one_hot(bins[labels.long()], num_classes=NUM_CLASSES + 1).float()
    votes = torch.zeros((GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2], NUM_CLASSES + 1), device=device)
    votes.index_add_(0, flat_indices, one_hot)
    return votes.view(*GRID_SIZE, NUM_CLASSES + 1)


def sample_free_space(points, labels):
    """Append free-space samples (label 0) every RAY_STEP meters along each sensor ray.

    Rays start at the sensor origin (0, 0, 0). Sampling stops one step before the hit
    point, so the samples never land in the voxel of the point itself.
    """
    norm = points.norm(dim=1)
    distances = norm - RAY_STEP
    directions = points / norm.unsqueeze(1)

    max_steps = int((distances.max() / RAY_STEP).ceil().item())
    t = torch.arange(0, max_steps, device=points.device).unsqueeze(0) * RAY_STEP  # [1, S]
    t = t.expand(points.shape[0], -1)                                             # [N, S]
    valid = t < distances.unsqueeze(1)
    samples = (directions.unsqueeze(1) * t.unsqueeze(2))[valid]                    # [M, 3]

    free_labels = torch.zeros(samples.shape[0], dtype=labels.dtype, device=labels.device)
    return torch.cat([points, samples], dim=0), torch.cat([labels, free_labels], dim=0)


def voxelize_vanilla(points, labels):
    """Visibility prior 'none': occupied voxels get their majority class, all others are 0.

    Voxels whose points all lack a valid class are set to 255.
    """
    grid = torch.zeros(GRID_SIZE, dtype=torch.int16, device=points.device)
    flat, mask = _voxel_indices(points)
    labels = labels[mask]
    votes = _class_votes(flat, labels)

    has_points = votes.sum(dim=-1) > 0
    has_class = votes[..., 1:NUM_CLASSES].sum(dim=-1) > 0
    majority = votes[..., 1:NUM_CLASSES].argmax(dim=-1) + 1

    grid[has_class] = majority[has_class].to(torch.int16)
    grid[has_points & ~has_class] = UNKNOWN
    return grid


def voxelize_dense(points, labels):
    """Visibility prior 'dense': occupied voxels get their majority class, voxels crossed
    by a ray (and not occupied) are observed-empty (0), and all others are unknown (255)."""
    points, labels = sample_free_space(points, labels)
    flat, mask = _voxel_indices(points)
    votes = _class_votes(flat, labels[mask])

    grid = torch.full(GRID_SIZE, UNKNOWN, dtype=torch.int16, device=points.device)
    has_class = votes[..., 1:NUM_CLASSES].sum(dim=-1) > 0
    has_free = votes[..., 0] > 0
    majority = votes[..., 1:NUM_CLASSES].argmax(dim=-1) + 1

    grid[has_class] = majority[has_class].to(torch.int16)
    grid[has_free & ~has_class] = 0
    return grid


def build_input(points, labels, visibility, device='cpu'):
    """Voxelize a scan with its per-point classes into the (256, 256, 32) uint8 network input."""
    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels, dtype=torch.int64, device=device)
    if visibility == 'none':
        grid = voxelize_vanilla(points, labels)
    elif visibility == 'dense':
        grid = voxelize_dense(points, labels)
    else:
        raise ValueError(f'Unknown visibility prior: {visibility}')
    return grid.cpu().numpy().astype(np.uint8)
