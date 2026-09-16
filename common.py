# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model construction, checkpoint I/O and prediction helpers shared by train.py and eval.py."""

import os

import numpy as np
import torch

import priors
from models.networks import AutoEncoderGroupSkip

# Architecture of the released SemCity-AE models.
MODEL_CONFIG = {
    'num_class': priors.NUM_CLASSES,
    'geo_feat_channels': 16,
    'feat_channel_up': 64,
    'mlp_hidden_channels': 256,
    'mlp_hidden_layers': 4,
    'padding_mode': 'replicate',
    'z_down': False,
    'pos': True,
}

# Which semantic priors a model may be evaluated with, given the one it was trained with.
# A model trained on WaffleIron pseudo-labels is evaluated with WaffleIron-TTA pseudo-labels
# in the paper: the input format is identical, only the labels are more accurate.
COMPATIBLE_SEMANTICS = {
    'none': ('none',),
    'waffleiron': ('waffleiron', 'waffleiron_tta'),
}


def build_model(model_config=None):
    return AutoEncoderGroupSkip(**(model_config or MODEL_CONFIG))


def make_checkpoint(model, epoch, semantics, visibility, **extra):
    """Checkpoint dict: weights, epoch and the config needed to rebuild and feed the model."""
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'config': {
            'model': dict(MODEL_CONFIG),
            'semantics': semantics,
            'visibility': visibility,
        },
    }
    checkpoint.update(extra)
    return checkpoint


def load_checkpoint(path, device='cuda'):
    """Build the model stored in a checkpoint. Returns (model, config, checkpoint)."""
    checkpoint = torch.load(path, map_location='cpu')
    if 'config' not in checkpoint:
        raise KeyError(f'{path} has no "config" entry; convert it with tools/export_checkpoint.py')
    config = checkpoint['config']
    model = build_model(config['model'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device), config, checkpoint


def check_semantics(trained_with, evaluated_with):
    if evaluated_with not in COMPATIBLE_SEMANTICS[trained_with]:
        raise ValueError(f"A model trained with semantics='{trained_with}' cannot be evaluated with "
                         f"semantics='{evaluated_with}' (allowed: {COMPATIBLE_SEMANTICS[trained_with]})")


def predict_grid(logits, query_coords):
    """Scatter the argmax of the per-query logits (B, N, C) into (B, 256, 256, 32) class grids."""
    labels = logits.argmax(dim=-1)
    grid = torch.zeros((logits.shape[0], *priors.GRID_SIZE), dtype=torch.long, device=logits.device)
    for b in range(logits.shape[0]):
        c = query_coords[b]
        grid[b, c[:, 0], c[:, 1], c[:, 2]] = labels[b]
    return grid


def save_prediction(grid, learning_map_inv, path):
    """Write a (256, 256, 32) class grid as a SemanticKITTI voxel label file (uint16, original ids)."""
    lut = np.zeros(max(learning_map_inv.keys()) + 1, dtype=np.uint16)
    lut[list(learning_map_inv.keys())] = list(learning_map_inv.values())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lut[grid.reshape(-1)].astype(np.uint16).tofile(path)


def ssc_scores(metrics):
    """(completion IoU %, semantic mIoU %, per-class IoU array) from an SSCMetrics accumulator."""
    conf = metrics.get_confusion()
    iou = conf[1:, 1:].sum() / (conf.sum() - conf[0, 0] + 1e-8) * 100
    _, per_class = metrics.getIoU()
    miou = per_class[1:].mean() * 100
    return iou, miou, per_class
