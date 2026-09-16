# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The inference loop follows eval_kitti.py of WaffleIron
# (https://github.com/valeoai/WaffleIron), Copyright 2022 Valeo Comfort and Driving
# Assistance, licensed under the Apache License 2.0. See THIRD-PARTY-NOTICES.txt.

"""
Generate WaffleIron semantic pseudo-labels for the SemanticKITTI SSC scans.

This runs an unmodified clone of WaffleIron (https://github.com/valeoai/WaffleIron) with
its public SemanticKITTI checkpoint. Compared with WaffleIron's own eval_kitti.py it
  - processes any sequences (eval_kitti.py only iterates over the split of --phase), always
    with deterministic inference-time processing, so the labels of the training sequences
    are produced exactly like those of the validation sequence;
  - only processes the scans that have SSC voxel ground truth (every 5th scan).
With --num_votes > 1 it applies WaffleIron's test-time augmentation (TTA): the per-point
softmax is averaged over the votes before the argmax, as in eval_kitti.py.

Output: {output_dir}/sequences/{seq}/predictions/{frame}.label, one uint32 SemanticKITTI
label id per point of the scan (the format of the SemanticKITTI point labels).

Examples (run from this repository's root):
    # Plain WaffleIron on all SSC scans of sequences 00-10 (training + per-epoch validation)
    python tools/generate_waffleiron_labels.py --waffleiron_root <WAFFLEIRON_REPO> --kitti_root <KITTI_ROOT> --output_dir <PSEUDO_LABELS_ROOT>/waffleiron

    # WaffleIron with 12-vote TTA on the validation sequence 08 (evaluation)
    python tools/generate_waffleiron_labels.py --waffleiron_root <WAFFLEIRON_REPO> --kitti_root <KITTI_ROOT> --output_dir <PSEUDO_LABELS_ROOT>/waffleiron_tta --sequences 08 --num_votes 12 --batch_size 12
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
import yaml
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Generate WaffleIron pseudo-labels for the SemanticKITTI SSC scans')
    parser.add_argument('--waffleiron_root', required=True, help='Clone of https://github.com/valeoai/WaffleIron')
    parser.add_argument('--kitti_root', required=True, help='SemanticKITTI root, containing dataset/sequences/')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', default='configs/WaffleIron-48-256__kitti.yaml',
                        help='WaffleIron config, relative to --waffleiron_root')
    parser.add_argument('--ckpt', default='pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth',
                        help='WaffleIron checkpoint, relative to --waffleiron_root')
    parser.add_argument('--sequences', nargs='+', default=[f'{s:02d}' for s in range(11)],
                        help='Sequences to process (default: 00-10)')
    parser.add_argument('--num_votes', type=int, default=1, help='Number of TTA votes (1 = no TTA)')
    parser.add_argument('--batch_size', type=int, default=1, help='Votes per forward pass; must divide --num_votes')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--skip_existing', action='store_true', help='Skip scans whose label file already exists')
    parser.add_argument('--limit', type=int, default=None, help='Only process the first N scans (smoke test)')
    return parser.parse_args()


def output_path(output_dir, scan_path):
    seq = os.path.basename(os.path.dirname(os.path.dirname(scan_path)))
    frame = os.path.splitext(os.path.basename(scan_path))[0]
    return os.path.join(output_dir, 'sequences', seq, 'predictions', f'{frame}.label')


def main():
    args = get_args()
    assert args.num_votes % args.batch_size == 0, '--batch_size must divide --num_votes'
    waffleiron_root = os.path.abspath(args.waffleiron_root)
    sys.path.insert(0, waffleiron_root)  # WaffleIron's `waffleiron`, `datasets` and `utils` packages
    import waffleiron
    from datasets import Collate, SemanticKITTI
    from waffleiron import Segmenter

    with open(os.path.join(waffleiron_root, args.config)) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(waffleiron_root, 'datasets', 'semantic-kitti.yaml')) as f:
        learning_map_inv = yaml.safe_load(f)['learning_map_inv']
    remap_lut = np.zeros(max(learning_map_inv.keys()) + 100, dtype=np.int32)
    remap_lut[list(learning_map_inv.keys())] = list(learning_map_inv.values())

    # SSC scans = scans with a voxel ground-truth file (every 5th scan of each sequence).
    scans = []
    for seq in args.sequences:
        voxel_files = sorted(glob(os.path.join(args.kitti_root, 'dataset', 'sequences', seq, 'voxels', '*.label')))
        if not voxel_files:
            raise FileNotFoundError(f'No SSC voxel files for sequence {seq} under {args.kitti_root}')
        scans += [p.replace('/voxels/', '/velodyne/')[:-len('.label')] + '.bin' for p in voxel_files]
    if args.skip_existing:
        scans = [s for s in scans if not os.path.exists(output_path(args.output_dir, s))]
    scans = scans[:args.limit]
    print(f'{len(scans)} scans, {args.num_votes} vote(s) each -> {args.output_dir}')
    if not scans:
        return

    # phase='val' selects WaffleIron's inference-time processing (deterministic voxelization,
    # predictions upsampled to every point of the scan, TTA allowed). The frame list of that
    # split is then replaced by the requested scans, each repeated once per vote.
    tta = args.num_votes > 1
    dataset = SemanticKITTI(
        rootdir=args.kitti_root,
        input_feat=config['embedding']['input_feat'],
        voxel_size=config['embedding']['voxel_size'],
        num_neighbors=config['embedding']['neighbors'],
        dim_proj=config['waffleiron']['dim_proj'],
        grids_shape=config['waffleiron']['grids_size'],
        fov_xyz=config['waffleiron']['fov_xyz'],
        phase='val',
        tta=tta,
    )
    dataset.im_idx = [s for s in scans for _ in range(args.num_votes)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True,
                                         drop_last=False, collate_fn=Collate())
    batches_per_scan = args.num_votes // args.batch_size

    net = Segmenter(
        input_channels=config['embedding']['size_input'],
        feat_channels=config['waffleiron']['nb_channels'],
        depth=config['waffleiron']['depth'],
        grid_shape=config['waffleiron']['grids_size'],
        nb_class=config['classif']['nb_class'],
        drop_path_prob=config['waffleiron']['drop'],
    ).cuda()
    state_dict = torch.load(os.path.join(waffleiron_root, args.ckpt), map_location='cuda:0')['net']
    state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    net.compress()
    net.eval()
    if tta:  # WaffleIron's TTA also keeps DropPath active
        for m in net.modules():
            if isinstance(m, waffleiron.backbone.DropPath):
                m.train()

    vote, id_vote = None, 0
    for batch in tqdm(loader, desc='WaffleIron', unit='batch', dynamic_ncols=True):
        net_inputs = (
            batch['feat'].cuda(non_blocking=True),
            batch['cell_ind'].cuda(non_blocking=True),
            batch['occupied_cells'].cuda(non_blocking=True),
            batch['neighbors_emb'].cuda(non_blocking=True),
        )
        upsample = [up.cuda(non_blocking=True) for up in batch['upsample']]
        with torch.autocast('cuda', enabled=True), torch.inference_mode():
            out = net(*net_inputs)
            for b in range(out.shape[0]):
                probs = torch.softmax(out[b, :, upsample[b]].T, dim=1)
                vote = probs if vote is None else vote + probs
        id_vote += 1

        if id_vote == batches_per_scan:
            assert batch['filename'][0] == batch['filename'][-1]
            # Network class c corresponds to learning class c + 1 (class 0 is "ignore").
            label = remap_lut[(vote.argmax(dim=1) + 1).cpu().numpy()].astype(np.uint32)
            path = output_path(args.output_dir, batch['filename'][0])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            label.tofile(path)
            vote, id_vote = None, 0


if __name__ == '__main__':
    main()
