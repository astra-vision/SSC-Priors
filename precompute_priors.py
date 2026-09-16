# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Precompute the network inputs (semantic + visibility priors) of every SemanticKITTI SSC scan.

Building the dense visibility prior takes ~1 s per scan on CPU and much less on a GPU, so
precomputing the inputs once makes training and evaluation much faster. Inputs are written to
    {output_root}/{semantics}/{visibility}/{seq}/{frame}.npy      (uint8, 256x256x32, see priors.py)
Existing files are skipped, so an interrupted run can simply be restarted.

Examples:
    # Training inputs: WaffleIron semantic prior + dense visibility prior (sequences 00-10)
    python precompute_priors.py --kitti_root <KITTI_ROOT> --pseudo_labels_root <PSEUDO_LABELS_ROOT> --semantics waffleiron --visibility dense --output_root <PRECOMPUTED_ROOT>

    # Evaluation inputs: WaffleIron-TTA semantic prior + dense visibility prior (sequence 08)
    python precompute_priors.py --kitti_root <KITTI_ROOT> --pseudo_labels_root <PSEUDO_LABELS_ROOT> --semantics waffleiron_tta --visibility dense --splits val --output_root <PRECOMPUTED_ROOT>
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import priors
from data.semantic_kitti import split_sequences, ssc_frames


def get_args():
    parser = argparse.ArgumentParser(description='Precompute the SSC network inputs')
    parser.add_argument('--kitti_root', required=True, help='SemanticKITTI root, containing dataset/sequences/')
    parser.add_argument('--semantics', required=True, choices=priors.SEMANTICS)
    parser.add_argument('--visibility', required=True, choices=priors.VISIBILITY)
    parser.add_argument('--pseudo_labels_root', default=None,
                        help='Root of the pseudo-label trees ({root}/{semantics}/sequences/...)')
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val'],
                        help="'train' = sequences 00-07, 09, 10 (all SSC scans); 'val' = sequence 08 (every 5th scan)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=None, help='Only process the first N scans (smoke test)')
    return parser.parse_args()


def main():
    args = get_args()
    kitti_yaml = priors.load_kitti_yaml()
    learning_map = priors.build_learning_map(kitti_yaml)

    frames = []
    for split in args.splits:
        frames += ssc_frames(args.kitti_root, split_sequences(split, kitti_yaml), every_5th=(split == 'val'))
    frames = frames[:args.limit]

    out_dir = os.path.join(args.output_root, args.semantics, args.visibility)
    print(f'{len(frames)} scans -> {out_dir} (device: {args.device})')
    skipped = 0
    for seq, frame in tqdm(frames, desc=f'{args.semantics}/{args.visibility}', unit='scan', dynamic_ncols=True):
        out_path = os.path.join(out_dir, seq, f'{frame}.npy')
        if os.path.exists(out_path):
            skipped += 1
            continue
        points = priors.load_scan(args.kitti_root, seq, frame)
        labels = priors.load_point_semantics(args.semantics, points.shape[0], learning_map,
                                             args.pseudo_labels_root, seq, frame)
        grid = priors.build_input(points, labels, args.visibility, device=args.device)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_path = out_path[:-len('.npy')] + '.tmp.npy'
        np.save(tmp_path, grid)
        os.replace(tmp_path, out_path)  # never leave a truncated file behind
    print(f'Done ({skipped} existing scans skipped).')


if __name__ == '__main__':
    main()
