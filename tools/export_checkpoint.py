# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Export a training checkpoint as a lightweight release checkpoint.

The exported file only contains what eval.py needs: the model weights, the epoch and the
model/prior config. Everything else (optimizer and scheduler states, training arguments,
paths, ...) is dropped. The weights are loaded strictly into the release model before saving,
so the export fails if the checkpoint does not match the released architecture.

Example (run from this repository's root):
    python tools/export_checkpoint.py --input runs/wi_dense/checkpoints/last.pt --output checkpoints/semcity_ae_wi_tta_dense.pt

For checkpoints that do not store their prior config, pass --semantics and --visibility.
"""

import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import priors  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description='Export a lightweight release checkpoint')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--semantics', default=None, choices=['none', 'waffleiron'],
                        help='Semantic prior used for training (read from the checkpoint if stored)')
    parser.add_argument('--visibility', default=None, choices=priors.VISIBILITY,
                        help='Visibility prior used for training (read from the checkpoint if stored)')
    parser.add_argument('--drop_prefix', nargs='*', default=[],
                        help='Drop state-dict entries starting with these prefixes (e.g. unused heads)')
    return parser.parse_args()


def main():
    args = get_args()
    checkpoint = torch.load(args.input, map_location='cpu')
    stored = checkpoint.get('config', {})
    semantics = args.semantics or stored.get('semantics')
    visibility = args.visibility or stored.get('visibility')
    if semantics is None or visibility is None:
        raise ValueError('The checkpoint does not store its priors: pass --semantics and --visibility')
    for key, value in (('semantics', semantics), ('visibility', visibility)):
        if key in stored and stored[key] != value:
            raise ValueError(f'--{key} {value} contradicts the checkpoint ({stored[key]})')

    print(f'Input keys: {sorted(checkpoint.keys())}')
    state_dict = checkpoint['model']
    dropped = [k for k in state_dict if any(k.startswith(p) for p in args.drop_prefix)]
    state_dict = {k: v for k, v in state_dict.items() if k not in dropped}
    if dropped:
        print(f'Dropped {len(dropped)} state-dict entries: {dropped}')

    model = common.build_model()
    model.load_state_dict(state_dict, strict=True)
    exported = common.make_checkpoint(model, checkpoint.get('epoch'), semantics, visibility)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(exported, args.output)
    with open(args.output, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f'Wrote {args.output}: epoch {exported["epoch"]}, config {exported["config"]}')
    print(f'Size {os.path.getsize(args.output) / 2**20:.1f} MiB, sha256 {sha256}')


if __name__ == '__main__':
    main()
