# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate a SemCity-AE checkpoint on the SemanticKITTI validation set (sequence 08, every 5th scan).

The visibility prior is read from the checkpoint. The semantic prior defaults to the one the
model was trained with; a model trained with WaffleIron pseudo-labels can be evaluated with
WaffleIron-TTA pseudo-labels (--semantics waffleiron_tta), as in the paper.

Examples:
    # Vanilla SemCity-AE
    python eval.py --kitti_root <KITTI_ROOT> --checkpoint checkpoints/semcity_ae_vanilla.pt

    # SemCity-AE + WaffleIron-TTA semantic prior + dense visibility prior
    python eval.py --kitti_root <KITTI_ROOT> --checkpoint checkpoints/semcity_ae_wi_tta_dense.pt --semantics waffleiron_tta --pseudo_labels_root <PSEUDO_LABELS_ROOT>

Add --save_predictions <DIR> to also write the predicted grids as
<DIR>/sequences/08/predictions/{frame}.label (uint16 SemanticKITTI ids, the format of the
official semantic-kitti-api evaluate_completion.py).
"""

import argparse
import os

import torch
from tqdm import tqdm

import common
import priors
from data.semantic_kitti import CLASS_NAMES, SemanticKITTISSC
from metrics import SSCMetrics


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate SemCity-AE on SemanticKITTI val')
    parser.add_argument('--kitti_root', required=True, help='SemanticKITTI root, containing dataset/sequences/')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint written by train.py or a released one')
    parser.add_argument('--semantics', default=None, choices=priors.SEMANTICS,
                        help='Semantic prior at evaluation (default: the one used for training)')
    parser.add_argument('--pseudo_labels_root', default=None,
                        help='Root of the pseudo-label trees ({root}/{semantics}/sequences/...)')
    parser.add_argument('--precomputed_root', default=None,
                        help='Root of the inputs written by precompute_priors.py. If omitted, inputs are built on the fly')
    parser.add_argument('--save_predictions', default=None, help='Directory where to write the predicted grids')
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    model, config, checkpoint = common.load_checkpoint(args.checkpoint)
    model.eval()
    semantics = args.semantics or config['semantics']
    common.check_semantics(config['semantics'], semantics)
    visibility = config['visibility']
    print(f'Checkpoint: {args.checkpoint} (epoch {checkpoint.get("epoch")})')
    print(f'Trained with semantics={config["semantics"]}, visibility={visibility}; '
          f'evaluating with semantics={semantics}, visibility={visibility}')

    dataset = SemanticKITTISSC(args.kitti_root, 'val', semantics, visibility,
                               pseudo_labels_root=args.pseudo_labels_root, precomputed_root=args.precomputed_root)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)

    metrics = SSCMetrics(priors.NUM_CLASSES)
    for batch in tqdm(loader, desc='Evaluating', unit='scan', dynamic_ncols=True):
        net_input = batch['input'].cuda(non_blocking=True)
        query = batch['query'].float().cuda(non_blocking=True)
        logits = model(net_input, query)
        pred = common.predict_grid(logits, batch['query_coords'].cuda()).cpu()

        mask = metrics.get_eval_mask(batch['target'].numpy(), batch['invalid'].numpy())
        metrics.addBatch(pred.numpy()[mask], batch['target'].numpy()[mask])

        if args.save_predictions:
            path = os.path.join(args.save_predictions, 'sequences', batch['seq'][0], 'predictions',
                                f'{batch["frame"][0]}.label')
            common.save_prediction(pred[0].numpy(), dataset.learning_map_inv, path)

    iou, miou, per_class = common.ssc_scores(metrics)
    print(f'\nSemanticKITTI val ({len(dataset)} scans)')
    print(f'  {"class":<16}{"IoU %":>8}')
    for name, value in zip(['empty'] + CLASS_NAMES, per_class):
        print(f'  {name:<16}{value * 100:>8.2f}')
    print(f'\n  IoU (completion): {iou:.2f}')
    print(f'  mIoU (semantics): {miou:.2f}')


if __name__ == '__main__':
    main()
