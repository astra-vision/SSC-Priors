# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from SemCity
# (https://github.com/zoomin-lee/SemCity), Copyright (c) 2024 Jumin Lee,
# licensed under the MIT License. See THIRD-PARTY-NOTICES.txt.

"""
Train SemCity-AE for semantic scene completion on SemanticKITTI, with or without priors.

The model is trained from scratch with cross-entropy + Lovasz-softmax losses and validated
on sequence 08 after every epoch. Outputs go to {save_path}/: training_args.json,
training_log.txt, tensorboard logs (tb/) and checkpoints/{last,best_miou}.pt.

Examples:
    # Vanilla SemCity-AE (no prior)
    python train.py --kitti_root <KITTI_ROOT> --semantics none --visibility none --save_path runs/vanilla

    # SemCity-AE + WaffleIron semantic prior + dense visibility prior
    python train.py --kitti_root <KITTI_ROOT> --semantics waffleiron --visibility dense --pseudo_labels_root <PSEUDO_LABELS_ROOT> --precomputed_root <PRECOMPUTED_ROOT> --save_path runs/wi_dense
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import common
import priors
from data.semantic_kitti import CLASS_NAMES, SemanticKITTISSC
from losses import lovasz_softmax
from metrics import SSCMetrics

# Number of voxels per class (0 = empty) over the training set; the cross-entropy weight of
# class c is (max_freq / freq_c) ** (1/3), and the weight of the empty class is set to 7.
CLASS_VOXEL_COUNTS = np.array([
    5417730211, 15783539, 125136, 118809, 646799, 821951, 262978, 283696, 204750, 61688703,
    4502961, 44883650, 2269923, 56840218, 15719652, 158442623, 2061623, 36970523, 1151988, 334146,
])
EMPTY_CLASS_WEIGHT = 7.0


def get_args():
    parser = argparse.ArgumentParser(description='Train SemCity-AE with semantic / visibility priors')
    parser.add_argument('--kitti_root', required=True, help='SemanticKITTI root, containing dataset/sequences/')
    parser.add_argument('--semantics', default='none', choices=['none', 'waffleiron'],
                        help='Semantic prior: none, or WaffleIron pseudo-labels')
    parser.add_argument('--visibility', default='none', choices=priors.VISIBILITY,
                        help="Visibility prior: 'none' (vanilla occupancy) or 'dense' (ray-cast free space)")
    parser.add_argument('--pseudo_labels_root', default=None,
                        help='Root of the pseudo-label trees ({root}/waffleiron/sequences/...); needed for --semantics waffleiron')
    parser.add_argument('--precomputed_root', default=None,
                        help='Root of the inputs written by precompute_priors.py. If omitted, inputs are built on the fly')
    parser.add_argument('--save_path', required=True, help='Output directory of this run')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', default=None, help='Resume from a checkpoints/last.pt of this script')
    parser.add_argument('--max_iters_per_epoch', type=int, default=None,
                        help='Stop each epoch (train and val) after this many iterations; for smoke tests only')
    return parser.parse_args()


class Tee:
    """Duplicate a stream into a log file."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def to_cuda(batch):
    return {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def compute_loss(logits, batch, ce_loss):
    """Cross-entropy on the query points + Lovasz-softmax on the voxel grid (255 = ignored)."""
    num_class = logits.shape[-1]
    labels, coords = batch['query_labels'], batch['query_coords']
    loss_ce = ce_loss(logits.reshape(-1, num_class), labels.reshape(-1))

    probs = torch.softmax(logits, dim=2)
    probs_grid = torch.zeros((logits.shape[0], *priors.GRID_SIZE, num_class), device=logits.device)
    labels_grid = torch.zeros((logits.shape[0], *priors.GRID_SIZE), device=logits.device)
    for b in range(logits.shape[0]):
        c = coords[b]
        probs_grid[b, c[:, 0], c[:, 1], c[:, 2]] = probs[b]
        labels_grid[b, c[:, 0], c[:, 1], c[:, 2]] = labels[b].float()
    loss_lovasz = lovasz_softmax(probs_grid.permute(0, 4, 1, 2, 3), labels_grid, ignore=priors.UNKNOWN)
    return loss_ce, loss_lovasz


def accumulate(metrics, logits, batch):
    pred = common.predict_grid(logits.detach(), batch['query_coords'])
    target, invalid = batch['target'], batch['invalid']
    mask = torch.from_numpy(metrics.get_eval_mask(target.cpu().numpy(), invalid.cpu().numpy())).to(target.device)
    metrics.addBatch(pred[mask].cpu().numpy(), target[mask].cpu().numpy())


def train_one_epoch(model, loader, optimizer, scaler, ce_loss, writer, epoch, global_step, max_iters):
    model.train()
    metrics = SSCMetrics(priors.NUM_CLASSES)
    pbar = tqdm(loader, desc=f'Train epoch {epoch}', unit='scan', dynamic_ncols=True)
    for it, batch in enumerate(pbar):
        if max_iters is not None and it >= max_iters:
            break
        batch = to_cuda(batch)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(batch['input'], batch['query'].float())
            loss_ce, loss_lovasz = compute_loss(logits, batch, ce_loss)
            loss = loss_ce + loss_lovasz

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        accumulate(metrics, logits, batch)
        pbar.set_postfix(loss_ce=f'{loss_ce.item():.3f}', loss_lovasz=f'{loss_lovasz.item():.3f}')
        writer.add_scalar('train/loss_ce', loss_ce.item(), global_step)
        writer.add_scalar('train/loss_lovasz', loss_lovasz.item(), global_step)
        writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
        global_step += 1

    iou, miou, _ = common.ssc_scores(metrics)
    writer.add_scalar('train/IoU', iou, epoch)
    writer.add_scalar('train/mIoU', miou, epoch)
    print(f'Epoch {epoch} train: IoU {iou:.2f}  mIoU {miou:.2f}')
    return global_step


@torch.no_grad()
def validate(model, loader, writer, epoch, max_iters):
    model.eval()
    metrics = SSCMetrics(priors.NUM_CLASSES)
    for it, batch in enumerate(tqdm(loader, desc=f'Val epoch {epoch}', unit='scan', dynamic_ncols=True)):
        if max_iters is not None and it >= max_iters:
            break
        batch = to_cuda(batch)
        logits = model(batch['input'], batch['query'].float())
        accumulate(metrics, logits, batch)

    iou, miou, per_class = common.ssc_scores(metrics)
    writer.add_scalar('val/IoU', iou, epoch)
    writer.add_scalar('val/mIoU', miou, epoch)
    for name, value in zip(CLASS_NAMES, per_class[1:]):
        writer.add_scalar(f'val_class_IoU/{name}', value * 100, epoch)
    print(f'Epoch {epoch} val:   IoU {iou:.2f}  mIoU {miou:.2f}')
    return miou


def main():
    args = get_args()
    if args.semantics != 'none' and args.pseudo_labels_root is None and args.precomputed_root is None:
        raise ValueError('--semantics waffleiron needs --pseudo_labels_root (or --precomputed_root)')

    os.makedirs(os.path.join(args.save_path, 'checkpoints'), exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(os.path.join(args.save_path, 'training_log.txt'), 'a'))
    with open(os.path.join(args.save_path, 'training_args.json'), 'w') as f:
        json.dump({'command': ' '.join(sys.argv), 'args': vars(args)}, f, indent=2)
    print(f'Arguments: {vars(args)}')

    dataset_kwargs = dict(kitti_root=args.kitti_root, semantics=args.semantics, visibility=args.visibility,
                          pseudo_labels_root=args.pseudo_labels_root, precomputed_root=args.precomputed_root)
    # Batch size 1: the number of supervised query points varies from scan to scan.
    train_set = SemanticKITTISSC(split='train', augment=True, **dataset_kwargs)
    val_set = SemanticKITTISSC(split='val', **dataset_kwargs)
    print(f'Train scans: {len(train_set)}  Val scans: {len(val_set)}')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True,
                                               persistent_workers=args.num_workers > 0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    model = common.build_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    weights = CLASS_VOXEL_COUNTS / CLASS_VOXEL_COUNTS.sum()
    weights = np.power(weights.max() / weights, 1 / 3.0)
    weights[0] = EMPTY_CLASS_WEIGHT
    ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).cuda(),
                                        ignore_index=priors.UNKNOWN)

    start_epoch, global_step, best_miou = 1, 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_miou = checkpoint['best_miou']
        print(f'Resumed from {args.resume} (epoch {checkpoint["epoch"]})')

    writer = SummaryWriter(os.path.join(args.save_path, 'tb'))
    for epoch in range(start_epoch, args.epochs + 1):
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        global_step = train_one_epoch(model, train_loader, optimizer, scaler, ce_loss, writer, epoch,
                                      global_step, args.max_iters_per_epoch)
        miou = validate(model, val_loader, writer, epoch, args.max_iters_per_epoch)
        scheduler.step()

        best_miou = max(best_miou, miou)
        checkpoint = common.make_checkpoint(model, epoch, args.semantics, args.visibility,
                                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                            global_step=global_step, best_miou=best_miou)
        ckpt_dir = os.path.join(args.save_path, 'checkpoints')
        torch.save(checkpoint, os.path.join(ckpt_dir, 'last.pt'))
        if miou >= best_miou:
            torch.save(checkpoint, os.path.join(ckpt_dir, 'best_miou.pt'))
    writer.close()
    print('Training done. The paper reports the last epoch (checkpoints/last.pt).')


if __name__ == '__main__':
    main()
