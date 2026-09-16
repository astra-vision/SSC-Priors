# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class SSCMetrics:
    """Confusion-matrix accumulator for semantic scene completion.

    Maintains an (n_classes x n_classes) confusion matrix indexed as
    ``conf[prediction, ground_truth]``. Class 0 is the empty/free class and
    classes ``1..n_classes-1`` are semantic classes. This is a standard
    intersection-over-union evaluator built from the usual confusion-matrix
    definitions (TP on the diagonal, FP across a prediction row, FN down a
    ground-truth column); the SemanticKITTI evaluation convention of dropping
    label-255 and invalid (never-observed) voxels is applied in
    ``get_eval_mask`` (see THIRD-PARTY-NOTICES.txt for the SemanticKITTI
    dev-kit attribution).
    """

    def __init__(self, n_classes, ignore=None):
        self.n_classes = n_classes
        self.ignore = np.array([] if ignore is None else ignore, dtype=np.int64)
        self.include = np.array(
            [c for c in range(n_classes) if c not in self.ignore], dtype=np.int64
        )
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def get_eval_mask(self, labels, invalid_voxels, specified_points=None):
        """Boolean mask selecting the voxels to score.

        Voxels labelled 255 (unknown) or flagged invalid (i.e. never observed
        by a laser ray) are excluded, following the SemanticKITTI dev-kit
        evaluation protocol. When ``specified_points`` is given, scoring is
        further restricted to those voxel coordinates.
        """
        mask = np.ones_like(labels, dtype=np.bool_)
        if specified_points is not None:
            mask[0, :, :, :] = False
            specified_points = specified_points.squeeze(0).cpu().numpy()
            mask[0, specified_points[:, 0], specified_points[:, 1], specified_points[:, 2]] = True
        mask[labels == 255] = False
        mask[invalid_voxels == 1] = False
        return mask

    def _confusion(self, preds, targets):
        """Confusion matrix for a single (preds, targets) pair, conf[pred, gt]."""
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        assert preds.shape == targets.shape
        k = self.n_classes
        flat = preds * k + targets
        return np.bincount(flat, minlength=k * k).reshape(k, k)

    @staticmethod
    def _completion_iou(conf):
        """Geometric completion IoU: any-occupied (class>=1) vs free (class 0)."""
        intersection = conf[1:, 1:].sum()          # predicted occupied AND gt occupied
        union = conf.sum() - conf[0, 0]            # predicted occupied OR gt occupied
        return intersection / (union + 1e-15)

    def _per_class_iou(self, conf):
        conf = conf.copy()
        conf[:, self.ignore] = 0
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp                 # predicted c but gt != c
        fn = conf.sum(axis=0) - tp                 # gt c but predicted != c
        return tp / (tp + fp + fn + 1e-15)

    def addBatch(self, preds, targets):
        """Accumulate one batch; return its (completion IoU, semantic mIoU) in %.

        The returned values are per-batch progress indicators; the reported
        metrics are computed from the accumulated matrix via :meth:`getIoU` and
        :meth:`get_confusion`.
        """
        conf = self._confusion(preds, targets)
        self.conf_matrix += conf

        iou = self._completion_iou(conf) * 100.0
        per_class = self._per_class_iou(conf)
        present = conf.sum(axis=0) > 0             # classes appearing in this batch's gt
        present[0] = False                         # exclude the empty class from mIoU
        miou = (per_class[present].mean() if present.any() else 0.0) * 100.0
        return iou, miou

    def getStats(self):
        conf = self.conf_matrix.copy()
        conf[:, self.ignore] = 0
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getIoU(self):
        """Return (mean IoU over included classes, per-class IoU array)."""
        tp, fp, fn = self.getStats()
        iou = tp / (tp + fp + fn + 1e-15)
        iou_mean = iou[self.include].mean()
        return iou_mean, iou

    def get_confusion(self):
        return self.conf_matrix.copy()
