# Copyright (c) 2026 Valeo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from SemCity
# (https://github.com/zoomin-lee/SemCity), Copyright (c) 2024 Jumin Lee,
# licensed under the MIT License. See THIRD-PARTY-NOTICES.txt.

import torch


def get_query(voxel_label, num_class=20, grid_size=(256, 256, 32), max_points=400000, num_invalid=None, is_train=True):
    """Voxels at which the decoder is queried, with their labels and normalized coordinates.

    Training: all occupied voxels, then randomly drawn empty voxels up to `max_points`, then
    the invalid voxels (label 255, ignored by the losses). Evaluation: every voxel of the grid.

    Returns:
        query: (N, 3) coordinates normalized to [-1, 1]
        xyz_label: (N,) class of each queried voxel
        xyz_center: (N, 3) integer voxel coordinates (as float)
    """
    voxel_label = torch.from_numpy(voxel_label).to(torch.int16)
    xyzl_list = []

    if is_train:
        mask = (voxel_label > 0) & (voxel_label < num_class)
    else:
        mask = voxel_label > -1

    xyz = mask.nonzero(as_tuple=False)
    labels = voxel_label[mask].unsqueeze(1)
    xyzl = torch.cat([labels, xyz], dim=1)
    xyzl = xyzl[torch.randperm(xyzl.shape[0])]   # shuffle
    xyzl_list.append(xyzl)

    if is_train:
        # check how much space is still in the query and randomly select voxels with label 0 to fill the query
        num_free = max_points - xyzl.shape[0]
        if num_free > 0:
            mask = voxel_label == 0
            xyz = mask.nonzero(as_tuple=False)
            labels = voxel_label[mask].unsqueeze(1)
            xyz0 = torch.cat([labels, xyz], dim=1)
            take = min(num_free, xyz0.shape[0]) # for safety, but normally it's equal to num_free
            idx = torch.randperm(xyz0.shape[0])[:take]
            xyzl_list.append(xyz0[idx])

        # finally fill the query with all of the unobserve voxels label 255
        mask = voxel_label == 255
        xyz = mask.nonzero(as_tuple=False)
        labels = voxel_label[mask].unsqueeze(1)
        xyz255 = torch.cat([labels, xyz], dim=1)
        if num_invalid:
            take = min(num_invalid, xyz255.shape[0])
        else:
            take = xyz255.shape[0]
        idx = torch.randperm(xyz255.shape[0])[:take]
        xyzl_list.append(xyz255[idx])

    xyzl = torch.cat(xyzl_list, dim=0)

    xyz_label = xyzl[:, 0]
    xyz_center = xyzl[:, 1:].float()

    norm = torch.tensor(grid_size, dtype=torch.float32)
    query = 2 * xyz_center / (norm - 1) - 1
    query = query.clamp(-1, 1)

    return query, xyz_label, xyz_center
