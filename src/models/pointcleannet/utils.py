"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: utils.py
Responsibility: Mathematical helper functions for PointCleanNet quaternion rotation and vector-angle computations.

Attribution:
    This module is adapted from the official PointCleanNet implementation:
    https://github.com/mrakotosaon/pointcleannet
    The adaptation keeps numerical behavior and exposes helpers used by local models.
"""

import torch


def batch_quat_to_rotmat(q, out=None):
    """Convert a batch of quaternions in [a, b, c, d] format to rotation matrices."""

    batchsize = q.size(0)

    if out is None:
        out = torch.FloatTensor(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2 / torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out


def cos_angle(v1, v2):
    """Compute cosine of the angle between batched vectors."""

    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(
        v1.norm(2, 1) * v2.norm(2, 1), min=0.000001
    )
