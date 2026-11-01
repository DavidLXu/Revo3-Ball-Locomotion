"""Remove world yaw without removing physical roll or pitch.

The horizontal polar factor gives a yaw-equivariant heading even with tilt.
Its only singularity is an upside-down orientation, outside the fall limit.
Vectors are expressed in the reset-aligned heading frame; physics stays world-frame.
"""
import numpy as np
import torch


def heading_inverse_numpy(rotation):
    angle = np.arctan2(rotation[1, 0] - rotation[0, 1], rotation[0, 0] + rotation[1, 1])
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])


def heading_inverse_torch(rotation):
    angle = torch.atan2(rotation[:, 1, 0] - rotation[:, 0, 1], rotation[:, 0, 0] + rotation[:, 1, 1])
    c, s = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, s, zero, -s, c, zero, zero, zero, one), -1).reshape(-1, 3, 3)
