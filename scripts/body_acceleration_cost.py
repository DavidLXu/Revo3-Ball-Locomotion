"""Causal substep acceleration cost; does not modify physics or actions."""
import math
import numpy as np
import torch
import warp as wp

CONFIG = dict(version=1, filter_tau_s=.02, angular_scale_radps2=50.,
              relative_vertical_scale_mps2=5., clip_per_component=16.,
              initial_filtered_velocity='zero', aggregation='substep_mean')


def validate_metadata(cfg, reward):
    declared = (reward or {}).get('body_acceleration_cost')
    weight = cfg.get('body_acceleration_weight', 0.)
    if weight or declared is not None:
        assert weight > 0 and declared == CONFIG
        assert reward['body_acceleration_weight'] == weight
        assert cfg['body_motion_substeps']


def native_update(previous, velocity, timestep):
    alpha = 1. - math.exp(-timestep / CONFIG['filter_tau_s'])
    filtered = (1. - alpha) * previous + alpha * velocity
    acceleration = (filtered - previous) / timestep
    cost = min(float(np.dot(acceleration[:3], acceleration[:3])) / 2500., 16.)
    cost += min(float(acceleration[3] ** 2) / 25., 16.)
    return filtered, cost


@wp.kernel
def accumulate_acceleration(qvel: wp.array2d[float], ball_dof: int,
                            timestep: float, alpha: float,
                            filtered: wp.array2d[float], total: wp.array[float]):
    i = wp.tid()
    angular = float(0.)
    vertical = float(0.)
    for j in range(4):
        velocity = float(0.)
        if j < 3:
            velocity = qvel[i, j + 3]
        else:
            velocity = qvel[i, 2] - qvel[i, ball_dof + 2]
        old = filtered[i, j]
        new = (1. - alpha) * old + alpha * velocity
        acceleration = (new - old) / timestep
        filtered[i, j] = new
        if j < 3:
            angular += acceleration * acceleration / 2500.
        else:
            vertical = acceleration * acceleration / 25.
    total[i] += wp.min(angular, 16.) + wp.min(vertical, 16.)


class WarpBodyAcceleration:
    def __init__(self, num_envs, device):
        self.filtered = torch.zeros((num_envs, 4), device=device)
        self.total = torch.zeros(num_envs, device=device)
        self.cost = torch.zeros_like(self.total)
        self.filtered_wp = wp.from_torch(self.filtered)
        self.total_wp = wp.from_torch(self.total)
        self.num_envs = num_envs

    def reset(self, ids):
        self.filtered[ids] = 0
        self.total[ids] = 0
        self.cost[ids] = 0

    def begin(self):
        self.total.zero_()

    def step(self, qvel, ball_dof, timestep, device):
        alpha = 1. - math.exp(-timestep / CONFIG['filter_tau_s'])
        wp.launch(accumulate_acceleration, dim=self.num_envs,
                  inputs=[qvel, ball_dof, timestep, alpha, self.filtered_wp, self.total_wp],
                  device=device)

    def finish(self, decimation):
        self.cost.copy_(self.total / decimation)
