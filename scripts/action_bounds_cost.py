"""Metadata for the existing unclipped-action excess reward term."""
import math

CONFIG = dict(version=1, input='raw_policy_action_before_clamp',
              formula='mean(relu(abs(action)-1)^2).clamp(max=4)',
              aggregation='mean_over_21_actuators', maximum=4.)


def validate_metadata(cfg, reward):
    # Older standalone action-bounds experiments predate smooth-reward metadata.
    if reward is None:
        return
    weight=cfg.get('direct_action_bounds_weight',0.)
    assert math.isfinite(weight) and weight>=0
    assert reward.get('direct_action_bounds_weight',0.)==weight, 'Action bounds weight disagrees with reward metadata'
    declared=reward.get('action_bounds_cost')
    if weight:
        assert declared==CONFIG, 'Action bounds cost definition must be preserved'
        assert not cfg.get('residual_teacher')
    else:
        assert declared is None
