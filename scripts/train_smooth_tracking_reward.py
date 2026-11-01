"""Bounded candidate with stronger speed tracking and target acceleration cost."""
from train_physical_smooth_reward import PhysicalSmoothRewardRunner
from train_moving_policy_smooth import main


REWARD_CONFIG = dict(version=2, body_motion_weight=3., velocity_error_huber_weight=32.,
    joint_target_rate_weight=.02, joint_target_acceleration_weight=4e-5,
    survival_reward_rate=10., survival_offset_position='after_rate_clamp_before_dt',
    failure_penalty=5.)


class SmoothTrackingRewardRunner(PhysicalSmoothRewardRunner):
    physical_smooth_reward_config = REWARD_CONFIG
    reward_revision = 'physical_smooth_reward_v2'
    reward_configuration_overrides = {key: REWARD_CONFIG[key] for key in (
        'body_motion_weight', 'velocity_error_huber_weight', 'joint_target_rate_weight',
        'joint_target_acceleration_weight', 'survival_reward_rate')}


if __name__ == '__main__':
    main(runner_type=SmoothTrackingRewardRunner)
