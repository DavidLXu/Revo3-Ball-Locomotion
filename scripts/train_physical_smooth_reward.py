"""Bounded moving-policy adaptation with explicit smoothness reward metadata."""
from train_target_startup_policy import StartupPracticeRunner
from train_moving_policy_smooth import main


REWARD_CONFIG = dict(version=1, body_motion_weight=3., velocity_error_huber_weight=24.,
    joint_target_rate_weight=.02, joint_target_acceleration_weight=2e-5,
    survival_reward_rate=5., survival_offset_position='after_rate_clamp_before_dt',
    failure_penalty=5.)


class PhysicalSmoothRewardRunner(StartupPracticeRunner):
    physical_smooth_reward_config = REWARD_CONFIG
    reward_revision = 'physical_smooth_reward_v1'
    reward_configuration_overrides = {key: REWARD_CONFIG[key] for key in (
        'body_motion_weight', 'velocity_error_huber_weight', 'joint_target_rate_weight',
        'joint_target_acceleration_weight', 'survival_reward_rate')}

    def __init__(self, env, *args, **kwargs):
        for key, value in self.reward_configuration_overrides.items():
            assert env.cfg[key] == value, key
        assert env.cfg['failure_penalty'] == REWARD_CONFIG['failure_penalty']
        super().__init__(env, *args, **kwargs)

    def additional_checkpoint_metadata(self):
        return {**super().additional_checkpoint_metadata(),
                'physical_smooth_reward': dict(self.physical_smooth_reward_config)}


if __name__ == '__main__':
    main(runner_type=PhysicalSmoothRewardRunner)
