"""Explicit acceleration-reward trial with half-ms physics and100Hz control."""
from body_acceleration_cost import CONFIG as COST_CONFIG, validate_metadata
from physics_integration_adaptation import CONFIG, validate_environment
from train_physical_smooth_reward import PhysicalSmoothRewardRunner
from train_moving_policy_smooth import main


REWARD_CONFIG = dict(version=4, body_motion_weight=3., velocity_error_huber_weight=48.,
    joint_target_rate_weight=.02, joint_target_acceleration_weight=4e-5,
    survival_reward_rate=20., survival_offset_position='after_rate_clamp_before_dt',
    failure_penalty=5., body_acceleration_weight=4., body_acceleration_cost=dict(COST_CONFIG))


class BodyAccelerationSmoothRunner(PhysicalSmoothRewardRunner):
    physical_smooth_reward_config = REWARD_CONFIG
    physics_integration_config = CONFIG
    reward_revision = 'physical_smooth_reward_v4_body_acceleration'
    reward_configuration_overrides = {key: REWARD_CONFIG[key] for key in (
        'body_motion_weight','velocity_error_huber_weight','joint_target_rate_weight',
        'joint_target_acceleration_weight','survival_reward_rate','body_acceleration_weight')}

    def __init__(self, env, *args, **kwargs):
        validate_environment(env.cfg, self.physics_integration_config)
        validate_metadata(env.cfg, self.physical_smooth_reward_config)
        super().__init__(env, *args, **kwargs)

    def additional_checkpoint_metadata(self):
        return {**super().additional_checkpoint_metadata(),
                'physics_integration_adaptation': dict(self.physics_integration_config)}


if __name__ == '__main__':main(runner_type=BodyAccelerationSmoothRunner)
