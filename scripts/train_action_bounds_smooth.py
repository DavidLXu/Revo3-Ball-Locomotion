"""Candidate half-ms smoothness trial with explicit raw-action excess cost."""
from action_bounds_cost import CONFIG as BOUNDS_CONFIG, validate_metadata
from train_body_acceleration_smooth import BodyAccelerationSmoothRunner, REWARD_CONFIG as BASE_REWARD
from train_moving_policy_smooth import main


REWARD_CONFIG = {**BASE_REWARD, 'version':5, 'direct_action_bounds_weight':16.,
                 'action_bounds_cost':dict(BOUNDS_CONFIG)}


class ActionBoundsSmoothRunner(BodyAccelerationSmoothRunner):
    physical_smooth_reward_config = REWARD_CONFIG
    reward_revision = 'physical_smooth_reward_v5_action_bounds'
    reward_configuration_overrides = {
        **BodyAccelerationSmoothRunner.reward_configuration_overrides,
        'direct_action_bounds_weight':REWARD_CONFIG['direct_action_bounds_weight']}

    def __init__(self,env,*args,**kwargs):
        validate_metadata(env.cfg,self.physical_smooth_reward_config)
        super().__init__(env,*args,**kwargs)


if __name__ == '__main__':main(runner_type=ActionBoundsSmoothRunner)
