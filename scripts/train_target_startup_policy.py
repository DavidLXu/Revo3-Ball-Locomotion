"""Bounded moving-policy PPO with explicit target-speed startup practice."""
from startup_target_practice import install_startup_target_practice
from train_moving_policy_smooth import MovingPolicyRunner, main


class StartupPracticeRunner(MovingPolicyRunner):
    startup_target_practice_config = dict(
        version=1, speed_mps=.1, population='startup_worlds',
        nonzero_commands_only=True, preserve_stops_directions_holds=True)

    def load(self, path, *args, **kwargs):
        result = super().load(path, *args, **kwargs)
        install_startup_target_practice(self.env, self.startup_target_practice_config['speed_mps'])
        return result

    def additional_checkpoint_metadata(self):
        return {'startup_target_practice': dict(self.startup_target_practice_config)}


if __name__ == '__main__':
    main(runner_type=StartupPracticeRunner)
