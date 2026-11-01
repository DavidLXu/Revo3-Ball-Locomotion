"""Bounded PPO with a mixture of immediate and settled target-speed starts."""
import copy

from startup_settle_practice import (SETTLE_CONFIG, install_settled_startup_practice,
                                    settled_startup_report)
from train_target_startup_policy import StartupPracticeRunner
from train_moving_policy_smooth import main


class SettledStartupRunner(StartupPracticeRunner):
    settled_startup_practice_config = SETTLE_CONFIG

    def load(self, path, *args, **kwargs):
        result = super().load(path, *args, **kwargs)
        install_settled_startup_practice(self.env)
        return result

    def additional_checkpoint_metadata(self):
        return dict(super().additional_checkpoint_metadata(),
                    settled_startup_practice=copy.deepcopy(self.settled_startup_practice_config),
                    settled_startup_exposure=settled_startup_report(self.env))


if __name__ == '__main__':
    main(runner_type=SettledStartupRunner)
