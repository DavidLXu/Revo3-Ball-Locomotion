"""Combine the tested smooth-tracking reward with mixed settled starts."""
from train_smooth_tracking_reward import SmoothTrackingRewardRunner
from train_settled_startup_policy import SettledStartupRunner
from train_moving_policy_smooth import main


class SettledSmoothTrackingRunner(SmoothTrackingRewardRunner, SettledStartupRunner):
    """Preserve both reward and startup-distribution metadata on save/resume."""


if __name__ == '__main__':
    main(runner_type=SettledSmoothTrackingRunner)
