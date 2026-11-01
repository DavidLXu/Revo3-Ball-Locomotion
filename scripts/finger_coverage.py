"""Recent per-finger contact participation, shared by training and playback."""
import math


def update_coverage(coverage, contacts, dt, tau):
    return coverage+(1.-math.exp(-dt/tau))*(contacts-coverage)


def coverage_deficit(coverage, target):
    # A finger can lift during a step. Sustained omission incurs a deficit.
    return (1.-coverage/target).clip(0.,1.).mean(-1)
