"""Training-only mixed settled starts using the ordinary command slew limiter."""
import types

import torch


SETTLE_CONFIG = dict(version=1, population='even_local_startup_ids',
                    delays_s=[.5, 1., 2.], moving_seconds=3.,
                    nonzero_initial_commands_only=True)


def install_settled_startup_practice(env):
    assert env.cfg['reversal_sequence_probability'] == 0
    assert env.cfg['command_target_observation']
    assert env.cfg['startup_episode_seconds'] == 3.
    for group in env.groups:
        assert hasattr(group, 'startup_practice_original_sampler')
        assert not hasattr(group, 'settle_original_sampler')
        assert group.startup_env_count >= 6 and not group.external_command
        original = group.sample_commands
        group.settle_original_sampler = original
        ids = torch.arange(group.num_envs, device=env.device)
        group.settle_eligible = (ids < group.startup_env_count) & (ids % 2 == 0)
        delays = torch.tensor(SETTLE_CONFIG['delays_s'], device=env.device)
        group.settle_delay_ticks = torch.round(delays[(ids // 2) % len(delays)] / group.dt).long()
        group.settle_base_horizons = group.episode_horizons.clone()
        group.settle_pending = torch.zeros(group.num_envs, device=env.device, dtype=torch.bool)
        group.settle_episode = torch.zeros_like(group.settle_pending)
        group.settle_target = torch.zeros_like(group.command_targets)
        # Counts at the reward boundary, before any reset: waiting, moving,
        # moving at target speed. The fourth counter counts scheduled episodes.
        group.settle_counts = torch.zeros(4, device=env.device, dtype=torch.long)

        def sample(this, sample_ids, original=original):
            fresh = sample_ids[this.settle_eligible[sample_ids] &
                               (this.episode_length_buf[sample_ids] == 0)]
            release = sample_ids[this.settle_pending[sample_ids] &
                                 (this.episode_length_buf[sample_ids] > 0)]
            original(sample_ids)
            # Resets cancel the previous episode's pending command, including
            # resets whose newly sampled target is a stop.
            this.settle_pending[fresh] = False
            this.settle_episode[fresh] = False
            this.settle_target[fresh] = 0
            this.episode_horizons[fresh] = this.settle_base_horizons[fresh]
            moving = fresh[this.command_targets[fresh].norm(dim=-1) > 1e-6]
            this.settle_target[moving] = this.command_targets[moving]
            this.settle_pending[moving] = True
            this.settle_episode[moving] = True
            this.command_targets[moving] = 0
            # Observations prepare the next interval and decrement once.
            this.command_countdown[moving] = this.settle_delay_ticks[moving] + 1
            move_ticks = round(SETTLE_CONFIG['moving_seconds'] / this.dt)
            this.episode_horizons[moving] = this.settle_delay_ticks[moving] + move_ticks
            this.settle_counts[3] += moving.numel()
            this.command_targets[release] = this.settle_target[release]
            this.command_countdown[release] = move_ticks
            this.settle_pending[release] = False

        group.sample_commands = types.MethodType(sample, group)
        original_reward = group.command_reward

        def reward(this, q, v, found, previous_ball_xy, original_reward=original_reward):
            moving = this.settle_episode & ~this.settle_pending
            at_target = (this.commands.norm(dim=-1) - this.cfg['target_speed']).abs() < 1e-6
            this.settle_counts[:3] += torch.stack((this.settle_pending.sum(),
                                                  moving.sum(), (moving & at_target).sum()))
            return original_reward(q, v, found, previous_ball_xy)

        group.command_reward = types.MethodType(reward, group)


def settled_startup_report(env):
    return dict(scope='Current process only; command exposure, not actual ball speed.',
        groups=[dict(physics_timestep_s=g.active_physics_timestep,
                     eligible_worlds=int(g.settle_eligible.sum()),
                     waiting_control_steps=int(g.settle_counts[0]),
                     moving_control_steps=int(g.settle_counts[1]),
                     moving_at_target_control_steps=int(g.settle_counts[2]),
                     scheduled_episodes=int(g.settle_counts[3])) for g in env.groups])
