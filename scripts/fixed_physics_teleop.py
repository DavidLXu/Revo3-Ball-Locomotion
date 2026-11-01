"""Two persistent physics groups sharing one policy batch and curriculum.

Each physical world retains its own integration timestep for its entire life.
Both groups advance one identical control interval before PPO receives the batch.
Identical timesteps require explicit opt-in and limit per-instance solver memory.
"""
from collections import deque
from dataclasses import asdict, replace

import torch
from rsl_rl.env import VecEnv

from teleop_env import TeleopEnv


class FixedPhysicsTeleopEnv(VecEnv):
    def __init__(self, cfg, device='cuda:0', *, allow_identical_timesteps=False):
        assert cfg.num_envs >= 2 and cfg.num_envs % 2 == 0
        assert cfg.alternate_physics_timestep > 0
        if allow_identical_timesteps:
            assert cfg.alternate_physics_timestep == cfg.physics_timestep
        else:
            assert cfg.alternate_physics_timestep != cfg.physics_timestep
        assert not cfg.residual_teacher, 'This adapter currently supports direct policies'
        self.cfg = asdict(cfg)
        self.device = device
        self.num_envs = cfg.num_envs
        self.dt = cfg.physics_timestep * cfg.decimation
        per_group = cfg.num_envs // 2
        assert 2 * int(per_group * cfg.startup_env_fraction) == int(cfg.num_envs * cfg.startup_env_fraction)
        self.groups = []
        for timestep in (cfg.physics_timestep, cfg.alternate_physics_timestep):
            decimation = round(self.dt / timestep)
            assert decimation >= 1 and abs(decimation * timestep - self.dt) < 1e-12
            group_cfg = replace(cfg, num_envs=per_group, physics_timestep=timestep,
                                decimation=decimation, alternate_physics_timestep=0.,
                                auto_curriculum=False)
            self.groups.append(TeleopEnv(group_cfg, device=device))
        self.num_actions = self.groups[0].num_actions
        self.max_episode_length = self.groups[0].max_episode_length
        assert all(g.max_episode_length == self.max_episode_length for g in self.groups)
        self.recent = deque(maxlen=2048)
        for group in self.groups:
            group.recent = self.recent
        self.curriculum_records = []
        self.last_promotion_step = 0
        self.teacher = None
        # A scalar would incorrectly imply that all worlds share one timestep.
        self.active_physics_timestep = None
        self.physics_decimation = None
        self.last_episode_s = self.last_return = self.last_full_episode_s = 0.
        self.last_startup_failure_rate = 0.
        self.last_residual_clipped_fraction = self.last_residual_excess_cost = 0.

    @property
    def physics_groups(self):
        offset = 0
        result = []
        for group in self.groups:
            result.append(dict(start=offset, end=offset + group.num_envs,
                               physics_timestep_s=group.active_physics_timestep,
                               decimation=group.physics_decimation,
                               control_timestep_s=group.dt))
            offset += group.num_envs
        return result

    @property
    def global_steps(self):
        values = [g.global_steps for g in self.groups]
        assert len(set(values)) == 1, 'Physics groups must advance together'
        return values[0]

    @global_steps.setter
    def global_steps(self, value):
        for group in self.groups:
            group.global_steps = value

    @property
    def speed_limit(self):
        values = [g.speed_limit for g in self.groups]
        assert len(set(values)) == 1, 'Physics groups must share the speed curriculum'
        return values[0]

    @speed_limit.setter
    def speed_limit(self, value):
        for group in self.groups:
            group.speed_limit = value

    @property
    def episode_length_buf(self):
        return torch.cat([g.episode_length_buf for g in self.groups])

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        assert value.shape == (self.num_envs,)
        for group, segment in zip(self.groups, value.split([g.num_envs for g in self.groups])):
            group.episode_length_buf.copy_(segment)

    @property
    def finger_coverage(self):
        return torch.cat([g.finger_coverage for g in self.groups])

    def initialize_recovery_resets(self):
        for group in self.groups:group.initialize_recovery_resets()

    def recovery_metrics(self):
        values=[g.recovery_metrics() for g in self.groups]
        return {k:sum(v[k] for v in values) for k in values[0]} if values and values[0] else {}

    def practice_metrics(self):
        values=[g.practice_metrics() for g in self.groups]
        assert all(v['speed_bin_upper_mps']==values[0]['speed_bin_upper_mps'] for v in values)
        return dict(num_worlds=sum(v['num_worlds'] for v in values),
                    speed_bin_upper_mps=values[0]['speed_bin_upper_mps'],
                    failed_episodes=sum(v['failed_episodes'] for v in values),
                    completed_episodes=sum(v['completed_episodes'] for v in values),
                    **{k:[sum(v[k][i] for v in values) for i in range(4)]
                       for k in ('requested_control_steps','applied_control_steps')})

    @property
    def last_tracking(self):
        return sum(g.last_tracking * g.num_envs for g in self.groups) / self.num_envs

    @property
    def last_velocity(self):
        return sum(g.last_velocity * g.num_envs for g in self.groups) / self.num_envs

    @property
    def last_tracking_error(self):
        return sum(g.last_tracking_error * g.num_envs for g in self.groups) / self.num_envs

    @property
    def last_direct_clipped_fraction(self):
        return sum(g.last_direct_clipped_fraction*g.num_envs for g in self.groups)/self.num_envs

    @property
    def last_direct_excess_cost(self):
        return sum(g.last_direct_excess_cost*g.num_envs for g in self.groups)/self.num_envs

    @property
    def last_direct_invalid_fraction(self):
        return sum(g.last_direct_invalid_fraction*g.num_envs for g in self.groups)/self.num_envs

    @property
    def last_hand_tilt_rate_cost(self):
        return sum(g.last_hand_tilt_rate_cost*g.num_envs for g in self.groups)/self.num_envs

    @property
    def last_body_motion_cost(self):
        return sum(g.last_body_motion_cost*g.num_envs for g in self.groups)/self.num_envs

    def _dispatch_ids(self, method, ids):
        offset = 0
        for group in self.groups:
            local = ids[(ids >= offset) & (ids < offset + group.num_envs)] - offset
            if len(local):
                getattr(group, method)(local)
            offset += group.num_envs

    def reset(self, ids):
        self._dispatch_ids('reset', ids)

    def sample_commands(self, ids):
        self._dispatch_ids('sample_commands', ids)

    def restart_commands(self, ids):
        self._dispatch_ids('restart_commands', ids)

    def set_command(self, vx, vy):
        for group in self.groups:
            group.set_command(vx, vy)

    def get_observations(self):
        return torch.cat([g.get_observations() for g in self.groups], dim=0)

    def step(self, actions):
        assert actions.shape == (self.num_envs, self.num_actions)
        results = [group.step(segment) for group, segment in
                   zip(self.groups, actions.split([g.num_envs for g in self.groups]))]
        _ = self.global_steps  # Check that both clocks advanced one control step.
        logs = {}
        keys = set().union(*(result[3]['log'].keys() for result in results))
        for key in keys:
            if key == 'Balance/nonfinite_world_count':
                logs[key] = sum(result[3]['log'][key] for result in results)
                continue
            weighted = []
            for group, result in zip(self.groups, results):
                if key not in result[3]['log']:
                    continue
                done = result[2]
                if key == 'Episode/full_duration_s':
                    weight = int(done[group.startup_env_count:].sum())
                elif key == 'Episode/startup_failure_rate':
                    weight = int(done[:group.startup_env_count].sum())
                elif key.startswith('Episode/'):
                    weight = int(done.sum())
                else:
                    weight = group.num_envs
                if weight:
                    weighted.append((result[3]['log'][key], weight))
            if weighted:
                logs[key] = sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted)
        for key, attribute in [('Episode/duration_s', 'last_episode_s'),
                               ('Episode/return', 'last_return'),
                               ('Episode/full_duration_s', 'last_full_episode_s'),
                               ('Episode/startup_failure_rate', 'last_startup_failure_rate')]:
            if key in logs:
                setattr(self, attribute, float(logs[key]))
        physics_logs = {}
        for group, result in zip(self.groups, results):
            for key in ('Balance/all_five', 'Balance/actual_fingers', 'Teleop/tracking_score',
                        'Teleop/velocity_error_mps'):
                label = f'Physics/{group.active_physics_timestep:g}/{key}'
                physics_logs.setdefault(label, []).append((result[3]['log'][key], group.num_envs))
        for label, values in physics_logs.items():
            logs[label] = sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
        # Child auto-curricula are disabled. This uses the original gate logic,
        # with shared completed-episode records and a broadcast speed setter.
        TeleopEnv.maybe_promote(self)
        logs['Teleop/speed_limit_mps'] = self.speed_limit
        if self.cfg.get('recovery_reset_probability',0)>0:
            logs.update({'Recovery/'+k:float(v) for k,v in self.recovery_metrics().items()})
        extras = {key: torch.cat([result[3][key] for result in results])
                  for key in ('time_outs', 'failures')}
        extras['log'] = logs
        return (torch.cat([result[0] for result in results], dim=0),
                torch.cat([result[1] for result in results]),
                torch.cat([result[2] for result in results]), extras)
