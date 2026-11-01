"""Deterministic GPU evaluation with native-matched resets and terminal samples."""
import argparse
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch

from evaluate_teleop import PROFILE_NAMES, profile
from fixed_physics_teleop import FixedPhysicsTeleopEnv
from history_lifecycle import HistoryTeleopEnv, pack_frame
from policy_loader import load_controller
from teleop_env import TeleopCfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--speed', type=float, default=.1)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--physics-timesteps', type=float, nargs=2,
                   help='Explicit evaluation timesteps, independent of training integration.')
    p.add_argument('--seed', type=int, default=31000)
    p.add_argument('--profiles', default='stop,forward,backward,left,right,diagonal,diagonal_forward_right,diagonal_backward_left,diagonal_backward_right,switch')
    a = p.parse_args()
    names = a.profiles.split(',')
    assert set(names) <= PROFILE_NAMES and a.episodes > 0 and a.seconds > 0
    assert not a.output.exists()
    torch.set_num_threads(1)
    torch.cuda.set_device(a.device)
    torch.manual_seed(a.seed)
    controller = load_controller(a.checkpoint)
    cfg = json.loads((a.checkpoint.parent / 'env_config.json').read_text())
    gamma = json.loads((a.checkpoint.parent / 'runner_config.json').read_text())['algorithm']['gamma']
    cfg.pop('history_observation')
    training_timesteps = [cfg['physics_timestep'], cfg['alternate_physics_timestep']]
    if a.physics_timesteps is not None:
        first, second = a.physics_timesteps
        control_dt = cfg['physics_timestep'] * cfg['decimation']
        assert first > 0 and second > 0 and first != second
        assert all(abs(round(control_dt/dt)*dt-control_dt)<1e-12 for dt in (first,second))
        cfg.update(physics_timestep=first,alternate_physics_timestep=second,
                   decimation=round(control_dt/first))
    assert cfg['physics_timestep'] != cfg['alternate_physics_timestep'], \
        'Supply distinct --physics-timesteps for this training configuration'
    assert 0 <= a.speed <= cfg['target_speed'] and cfg['reset_velocity'] == 0
    cases = [(name, a.seed + ep) for name in names for ep in range(a.episodes)]
    n = len(cases)
    cfg.update(num_envs=2*n, physics_backend='warp', episode_seconds=a.seconds,
               startup_env_fraction=0., practice_env_fraction=0., practice_speeds=[],
               practice_stop_hold_max=0., auto_curriculum=False,
               curriculum_gate_file='', push_force=0., recovery_reset_probability=0.)
    env = HistoryTeleopEnv(FixedPhysicsTeleopEnv(TeleopCfg(**cfg), a.device))
    policy = controller.policy.to(a.device).eval()
    poses = []
    for name, seed in cases:
        controller.env.reset(seed, 1.)
        poses.append(controller.env.d.qpos.copy())
    initial = torch.tensor(np.asarray(poses), device=a.device, dtype=torch.float32)
    snapshots = [None, None]
    for index, group in enumerate(env.groups):
        assert torch.all(group.episode_horizons == round(a.seconds/env.dt))
        group.sim.data.qpos[:] = initial
        group.sim.data.qvel.zero_()
        group.sim.data.ctrl[:] = group.joint0
        group.set_command(0., 0.)
        group.sim.forward()
        original = group.command_reward
        def capture(self, q, v, found, previous_ball_xy, original=original, index=index):
            reward = original(q, v, found, previous_ball_xy)
            # Called after physics and filtering, before any automatic episode reset.
            snapshots[index] = dict(ball=q[:, self.ballqa:self.ballqa+2].clone(),
                velocity=self.reward_ball_velocity.clone(), command=self.commands.clone(),
                motion=self.sim.body_motion_cost.clone(), contacts=found.clone(),
                finite=torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1))
            return reward
        group.command_reward = types.MethodType(capture, group)
    env.history.reset(torch.arange(2*n, device=a.device), pack_frame(env.get_observations()))
    active = torch.ones(2*n, device=a.device, dtype=torch.bool)
    traces = [[] for _ in range(2*n)]
    success = [False] * (2*n)
    reasons = ['incomplete'] * (2*n)
    for step in range(round(a.seconds/env.dt)):
        targets = torch.tensor([profile(name, step*env.dt, a.speed, a.seconds)
                                for name, seed in cases], device=a.device)
        for group in env.groups:
            group.command_targets.copy_(targets)
        with torch.inference_mode():
            actions = policy(env.get_observations(), stochastic_output=False)
            _, reward, done, extra = env.step(actions)
        captured = {key: torch.cat([s[key] for s in snapshots]).cpu().numpy()
                    for key in snapshots[0]}
        captured['reward'] = reward.cpu().numpy()
        ids = active.nonzero().flatten().cpu().tolist()
        done_cpu = done.cpu().numpy()
        failure_cpu = extra['failures'].cpu().numpy()
        for i in ids:
            traces[i].append({key: value[i].tolist() for key, value in captured.items()})
            if done_cpu[i]:
                success[i] = not bool(failure_cpu[i])
                reasons[i] = 'none' if success[i] else ('nonfinite' if not captured['finite'][i] else 'physical_or_reward_failure')
        active &= ~done.bool()
        if not active.any():
            break
        if (step+1) % 500 == 0:
            print(json.dumps(dict(step=step+1, active_worlds=int(active.sum()))), flush=True)
    rows = []
    for i, trace in enumerate(traces):
        steady = trace[min(200, len(trace)//2):]
        velocity = np.array([r['velocity'] for r in steady])
        command = np.array([r['command'] for r in steady])
        initial_xy = initial[i % n, env.groups[i//n].ballqa:env.groups[i//n].ballqa+2].cpu().numpy()
        rewards = np.array([r['reward'] for r in trace])
        discount = gamma ** np.arange(len(trace))
        discounted_return = float((rewards*discount).sum())
        # A diagnostic only: hypothetical earlier failure at each prefix does
        # not establish that the policy can physically realize that failure.
        prefix = np.r_[0., np.cumsum((rewards*discount)[:-1])]
        early_failure_returns = prefix - cfg['failure_penalty']*discount
        best_early = int(np.argmax(early_failure_returns))
        rows.append(dict(profile=cases[i % n][0], seed=cases[i % n][1],
            physics_timestep_s=env.groups[i//n].active_physics_timestep,
            completed=success[i], survival_s=len(trace)*env.dt, failure_reason=reasons[i],
            mean_velocity_error_mps=float(np.linalg.norm(velocity-command, axis=1).mean()),
            mean_actual_velocity_mps=velocity.mean(0).tolist(),
            ball_displacement_xy_m=(np.array(trace[-1]['ball'])-initial_xy).tolist(),
            mean_body_motion_cost=float(np.mean([r['motion'] for r in trace])),
            discounted_return=discounted_return,
            best_hypothetical_early_failure_return=float(early_failure_returns[best_early]),
            hypothetical_early_failure_time_s=best_early*env.dt,
            hypothetical_early_failure_gain=float(early_failure_returns[best_early]-discounted_return),
            mean_finger_contacts=float(np.mean([sum(r['contacts']) for r in trace]))))
    result = dict(engine='MjLab / MuJoCo Warp', checkpoint=str(a.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        policy_mode='deterministic', reset_mode='native_matched_numpy_seeds_noise1',
        terminal_samples_preserved=True, startup_short_horizons=False,
        reward_discount_gamma=gamma,control_timestep_s=env.dt,
        training_physics_timesteps_s=training_timesteps,
        evaluation_physics_timesteps_s=[g.active_physics_timestep for g in env.groups],
        hypothetical_immediate_failure_return=-cfg['failure_penalty'],
        early_failure_diagnostic='Counterfactual prefix truncation; not a physically demonstrated alternative policy.',
        requested_speed_mps=a.speed, requested_seconds=a.seconds, episodes=rows,
        completed=sum(success), trials=len(rows), goal_achieved=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
