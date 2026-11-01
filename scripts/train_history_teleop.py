"""Bounded history-feedback PPO, retaining the selected contact policy physics."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from balance_model import ROOT
from fixed_physics_teleop import FixedPhysicsTeleopEnv
from history_context import HISTORY_CONFIG, migrate_checkpoint, split_history_optimizer
from history_lifecycle import HistoryTeleopEnv
from teleop_env import TeleopCfg
from train_teleop import TeleopRunner


class HistoryRunner(TeleopRunner):
    def __init__(self, env, *args, **kwargs):
        assert env.cfg.get('history_observation') == HISTORY_CONFIG
        # TeleopRunner.__init__ already splits contact groups. Start with the
        # unsplit optimizer so every inherited Adam slot keeps its original id.
        OnPolicyRunner.__init__(self, env, *args, **kwargs)
        split_history_optimizer(self.alg)

    def release_contact_backbone(self):
        super().release_contact_backbone()
        self.alg.get_policy().history_projection.requires_grad_(True)

    def load(self, path, *args, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['infos'].get('history_observation') == HISTORY_CONFIG
        assert saved['infos'].get('contact_backbone_trainable') is True
        assert not saved['infos'].get('contact_branch_only')
        startup = saved['infos'].get('startup_target_practice')
        if startup is not None:
            assert startup == getattr(self, 'startup_target_practice_config', None), \
                'Resume must explicitly preserve target-speed startup practice'
        settled = saved['infos'].get('settled_startup_practice')
        if settled is not None:
            assert settled == getattr(self, 'settled_startup_practice_config', None), \
                'Resume must explicitly preserve settled-startup practice'
        smooth_reward = saved['infos'].get('physical_smooth_reward')
        if smooth_reward is not None:
            assert smooth_reward == getattr(self, 'physical_smooth_reward_config', None), \
                'Resume must explicitly preserve the physical smoothness reward'
            from body_acceleration_cost import validate_metadata as validate_acceleration
            validate_acceleration(self.env.cfg, smooth_reward)
            from action_bounds_cost import validate_metadata as validate_action_bounds
            validate_action_bounds(self.env.cfg, smooth_reward)
        optimizer_adaptation = saved['infos'].get('optimizer_adaptation')
        stop_adaptation = saved['infos'].get('practice_stop_adaptation')
        if stop_adaptation is not None:
            assert stop_adaptation == getattr(self, 'practice_stop_adaptation_config', None), \
                'Resume must explicitly preserve practice stop adaptation'
            from practice_stop_adaptation import validate_environment as validate_stop
            validate_stop(self.env.cfg, stop_adaptation)
        integration = saved['infos'].get('physics_integration_adaptation')
        if integration is not None:
            assert integration == getattr(self, 'physics_integration_config', None), \
                'Resume must explicitly preserve physics integration adaptation'
            from physics_integration_adaptation import validate_environment
            validate_environment(self.env.cfg, integration)
        if optimizer_adaptation is not None:
            assert optimizer_adaptation == getattr(self, 'optimizer_adaptation_config', None), \
                'Resume must explicitly preserve optimizer learning-rate adaptation'
            from optimizer_adaptation import validate
            validate(optimizer_adaptation)
            assert [g['lr'] for g in saved['optimizer_state_dict']['param_groups']] == \
                optimizer_adaptation['group_learning_rates'], 'Saved optimizer rates disagree with metadata'
        return super().load(path, *args, **kwargs)

    def save(self, path, infos=None):
        # This isolated runner deliberately supports only the selected direct
        # contact-feedback phase. Do not silently drop unsupported metadata.
        assert infos is None
        assert self.contact_backbone_trainable and not self.contact_branch_only
        assert self.actor_normalization_frozen
        assert not any(getattr(self, k, False) for k in ('actor_frozen_for_critic_warmup',
            'observation_normalization_frozen', 'command_branch_only', 'command_backbone_trainable'))
        assert self.policy_reference_config() is None
        assert self.recovery_reset_config() is None
        assert self.temporal_smoothness_config() is None
        curriculum = dict(speed_limit=self.env.speed_limit, records=self.env.curriculum_records,
            global_steps=self.env.global_steps, last_promotion_step=self.env.last_promotion_step)
        metadata = dict(teleop_curriculum=curriculum, observation_normalization_frozen=False,
            actor_frozen_for_critic_warmup=False, actor_normalization_frozen=True,
            hand_inertia_mode=self.env.cfg['hand_inertia_mode'], contact_load_observation=True,
            contact_branch_only=False, contact_backbone_trainable=True,
            command_branch_only=False, command_backbone_trainable=False,
            history_observation=dict(HISTORY_CONFIG))
        # Publish only complete checkpoints. Evaluation never sees a file that
        # still lacks its history metadata or is being serialized.
        path = Path(path); temporary = path.with_suffix('.tmp.pt')
        OnPolicyRunner.save(self, str(temporary), metadata)
        temporary.replace(path)
        state = dict(iteration=self.current_learning_iteration, checkpoint=str(path),
            time_unix=time.time(), pid=os.getpid(), target_speed_mps=self.env.cfg['target_speed'],
            curriculum=curriculum, environment_steps=self.env.global_steps*self.env.num_envs,
            recent_episode_seconds=self.env.last_episode_s, recent_return=self.env.last_return,
            recent_tracking_score=self.env.last_tracking,
            recent_actual_speed_mps=self.env.last_velocity,
            recent_velocity_error_mps=self.env.last_tracking_error,
            recent_full_episode_seconds=self.env.last_full_episode_s,
            recent_startup_failure_rate=self.env.last_startup_failure_rate,
            recent_body_motion_cost=self.env.last_body_motion_cost,
            actor_normalization_frozen=True, observation_normalization_frozen=False,
            training_phase='history_contact_feedback_actor_and_critic',
            history_observation=dict(HISTORY_CONFIG), physics_backend='native',
            active_physics_timestep_s=None, physics_decimation=None,
            physics_groups=self.env.physics_groups)
        progress = path.parent / 'progress.json'
        progress.with_suffix('.tmp').write_text(json.dumps(state, indent=2))
        progress.with_suffix('.tmp').replace(progress)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--initialize-history', action='store_true')
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--iterations', type=int, default=31)
    p.add_argument('--seed', type=int, default=108)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--history-learning-rate', type=float, default=.0003)
    p.add_argument('--contact-learning-rate', type=float, default=.0003)
    p.add_argument('--body-motion-weight', type=float)
    p.add_argument('--target-speed', type=float)
    p.add_argument('--practice-env-fraction', type=float, default=0.)
    p.add_argument('--practice-speeds', type=float, nargs='*', default=[])
    p.add_argument('--practice-stop-hold-max', type=float, default=0.)
    p.add_argument('--transition-practice', action='store_true',
                   help='Use reviewed 2–4s command holds and 50%% reversal sampling.')
    p.add_argument('--upper-boundary-practice', action='store_true',
                   help='Practice 0.09, 0.095 and 0.1 m/s from the reviewed target-speed policy.')
    args = p.parse_args()
    assert not (args.transition_practice and args.upper_boundary_practice)
    assert 1 <= args.iterations <= 101 and args.num_envs >= 8 and args.num_envs % 8 == 0
    assert 0 < args.history_learning_rate <= .0003
    assert 0 < args.contact_learning_rate <= .0003
    assert 0 <= args.practice_env_fraction <= .5
    assert bool(args.practice_speeds) == bool(args.practice_env_fraction)
    if args.practice_env_fraction > .25:
        assert args.practice_env_fraction == .5
        assert args.practice_speeds in ([.1], [.0775, .08, .0825, .085])
        assert args.practice_stop_hold_max == 1. and args.body_motion_weight == .75
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    source = args.resume.resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == args.source_sha256
    saved = torch.load(source, map_location='cpu', weights_only=False)
    cfg = json.loads((source.parent / 'env_config.json').read_text())
    rcfg = json.loads((source.parent / 'runner_config.json').read_text())
    precision = json.loads((source.parent / 'numeric_precision.json').read_text())
    assert precision['cuda_matmul_allow_tf32'] is False
    assert precision['float32_matmul_precision'] == 'highest'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    assert cfg['body_motion_weight'] in (.5, .75)
    if cfg['body_motion_weight'] == .75:
        # Continue an already trained smooth-motion policy without migrating
        # weights or resetting Adam. Only the two reviewed practice layouts
        # are supported; this does not authorize other source configurations.
        assert not args.initialize_history
        assert cfg.get('history_observation') == HISTORY_CONFIG
        assert cfg['body_motion_substeps']
        assert cfg['practice_env_fraction'] == .5
        assert cfg['practice_speeds'] in ([.1], [.0775, .08, .0825, .085])
    if cfg['practice_env_fraction'] != 0:
        # A saved practice policy can continue without resetting its command
        # distribution, weights, optimizer slots, or curriculum.
        if cfg['body_motion_weight'] == .5:
            assert cfg['practice_env_fraction'] == .125
            assert cfg['practice_speeds'] == [.085, .09, .1]
        assert cfg['practice_stop_hold_max'] == 1.
        assert cfg['target_speed'] == .1
        assert cfg['curriculum_speed_levels'] == [.025, .035, .04, .05, .075, .1]
    assert cfg['physics_backend'] == 'native' and cfg['alternate_physics_timestep'] == .0005
    assert cfg['physics_timestep'] == .001 and cfg['decimation'] == 10
    assert not cfg.get('recovery_reset_probability') and not cfg.get('residual_teacher')
    assert rcfg['algorithm']['class_name'] == 'PPO'
    assert rcfg['num_steps_per_env'] == 128
    run = ROOT / 'runs' / args.run
    run.mkdir(exist_ok=False)
    (run / 'numeric_precision.json').write_text(json.dumps(dict(
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        torch_version=torch.__version__,
        scope='Preserved source float32 policy and PPO matrix arithmetic.'), indent=2))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    cfg.update(num_envs=args.num_envs, seed=args.seed,
               curriculum_gate_file=str(run / 'curriculum_approval.json'))
    if args.upper_boundary_practice:
        assert cfg['body_motion_weight'] == .75 and cfg['practice_env_fraction'] == .5
        assert cfg['practice_speeds'] == [.1] and not args.initialize_history
        assert not args.practice_env_fraction and args.body_motion_weight is None
        assert args.target_speed is None
        assert (cfg['command_hold_min'], cfg['command_hold_max']) == (4., 10.)
        assert cfg['reverse_command_probability'] == 0.
        assert not cfg['rightward_command_probability'] and not cfg['reversal_sequence_probability']
        cfg['practice_speeds'] = [.09, .095, .1]
    if args.transition_practice:
        assert cfg['body_motion_weight'] == .75 and cfg['practice_env_fraction'] == .5
        assert cfg['practice_speeds'] == [.1] and not args.initialize_history
        assert not args.practice_env_fraction and args.body_motion_weight is None
        assert args.target_speed is None
        assert (cfg['command_hold_min'], cfg['command_hold_max']) == (4., 10.)
        assert cfg['reverse_command_probability'] == 0.
        assert not cfg['rightward_command_probability'] and not cfg['reversal_sequence_probability']
        cfg.update(command_hold_min=2., command_hold_max=4., reverse_command_probability=.5)
    if args.body_motion_weight is not None:
        assert args.body_motion_weight == .75 and cfg['body_motion_substeps']
        cfg['body_motion_weight'] = args.body_motion_weight
    if args.target_speed is not None:
        assert saved['infos']['teleop_curriculum']['speed_limit'] <= args.target_speed <= cfg['target_speed']
        cfg['target_speed'] = args.target_speed
        cfg['curriculum_speed_levels'] = [s for s in cfg['curriculum_speed_levels'] if s <= args.target_speed]
        assert cfg['curriculum_speed_levels'][-1] == args.target_speed
    if args.practice_env_fraction:
        assert 0 < args.practice_stop_hold_max <= cfg['command_hold_max']
        cfg.update(practice_env_fraction=args.practice_env_fraction,
                   practice_speeds=args.practice_speeds,
                   practice_stop_hold_max=args.practice_stop_hold_max)
    if args.initialize_history:
        assert 'history_observation' not in cfg
        assert rcfg['actor']['class_name'] == 'contact_context:ContactContextModel'
        saved = migrate_checkpoint(saved, args.history_learning_rate)
        load_path = run / 'history_initialization.pt'
        torch.save(saved, load_path)
        (run / 'history_initialization.json').write_text(json.dumps(dict(source=str(source),
            source_sha256=args.source_sha256,
            migrated_sha256=hashlib.sha256(load_path.read_bytes()).hexdigest(),
            optimizer_updates=0, zero_initialized=True), indent=2))
    else:
        assert cfg.pop('history_observation') == HISTORY_CONFIG
        assert rcfg['actor']['class_name'] == 'history_context:HistoryContextModel'
        load_path = source
    rcfg.update(seed=args.seed, max_iterations=args.iterations, run_name=args.run,
                save_interval=args.save_interval)
    for role in ('actor', 'critic'): rcfg[role]['class_name'] = 'history_context:HistoryContextModel'
    assert rcfg['algorithm']['learning_rate'] == 1e-5 and rcfg['algorithm']['schedule'] == 'fixed'
    for name in ('pose_initial_qpos.npy', 'pose_initial_state.json', 'pose_scene.xml'):
        shutil.copy2(source.parent / name, run / name)
    # The environment constructs the canonical opposed pose. Verify it matches
    # the checkpoint's saved pose instead of silently changing the reset state.
    for name in ('initial_qpos.npy', 'initial_state.json', 'scene.xml'):
        assert (ROOT / 'outputs/opposed_thumb' / name).read_bytes() == (run / ('pose_' + name)).read_bytes()
    snapshot = run / 'source'; snapshot.mkdir()
    for file in (ROOT / 'scripts').glob('*.py'): shutil.copy2(file, snapshot / file.name)
    (run / 'env_config.json').write_text(json.dumps({**cfg, 'history_observation': HISTORY_CONFIG}, indent=2))
    (run / 'runner_config.json').write_text(json.dumps(rcfg, indent=2))
    (run / 'launch_args.json').write_text(json.dumps({k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, indent=2))
    env = HistoryTeleopEnv(FixedPhysicsTeleopEnv(TeleopCfg(**cfg), args.device))
    try:
        runner = HistoryRunner(env, copy.deepcopy(rcfg), str(run), device=args.device)
        runner.load(load_path)
        curriculum = saved['infos']['teleop_curriculum']
        env.speed_limit = curriculum['speed_limit']; env.curriculum_records = copy.deepcopy(curriculum['records'])
        env.global_steps = curriculum['global_steps']; env.last_promotion_step = curriculum['last_promotion_step']
        ids = torch.arange(env.num_envs, device=env.device)
        env.sample_commands(ids)
        # Commands were restored before the first action. Start all past frames
        # from this exact initial observation, without fabricated pre-rollout data.
        env.history.reset(ids, pack_current(env))
        runner.alg.learning_rate = 1e-5; runner.alg.schedule = 'fixed'
        for group, lr in zip(runner.alg.optimizer.param_groups,
                             (1e-5, .0003, args.contact_learning_rate, args.history_learning_rate)):
            group['lr'] = lr
        assert len(runner.alg.optimizer.param_groups) == 4
        assert runner.alg.get_policy().history_projection.requires_grad
        (run / 'process.json').write_text(json.dumps(dict(pid=os.getpid(), start_time=time.time(), num_envs=args.num_envs)))
        runner.learn(args.iterations, init_at_random_ep_len=False)
        (run / 'finished.json').write_text(json.dumps(dict(iteration=runner.current_learning_iteration, time_unix=time.time())))
    finally:
        for group in env.groups: group.sim.pool.shutdown()


def pack_current(env):
    from history_lifecycle import pack_frame
    return pack_frame(env.get_observations())


if __name__ == '__main__': main()
