"""Bounded GPU PPO adaptation of a moving policy; explicitly not fresh training."""
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
from command_exposure import CommandExposure
from fixed_physics_teleop import FixedPhysicsTeleopEnv
from history_context import HISTORY_CONFIG
from history_lifecycle import HistoryTeleopEnv, pack_frame
from teleop_env import TeleopCfg
from train_history_teleop import HistoryRunner
from train_scratch_tracking_branch import equal_tree
from train_smooth_from_scratch import write_json


class MovingPolicyRunner(HistoryRunner):
    def additional_checkpoint_metadata(self):
        return {}

    def save(self, path, infos=None):
        assert infos is None
        assert self.contact_backbone_trainable and not self.contact_branch_only
        assert self.actor_normalization_frozen
        assert self.policy_reference_config() is None
        assert self.recovery_reset_config() is None
        assert self.temporal_smoothness_config() is None
        curriculum = dict(speed_limit=self.env.speed_limit, records=self.env.curriculum_records,
            global_steps=self.env.global_steps, last_promotion_step=self.env.last_promotion_step)
        samples = (self.env.global_steps-self.source_steps)*self.env.num_envs
        exposure = self.command_exposure.report()
        assert exposure['world_control_steps'] == samples
        metadata = dict(teleop_curriculum=curriculum, history_observation=HISTORY_CONFIG,
            observation_normalization_frozen=False, actor_frozen_for_critic_warmup=False,
            actor_normalization_frozen=True, hand_inertia_mode=self.env.cfg['hand_inertia_mode'],
            contact_load_observation=True, contact_branch_only=False, contact_backbone_trainable=True,
            command_branch_only=False, command_backbone_trainable=False,
            fresh_training=False, pretrained_weights_loaded=True, goal_qualified=False,
            initialization='moving_ball_policy_warm_start', parent_checkpoint=str(self.source),
            parent_sha256=self.source_hash, source_global_steps=self.source_steps,
            branch_environment_steps=samples, command_exposure=exposure,
            curriculum_recent=list(self.env.recent),
            reward_revision=getattr(self, 'reward_revision', 'moving_policy_smooth_tracking_v1'),
            velocity_error_huber_weight=self.env.cfg['velocity_error_huber_weight'])
        additional = self.additional_checkpoint_metadata()
        assert not metadata.keys() & additional.keys()
        metadata.update(additional)
        path = Path(path)
        temporary = path.with_suffix('.tmp.pt')
        OnPolicyRunner.save(self, str(temporary), metadata)
        temporary.replace(path)
        write_json(path.parent/'progress.json', dict(pid=os.getpid(), time_unix=time.time(),
            iteration=self.current_learning_iteration, checkpoint=str(path), curriculum=curriculum,
            branch_environment_steps=samples, command_exposure=exposure,
            recent_episode_seconds=self.env.last_episode_s,
            recent_full_episode_seconds=self.env.last_full_episode_s,
            recent_tracking_score=self.env.last_tracking,
            recent_actual_speed_mps=self.env.last_velocity,
            recent_velocity_error_mps=self.env.last_tracking_error,
            recent_body_motion_cost=self.env.last_body_motion_cost,
            physics_backend=self.env.cfg['physics_backend'], physics_groups=self.env.physics_groups,
            fresh_training=False, goal_qualified=False))


def main(runner_type=MovingPolicyRunner):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--seed', type=int, default=137)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--capture-numerical-failures', action='store_true',
                   help='Passively save pre-reset state/actions for up to four numerical faults.')
    a = p.parse_args()
    assert 1 <= a.iterations <= 200 and a.num_envs >= 32 and a.num_envs % 16 == 0
    assert Path(a.run).name == a.run and a.run not in ('.', '..')
    source = a.source.resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == a.source_sha256
    saved = torch.load(source, map_location='cpu', weights_only=False)
    cfg = json.loads((source.parent/'env_config.json').read_text())
    original = copy.deepcopy(cfg)
    stop_config = getattr(runner_type, 'practice_stop_adaptation_config', None)
    from practice_stop_adaptation import validate_source as validate_stop_source
    validate_stop_source(cfg, saved['infos'].get('practice_stop_adaptation'), stop_config)
    rcfg = json.loads((source.parent/'runner_config.json').read_text())
    assert cfg.pop('history_observation') == HISTORY_CONFIG
    assert cfg['initial_pose'] == 'opposed_thumb' and cfg['hand_inertia_mode'] == 'collision_only'
    integration_config = getattr(runner_type, 'physics_integration_config', None)
    from physics_integration_adaptation import validate_source
    validate_source(cfg, saved['infos'].get('physics_integration_adaptation'), integration_config)
    assert cfg['body_motion_substeps']
    assert cfg['target_speed'] == .1 and cfg['practice_speeds'] == [.1]
    assert cfg['action_filter_alpha'] == .3 and cfg['pd_derivative_gain'] == .18
    assert not cfg['push_force'] and not cfg.get('residual_teacher')
    assert not cfg.get('recovery_reset_probability')
    assert saved['infos']['contact_backbone_trainable'] and saved['infos']['actor_normalization_frozen']
    assert rcfg['num_steps_per_env'] == 128 and rcfg['algorithm']['schedule'] == 'fixed'
    run = ROOT/'runs'/a.run
    run.mkdir(exist_ok=False)
    cfg.update(num_envs=a.num_envs, seed=a.seed, physics_backend='warp',
        body_motion_weight=1.5, velocity_error_huber_weight=16.,
        joint_target_rate_weight=.02, joint_target_acceleration_weight=.00002,
        curriculum_gate_file=str(run/'curriculum_approval.json'))
    reward_overrides = getattr(runner_type, 'reward_configuration_overrides', {})
    assert set(reward_overrides) <= {'body_motion_weight', 'velocity_error_huber_weight',
        'joint_target_rate_weight', 'joint_target_acceleration_weight', 'survival_reward_rate',
        'body_acceleration_weight', 'direct_action_bounds_weight'}
    cfg.update(reward_overrides)
    stop_migration = None
    if stop_config is not None:
        from practice_stop_adaptation import configure as configure_stop
        stop_migration = configure_stop(cfg, stop_config)
    from body_acceleration_cost import validate_metadata as validate_acceleration
    validate_acceleration(cfg, getattr(runner_type,'physical_smooth_reward_config',None))
    from action_bounds_cost import validate_metadata as validate_action_bounds
    reward_metadata=getattr(runner_type,'physical_smooth_reward_config',None)
    if reward_overrides.get('direct_action_bounds_weight',0)>0:
        assert reward_metadata is not None
    validate_action_bounds(cfg,reward_metadata)
    integration_migration = None
    if integration_config is not None:
        from physics_integration_adaptation import configure as configure_integration
        integration_migration = configure_integration(cfg, integration_config)
    optimizer_config = getattr(runner_type, 'optimizer_adaptation_config', None)
    if optimizer_config is not None:
        from optimizer_adaptation import validate
        validate(optimizer_config)
        rcfg['algorithm']['learning_rate'] = optimizer_config['group_learning_rates'][0]
        rcfg['algorithm']['schedule'] = optimizer_config['schedule']
    rcfg.update(seed=a.seed, run_name=a.run, max_iterations=a.iterations,
        save_interval=25, upload_model=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    for name in ('pose_initial_qpos.npy', 'pose_initial_state.json', 'pose_scene.xml', 'numeric_precision.json'):
        shutil.copy2(source.parent/name, run/name)
    for name in ('initial_qpos.npy', 'initial_state.json', 'scene.xml'):
        assert (ROOT/'outputs/opposed_thumb'/name).read_bytes() == (run/('pose_'+name)).read_bytes()
    snapshot = run/'source'; snapshot.mkdir()
    for f in (ROOT/'scripts').glob('*.py'): shutil.copy2(f, snapshot/f.name)
    final_cfg = {**cfg, 'history_observation': HISTORY_CONFIG}
    changes = {k:[original.get(k),final_cfg.get(k)] for k in original.keys()|final_cfg.keys()
               if original.get(k) != final_cfg.get(k)}
    permitted_changes = {'num_envs','seed','physics_backend','body_motion_weight',
        'velocity_error_huber_weight','joint_target_rate_weight','joint_target_acceleration_weight',
        'curriculum_gate_file','survival_reward_rate'}
    if integration_config is not None:
        permitted_changes |= {'physics_timestep','alternate_physics_timestep','decimation'}
    if cfg.get('body_acceleration_weight',0)>0:
        permitted_changes.add('body_acceleration_weight')
    if reward_overrides.get('direct_action_bounds_weight',0)>0:
        permitted_changes.add('direct_action_bounds_weight')
    if stop_config is not None:
        permitted_changes.add('practice_stop_hold_max')
    assert set(changes) <= permitted_changes
    write_json(run/'env_config.json', final_cfg)
    write_json(run/'runner_config.json', rcfg)
    write_json(run/'launch_args.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    env = HistoryTeleopEnv(FixedPhysicsTeleopEnv(TeleopCfg(**cfg), a.device,
        allow_identical_timesteps=integration_config is not None))
    try:
        runner = runner_type(env, copy.deepcopy(rcfg), str(run), device=a.device)
        runner.load(str(source), map_location=a.device)
        for role, state in (('actor_state_dict',runner.alg.get_policy().state_dict()),
                            ('critic_state_dict',runner.alg.critic.state_dict()),
                            ('optimizer_state_dict',runner.alg.optimizer.state_dict())):
            assert equal_tree(saved[role], state), role
        assert len(runner.alg.optimizer.param_groups) == 4
        optimizer_migration = None
        if optimizer_config is not None:
            from optimizer_adaptation import configure, expected_optimizer_state
            optimizer_migration = configure(runner, optimizer_config)
            assert equal_tree(expected_optimizer_state(saved['optimizer_state_dict'], optimizer_config),
                              runner.alg.optimizer.state_dict())
        assert runner.alg.get_policy().history_projection.requires_grad
        runner.current_learning_iteration = saved['iter']+1
        curriculum = saved['infos']['teleop_curriculum']
        env.speed_limit = curriculum['speed_limit']; env.global_steps = curriculum['global_steps']
        env.curriculum_records = copy.deepcopy(curriculum['records'])
        env.last_promotion_step = curriculum['last_promotion_step']
        env.recent.extend(saved['infos'].get('curriculum_recent', []))
        ids = torch.arange(env.num_envs, device=env.device)
        env.sample_commands(ids); env.history.reset(ids, pack_frame(env.get_observations()))
        runner.source = source; runner.source_hash = a.source_sha256
        runner.source_steps = env.global_steps; runner.command_exposure = CommandExposure(env)
        runner.save(run/'branch_initial.pt')
        write_json(run/'initialization_audit.json', dict(actor_exact=True, critic_exact=True,
            adam_exact=equal_tree(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict()),
            adam_slots_exact=True, optimizer_migration=optimizer_migration,
            integration_migration=integration_migration,
            practice_stop_migration=stop_migration,
            actor_normalization_frozen=True, source_iteration=saved['iter'],
            source_sha256=a.source_sha256, config_differences=changes,
            optimizer_learning_rates=[g['lr'] for g in runner.alg.optimizer.param_groups],
            original_pose=True, fresh_training=False, goal_achieved=False,
            additional_training_configuration=runner.additional_checkpoint_metadata()))
        write_json(run/'process.json', dict(pid=os.getpid(), start_time=time.time(), num_envs=a.num_envs))
        capture = None
        if a.capture_numerical_failures:
            from training_failure_capture import TrainingFailureCapture
            capture = TrainingFailureCapture(env, run/'numerical_capture', runner.alg.get_policy())
        runner.learn(a.iterations, init_at_random_ep_len=False)
        if capture is not None:
            capture.publish()
        write_json(run/'finished.json', dict(iteration=runner.current_learning_iteration, time_unix=time.time()))
    finally:
        for group in env.groups:
            if hasattr(group.sim, 'pool'): group.sim.pool.shutdown()


if __name__ == '__main__': main()
