"""Fresh PPO on the original free-ball task, with smoothness from update zero."""
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
from teleop_env import TeleopCfg
from fixed_physics_teleop import FixedPhysicsTeleopEnv
from history_lifecycle import HistoryTeleopEnv
from history_context import HISTORY_CONFIG


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


class ScratchRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        assert infos is None
        curriculum = dict(speed_limit=self.env.speed_limit, records=self.env.curriculum_records,
            global_steps=self.env.global_steps, last_promotion_step=self.env.last_promotion_step)
        metadata = dict(teleop_curriculum=curriculum, history_observation=HISTORY_CONFIG,
            hand_inertia_mode=self.env.cfg['hand_inertia_mode'], contact_load_observation=True,
            actor_normalization_frozen=False, observation_normalization_frozen=False,
            fresh_training=True, pretrained_weights_loaded=False,
            goal_qualified=False, reward_revision='smooth_from_scratch_v1')
        metadata.update(velocity_error_huber_weight=self.env.cfg.get('velocity_error_huber_weight', 0.),
            curriculum_recent=list(self.env.recent))
        if metadata['velocity_error_huber_weight']:
            metadata['reward_revision'] = 'smooth_scratch_velocity_huber_v1'
        metadata.update(getattr(self, 'branch_metadata', {}))
        path = Path(path)
        temporary = path.with_suffix('.tmp.pt')
        super().save(str(temporary), metadata)
        temporary.replace(path)
        write_json(path.parent / 'progress.json', dict(pid=os.getpid(), time_unix=time.time(),
            iteration=self.current_learning_iteration, checkpoint=str(path), curriculum=curriculum,
            environment_steps=self.env.global_steps*self.env.num_envs,
            recent_episode_seconds=self.env.last_episode_s,
            recent_full_episode_seconds=self.env.last_full_episode_s,
            recent_tracking_score=self.env.last_tracking,
            recent_velocity_error_mps=self.env.last_tracking_error,
            recent_actual_speed_mps=self.env.last_velocity,
            recent_body_motion_cost=self.env.last_body_motion_cost,
            physics_groups=self.env.physics_groups, goal_qualified=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--config-source', type=Path, required=True,
                        help='Read JSON configuration and reset assets only; never load weights.')
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--iterations', type=int, default=1001)
    parser.add_argument('--seed', type=int, default=121)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-interval', type=int, default=25)
    parser.add_argument('--velocity-huber-weight', type=float, choices=[0., 4., 8.], default=0.,
                        help='Velocity-error penalty from random initialization; keep smoothness costs unchanged.')
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    assert args.num_envs >= 32 and args.num_envs % 16 == 0 and args.iterations > 0
    run = ROOT / 'runs' / args.run
    run.mkdir(exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((args.config_source / 'env_config.json').read_text())
    assert cfg.pop('history_observation') == HISTORY_CONFIG
    assert cfg['initial_pose'] == 'opposed_thumb' and cfg['hand_inertia_mode'] == 'collision_only'
    assert cfg['physics_timestep'] == .001 and cfg['alternate_physics_timestep'] == .0005
    cfg.update(num_envs=args.num_envs, seed=args.seed, body_motion_weight=1.5,
        joint_target_rate_weight=.02, joint_target_acceleration_weight=.00002,
        initial_speed=.025, target_speed=.1, curriculum_speed_levels=[.025,.035,.05,.075,.1],
        curriculum_gate_file='', auto_curriculum=True,
        practice_env_fraction=.125, practice_speeds=[.05,.075,.1], practice_stop_hold_max=1.,
        zero_command_probability=.2, velocity_error_huber_weight=args.velocity_huber_weight)
    rcfg = json.loads((args.config_source / 'runner_config.json').read_text())
    rcfg.update(seed=args.seed, max_iterations=args.iterations, run_name=args.run,
        save_interval=args.save_interval, upload_model=False)
    assert rcfg['algorithm']['class_name'] == 'PPO'
    rcfg['algorithm'].update(learning_rate=.0003, schedule='adaptive', entropy_coef=.001)
    rcfg['actor']['distribution_cfg']['init_std'] = .25
    for name in ('initial_qpos.npy','initial_state.json','scene.xml'):
        source = ROOT / 'outputs/opposed_thumb' / name
        assert source.read_bytes() == (args.config_source / ('pose_'+name)).read_bytes()
        shutil.copy2(source, run / ('pose_'+name))
    snapshot = run / 'source'; snapshot.mkdir()
    for file in (ROOT / 'scripts').glob('*.py'): shutil.copy2(file, snapshot / file.name)
    write_json(run/'env_config.json', {**cfg, 'history_observation': HISTORY_CONFIG})
    write_json(run/'runner_config.json', rcfg)
    write_json(run/'numeric_precision.json', dict(cuda_matmul_allow_tf32=False,
        float32_matmul_precision='highest', torch_version=torch.__version__))
    write_json(run/'launch_args.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    env = HistoryTeleopEnv(FixedPhysicsTeleopEnv(TeleopCfg(**cfg), args.device))
    try:
        runner = ScratchRunner(env, copy.deepcopy(rcfg), str(run), device=args.device)
        assert not runner.alg.optimizer.state
        assert env.global_steps == 0 and not env.curriculum_records
        assert all(p.requires_grad for p in runner.alg.get_policy().parameters())
        # Fresh constructor initializes network/normalizers/Adam; no checkpoint is read.
        runner.save(run/'initial_random.pt')
        write_json(run/'initialization_audit.json', dict(pretrained_weights_loaded=False,
            pretrained_optimizer_loaded=False, optimizer_state_entries=0,
            environment_steps=0, actor_and_critic_trainable=True,
            observation_normalization='learned online from fresh rollouts',
            initial_checkpoint_sha256=hashlib.sha256((run/'initial_random.pt').read_bytes()).hexdigest(),
            velocity_error_huber_weight=args.velocity_huber_weight,
            goal_speed_mps=.1, training_curriculum_only=True,
            qualification='Independent deterministic multidirection rollouts at both physics timesteps required.'))
        write_json(run/'process.json', dict(pid=os.getpid(), start_time=time.time(), num_envs=args.num_envs))
        runner.learn(args.iterations, init_at_random_ep_len=False)
        write_json(run/'finished.json', dict(iteration=runner.current_learning_iteration, time_unix=time.time()))
    finally:
        for group in env.groups: group.sim.pool.shutdown()


if __name__ == '__main__': main()
