"""Fresh command-conditioned PPO training on the opposed-thumb scene."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time,random,shutil
from dataclasses import asdict
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from mjlab.rl.config import RslRlOnPolicyRunnerCfg,RslRlModelCfg,RslRlPpoAlgorithmCfg
from teleop_env import TeleopCfg,TeleopEnv
from balance_model import ROOT

class TeleopRunner(OnPolicyRunner):
    def __init__(self,env,*args,**kwargs):
        super().__init__(env,*args,**kwargs)
        if env.cfg.get('contact_load_observation',False):
            from contact_context import split_contact_optimizer
            split_contact_optimizer(self.alg)
        elif env.cfg.get('command_target_observation',False):
            from command_context import split_command_optimizer
            split_command_optimizer(self.alg)

    def train_contact_branch_only(self):
        actor=self.alg.get_policy()
        assert self.env.cfg.get('contact_load_observation',False) and hasattr(actor,'contact_projection')
        actor.requires_grad_(False)
        actor.contact_projection.requires_grad_(True)
        self.freeze_actor_normalization()
        self.contact_branch_only=True
        self.contact_backbone_trainable=False
        self.command_branch_only=False
        self.command_backbone_trainable=False

    def release_contact_backbone(self):
        """Jointly adapt state, command and contact feedback; preserve exploration/statistics."""
        self.train_contact_branch_only()
        actor=self.alg.get_policy()
        actor.mlp.requires_grad_(True)
        actor.command_projection.requires_grad_(True)
        self.contact_branch_only=False
        self.contact_backbone_trainable=True

    def train_command_branch_only(self):
        actor=self.alg.get_policy()
        assert hasattr(actor,'command_projection')
        actor.requires_grad_(False)
        actor.command_projection.requires_grad_(True)
        self.freeze_actor_normalization()
        self.command_branch_only=True
        self.command_backbone_trainable=False

    def release_command_backbone(self):
        """Learn feedback as well as command transients, preserving exploration/statistics."""
        self.train_command_branch_only()
        self.alg.get_policy().mlp.requires_grad_(True)
        self.command_branch_only=False
        self.command_backbone_trainable=True

    def freeze_actor_for_value_warmup(self):
        """Fit values without changing the sampled physical policy."""
        actor = self.alg.get_policy()
        if self.env.teacher is None:
            # A nonzero pretrained actor also depends on its input statistics.
            self.freeze_actor_normalization()
        else:
            final = [m for m in actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
            assert torch.count_nonzero(final.weight) == 0 and torch.count_nonzero(final.bias) == 0
        actor.requires_grad_(False)
        self.actor_frozen_for_critic_warmup = True

    def freeze_actor_normalization(self):
        normalizer = self.alg.get_policy().obs_normalizer
        assert hasattr(normalizer, 'until') and int(normalizer.count) > 0
        normalizer.until = int(normalizer.count)
        self.actor_normalization_frozen = True

    def release_actor_after_value_warmup(self):
        assert getattr(self, 'actor_frozen_for_critic_warmup', False)
        self.alg.get_policy().requires_grad_(True)
        self.actor_frozen_for_critic_warmup = False

    def freeze_observation_normalization(self):
        """Keep pretrained input coordinates while actor/critic weights learn."""
        for model in [self.alg.get_policy(),self.alg.critic]:
            normalizer=model.obs_normalizer
            assert hasattr(normalizer,'until') and int(normalizer.count)>0
            normalizer.until=int(normalizer.count)
        self.observation_normalization_frozen=True

    def load(self,*args,**kwargs):
        infos=super().load(*args,**kwargs)
        assert bool((infos or {}).get('contact_load_observation',False))==bool(self.env.cfg.get('contact_load_observation',False)), 'Contact-load migration must be explicit'
        previous_inertia=(infos or {}).get('hand_inertia_mode','legacy')
        current_inertia=self.env.cfg.get('hand_inertia_mode','legacy')
        changing_inertia=previous_inertia!=current_inertia
        if changing_inertia:
            assert getattr(self,'allow_inertia_transition',False), 'Changing checkpoint inertia requires --adapt-hand-inertia'
            assert previous_inertia=='legacy' and current_inertia=='collision_only'
            assert self.recovery_reset_config() is None, 'Do not restore old-physics recovery states during inertia adaptation'
            self.inertia_transition=dict(from_mode=previous_inertia,to_mode=current_inertia,
                discarded_recovery_config=(infos or {}).get('recovery_reset_config'))
        recovery=(infos or {}).get('recovery_reset_config')
        if recovery is not None:
            assert recovery==self.recovery_reset_config() or changing_inertia, 'Resume must preserve the saved recovery-state configuration'
        reference=(infos or {}).get('policy_reference')
        if reference is not None:
            assert reference==self.policy_reference_config(), 'Resume must preserve the saved reference constraint'
        replay=getattr(self.alg,'replay',None)
        if replay is not None and 'reference_replay_cursor' in (infos or {}):
            cursor=infos['reference_replay_cursor']
            assert isinstance(cursor,int) and 0<=cursor<len(replay.observations)
            replay.cursor=cursor
        if (infos or {}).get('observation_normalization_frozen',False):
            self.freeze_observation_normalization()
        if (infos or {}).get('actor_normalization_frozen',False):
            self.freeze_actor_normalization()
        if (infos or {}).get('actor_frozen_for_critic_warmup',False):
            self.freeze_actor_for_value_warmup()
        if (infos or {}).get('command_branch_only',False):
            assert not (infos or {}).get('command_backbone_trainable',False)
            self.train_command_branch_only()
        if (infos or {}).get('command_backbone_trainable',False):
            self.release_command_backbone()
        if self.env.cfg.get('contact_load_observation',False):
            branch=bool((infos or {}).get('contact_branch_only',False))
            backbone=bool((infos or {}).get('contact_backbone_trainable',False))
            assert branch != backbone, 'Resume needs one explicit contact training phase'
            assert not any((infos or {}).get(k,False) for k in ('command_branch_only','command_backbone_trainable','actor_frozen_for_critic_warmup'))
            if backbone:self.release_contact_backbone()
            else:self.train_contact_branch_only()
        temporal=(infos or {}).get('temporal_smoothness')
        if temporal is not None:
            assert self.temporal_smoothness_config()=={k:temporal[k] for k in ('weight','ramp_updates')}, 'Resume must preserve temporal regularization'
            assert isinstance(temporal['completed_updates'],int) and temporal['completed_updates']>=0
            self.alg.temporal_updates=temporal['completed_updates']
        return infos

    def temporal_smoothness_config(self):
        if not hasattr(self.alg,'temporal_weight'):return None
        return dict(weight=self.alg.temporal_weight,ramp_updates=self.alg.temporal_ramp_updates)

    def policy_reference_config(self):
        if getattr(self.alg,'reference',None) is None:return None
        result=dict(checkpoint=self.alg.reference_checkpoint,
                    sha256=self.alg.reference_sha256,weight=self.alg.reference_weight)
        if getattr(self.alg,'replay',None) is not None:
            result['replay']={**self.alg.replay.config,'weight':self.alg.replay_weight}
        return result

    def recovery_reset_config(self):
        cfg=self.env.cfg
        if cfg.get('recovery_reset_probability',0)<=0:return None
        return {k:cfg[k] for k in ('recovery_state_bank','recovery_state_bank_sha256',
                                  'recovery_reset_probability','recovery_prefix_seconds')}

    def save(self,path,infos=None):
        curriculum={'speed_limit':self.env.speed_limit,'records':self.env.curriculum_records,
                    'global_steps':self.env.global_steps,'last_promotion_step':self.env.last_promotion_step}
        frozen=getattr(self,'observation_normalization_frozen',False)
        warmup=getattr(self,'actor_frozen_for_critic_warmup',False)
        actor_norm_frozen=getattr(self,'actor_normalization_frozen',False)
        metadata={'teleop_curriculum':curriculum,'observation_normalization_frozen':frozen,
                          'actor_frozen_for_critic_warmup':warmup,
                          'actor_normalization_frozen':actor_norm_frozen}
        metadata['hand_inertia_mode']=self.env.cfg.get('hand_inertia_mode','legacy')
        if self.env.cfg.get('contact_load_observation',False):
            branch=getattr(self,'contact_branch_only',False)
            backbone=getattr(self,'contact_backbone_trainable',False)
            assert branch != backbone
            metadata.update(contact_load_observation=True,contact_branch_only=branch,
                            contact_backbone_trainable=backbone)
        if hasattr(self,'inertia_transition'):
            metadata['inertia_transition']=self.inertia_transition
        branch_only=getattr(self,'command_branch_only',False)
        backbone_trainable=getattr(self,'command_backbone_trainable',False)
        if self.env.cfg.get('command_target_observation',False):
            metadata['command_branch_only']=branch_only
            metadata['command_backbone_trainable']=backbone_trainable
        if self.policy_reference_config() is not None:
            metadata['policy_reference']=self.policy_reference_config()
        if self.recovery_reset_config() is not None:
            metadata['recovery_reset_config']=self.recovery_reset_config()
        if self.temporal_smoothness_config() is not None:
            metadata['temporal_smoothness']={**self.temporal_smoothness_config(),
                'completed_updates':self.alg.temporal_updates}
        if getattr(self.alg,'replay',None) is not None:
            metadata['reference_replay_cursor']=self.alg.replay.cursor
        super().save(path,metadata)
        state={'iteration':self.current_learning_iteration,'checkpoint':path,'time_unix':time.time(),
               'recent_episode_seconds':self.env.last_episode_s,'recent_return':self.env.last_return,
               'environment_steps':self.env.global_steps*self.env.num_envs,'pid':os.getpid(),
               'target_speed_mps':self.env.cfg['target_speed'],'curriculum':curriculum,
               'recent_tracking_score':self.env.last_tracking,'recent_actual_speed_mps':self.env.last_velocity,
               'recent_velocity_error_mps':self.env.last_tracking_error,
               'observation_normalization_frozen':frozen}
        state['training_phase']='critic_warmup' if warmup else 'actor_and_critic'
        if branch_only:state['training_phase']='command_branch_and_critic'
        if backbone_trainable:state['training_phase']='command_actor_and_critic'
        if getattr(self,'contact_branch_only',False):state['training_phase']='contact_branch_and_critic'
        if getattr(self,'contact_backbone_trainable',False):state['training_phase']='contact_feedback_actor_and_critic'
        if self.temporal_smoothness_config() is not None:
            state['temporal_smoothness']=metadata['temporal_smoothness']
            state['last_temporal_coefficient']=self.alg.temporal_coefficient
        state['actor_normalization_frozen']=actor_norm_frozen
        state['active_physics_timestep_s']=self.env.active_physics_timestep
        state['physics_decimation']=self.env.physics_decimation
        if self.env.cfg['physics_backend']=='native':
            state['physics_backend']='native'
        if hasattr(self.env,'physics_groups'):
            state['physics_groups']=self.env.physics_groups
        if self.env.cfg.get('startup_env_fraction',0)>0:
            state['recent_full_episode_seconds']=self.env.last_full_episode_s
            state['recent_startup_failure_rate']=self.env.last_startup_failure_rate
        if self.env.cfg.get('finger_coverage_weight',0)>0:
            from finger_coverage import coverage_deficit
            coverage=self.env.finger_coverage
            state['recent_finger_coverage']=coverage.mean(0).cpu().tolist()
            state['recent_finger_coverage_deficit']=float(coverage_deficit(coverage,self.env.cfg['finger_coverage_target']).mean())
        if self.env.cfg.get('residual_bounds_weight',0)>0:
            state['recent_residual_clipped_fraction']=self.env.last_residual_clipped_fraction
            state['recent_residual_excess_cost']=self.env.last_residual_excess_cost
        if self.env.cfg.get('direct_action_bounds_weight',0)>0:
            state['recent_direct_clipped_fraction']=self.env.last_direct_clipped_fraction
            state['recent_direct_excess_cost']=self.env.last_direct_excess_cost
            state['recent_direct_invalid_fraction']=self.env.last_direct_invalid_fraction
        if self.env.cfg.get('hand_tilt_rate_weight',0)>0:
            state['recent_hand_tilt_rate_cost']=self.env.last_hand_tilt_rate_cost
        if self.env.cfg.get('body_motion_weight',0)>0:
            state['recent_body_motion_cost']=self.env.last_body_motion_cost
        if self.env.cfg.get('practice_env_fraction',0)>0:
            state['practice_exposure']=self.env.practice_metrics()
        if self.env.cfg.get('recovery_reset_probability',0)>0:
            state['recovery_resets']=self.env.recovery_metrics()
        p=ROOT/'runs'/self.env.cfg['run_name']/'progress.json' if 'run_name' in self.env.cfg else __import__('pathlib').Path(path).parent/'progress.json'
        p.with_suffix('.tmp').write_text(json.dumps(state,indent=2));p.with_suffix('.tmp').replace(p)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='teleop_opposed_v1');p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--iterations',type=int,default=5000);p.add_argument('--target-speed',type=float,default=1.)
    p.add_argument('--save-interval',type=int,default=50,help='Checkpoint interval in PPO iterations; does not change the rollout or optimizer settings')
    p.add_argument('--free-heading',action='store_true',help='Allow physical yaw while preserving true tilt and support limits')
    p.add_argument('--heading-observations',action='store_true',help='Optional heading-canonical policy inputs; requires free heading and a compatible policy')
    p.add_argument('--filtered-velocity-observations',action='store_true',help='Append displacement-EMA ball XY velocity; requires a compatible 124-input checkpoint for the current free-heading task')
    p.add_argument('--physics-timestep-observation',action='store_true',help='Append actual physics timestep in milliseconds; requires a compatible expanded checkpoint')
    p.add_argument('--split-timestep-projection',action='store_true',help='Evaluate the appended timestep column separately to preserve original input arithmetic')
    p.add_argument('--command-target-observation',action='store_true')
    p.add_argument('--command-branch-learning-rate',type=float,default=0.0003)
    p.add_argument('--contact-load-observation',action='store_true')
    p.add_argument('--contact-branch-only',action='store_true')
    p.add_argument('--release-contact-backbone',action='store_true',help='Jointly train state/command/contact feedback while retaining actor normalization and exploration.')
    p.add_argument('--contact-branch-learning-rate',type=float,default=0.0003)
    p.add_argument('--initialize-contact-feedback',action='store_true',help='Explicitly migrate a legacy command-context checkpoint with zero contact projections.')
    p.add_argument('--release-command-backbone',action='store_true',
        help='Train the original actor MLP with the command branch; keep exploration and input statistics frozen.')
    p.add_argument('--initial-speed',type=float,default=0.);p.add_argument('--episode-seconds',type=float,default=15.)
    p.add_argument('--startup-env-fraction',type=float,default=0.)
    p.add_argument('--startup-episode-seconds',type=float,default=3.)
    p.add_argument('--command-hold-min',type=float,default=4.)
    p.add_argument('--command-hold-max',type=float,default=8.)
    p.add_argument('--long-command-probability',type=float,default=0.,help='Fraction of command draws held for the optional long duration, including stops.')
    p.add_argument('--long-command-hold-min',type=float,default=20.)
    p.add_argument('--long-command-hold-max',type=float,default=35.)
    p.add_argument('--fixed-learning-rate',type=float)
    p.add_argument('--freeze-observation-normalization',action='store_true')
    phase=p.add_mutually_exclusive_group()
    phase.add_argument('--critic-warmup',action='store_true')
    phase.add_argument('--release-actor',action='store_true')
    p.add_argument('--reset-curriculum',action='store_true')
    p.add_argument('--dense-command-tracking',action='store_true')
    p.add_argument('--independent-curriculum-gate',action='store_true')
    p.add_argument('--zero-command-probability',type=float,default=.2)
    p.add_argument('--full-speed-probability',type=float,default=0.,
                   help='Fraction of moving commands sampled at the current curriculum speed cap.')
    p.add_argument('--diagonal-command-probability',type=float,default=0.,
                   help='Fraction of commands assigned to exact keyboard diagonals (0–0.5); cardinal share stays0.5.')
    p.add_argument('--reverse-command-probability',type=float,default=0.,
                   help='Chance to reverse an ongoing nonzero command on a moving-to-moving transition; retains sampled speed and hold duration.')
    p.add_argument('--rightward-command-probability',type=float,default=0.,
                   help='Fraction of moving targets replaced equally by right and forward-right directions; preserves sampled speed and duration.')
    p.add_argument('--reversal-sequence-probability',type=float,default=0.,
                   help='Chance for a fresh moving target to start stop1s/move2s/reverse2s/stop2s, then return to normal sampling.')
    p.add_argument('--curriculum-speed-levels',type=float,nargs='+',
                   help='Optional increasing speed stages ending at target speed; existing promotion gates still apply.')
    p.add_argument('--exploration-scale',type=float,default=1.)
    p.add_argument('--action-filter-alpha',type=float,default=.3,
                   help='New-action fraction in the 100 Hz joint target EMA; smaller values add smoothing and delay.')
    p.add_argument('--pd-derivative-gain',type=float,default=.18,
                   help='Joint position-controller derivative gain; saved for evaluation and keyboard control.')
    p.add_argument('--joint-target-rate-weight',type=float,default=0.,
                   help='Penalty on mean squared bounded actuator target velocity, in (rad/s)^2.')
    p.add_argument('--temporal-smoothness-weight',type=float,default=0.,
                   help='Optional direct actor loss on consecutive clipped means; excludes resets and requested-command changes.')
    p.add_argument('--temporal-smoothness-ramp-updates',type=int,default=100)
    p.add_argument('--entropy',type=float,default=.003)
    p.add_argument('--rollout-steps',type=int,default=32,
                   help='Control steps collected per environment before each PPO update.')
    p.add_argument('--gae-lambda',type=float,default=.95,
                   help='GAE trace parameter; defaults preserve the existing training setup.')
    p.add_argument('--residual-teacher',default='')
    p.add_argument('--residual-scale',type=float,default=.35)
    p.add_argument('--residual-target-space',action='store_true')
    p.add_argument('--command-path-limit',type=float,default=0.)
    p.add_argument('--path-reward-weight',type=float,default=4.)
    p.add_argument('--finger-coverage-weight',type=float,default=0.)
    p.add_argument('--finger-coverage-tau',type=float,default=1.)
    p.add_argument('--finger-coverage-target',type=float,default=.25)
    p.add_argument('--residual-bounds-weight',type=float,default=0.)
    p.add_argument('--direct-action-bounds-weight',type=float,default=0.,
                   help='Penalty on raw direct-actor outputs beyond +/-1; physical clipping and limits are unchanged.')
    p.add_argument('--hand-tilt-rate-weight',type=float,default=0.,
                   help='Extra cost on world-horizontal hand angular velocity squared (capped at100); yaw remains free.')
    p.add_argument('--body-motion-weight',type=float,default=0.,
                   help='Physical all-axis angular and relative vertical velocity cost; does not constrain heading or change control.')
    p.add_argument('--body-motion-substeps',action='store_true',
                   help='Average existing body-motion cost at every native physics substep; default uses the control endpoint.')
    p.add_argument('--hand-inertia-mode', choices=['legacy', 'collision_only'], default='legacy',
                   help='Explicit hand inertia source; legacy preserves existing checkpoints, collision_only excludes visual mesh inertia inference.')
    p.add_argument('--adapt-hand-inertia',action='store_true',
                   help='Explicit legacy-to-collision-only checkpoint transition; requires recovery resets disabled.')
    p.add_argument('--practice-env-fraction',type=float,default=0.,
                   help='Fraction of main worlds practicing specified speeds; excluded from curriculum statistics.')
    p.add_argument('--practice-stop-hold-max',type=float,default=0.,
                   help='Cap sampled stop holds only in practice worlds; zero disables. Keyboard stops and recovery prefixes are unchanged.')
    p.add_argument('--practice-speeds',type=float,nargs='+',default=[],
                   help='Explicit training speeds for practice worlds; does not grant independent approval.')
    p.add_argument('--recovery-state-bank',default='')
    p.add_argument('--recovery-state-bank-sha256',default='',help='SHA256 of the bank metadata JSON, which locks its NPZ hash.')
    p.add_argument('--recovery-reset-probability',type=float,default=0.,help='Fraction of main-world resets sampled from the native recovery bank; startup worlds are excluded.')
    p.add_argument('--recovery-prefix-seconds',type=float,choices=[2.,4.],default=2.,help='Recorded command duration after a recovery reset, capped at the original profile end.')
    p.add_argument('--physics-timestep',type=float,default=.002,
                   help='Physics substep; control remains 100 Hz (must divide 0.01 seconds).')
    p.add_argument('--physics-backend',choices=['warp','native'],default='warp',
                   help='Opt in to persistent native MuJoCo worlds for fine-tuning; default retains mjlab Warp.')
    p.add_argument('--alternate-physics-timestep',type=float,default=0.)
    p.add_argument('--physics-timestep-switch-steps',type=int,default=1000)
    p.add_argument('--fixed-physics-groups',action='store_true',
                   help='Use two equal persistent simulation groups in every PPO batch.')
    p.add_argument('--allow-identical-physics-groups',action='store_true',
                   help='Explicitly split one timestep into two simulation groups to bound solver memory.')
    p.add_argument('--disable-tf32',action='store_true',help='Use full float32 matrix multiplication for actor, critic and PPO; preserve the existing TF32 default unless requested')
    p.add_argument('--reference-policy-weight',type=float,default=0.,help='Optional frozen-reference KL coefficient in PPO; zero preserves the original algorithm')
    p.add_argument('--reference-policy-checkpoint',help='Frozen reference actor checkpoint; required with a positive reference weight')
    p.add_argument('--reference-policy-sha256',help='Expected SHA256 of the frozen reference checkpoint')
    p.add_argument('--reference-replay-weight',type=float,default=0.)
    p.add_argument('--reference-replay-path')
    p.add_argument('--reference-replay-sha256')
    p.add_argument('--reference-replay-batch-size',type=int,default=256)
    p.add_argument('--seed',type=int,default=73);p.add_argument('--resume');p.add_argument('--no-curriculum',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    assert args.rollout_steps>0 and 0<args.gae_lambda<=1
    assert args.save_interval>0
    assert not args.allow_identical_physics_groups or args.fixed_physics_groups
    torch.backends.cuda.matmul.allow_tf32=not args.disable_tf32
    run=ROOT/'runs'/args.run
    if run.exists() and not args.resume:raise RuntimeError(f'Refusing to overwrite existing run {run}')
    run.mkdir(parents=True,exist_ok=True)
    (run/'numeric_precision.json').write_text(json.dumps({
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'torch_version':torch.__version__,
        'scope':'Process-wide matrix multiplication used by actor, critic and PPO; physics settings unchanged.'},indent=2)+'\n')
    cfg=TeleopCfg(num_envs=args.num_envs,episode_seconds=args.episode_seconds,target_speed=args.target_speed,
                  initial_speed=args.initial_speed,seed=args.seed,auto_curriculum=not args.no_curriculum)
    assert args.physics_timestep>0
    cfg.physics_timestep=args.physics_timestep
    cfg.physics_backend=args.physics_backend
    cfg.alternate_physics_timestep=args.alternate_physics_timestep
    cfg.physics_timestep_switch_steps=args.physics_timestep_switch_steps
    cfg.fixed_physics_groups=args.fixed_physics_groups
    cfg.free_heading=args.free_heading
    cfg.heading_observations=args.heading_observations
    cfg.filtered_velocity_observations=args.filtered_velocity_observations
    cfg.physics_timestep_observation=args.physics_timestep_observation
    cfg.split_timestep_projection=args.split_timestep_projection
    cfg.command_target_observation=args.command_target_observation
    assert not args.release_command_backbone or cfg.command_target_observation
    if cfg.command_target_observation:
        assert args.resume and args.fixed_learning_rate
        assert not args.critic_warmup and not args.release_actor and not args.reference_policy_weight
        assert args.command_branch_learning_rate>0 and np.isfinite(args.command_branch_learning_rate)
    assert not cfg.split_timestep_projection or cfg.physics_timestep_observation
    cfg.decimation=round(.01/args.physics_timestep)
    assert cfg.decimation>=1 and abs(cfg.decimation*cfg.physics_timestep-.01)<1e-10
    cfg.residual_teacher=args.residual_teacher;cfg.residual_scale=args.residual_scale
    cfg.startup_env_fraction=args.startup_env_fraction
    cfg.startup_episode_seconds=args.startup_episode_seconds
    cfg.residual_target_space=args.residual_target_space
    cfg.command_path_limit=args.command_path_limit;cfg.path_reward_weight=args.path_reward_weight
    cfg.command_hold_min=args.command_hold_min;cfg.command_hold_max=args.command_hold_max
    cfg.long_command_probability=args.long_command_probability
    cfg.long_command_hold_min=args.long_command_hold_min;cfg.long_command_hold_max=args.long_command_hold_max
    cfg.finger_coverage_weight=args.finger_coverage_weight
    cfg.finger_coverage_tau=args.finger_coverage_tau;cfg.finger_coverage_target=args.finger_coverage_target
    cfg.residual_bounds_weight=args.residual_bounds_weight
    cfg.direct_action_bounds_weight=args.direct_action_bounds_weight
    cfg.hand_tilt_rate_weight=args.hand_tilt_rate_weight
    cfg.body_motion_weight=args.body_motion_weight
    cfg.body_motion_substeps=args.body_motion_substeps
    cfg.contact_load_observation=args.contact_load_observation
    assert not (args.contact_branch_only and args.release_contact_backbone)
    assert (args.contact_branch_only or args.release_contact_backbone)==cfg.contact_load_observation
    assert not args.initialize_contact_feedback or cfg.contact_load_observation
    assert not args.release_contact_backbone or not args.initialize_contact_feedback
    if cfg.contact_load_observation:
        assert args.resume and args.fixed_learning_rate and cfg.command_target_observation and cfg.physics_backend=='native'
        assert not any((args.release_command_backbone,args.critic_warmup,args.release_actor,args.adapt_hand_inertia,args.reset_curriculum))
        assert args.exploration_scale==1 and not args.recovery_reset_probability
        assert not args.temporal_smoothness_weight and not args.reference_policy_weight and not args.reference_replay_weight
        assert np.isfinite(args.contact_branch_learning_rate) and args.contact_branch_learning_rate>0
    cfg.hand_inertia_mode=args.hand_inertia_mode
    cfg.practice_env_fraction=args.practice_env_fraction
    cfg.practice_speeds=tuple(args.practice_speeds)
    cfg.practice_stop_hold_max=args.practice_stop_hold_max
    cfg.action_filter_alpha=args.action_filter_alpha
    cfg.pd_derivative_gain=args.pd_derivative_gain
    cfg.joint_target_rate_weight=args.joint_target_rate_weight
    cfg.recovery_state_bank=args.recovery_state_bank
    cfg.recovery_state_bank_sha256=args.recovery_state_bank_sha256
    cfg.recovery_reset_probability=args.recovery_reset_probability
    cfg.recovery_prefix_seconds=args.recovery_prefix_seconds
    if cfg.recovery_reset_probability>0:
        assert args.resume and cfg.physics_backend=='native' and cfg.hand_tilt_rate_weight==0
    cfg.dense_command_tracking=args.dense_command_tracking
    cfg.zero_command_probability=args.zero_command_probability
    cfg.full_speed_probability=args.full_speed_probability
    cfg.diagonal_command_probability=args.diagonal_command_probability
    cfg.reverse_command_probability=args.reverse_command_probability
    cfg.rightward_command_probability=args.rightward_command_probability
    cfg.reversal_sequence_probability=args.reversal_sequence_probability
    cfg.curriculum_speed_levels=tuple(args.curriculum_speed_levels or ())
    cfg.curriculum_gate_file=str(run/'curriculum_approval.json') if args.independent_curriculum_gate else ''
    assert 0<=cfg.zero_command_probability<1 and args.exploration_scale>0
    assert 0<=cfg.full_speed_probability<=1
    assert 0<=cfg.diagonal_command_probability<=.5
    assert 0<=cfg.long_command_probability<=1
    assert cfg.long_command_hold_max>=cfg.long_command_hold_min>0
    assert not cfg.long_command_probability or cfg.long_command_hold_min>cfg.command_hold_max
    (run/'env_config.json').write_text(json.dumps(asdict(cfg),indent=2));(run/'launch_args.json').write_text(json.dumps(vars(args),indent=2))
    (run/'source').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'scripts/heading_frame.py',run/'source/heading_frame.py')
    shutil.copy2(ROOT/'scripts/timestep_input.py',run/'source/timestep_input.py')
    shutil.copy2(ROOT/'scripts/command_context.py',run/'source/command_context.py')
    if cfg.contact_load_observation:
        for name in ('contact_context.py','contact_load.py'):
            shutil.copy2(ROOT/'scripts'/name,run/'source'/name)
    if args.temporal_smoothness_weight:
        shutil.copy2(ROOT/'scripts/temporal_smoothness.py',run/'source/temporal_smoothness.py')
    for name in ['balance_model.py','balance_env.py','teleop_env.py','train_teleop.py','native_sim.py','teleop_control.py','evaluate_balance.py','residual_policy.py','finger_coverage.py','fixed_physics_teleop.py','evaluate_teleop.py','evaluate_teleop_parallel.py','watch_teleop_training.py','reference_ppo.py','reference_replay.py','recovery_resets.py']:
        if (ROOT/'scripts'/name).exists():shutil.copy2(ROOT/'scripts'/name,run/'source'/name)
    if cfg.recovery_reset_probability>0:
        from pathlib import Path
        bank_path=Path(cfg.recovery_state_bank)
        shutil.copy2(bank_path,run/'recovery_bank.json')
        shutil.copy2(bank_path.with_suffix('.npz'),run/'recovery_bank.npz')
    # Preserve the exact reset state with the run. Scene meshes remain the official assets.
    for name in ['initial_qpos.npy','initial_state.json','scene.xml']:
        shutil.copy2(ROOT/'outputs/opposed_thumb'/name,run/('pose_'+name))
    if args.fixed_physics_groups:
        from fixed_physics_teleop import FixedPhysicsTeleopEnv
        env=FixedPhysicsTeleopEnv(cfg,allow_identical_timesteps=args.allow_identical_physics_groups)
    else:
        env=TeleopEnv(cfg)
    rcfg=asdict(RslRlOnPolicyRunnerCfg(seed=args.seed,num_steps_per_env=args.rollout_steps,max_iterations=args.iterations,save_interval=args.save_interval,
        experiment_name='revo3_ball_teleop',run_name=args.run,logger='tensorboard',
        actor=RslRlModelCfg(hidden_dims=(256,128,128),obs_normalization=True,distribution_cfg={'class_name':'GaussianDistribution','init_std':.35,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(256,128,128),obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=args.fixed_learning_rate or 3e-4,schedule='fixed' if args.fixed_learning_rate else 'adaptive',entropy_coef=args.entropy,num_learning_epochs=5,
            num_mini_batches=8,gamma=.998,lam=args.gae_lambda,desired_kl=.01)))
    for role in ['actor','critic']:
        for key in ['cnn_cfg','rnn_type','rnn_hidden_dim','rnn_num_layers']:rcfg[role].pop(key,None)
        if cfg.command_target_observation:rcfg[role]['class_name']='command_context:CommandContextModel'
        if cfg.contact_load_observation:rcfg[role]['class_name']='contact_context:ContactContextModel'
    assert np.isfinite(args.reference_policy_weight) and args.reference_policy_weight>=0
    if args.reference_policy_weight:
        assert args.resume and args.reference_policy_checkpoint and args.reference_policy_sha256
        assert not args.residual_teacher and not args.split_timestep_projection
        rcfg['algorithm'].update(class_name='reference_ppo:ReferencePPO',
            reference_weight=args.reference_policy_weight,
            reference_checkpoint=args.reference_policy_checkpoint,
            reference_sha256=args.reference_policy_sha256)
    else:
        assert not args.reference_policy_checkpoint and not args.reference_policy_sha256
    assert np.isfinite(args.reference_replay_weight) and args.reference_replay_weight>=0
    if args.reference_replay_weight:
        assert args.reference_policy_weight and args.reference_replay_path and args.reference_replay_sha256
        assert 1<=args.reference_replay_batch_size<=2048
        rcfg['algorithm'].update(replay_weight=args.reference_replay_weight,
            replay_path=args.reference_replay_path,replay_sha256=args.reference_replay_sha256,
            replay_batch_size=args.reference_replay_batch_size)
    else:
        assert not args.reference_replay_path and not args.reference_replay_sha256
    assert np.isfinite(args.temporal_smoothness_weight) and args.temporal_smoothness_weight>=0
    assert args.temporal_smoothness_ramp_updates>0
    if args.temporal_smoothness_weight:
        assert args.resume and cfg.command_target_observation and args.release_command_backbone
        assert args.fixed_learning_rate and cfg.physics_backend=='native'
        assert not args.reference_policy_weight and not args.reference_replay_weight
        assert cfg.joint_target_rate_weight==0, 'Keep direct regularization separate from target-rate reward trials'
        rcfg['algorithm'].update(class_name='temporal_smoothness:TemporalPPO',
            temporal_weight=args.temporal_smoothness_weight,
            temporal_ramp_updates=args.temporal_smoothness_ramp_updates)
    (run/'runner_config.json').write_text(json.dumps(rcfg,indent=2))
    runner=TeleopRunner(env,rcfg,str(run),device='cuda:0')
    if cfg.split_timestep_projection:
        from timestep_input import install_timestep_projection
        for model in (runner.alg.get_policy(),runner.alg.critic):
            install_timestep_projection(model)
    if args.residual_teacher and not args.resume:
        # Exactly reproduce the validated teacher before the first update.
        policy=runner.alg.get_policy()
        final=[module for module in policy.mlp.modules() if isinstance(module,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(final.weight);torch.nn.init.zeros_(final.bias)
    if args.resume:
        runner.allow_inertia_transition=args.adapt_hand_inertia
        load_path=args.resume
        if args.initialize_contact_feedback:
            from contact_context import migrate_checkpoint
            import hashlib
            source=torch.load(args.resume,map_location='cpu',weights_only=False)
            assert not (source.get('infos') or {}).get('contact_load_observation',False)
            load_path=run/'contact_initialization.pt'
            assert not load_path.exists()
            torch.save(migrate_checkpoint(source,args.contact_branch_learning_rate),load_path)
            (run/'contact_initialization.json').write_text(json.dumps(dict(source=str(args.resume),
                source_sha256=hashlib.sha256(open(args.resume,'rb').read()).hexdigest(),
                migrated_sha256=hashlib.sha256(load_path.read_bytes()).hexdigest(),
                optimizer_updates=0,zero_initialized=True),indent=2))
        runner.load(load_path)
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        c=(saved.get('infos') or {}).get('teleop_curriculum',{})
        env.speed_limit=c.get('speed_limit',cfg.initial_speed);env.curriculum_records=c.get('records',[])
        env.global_steps=c.get('global_steps',0);env.last_promotion_step=c.get('last_promotion_step',env.global_steps)
        if args.reset_curriculum:
            env.speed_limit=cfg.initial_speed;env.curriculum_records=[];env.last_promotion_step=env.global_steps
        ids=torch.arange(env.num_envs,device=env.device)
        if cfg.reversal_sequence_probability>0:
            env.restart_commands(ids)
        else:
            env.sample_commands(ids)
        if cfg.recovery_reset_probability>0:
            env.initialize_recovery_resets()
    if args.freeze_observation_normalization:
        assert args.resume,'Frozen normalization requires pretrained statistics'
        runner.freeze_observation_normalization()
    if args.fixed_learning_rate:
        runner.alg.learning_rate=args.fixed_learning_rate;runner.alg.schedule='fixed'
        for group in runner.alg.optimizer.param_groups:group['lr']=args.fixed_learning_rate
    if cfg.command_target_observation:
        assert any(getattr(runner,k,False) for k in ('command_branch_only','command_backbone_trainable','contact_branch_only','contact_backbone_trainable')), 'Initialize or resume an explicit command-context checkpoint'
        runner.alg.optimizer.param_groups[1]['lr']=args.command_branch_learning_rate
        if args.release_command_backbone:runner.release_command_backbone()
    if cfg.contact_load_observation:
        runner.alg.optimizer.param_groups[2]['lr']=args.contact_branch_learning_rate
        if args.release_contact_backbone:
            assert (saved.get('infos') or {}).get('contact_load_observation',False), 'Initialize contact inputs before joint adaptation'
            runner.release_contact_backbone()
        else:
            assert not getattr(runner,'contact_backbone_trainable',False), 'Do not silently refreeze a joint-adaptation checkpoint'
            runner.train_contact_branch_only()
    if args.exploration_scale!=1.:
        with torch.no_grad():
            for name,param in runner.alg.get_policy().named_parameters():
                if name=='distribution.std_param':param.mul_(args.exploration_scale)
                elif name=='distribution.log_std':param.add_(np.log(args.exploration_scale))
    if args.critic_warmup:
        runner.freeze_actor_for_value_warmup()
    if args.release_actor:
        assert args.resume, 'Releasing a warmed actor requires its checkpoint'
        runner.release_actor_after_value_warmup()
    (run/'process.json').write_text(json.dumps({'pid':os.getpid(),'start_time':time.time(),'num_envs':args.num_envs}))
    runner.learn(args.iterations,init_at_random_ep_len=False)
    (run/'finished.json').write_text(json.dumps({'iteration':runner.current_learning_iteration,'time_unix':time.time()}))

if __name__=='__main__':main()
