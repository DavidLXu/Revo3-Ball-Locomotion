"""Command-conditioned planar ball locomotion, using the opposed-thumb keyframe.

World +X/-X and +Y/-Y commands affect observations and rewards only. No root,
ball, or external-force actuators are added. Curriculum promotions are recorded.
"""
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from balance_env import BalanceEnv, BalanceCfg


def hand_tilt_rate_cost(rotation, local_angular_velocity):
    """Squared world-horizontal angular speed; vertical yaw remains free."""
    world = (rotation @ local_angular_velocity[..., None]).squeeze(-1)
    return world[..., :2].square().sum(-1).clamp(max=100.)


def body_motion_cost(local_angular_velocity, relative_vertical_velocity):
    """Physical movement cost; all-axis angular norm is frame invariant.

    Reference scales are 1 rad/s and .08 m/s. Penalize rates, not heading:
    no orientation constraint, observation filter, or physical force is added.
    """
    angular = local_angular_velocity.square().sum(-1).clamp(max=16.)
    vertical = (relative_vertical_velocity / .08).square().clamp(max=16.)
    return angular + vertical

def velocity_error_huber_cost(velocity, command):
    """Dense radial velocity error; reward-only, using the existing measured velocity."""
    scale = command.norm(dim=-1).clamp(min=.025)
    error = (velocity-command).norm(dim=-1) / scale
    return torch.where(error <= 1., .5*error.square(), error-.5).clamp(max=4.)


@dataclass
class TeleopCfg(BalanceCfg):
    initial_pose:str='opposed_thumb'
    task:str='teleop'
    num_envs:int=1024
    episode_seconds:float=15.
    contact_weight:float=2.
    five_weight:float=.2
    failure_penalty:float=5.
    reject_palm_support:bool=True
    ball_angular_world:bool=True
    velocity_reward_tau:float=.1
    target_speed:float=1.
    initial_speed:float=0.
    command_acceleration:float=.5
    command_hold_min:float=4.
    command_hold_max:float=8.
    long_command_probability:float=0.
    long_command_hold_min:float=20.
    long_command_hold_max:float=35.
    zero_command_probability:float=.2
    full_speed_probability:float=0.
    diagonal_command_probability:float=0.
    reverse_command_probability:float=0.
    rightward_command_probability:float=0.
    reversal_sequence_probability:float=0.
    curriculum_speed_levels:tuple[float,...]=()
    auto_curriculum:bool=True
    dense_command_tracking:bool=False
    curriculum_gate_file:str=""
    residual_teacher:str=""
    residual_scale:float=.35
    residual_target_space:bool=False
    command_path_limit:float=0.
    path_reward_weight:float=4.
    finger_coverage_weight:float=0.
    finger_coverage_tau:float=1.
    finger_coverage_target:float=.25
    residual_bounds_weight:float=0.
    direct_action_bounds_weight:float=0.
    hand_tilt_rate_weight:float=0.
    body_motion_weight:float=0.
    body_motion_substeps:bool=False
    velocity_error_huber_weight:float=0.
    contact_load_observation:bool=False
    practice_env_fraction:float=0.
    practice_speeds:tuple[float,...]=()
    practice_stop_hold_max:float=0.
    recovery_state_bank:str=''
    recovery_state_bank_sha256:str=''
    recovery_reset_probability:float=0.
    recovery_prefix_seconds:float=2.
    fixed_physics_groups:bool=False
    free_heading:bool=False
    heading_observations:bool=False
    filtered_velocity_observations:bool=False
    physics_timestep_observation:bool=False
    split_timestep_projection:bool=False
    command_target_observation:bool=False

class TeleopEnv(BalanceEnv):
    def __init__(self,cfg=TeleopCfg(),device='cuda:0'):
        assert cfg.target_speed>0 and 0<=cfg.initial_speed<=cfg.target_speed
        assert cfg.command_acceleration>0 and cfg.command_hold_max>=cfg.command_hold_min>0
        assert 0<=cfg.long_command_probability<=1
        assert cfg.long_command_hold_max>=cfg.long_command_hold_min>0
        assert not cfg.long_command_probability or cfg.long_command_hold_min>cfg.command_hold_max
        assert 0<=cfg.full_speed_probability<=1
        assert 0<=cfg.diagonal_command_probability<=.5
        levels=cfg.curriculum_speed_levels
        assert not levels or (all(np.isfinite(v) and 0<v<=cfg.target_speed for v in levels)
                              and all(a<b for a,b in zip(levels,levels[1:]))
                              and abs(levels[-1]-cfg.target_speed)<1e-12)
        if cfg.free_heading:
            assert cfg.ball_angular_world and not cfg.residual_teacher and cfg.command_path_limit==0
        assert not cfg.heading_observations or cfg.free_heading
        assert not cfg.filtered_velocity_observations or cfg.velocity_reward_tau>0
        assert not cfg.split_timestep_projection or cfg.physics_timestep_observation
        if cfg.command_target_observation:
            assert cfg.free_heading and cfg.filtered_velocity_observations
            assert not cfg.heading_observations and not cfg.physics_timestep_observation
            assert not cfg.residual_teacher and not cfg.reversal_sequence_probability
        self.commands=torch.zeros((cfg.num_envs,2),device=device)
        self.command_prepared=torch.zeros(cfg.num_envs,device=device,dtype=torch.bool)
        self.command_targets=torch.zeros_like(self.commands)
        self.command_path_error=torch.zeros_like(self.commands)
        self.command_countdown=torch.zeros(cfg.num_envs,device=device,dtype=torch.long)
        self.reversal_sequence_stage=torch.zeros(cfg.num_envs,device=device,dtype=torch.long)
        self.reversal_sequence_velocity=torch.zeros_like(self.commands)
        self.finger_coverage=torch.full((cfg.num_envs,5),cfg.finger_coverage_target,device=device)
        self.tracking_sum=torch.zeros(cfg.num_envs,device=device)
        self.command_error_sum=torch.zeros(cfg.num_envs,device=device)
        self.speed_limit=cfg.initial_speed
        self.curriculum_records=[];self.recent=deque(maxlen=2048)
        self.last_promotion_step=0;self.last_tracking=0.;self.last_velocity=0.
        self.last_tracking_error=0.;self.external_command=False
        self.last_residual_clipped_fraction=0.;self.last_residual_excess_cost=0.
        self.residual_excess_cost=torch.zeros(cfg.num_envs,device=device)
        self.direct_action_excess_cost=torch.zeros(cfg.num_envs,device=device)
        self.last_direct_clipped_fraction=0.;self.last_direct_excess_cost=0.
        self.last_direct_invalid_fraction=0.
        self.last_hand_tilt_rate_cost=0.
        self.last_body_motion_cost=0.
        assert np.isfinite(cfg.practice_env_fraction) and 0<=cfg.practice_env_fraction<1
        assert bool(cfg.practice_speeds)==bool(cfg.practice_env_fraction)
        assert np.isfinite(cfg.practice_stop_hold_max) and cfg.practice_stop_hold_max>=0
        if cfg.practice_stop_hold_max:
            assert cfg.practice_env_fraction>0 and not cfg.reversal_sequence_probability
            assert cfg.practice_stop_hold_max>=cfg.physics_timestep*cfg.decimation
        assert all(np.isfinite(s) and 0<s<=cfg.target_speed for s in cfg.practice_speeds)
        assert all(a<b for a,b in zip(cfg.practice_speeds,cfg.practice_speeds[1:]))
        main_count=cfg.num_envs-int(cfg.num_envs*cfg.startup_env_fraction)
        self.practice_env_count=int(main_count*cfg.practice_env_fraction)
        assert not cfg.practice_env_fraction or self.practice_env_count>0
        self.practice_env_mask=torch.arange(cfg.num_envs,device=device)>=cfg.num_envs-self.practice_env_count
        # Per-process exposure telemetry; neither array enters observations or
        # the independent approval gate. Counts include stop/ramp intervals.
        self.practice_requested_steps=torch.zeros(4,device=device,dtype=torch.long)
        self.practice_applied_steps=torch.zeros(4,device=device,dtype=torch.long)
        self.practice_terminations=torch.zeros(2,device=device,dtype=torch.long)
        self.practice_bins=torch.tensor([.07501,.125,.175],device=device)
        self.recovery=None
        assert cfg.command_path_limit>=0 and cfg.path_reward_weight>=0
        assert cfg.command_path_limit==0 or cfg.velocity_reward_tau>0
        assert not cfg.residual_target_space or cfg.residual_teacher
        assert cfg.finger_coverage_weight>=0 and cfg.finger_coverage_tau>0
        assert 0<cfg.finger_coverage_target<=1
        assert cfg.residual_bounds_weight>=0
        assert np.isfinite(cfg.direct_action_bounds_weight) and cfg.direct_action_bounds_weight>=0
        assert not cfg.direct_action_bounds_weight or not cfg.residual_teacher
        assert np.isfinite(cfg.hand_tilt_rate_weight) and cfg.hand_tilt_rate_weight>=0
        assert np.isfinite(cfg.body_motion_weight) and cfg.body_motion_weight>=0
        assert np.isfinite(cfg.velocity_error_huber_weight) and cfg.velocity_error_huber_weight>=0
        assert not cfg.body_motion_substeps or (cfg.physics_backend in ('native','warp') and cfg.body_motion_weight>0)
        assert not cfg.contact_load_observation or (cfg.physics_backend in ('native','warp') and cfg.command_target_observation)
        assert np.isfinite(cfg.recovery_reset_probability) and 0<=cfg.recovery_reset_probability<=1
        assert 0<=cfg.reverse_command_probability<=1
        assert 0<=cfg.rightward_command_probability<=1
        assert not (cfg.reverse_command_probability and cfg.rightward_command_probability)
        assert 0<=cfg.reversal_sequence_probability<=1
        assert not cfg.reversal_sequence_probability or not (cfg.reverse_command_probability or cfg.rightward_command_probability)
        assert not cfg.residual_bounds_weight or cfg.residual_teacher
        super().__init__(cfg,device)
        self.teacher=None
        if cfg.residual_teacher:
            from residual_policy import load_balance_teacher
            self.teacher=load_balance_teacher(cfg.residual_teacher,device)
        if cfg.recovery_reset_probability>0:
            from recovery_resets import RecoveryResets
            self.recovery=RecoveryResets(self)

    def initialize_recovery_resets(self):
        assert not self.episode_length_buf.any()
        self.command_prepared[:]=False
        if self.recovery is not None:
            self.recovery.on_reset(torch.arange(self.num_envs,device=self.device))

    def recovery_metrics(self):
        return self.recovery.metrics() if self.recovery is not None else {}

    def practice_metrics(self):
        return dict(num_worlds=self.practice_env_count,
                    speed_bin_upper_mps=[.07501,.125,.175,self.cfg['target_speed']],
                    requested_control_steps=self.practice_requested_steps.cpu().tolist(),
                    applied_control_steps=self.practice_applied_steps.cpu().tolist(),
                    failed_episodes=int(self.practice_terminations[0]),
                    completed_episodes=int(self.practice_terminations[1]))

    def sample_commands(self,ids):
        self.command_prepared[ids]=False
        n=len(ids)
        # Half cardinal commands (keyboard), half arbitrary headings/diagonals.
        theta=torch.rand(n,device=self.device)*2*torch.pi
        cardinal=torch.randint(0,4,(n,),device=self.device)*torch.pi/2
        direction_draw=torch.rand(n,device=self.device)
        theta=torch.where(direction_draw<.5,cardinal,theta)
        # Reserve part of the non-cardinal half for exact two-key commands.
        # With this option disabled, the original values and RNG stream remain unchanged.
        diagonal_probability=self.cfg.get('diagonal_command_probability',0.)
        if diagonal_probability>0:
            diagonal=torch.randint(0,4,(n,),device=self.device)*torch.pi/2+torch.pi/4
            use_diagonal=(direction_draw>=.5)&(direction_draw<.5+diagonal_probability)
            theta=torch.where(use_diagonal,diagonal,theta)
        speed=self.speed_limit*(.3+.7*torch.rand(n,device=self.device))
        # Opt in to more course-cap commands; default preserves the old RNG stream.
        if self.cfg.get('full_speed_probability',0)>0:
            speed=torch.where(torch.rand(n,device=self.device)<self.cfg['full_speed_probability'],
                              self.speed_limit,speed)
        speed*=torch.rand(n,device=self.device)>=self.cfg['zero_command_probability']
        self.command_targets[ids]=torch.stack((speed*torch.cos(theta),speed*torch.sin(theta)),-1)
        self.command_path_error[ids]=0
        self.direct_action_excess_cost[ids]=0
        self.command_countdown[ids]=torch.randint(round(self.cfg['command_hold_min']/self.dt),round(self.cfg['command_hold_max']/self.dt)+1,(n,),device=self.device)
        # Mix sustained commands with short transitions without changing the default RNG.
        long_probability=self.cfg.get('long_command_probability',0.)
        if long_probability>0:
            use_long=torch.rand(n,device=self.device)<long_probability
            long_ticks=torch.randint(round(self.cfg['long_command_hold_min']/self.dt),round(self.cfg['long_command_hold_max']/self.dt)+1,(n,),device=self.device)
            self.command_countdown[ids]=torch.where(use_long,long_ticks,self.command_countdown[ids])
        # Bias only the temporal direction relationship. Keep the already drawn
        # speeds, stops and hold lengths; never reverse across an episode reset.
        probability=self.cfg.get('reverse_command_probability',0.)
        if probability>0:
            previous=self.commands[ids]
            magnitude=previous.norm(dim=-1)
            eligible=((magnitude>1e-6)&(speed>0)).nonzero().flatten()
            if len(eligible):
                chosen=eligible[torch.rand(len(eligible),device=self.device)<probability]
                self.command_targets[ids[chosen]]=-previous[chosen]/magnitude[chosen,None]*speed[chosen,None]
        # Keep the original speed/stop/duration draws, then focus some moving
        # targets on the two persistently weak world-frame directions.
        focus_probability=self.cfg.get('rightward_command_probability',0.)
        if focus_probability>0:
            eligible=(speed>0).nonzero().flatten()
            if len(eligible):
                chosen=eligible[torch.rand(len(eligible),device=self.device)<focus_probability]
                directions=torch.tensor([[0.,-1.],[2**-.5,-2**-.5]],device=self.device,dtype=speed.dtype)
                choice=torch.randint(0,2,(len(chosen),),device=self.device)
                self.command_targets[ids[chosen]]=directions[choice]*speed[chosen,None]

        if self.practice_env_count:
            selected=ids[self.practice_env_mask[ids]&(speed>0)]
            if len(selected):
                levels=torch.tensor(self.cfg['practice_speeds'],device=self.device)
                magnitude=levels[torch.randint(len(levels),(len(selected),),device=self.device)]
                magnitude=magnitude.clamp(min=self.speed_limit)
                target=self.command_targets[selected]
                self.command_targets[selected]=target/target.norm(dim=-1,keepdim=True)*magnitude[:,None]

        if self.cfg.get('reversal_sequence_probability',0.)>0:
            self.apply_reversal_sequence(ids)
        stop_cap=self.cfg.get('practice_stop_hold_max',0.)
        if stop_cap:
            stopped=ids[self.practice_env_mask[ids]&(self.command_targets[ids].norm(dim=-1)<1e-6)]
            self.command_countdown[stopped]=self.command_countdown[stopped].clamp(max=round(stop_cap/self.dt))

    def apply_reversal_sequence(self,ids):
        # Stage records the next phase, not a policy observation. Every phase
        # uses the existing command slew limiter and ordinary joint controller.
        stage=self.reversal_sequence_stage[ids].clone()
        for phase,sign in ((1,1.),(2,-1.),(3,0.)):
            selected=ids[stage==phase]
            self.command_targets[selected]=sign*self.reversal_sequence_velocity[selected]
            self.command_countdown[selected]=round(2./self.dt)
            self.reversal_sequence_stage[selected]=phase+1
        fresh=ids[(stage==0)|(stage==4)]
        self.clear_reversal_sequences(fresh)
        moving=fresh[self.command_targets[fresh].norm(dim=-1)>0]
        if len(moving):
            selected=moving[torch.rand(len(moving),device=self.device)<self.cfg['reversal_sequence_probability']]
            self.reversal_sequence_velocity[selected]=self.command_targets[selected]
            self.command_targets[selected]=0
            self.command_countdown[selected]=round(1./self.dt)
            self.reversal_sequence_stage[selected]=1

    def clear_reversal_sequences(self,ids):
        self.reversal_sequence_stage[ids]=0
        self.reversal_sequence_velocity[ids]=0

    def restart_commands(self,ids):
        self.clear_reversal_sequences(ids)
        self.sample_commands(ids)
        if self.cfg.get('reversal_sequence_probability',0.)>0:
            # Reset/resume happens between control intervals. The next step
            # decrements the countdown before applying its target; keep a full
            # one-second initial stop, as for sequences started inside step().
            selected=ids[self.reversal_sequence_stage[ids]==1]
            self.command_countdown[selected]+=1

    def reset(self,ids):
        self.command_prepared[ids]=False
        if hasattr(self,'episode_length_buf'):
            finished=ids[self.episode_length_buf[ids]>0]
            # Artificial startup timeouts must not count as failed full-length
            # episodes in the speed curriculum. Their failures remain logged
            # separately; independent full rollouts still gate every promotion.
            finished=finished[finished>=self.startup_env_count]
            # Training-only teacher-generated starts must not qualify the
            # ordinary-initial-state speed curriculum. Clear the marker when
            # a world subsequently resets to the original pose below.
            if hasattr(self,'teacher_start_mask'):
                finished=finished[~self.teacher_start_mask[finished]]
            # Practice above the approved speed must not grant a promotion or
            # contaminate ordinary-initial-state curriculum statistics.
            finished=finished[~self.practice_env_mask[finished]]
            if self.recovery is not None:
                finished=finished[~self.recovery.episode_mask[finished]]
            if len(finished):
                duration=self.episode_length_buf[finished].float()*self.dt
                tracking=self.tracking_sum[finished]/self.episode_length_buf[finished]
                error=self.command_error_sum[finished]/self.episode_length_buf[finished]
                self.recent.extend(zip(duration.cpu().tolist(),tracking.cpu().tolist(),error.cpu().tolist()))
        super().reset(ids)
        self.commands[ids]=0;self.tracking_sum[ids]=0;self.command_error_sum[ids]=0
        self.command_path_error[ids]=0
        self.restart_commands(ids)
        self.finger_coverage[ids]=self.cfg['finger_coverage_target']
        if hasattr(self,'teacher_start_mask'):
            self.teacher_start_mask[ids]=False
        if self.recovery is not None:
            self.recovery.on_reset(ids)

    def set_command(self,vx,vy):
        """Interactive API: direction, stop, and magnitude; no physical force."""
        target=torch.tensor([vx,vy],device=self.device,dtype=torch.float32)
        target*=min(1.,self.cfg['target_speed']/max(float(target.norm()),1e-9))
        changed=(self.command_targets-target).norm(dim=-1)>1e-6
        self.command_path_error[changed]=0
        self.external_command=True;self.command_targets[:]=target
        if self.recovery is not None:self.recovery.cancel_prefixes()

    def get_observations(self):
        if self.cfg.get('command_target_observation',False):
            self.prepare_observed_commands()
        base=super().get_observations()['actor']
        if self.cfg.get('command_path_limit',0)>0:
            base[:,60:62]=self.command_path_error*5
        commands=self.commands
        if self.cfg.get('heading_observations',False):
            from heading_frame import heading_inverse_torch
            rotation=self.sim.data.xmat[:,self.handbody].reshape(-1,3,3)@self.rot0.T
            inverse=heading_inverse_torch(rotation)
            commands=torch.einsum('nij,nj->ni',inverse[:,:2,:2],commands)
        obs=torch.cat((base,commands*2),-1)
        if self.cfg.get('finger_coverage_weight',0)>0:
            obs=torch.cat((obs,self.finger_coverage),-1)
        if self.cfg.get('filtered_velocity_observations',False):
            velocity=self.reward_ball_velocity
            if self.cfg.get('heading_observations',False):
                velocity=torch.einsum('nij,nj->ni',inverse[:,:2,:2],velocity)
            obs=torch.cat((obs,velocity*10),-1)
        if self.cfg.get('physics_timestep_observation',False):
            timestep_ms=1000.*self.active_physics_timestep
            obs=torch.cat((obs,torch.full((self.num_envs,1),timestep_ms,
                                         dtype=obs.dtype,device=obs.device)),-1)
        result={'actor':obs,'critic':obs}
        if self.cfg.get('command_target_observation',False):
            result['command_delta']=2*(self.command_targets-self.commands)
        if self.cfg.get('contact_load_observation',False):
            result['contact_load']=self.sim.contact_load
        return TensorDict(result,batch_size=[self.num_envs],device=self.device)

    def prepare_observed_commands(self):
        """Publish this interval's target once; reading observations is idempotent."""
        ids=(~self.command_prepared).nonzero().flatten()
        if not len(ids):return
        if not self.external_command:
            if self.recovery is not None:
                self.recovery.prepare_commands(ids)
            else:
                self.command_countdown[ids]-=1
                self.sample_commands(ids[self.command_countdown[ids]<=0])
        self.command_prepared[ids]=True

    def command_reward(self,q,v,found,previous_ball_xy):
        velocity=v[:,self.ballva:self.ballva+2]
        if previous_ball_xy is not None:
            measured=(q[:,self.ballqa:self.ballqa+2]-previous_ball_xy)/self.dt
            self.reward_ball_velocity.lerp_(measured,1.-np.exp(-self.dt/self.cfg['velocity_reward_tau']))
            velocity=self.reward_ball_velocity
            limit=self.cfg.get('command_path_limit',0)
            if limit>0:
                # Bounded accumulated displacement error, not a force or root target.
                self.command_path_error+=(measured-self.commands)*self.dt
                self.command_path_error*=torch.clamp(limit/self.command_path_error.norm(dim=-1,keepdim=True).clamp(min=1e-9),max=1.)
                self.command_path_error[self.command_targets.norm(dim=-1)<1e-6]=0
        error=(velocity-self.commands).norm(dim=-1)
        # Absolute low-speed accuracy, then relative accuracy as speed increases.
        magnitude=self.commands.norm(dim=-1)
        sigma=.04+.3*magnitude
        if self.cfg.get('dense_command_tracking',False):
            sigma=torch.where(magnitude<1e-4,torch.full_like(magnitude,.025),.003+.35*magnitude)
        tracking=torch.exp(-(error/sigma).square())
        angular=self.sim.data.xmat[:,self.ballbody].reshape(-1,3,3)@v[:,self.ballva+3:self.ballva+6,None]
        rolling_velocity=torch.stack((angular[:,1,0],-angular[:,0,0]),-1)*.12
        slip=(velocity-rolling_velocity).square().sum(-1)
        self.tracking_sum+=torch.nan_to_num(tracking,nan=0.,posinf=0.,neginf=0.)
        self.command_error_sum+=torch.nan_to_num(error,nan=10.,posinf=10.,neginf=10.).clamp(max=10)
        # A terminated solver failure must remain visible without making every
        # world in the logged batch appear NaN. These are telemetry only; the
        # reward below and the failure/reset path retain the original tensors.
        self.last_invalid_velocity_fraction=float((~torch.isfinite(velocity).all(-1)).float().mean())
        self.last_tracking=float(torch.nan_to_num(tracking,nan=0.,posinf=0.,neginf=0.).mean())
        self.last_velocity=float(torch.nan_to_num(velocity.norm(dim=-1),nan=0.,posinf=0.,neginf=0.).mean())
        self.last_tracking_error=float(torch.nan_to_num(error,nan=10.,posinf=10.,neginf=10.).mean())
        reward=6*tracking-2*(error/(.08+self.commands.norm(dim=-1))).square().clamp(max=4)
        if self.cfg.get('dense_command_tracking',False):
            progress=(velocity*self.commands).sum(-1)/magnitude.square().clamp(min=1e-6)
            reward+=2*progress.clamp(-1.,1.)*(magnitude>1e-4)
        reward-=.3*slip.clamp(max=4)
        if self.cfg.get('velocity_error_huber_weight',0)>0:
            cost=velocity_error_huber_cost(velocity,self.commands)
            reward-=self.cfg['velocity_error_huber_weight']*cost
            self.last_velocity_huber_cost=float(torch.nan_to_num(cost.detach(),nan=4.,posinf=4.).mean())
        # Permit stepping; maintaining three/four support contacts is preferable
        # to locking all five fingertips on the sphere during locomotion.
        reward-=.5*(3-found.sum(-1)).clamp(min=0)
        if self.cfg.get('command_path_limit',0)>0:
            reward+=self.cfg['path_reward_weight']*torch.exp(-(self.command_path_error/.04).square().sum(-1))*(self.command_targets.norm(dim=-1)>1e-6)
        if self.cfg.get('finger_coverage_weight',0)>0:
            from finger_coverage import update_coverage,coverage_deficit
            self.finger_coverage=update_coverage(self.finger_coverage,found,self.dt,self.cfg['finger_coverage_tau'])
            reward-=self.cfg['finger_coverage_weight']*coverage_deficit(self.finger_coverage,self.cfg['finger_coverage_target'])
        # Reward sees pre-clipped samples: different saturated outputs must not
        # remain indistinguishable when their physical joint action is identical.
        reward-=self.cfg.get('residual_bounds_weight',0)*self.residual_excess_cost
        if self.cfg.get('direct_action_bounds_weight',0)>0:
            reward-=self.cfg['direct_action_bounds_weight']*self.direct_action_excess_cost
        if self.cfg.get('hand_tilt_rate_weight',0)>0:
            cost=hand_tilt_rate_cost(self.sim.data.xmat[:,self.handbody].reshape(-1,3,3),v[:,3:6])
            reward-=self.cfg['hand_tilt_rate_weight']*cost
            self.last_hand_tilt_rate_cost=float(torch.nan_to_num(cost.detach(),nan=0.,posinf=0.,neginf=0.).mean())
        if self.cfg.get('body_motion_weight',0)>0:
            cost=(self.sim.body_motion_cost if self.cfg.get('body_motion_substeps',False)
                  else body_motion_cost(v[:,3:6],v[:,2]-v[:,self.ballva+2]))
            reward-=self.cfg['body_motion_weight']*cost
            self.last_body_motion_cost=float(torch.nan_to_num(cost.detach(),nan=0.,posinf=0.,neginf=0.).mean())
        if self.cfg.get('body_acceleration_weight',0)>0:
            cost=self.sim.body_acceleration_cost
            reward-=self.cfg['body_acceleration_weight']*cost
            self.last_body_acceleration_cost=float(torch.nan_to_num(cost.detach(),nan=0.,posinf=0.,neginf=0.).mean())
        return reward

    def maybe_promote(self):
        if not self.cfg['auto_curriculum'] or len(self.recent)<1024:return
        if self.global_steps-self.last_promotion_step<1500:return
        a=np.array(self.recent)
        completion=float(np.mean(a[:,0]>=self.cfg['episode_seconds']-.02))
        tracking=float(a[:,1].mean())
        if not np.isfinite(a).all() or not np.isfinite(tracking):return
        if completion<.8 or tracking<.7:return
        gate_file=self.cfg.get('curriculum_gate_file','')
        if gate_file:
            if self.global_steps%100:return
            try:gate=json.loads(Path(gate_file).read_text())
            except (OSError,ValueError):return
            if not gate.get('passed',False) or abs(gate.get('speed_mps',-1)-self.speed_limit)>1e-6:return
        levels=self.cfg.get('curriculum_speed_levels') or [0.,.025,.05,.1,.2,.35,.5,.75,self.cfg['target_speed']]
        higher=sorted(set(x for x in levels if self.speed_limit+1e-6<x<=self.cfg['target_speed']))
        if not higher:return
        old=self.speed_limit;self.speed_limit=higher[0]
        record={'global_step':self.global_steps,'from_mps':old,'to_mps':self.speed_limit,
                'completed_episode_fraction':completion,'mean_tracking_score':tracking}
        self.curriculum_records.append(record);print('CURRICULUM',record,flush=True)
        self.last_promotion_step=self.global_steps;self.recent.clear()

    def control_targets(self,previous_actions):
        if not self.cfg.get('residual_target_space',False):
            return super().control_targets(previous_actions)
        from residual_policy import residual_joint_target
        return residual_joint_target(self.joint0,previous_actions,self.actions,
            self.residual_teacher_action,self.low,self.high,self.cfg['action_scale'])

    def step(self,actions):
        if self.cfg.get('direct_action_bounds_weight',0)>0:
            assert self.teacher is None
            self.direct_action_excess_cost=(actions.abs()-1.).clamp(min=0).square().mean(-1).clamp(max=4.)
            self.last_direct_clipped_fraction=float((actions.abs()>=1).float().mean())
            self.last_direct_invalid_fraction=float((~torch.isfinite(actions)).float().mean())
            # Sanitize logging only. A NaN cost still reaches the original
            # invalid-reward termination path, never becomes a valid reward.
            self.last_direct_excess_cost=float(torch.nan_to_num(self.direct_action_excess_cost,
                nan=4.,posinf=4.,neginf=4.).mean())
        if self.teacher is not None:
            from residual_policy import teacher_action,combine_actions
            if self.cfg.get('residual_bounds_weight',0)>0:
                self.residual_excess_cost=(actions.abs()-1.).clamp(min=0).square().mean(-1).clamp(max=4.)
                self.last_residual_clipped_fraction=float((actions.abs()>=1).float().mean())
                self.last_residual_excess_cost=float(self.residual_excess_cost.mean())
            base=teacher_action(self.teacher,self.get_observations()["actor"])
            self.residual_teacher_action=base
            actions=combine_actions(base,actions,self.cfg["residual_scale"])
        if self.cfg.get('command_target_observation',False):
            self.prepare_observed_commands()
            # super().step returns the next observation after reward and reset.
            self.command_prepared[:]=False
        elif not self.external_command:
            if self.recovery is not None:
                self.recovery.prepare_commands()
            else:
                self.command_countdown-=1
                self.sample_commands((self.command_countdown<=0).nonzero().flatten())
        # The same slew rate is used in native evaluation and keyboard control.
        delta=self.command_targets-self.commands
        delta*=torch.clamp(self.cfg['command_acceleration']*self.dt/delta.norm(dim=-1,keepdim=True).clamp(min=1e-9),max=1.)
        self.commands+=delta
        if self.practice_env_count:
            for values,counts in ((self.command_targets,self.practice_requested_steps),
                                  (self.commands,self.practice_applied_steps)):
                bins=torch.bucketize(values[self.practice_env_mask].norm(dim=-1),self.practice_bins)
                counts+=torch.bincount(bins,minlength=4)
        result=super().step(actions)
        if self.practice_env_count:
            failures=result[3]['failures']
            self.practice_terminations[0]+=failures[self.practice_env_mask].sum()
            self.practice_terminations[1]+=(result[2]&~failures)[self.practice_env_mask].sum()
        self.maybe_promote()
        result[3]['log'].update({'Teleop/speed_limit_mps':self.speed_limit,
            'Teleop/tracking_score':self.last_tracking,'Teleop/actual_speed_mps':self.last_velocity,
            'Teleop/velocity_error_mps':self.last_tracking_error,
            'Teleop/invalid_velocity_fraction':self.last_invalid_velocity_fraction})
        if self.cfg.get('direct_action_bounds_weight',0)>0:
            result[3]['log'].update({'DirectAction/clipped_fraction':self.last_direct_clipped_fraction,
                'DirectAction/excess_cost':self.last_direct_excess_cost,
                'DirectAction/invalid_fraction':self.last_direct_invalid_fraction})
        if self.cfg.get('hand_tilt_rate_weight',0)>0:
            result[3]['log']['Hand/tilt_rate_cost']=self.last_hand_tilt_rate_cost
        if self.cfg.get('body_motion_weight',0)>0:
            result[3]['log']['Smoothness/body_motion_cost']=self.last_body_motion_cost
        if self.cfg.get('body_acceleration_weight',0)>0:
            result[3]['log']['Smoothness/body_acceleration_cost']=self.last_body_acceleration_cost
        if self.cfg.get('velocity_error_huber_weight',0)>0:
            result[3]['log']['Teleop/velocity_huber_cost']=self.last_velocity_huber_cost
        if self.cfg.get('joint_target_rate_weight',0)>0:
            result[3]['log']['Smoothness/joint_target_rate_squared_radps']=self.last_joint_target_rate_cost
        if self.recovery is not None:
            result[3]['log'].update({'Recovery/'+k:float(v) for k,v in self.recovery.metrics().items()})
        return result
