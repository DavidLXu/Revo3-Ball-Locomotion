"""Vectorized mjlab environment: only the 21 hand joints are actuated."""
from dataclasses import dataclass,asdict
import time
import numpy as np
import mujoco
import torch
import warp as wp
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from mjlab.sim import Simulation,SimulationCfg,MujocoCfg
from balance_model import make_model,ACTION_SCALE,KV,configure_pd_derivative_gain

@dataclass
class BalanceCfg:
    initial_pose:str="original"
    hand_inertia_mode:str="legacy"
    num_envs:int=256
    episode_seconds:float=10.
    decimation:int=5
    physics_timestep:float=.002
    alternate_physics_timestep:float=0.
    physics_timestep_switch_steps:int=1000
    reset_noise:float=1.
    action_scale:float=ACTION_SCALE
    action_filter_alpha:float=.3
    joint_target_rate_weight:float=0.
    joint_target_acceleration_weight:float=0.
    body_acceleration_weight:float=0.
    pd_derivative_gain:float=KV
    seed:int=42
    contact_weight:float=1.
    five_weight:float=.5
    push_force:float=0.
    reset_velocity:float=0.
    ball_center_weight:float=0.
    stationary_weight:float=0.
    physics_backend:str="warp"
    task:str="balance"
    forward_speed:float=.04
    roll_velocity_cost:float=0.
    support_deficit_weight:float=0.
    failure_penalty:float=1.
    survival_reward_rate:float=0.
    reject_palm_support:bool=False
    velocity_command_frame:bool=False
    ball_angular_world:bool=False
    path_observation_limit:float=0.
    startup_env_fraction:float=0.
    startup_episode_seconds:float=3.
    velocity_reward_tau:float=0.

def quatmul(a,b):
    aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)

class BalanceEnv(VecEnv):
    def __init__(self,cfg=BalanceCfg(),device='cuda:0'):
        self.cfg=asdict(cfg);self.device=device;self.num_envs=cfg.num_envs;self.num_actions=21
        assert cfg.physics_timestep>0 and cfg.decimation>=1
        assert 0 < cfg.action_filter_alpha <= 1
        assert np.isfinite(cfg.joint_target_rate_weight) and cfg.joint_target_rate_weight >= 0
        assert np.isfinite(cfg.joint_target_acceleration_weight) and cfg.joint_target_acceleration_weight >= 0
        assert np.isfinite(cfg.body_acceleration_weight) and cfg.body_acceleration_weight >= 0
        assert not cfg.body_acceleration_weight or (cfg.task == 'teleop' and self.cfg.get('body_motion_substeps',False))
        assert np.isfinite(cfg.survival_reward_rate) and cfg.survival_reward_rate >= 0
        assert not cfg.survival_reward_rate or cfg.task == 'teleop'
        self.dt=cfg.physics_timestep*cfg.decimation;self.max_episode_length=round(cfg.episode_seconds/self.dt)
        assert cfg.alternate_physics_timestep>=0 and cfg.physics_timestep_switch_steps>0
        if cfg.alternate_physics_timestep:
            assert abs(round(self.dt/cfg.alternate_physics_timestep)*cfg.alternate_physics_timestep-self.dt)<1e-10
        self.active_physics_timestep=cfg.physics_timestep
        self.physics_decimation=cfg.decimation
        assert 0<=cfg.startup_env_fraction<1
        assert self.dt<=cfg.startup_episode_seconds<=cfg.episode_seconds or cfg.startup_env_fraction==0
        self.episode_horizons=torch.full((cfg.num_envs,),self.max_episode_length,device=device,dtype=torch.long)
        self.startup_env_count=int(cfg.num_envs*cfg.startup_env_fraction)
        self.episode_horizons[:self.startup_env_count]=round(cfg.startup_episode_seconds/self.dt)
        self.m,q,qa,va=make_model(cfg.initial_pose,cfg.hand_inertia_mode)
        configure_pd_derivative_gain(self.m,cfg.pd_derivative_gain)
        self.m.opt.timestep=cfg.physics_timestep
        self.q0=torch.tensor(q,device=device,dtype=torch.float32)
        self.qa=torch.tensor(qa,device=device);self.va=torch.tensor(va,device=device)
        self.joint0=self.q0[self.qa]
        js=np.flatnonzero(self.m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        self.low=torch.tensor(self.m.jnt_range[js,0],device=device,dtype=torch.float32)
        self.high=torch.tensor(self.m.jnt_range[js,1],device=device,dtype=torch.float32)
        self.handbody=self.m.body('hand/right_hand_base_link').id
        self.ballbody=self.m.body('ball/ball').id
        self.ballqa=int(self.m.jnt_qposadr[self.m.joint('ball/free_ball').id])
        self.ballva=int(self.m.jnt_dofadr[self.m.joint('ball/free_ball').id])
        # Contact anchors taken from the actual five mesh/sphere contacts at reset.
        d=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,d,0);mujoco.mj_forward(self.m,d)
        bg=self.m.geom('ball/surface').id
        self.fingers=['thumb','index','middle','ring','little']
        bids=[];anchors=[]
        for f in self.fingers:
            found=[]
            for c in d.contact:
                if bg not in c.geom:continue
                g=c.geom2 if c.geom1==bg else c.geom1
                if 'right_'+f+'_DIP' in self.m.geom(g).name:
                    b=self.m.geom_bodyid[g];found.append((b,d.xmat[b].reshape(3,3).T@(c.pos-d.xpos[b])))
            assert found,f
            bids.append(found[0][0]);anchors.append(found[0][1])
        self.tipb=torch.tensor(bids,device=device)
        self.anchors=torch.tensor(np.array(anchors),device=device,dtype=torch.float32)
        self.rot0=torch.tensor(d.xmat[self.handbody].reshape(3,3),device=device,dtype=torch.float32)
        self.rel0=self.q0[:3]-self.q0[self.ballqa:self.ballqa+3]
        if cfg.physics_backend=='native':
            from native_sim import NativeSimulation
            self.sim=NativeSimulation(self.m,cfg.num_envs,device,cfg.decimation,
                body_acceleration_enabled=cfg.body_acceleration_weight>0,
                body_motion_ball_dof=self.ballva if self.cfg.get('body_motion_substeps',False) else None,
                contact_load_observation=self.cfg.get('contact_load_observation',False))
        else:
            assert cfg.physics_backend=='warp'
            from mjlab_balance_sim import MjlabBalanceSimulation
            self.sim=MjlabBalanceSimulation(num_envs=cfg.num_envs,model=self.m,device=device,
                body_acceleration_enabled=cfg.body_acceleration_weight>0,
                decimation=cfg.decimation,
                body_motion_ball_dof=self.ballva if self.cfg.get('body_motion_substeps',False) else None,
                contact_load_observation=self.cfg.get('contact_load_observation',False),
                cfg=SimulationCfg(nconmax=256,njmax=2048,mujoco=MujocoCfg(timestep=cfg.physics_timestep,cone='elliptic',iterations=30,ls_iterations=20)))
        self.episode_length_buf=torch.zeros(cfg.num_envs,device=device,dtype=torch.long)
        self.actions=torch.zeros((cfg.num_envs,21),device=device)
        self.previous_target_rate=torch.zeros_like(self.actions)
        self.target_rate_valid=torch.zeros(cfg.num_envs,device=device,dtype=torch.bool)
        assert cfg.velocity_reward_tau>=0
        self.reward_ball_velocity=torch.zeros((cfg.num_envs,2),device=device)
        self.returns=torch.zeros(cfg.num_envs,device=device)
        self.push_remaining=torch.zeros(cfg.num_envs,device=device,dtype=torch.long)
        self.next_push=torch.zeros(cfg.num_envs,device=device,dtype=torch.long)
        self.push_vectors=torch.zeros((cfg.num_envs,3),device=device)
        self.last_episode_s=0.;self.last_return=0.;self.completed=0;self.global_steps=0
        self.last_full_episode_s=0.;self.last_startup_failure_rate=0.
        self.reset(torch.arange(cfg.num_envs,device=device));self.sim.forward()

    def reset(self,ids):
        if len(ids)==0:return
        self.sim.reset(ids)
        n=len(ids);q=self.q0.repeat(n,1)
        noise=self.cfg['reset_noise']
        q[:,self.qa]+=torch.randn((n,21),device=self.device)*.008*noise
        q[:,:2]+=torch.randn((n,2),device=self.device)*.0008*noise
        rv=torch.randn((n,3),device=self.device)*.006*noise
        angle=rv.norm(dim=-1,keepdim=True)
        dq=torch.cat((torch.cos(angle/2),rv*torch.sinc(angle/(2*torch.pi))/2),-1)
        q[:,3:7]=quatmul(dq,q[:,3:7])
        self.sim.data.qpos[ids]=q
        self.sim.data.qvel[ids]=0
        self.sim.data.qvel[ids,:2]=torch.randn((n,2),device=self.device)*self.cfg['reset_velocity']
        self.push_remaining[ids]=0;self.push_vectors[ids]=0
        self.next_push[ids]=torch.randint(round(4/self.dt),round(8/self.dt),(n,),device=self.device)
        self.sim.data.ctrl[ids]=self.joint0
        self.actions[ids]=0;self.episode_length_buf[ids]=0;self.returns[ids]=0
        self.previous_target_rate[ids]=0;self.target_rate_valid[ids]=False
        self.reward_ball_velocity[ids]=0

    def features(self):
        q=self.sim.data.qpos;v=self.sim.data.qvel
        rot=self.sim.data.xmat[:,self.handbody].reshape(-1,3,3)
        rel=q[:,:3]-q[:,self.ballqa:self.ballqa+3]
        rot_delta=rot@self.rot0.T
        tilt=(3-rot_delta.diagonal(dim1=-2,dim2=-1).sum(-1)).clamp(min=0)
        tiprot=self.sim.data.xmat[:,self.tipb].reshape(self.num_envs,5,3,3)
        tips=self.sim.data.xpos[:,self.tipb]+torch.einsum('nfij,fj->nfi',tiprot,self.anchors)
        tiprel=tips-q[:,None,self.ballqa:self.ballqa+3]
        gaps=tiprel.norm(dim=-1)-.12
        if self.cfg.get('free_heading',False):
            tilt=(2*(1-rot_delta[:,2,2])).clamp(min=0)
        if self.cfg.get('heading_observations',False):
            from heading_frame import heading_inverse_torch
            inverse=heading_inverse_torch(rot_delta)
            rel=torch.einsum('nij,nj->ni',inverse,rel)
            tiprel=torch.einsum('nij,nfj->nfi',inverse,tiprel)
            rot_delta=inverse@rot_delta
        return q,v,rel,rot_delta,tilt,gaps,tiprel

    def get_observations(self):
        q,v,rel,rot,tilt,gaps,tiprel=self.features()
        ball_xy=q[:,self.ballqa:self.ballqa+2].clone()
        hand_velocity=v[:,:3].clone()
        ball_velocity=v[:,self.ballva:self.ballva+3].clone()
        ball_angular_velocity=v[:,self.ballva+3:self.ballva+6].clone()
        ball_rotation=self.sim.data.xmat[:,self.ballbody].reshape(-1,3,3)
        if self.cfg['ball_angular_world']:
            ball_angular_velocity=torch.einsum('nij,nj->ni',ball_rotation,ball_angular_velocity)
        if self.cfg['task']=='roll':
            ball_xy[:,0]-=self.episode_length_buf*self.dt*self.cfg['forward_speed']
            limit=self.cfg['path_observation_limit']
            if limit>0:ball_xy[:,0].clamp_(-limit,limit)
            if self.cfg['velocity_command_frame']:
                speed=self.cfg['forward_speed']
                hand_velocity[:,0]-=speed;ball_velocity[:,0]-=speed
                # Free-joint angular qvel is in the ball's local frame.
                if self.cfg['ball_angular_world']:ball_angular_velocity[:,1]-=speed/.12
                else:ball_angular_velocity-=ball_rotation[:,1,:]*(speed/.12)
        if self.cfg['task']=='teleop':ball_xy.zero_()
        if self.cfg.get('heading_observations',False):
            from heading_frame import heading_inverse_torch
            inverse=heading_inverse_torch(self.sim.data.xmat[:,self.handbody].reshape(-1,3,3)@self.rot0.T)
            hand_velocity=torch.einsum('nij,nj->ni',inverse,hand_velocity)
            ball_velocity=torch.einsum('nij,nj->ni',inverse,ball_velocity)
            ball_angular_velocity=torch.einsum('nij,nj->ni',inverse,ball_angular_velocity)
            # Free-joint angular qvel remains in the hand's local frame.
        obs=torch.cat(((q[:,self.qa]-self.joint0)*2,v[:,self.va]*.08,
                       (rel-self.rel0)*10,rot.reshape(-1,9),hand_velocity*2,v[:,3:6]*.2,
                       ball_xy*5,ball_velocity*2,
                       ball_angular_velocity*.1,gaps*30,tiprel.reshape(-1,15)*5,self.actions,(self.sim.data.sensordata[:,:5]>0).float(),(self.sim.data.sensordata[:,5:]>0).any(-1,keepdim=True).float()),-1)
        obs=torch.nan_to_num(obs,nan=0.,posinf=20.,neginf=-20.).clamp(-20,20)
        return TensorDict({'actor':obs,'critic':obs},batch_size=[self.num_envs],device=self.device)

    def control_targets(self,previous_actions):
        return (self.joint0+self.cfg['action_scale']*self.actions).clamp(self.low,self.high)

    def select_physics_timestep(self):
        """Vary integration resolution, preserving the physical control interval."""
        alternate=self.cfg.get('alternate_physics_timestep',0.)
        if not alternate:return
        phase=(self.global_steps//self.cfg['physics_timestep_switch_steps'])%2
        timestep=alternate if phase else self.cfg['physics_timestep']
        if timestep==self.active_physics_timestep:return
        decimation=round(self.dt/timestep)
        assert decimation>=1 and abs(decimation*timestep-self.dt)<1e-10
        self.m.opt.timestep=timestep
        if self.cfg['physics_backend']=='native':
            assert self.sim.pending==0
            self.sim.decimation=decimation
        else:
            # Update existing device memory read by the captured CUDA graph.
            # Replacing the array would leave the graph reading the old value.
            wp.to_torch(self.sim.wp_model.opt.timestep).fill_(timestep)
            self.sim.decimation=decimation
        self.active_physics_timestep=timestep
        self.physics_decimation=decimation

    def step(self,actions):
        self.select_physics_timestep()
        prev=self.actions.clone()
        previous_ball_xy=self.sim.data.qpos[:,self.ballqa:self.ballqa+2].clone() if self.cfg['velocity_reward_tau']>0 else None
        alpha=self.cfg.get('action_filter_alpha',.3)
        self.actions[:]=(1.-alpha)*prev+alpha*actions.clamp(-1,1)
        target=self.control_targets(prev)
        target_rate_cost=None
        target_acceleration_cost=None
        if self.cfg.get('joint_target_rate_weight',0)>0 or self.cfg.get('joint_target_acceleration_weight',0)>0:
            # Penalize the actual bounded actuator target, in rad/s. Existing
            # ctrl is the previous target, including immediately after reset.
            target_rate=(target-self.sim.data.ctrl)/self.dt
            target_rate_cost=target_rate.square().mean(-1)
            if self.cfg.get('joint_target_acceleration_weight',0)>0:
                # Do not join a new episode to the previous episode's targets.
                target_acceleration_cost=((target_rate-self.previous_target_rate)/self.dt).square().mean(-1)
                target_acceleration_cost=torch.where(self.target_rate_valid,target_acceleration_cost,0.)
            self.previous_target_rate.copy_(target_rate)
            self.target_rate_valid.fill_(True)
        self.sim.data.ctrl[:]=target
        if self.cfg['push_force']>0:
            trigger=(self.episode_length_buf>=self.next_push)
            idx=trigger.nonzero().flatten()
            if len(idx):
                angle=torch.rand(len(idx),device=self.device)*2*torch.pi
                magnitude=self.cfg['push_force']*(.5+.5*torch.rand(len(idx),device=self.device))
                self.push_vectors[idx,0]=magnitude*torch.cos(angle)
                self.push_vectors[idx,1]=magnitude*torch.sin(angle)
                self.push_remaining[idx]=round(.1/self.dt)
                self.next_push[idx]+=torch.randint(round(4/self.dt),round(8/self.dt),(len(idx),),device=self.device)
            self.sim.data.xfrc_applied.zero_()
            self.sim.data.xfrc_applied[:,self.handbody,:3]=self.push_vectors*(self.push_remaining>0)[:,None]
            self.push_remaining=(self.push_remaining-1).clamp(min=0)
        for _ in range(self.physics_decimation):self.sim.step()
        self.sim.forward()
        self.episode_length_buf+=1;self.global_steps+=1
        q,v,rel,rot,tilt,gaps,tiprel=self.features()
        contact=torch.exp(-(gaps/.009).square()).mean(-1)
        if self.cfg.get('free_heading',False) and not self.cfg.get('heading_observations',False):
            from heading_frame import heading_inverse_torch
            rel=torch.einsum('nij,nj->ni',heading_inverse_torch(rot),rel)
        centered=torch.exp(-((rel[:,:2]-self.rel0[:2])/.045).square().sum(-1))
        height=torch.exp(-((rel[:,2]-self.rel0[2])/.035).square())
        upright=torch.exp(-tilt/.18)
        hand_speed=v[:,:3].clone()
        if self.cfg['task']=='teleop':hand_speed[:,:2]-=self.commands
        speed=hand_speed.square().sum(-1)+.03*v[:,3:6].square().sum(-1)
        found=(self.sim.data.sensordata[:,:5]>0).float()
        palm=(self.sim.data.sensordata[:,5:]>0).any(-1)
        reward=(1.+self.cfg['contact_weight']*found.mean(-1)+self.cfg['five_weight']*found.prod(-1)-2*palm.float()+2*upright+1.5*height+1.5*centered+2*contact
                +self.cfg['ball_center_weight']*torch.exp(-(q[:,self.ballqa:self.ballqa+2]/.06).square().sum(-1))
                +self.cfg['stationary_weight']*torch.exp(-(v[:,self.ballva:self.ballva+2]/.07).square().sum(-1))
                -.3*speed-.02*(self.actions-prev).square().sum(-1)
                -.003*self.actions.square().sum(-1)-.0003*v[:,self.va].square().sum(-1))
        if target_rate_cost is not None:
            reward-=self.cfg['joint_target_rate_weight']*target_rate_cost
            self.last_joint_target_rate_cost=float(torch.nan_to_num(target_rate_cost.detach(),nan=0.,posinf=0.,neginf=0.).mean())
        if target_acceleration_cost is not None:
            reward-=self.cfg['joint_target_acceleration_weight']*target_acceleration_cost
        if self.cfg['task']=='roll':
            # Only joint torques propel the ball. The command is a moving
            # reference in the observation/reward, never a physical force.
            ball_velocity=v[:,self.ballva:self.ballva+2]
            if previous_ball_xy is not None:
                # Reward net translation over control intervals rather than a
                # single phase of high-frequency contact-induced velocity.
                measured=(q[:,self.ballqa:self.ballqa+2]-previous_ball_xy)/self.dt
                alpha=1.-np.exp(-self.dt/self.cfg['velocity_reward_tau'])
                self.reward_ball_velocity.lerp_(measured,alpha)
                ball_velocity=self.reward_ball_velocity
            target=torch.zeros_like(ball_velocity);target[:,0]=self.cfg['forward_speed']
            velocity_tracking=torch.exp(-((ball_velocity-target)/.035).square().sum(-1))
            path_error=q[:,self.ballqa:self.ballqa+2].clone()
            path_error[:,0]-=self.episode_length_buf*self.dt*self.cfg['forward_speed']
            path_tracking=torch.exp(-(path_error/.12).square().sum(-1))
            progress=(ball_velocity[:,0]/self.cfg['forward_speed']).clamp(-1,1)
            # Penalize lateral slip and excessive speed, rather than anchoring
            # the free sphere to its starting point.
            reward+=6*velocity_tracking+3*path_tracking+3*progress
            velocity_error=((ball_velocity-target)/self.cfg['forward_speed']).square().sum(-1).clamp(max=4)
            reward-=self.cfg['roll_velocity_cost']*velocity_error
            reward-=self.cfg['support_deficit_weight']*(4-found.sum(-1)).clamp(min=0)
            escaped=(path_error.norm(dim=-1)>.75)|(q[:,self.ballqa+1].abs()>.5)
        elif self.cfg['task']=='teleop':
            reward+=self.command_reward(q,v,found,previous_ball_xy)
            escaped=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        else:
            escaped=q[:,self.ballqa:self.ballqa+2].norm(dim=-1)>.35
        failure=((rel[:,2]<.115)|(rel[:,2]>.28)|(tilt>1.2)|
                 ((rel[:,:2]-self.rel0[:2]).norm(dim=-1)>.14)|
                 escaped|
                 (~torch.isfinite(q).all(-1))|(~torch.isfinite(v).all(-1)))
        if self.cfg['reject_palm_support']:failure|=palm
        timeout=self.episode_length_buf>=self.episode_horizons
        done=failure|timeout
        invalid=(~torch.isfinite(reward))
        failure|=invalid
        done=failure|timeout
        reward=torch.nan_to_num(reward,nan=-100.,posinf=-100.,neginf=-100.)
        if self.cfg['task']=='teleop':
            # Exploding but finite physics values must not produce unbounded
            # PPO targets. Normal task rewards are inside this interval.
            reward=reward.clamp(-25.,25.)
            # A constant per-second offset changes the incentive to terminate,
            # without changing fixed-duration moving-versus-standing margins.
            # Keep it after clipping, matching the frozen-trace reward screen.
            if self.cfg.get('survival_reward_rate',0)>0:
                reward=reward+self.cfg['survival_reward_rate']
        reward=reward*self.dt-failure.float()*self.cfg['failure_penalty']
        self.returns+=reward
        extras={'time_outs':timeout&~failure,'failures':failure,'log':{
            'Balance/invalid':invalid.float().mean(),'Balance/actual_fingers':found.sum(-1).mean(),'Balance/all_five':found.prod(-1).mean(),'Balance/palm_contact':palm.float().mean(),
            'Balance/nonfinite_world_count':((~torch.isfinite(q).all(-1))|(~torch.isfinite(v).all(-1))|invalid).sum(),
            'Balance/upright':torch.nan_to_num(upright).mean(),
            'Balance/finger_proximity':torch.nan_to_num(contact,nan=0.,posinf=0.,neginf=0.).mean(),
            'Balance/height':torch.nan_to_num(height,nan=0.,posinf=0.,neginf=0.).mean(),
            'Balance/centered':torch.nan_to_num(centered,nan=0.,posinf=0.,neginf=0.).mean()}}
        if self.cfg['task']=='roll':
            extras['log'].update({'Roll/forward_speed_mps':ball_velocity[:,0].mean(),
                'Roll/distance_m':q[:,self.ballqa].mean(),'Roll/velocity_tracking':velocity_tracking.mean(),
                'Roll/path_tracking':path_tracking.mean()})
        if target_rate_cost is not None:
            extras['log']['Smoothness/target_rate_squared']=target_rate_cost.mean()
        if target_acceleration_cost is not None:
            extras['log']['Smoothness/target_acceleration_squared']=target_acceleration_cost.mean()
        ids=done.nonzero().flatten()
        if len(ids):
            self.last_episode_s=float(self.episode_length_buf[ids].float().mean()*self.dt)
            self.last_return=float(self.returns[ids].mean());self.completed+=len(ids)
            extras['log']['Episode/duration_s']=self.last_episode_s
            extras['log']['Episode/return']=self.last_return
            full_ids=ids[ids>=self.startup_env_count]
            if len(full_ids):
                self.last_full_episode_s=float(self.episode_length_buf[full_ids].float().mean()*self.dt)
                extras['log']['Episode/full_duration_s']=self.last_full_episode_s
            startup_ids=ids[ids<self.startup_env_count]
            if len(startup_ids):
                self.last_startup_failure_rate=float(failure[startup_ids].float().mean())
                extras['log']['Episode/startup_failure_rate']=self.last_startup_failure_rate
            self.reset(ids);self.sim.forward()
        return self.get_observations(),reward,done,extras

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,default=64);p.add_argument('--steps',type=int,default=200);a=p.parse_args()
    env=BalanceEnv(BalanceCfg(num_envs=a.num_envs))
    t=time.time()
    for i in range(a.steps):
        obs,r,done,extra=env.step(torch.zeros((a.num_envs,21),device='cuda:0'))
        if i%50==0:print(i,'reward',r.mean().item(),'ep',env.last_episode_s,flush=True)
    print('FPS',a.steps*a.num_envs/(time.time()-t),'observations',obs['actor'].shape,flush=True)
