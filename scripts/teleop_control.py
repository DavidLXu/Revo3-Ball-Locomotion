"""Native MuJoCo command-conditioned inference shared by keyboard and evaluation."""
from pathlib import Path
import json
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from evaluate_balance import NativeBalance

class NativeTeleop(NativeBalance):
    def __init__(self,envcfg):
        assert envcfg['task']=='teleop' and envcfg['initial_pose']=='opposed_thumb'
        if envcfg.get('free_heading',False):
            assert envcfg.get('ball_angular_world') and not envcfg.get('residual_teacher') and not envcfg.get('command_path_limit',0)
        assert not envcfg.get('heading_observations',False) or envcfg.get('free_heading',False)
        assert not envcfg.get('filtered_velocity_observations',False) or envcfg['velocity_reward_tau']>0
        super().__init__(envcfg)
        assert 0 < envcfg.get('action_filter_alpha',.3) <= 1
        self.commands=np.zeros(2);self.command_targets=np.zeros(2);self.measured_velocity=np.zeros(2)
        self.command_path_error=np.zeros(2)
        self.teacher_base_action=None
        self.finger_coverage=np.full(5,envcfg.get('finger_coverage_target',.25))
        self.dt=envcfg.get('physics_timestep',.002)*envcfg.get('decimation',5)
        if envcfg.get('contact_load_observation',False):
            from contact_load import ContactLoadReader
            self.contact_reader=ContactLoadReader(self.m)

    def reset(self,seed=10000,noise=1.):
        super().reset(seed,noise);self.commands[:]=0;self.command_targets[:]=0;self.measured_velocity[:]=0
        self.command_path_error[:]=0
        self.teacher_base_action=None
        self.finger_coverage[:]=self.envcfg.get('finger_coverage_target',.25)

    def set_command(self,vx,vy):
        target=np.array([vx,vy],dtype=float)
        target*=min(1.,self.envcfg['target_speed']/max(np.linalg.norm(target),1e-9))
        if np.linalg.norm(target-self.command_targets)>1e-6:self.command_path_error[:]=0
        self.command_targets[:]=target

    def observation(self):
        base=super().observation();base[60:62]=0
        if self.envcfg.get('command_path_limit',0)>0:base[60:62]=self.command_path_error*5
        commands=self.commands
        if self.envcfg.get('heading_observations',False):
            from heading_frame import heading_inverse_numpy
            inverse=heading_inverse_numpy(self.d.xmat[self.handbody].reshape(3,3)@self.rot0.T)
            commands=inverse[:2,:2]@commands
        obs=np.r_[base,commands*2]
        if self.envcfg.get('finger_coverage_weight',0)>0:
            obs=np.r_[obs,self.finger_coverage]
        if self.envcfg.get('filtered_velocity_observations',False):
            velocity=self.measured_velocity
            if self.envcfg.get('heading_observations',False):
                velocity=inverse[:2,:2]@velocity
            obs=np.r_[obs,velocity*10]
        if self.envcfg.get('physics_timestep_observation',False):
            # Read the actual model setting, including evaluation overrides.
            obs=np.r_[obs,1000.*self.m.opt.timestep]
        return obs.astype(np.float32)

    def step(self,action,push=None):
        assert push is None,'Teleoperation cannot use external propulsion.'
        delta=self.command_targets-self.commands
        delta*=min(1.,self.envcfg['command_acceleration']*self.dt/max(np.linalg.norm(delta),1e-9))
        self.commands+=delta
        before=self.d.qpos[self.ballqa:self.ballqa+2].copy()
        previous_actions=self.actions.copy()
        alpha=self.envcfg.get('action_filter_alpha',.3)
        self.actions=(1.-alpha)*self.actions+alpha*np.clip(action,-1,1)
        if self.envcfg.get('residual_target_space',False):
            assert self.teacher_base_action is not None,'Compute the teacher action before every residual control step'
            from residual_policy import residual_joint_target
            self.d.ctrl[:]=residual_joint_target(self.joint0,previous_actions,self.actions,
                self.teacher_base_action,self.low,self.high,self.envcfg['action_scale'])
            self.teacher_base_action=None
        else:
            self.d.ctrl[:]=np.clip(self.joint0+self.envcfg['action_scale']*self.actions,self.low,self.high)
        self.d.xfrc_applied[:]=0
        import mujoco
        for _ in range(self.envcfg.get('decimation',5)):mujoco.mj_step(self.m,self.d)
        mujoco.mj_forward(self.m,self.d)
        if self.envcfg.get('finger_coverage_weight',0)>0:
            from finger_coverage import update_coverage
            self.finger_coverage=update_coverage(self.finger_coverage,(self.d.sensordata[:5]>0).astype(float),self.dt,self.envcfg['finger_coverage_tau'])
        measured=(self.d.qpos[self.ballqa:self.ballqa+2]-before)/self.dt
        limit=self.envcfg.get('command_path_limit',0)
        if limit>0:
            self.command_path_error+=(measured-self.commands)*self.dt
            self.command_path_error*=min(1.,limit/max(np.linalg.norm(self.command_path_error),1e-9))
            if np.linalg.norm(self.command_targets)<1e-6:self.command_path_error[:]=0
        tau=self.envcfg['velocity_reward_tau']
        self.measured_velocity+=(1.-np.exp(-self.dt/tau))*(measured-self.measured_velocity) if tau>0 else measured-self.measured_velocity

    def metrics(self):
        r=super().metrics()
        r['failed']=r['physical_fall'] or (self.envcfg['reject_palm_support'] and r['palm_contact'])
        r['failure_reason']='fall' if r['physical_fall'] else ('palm_support' if r['failed'] else 'none')
        r.update(command_mps=self.commands.tolist(),target_mps=self.command_targets.tolist(),
                 measured_velocity_mps=self.measured_velocity.tolist(),
                 velocity_error_mps=float(np.linalg.norm(self.measured_velocity-self.commands)))
        return r

class PolicyController:
    def __init__(self,checkpoint):
        self.checkpoint=Path(checkpoint)
        cfg=json.loads((self.checkpoint.parent/'runner_config.json').read_text())
        self.envcfg=json.loads((self.checkpoint.parent/'env_config.json').read_text())
        self.env=NativeTeleop(self.envcfg)
        obs=self.observations()
        acfg=dict(cfg['actor']);acfg.pop('class_name',None)
        model_type=MLPModel
        if self.envcfg.get('command_target_observation',False):
            from command_context import CommandContextModel
            if self.envcfg.get('contact_load_observation',False):
                from contact_context import ContactContextModel
                assert cfg['actor']['class_name']=='contact_context:ContactContextModel'
                model_type=ContactContextModel
            else:
                assert cfg['actor']['class_name']=='command_context:CommandContextModel'
                model_type=CommandContextModel
        self.policy=model_type(obs,cfg['obs_groups'],'actor',21,**acfg)
        if self.envcfg.get('split_timestep_projection',False):
            assert self.envcfg.get('physics_timestep_observation',False)
            from timestep_input import install_timestep_projection
            install_timestep_projection(self.policy)
        self.saved=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        saved_inertia=(self.saved.get('infos') or {}).get('hand_inertia_mode')
        if saved_inertia is not None:
            assert saved_inertia==self.envcfg.get('hand_inertia_mode','legacy'), 'Checkpoint and evaluation hand inertia modes disagree'
        assert bool((self.saved.get('infos') or {}).get('contact_load_observation',False))==bool(self.envcfg.get('contact_load_observation',False)), 'Checkpoint and evaluation contact-load inputs disagree'
        self.policy.load_state_dict(self.saved['actor_state_dict']);self.policy.eval()
        self.teacher=None
        if self.envcfg.get('residual_teacher'):
            from residual_policy import load_balance_teacher
            self.teacher=load_balance_teacher(self.envcfg['residual_teacher'],'cpu')
        self.trained_speed_limit=(self.saved.get('infos') or {}).get('teleop_curriculum',{}).get('speed_limit',0.)

    def observations(self):
        o=torch.from_numpy(self.env.observation()[None])
        data={'actor':o,'critic':o}
        if self.envcfg.get('command_target_observation',False):
            data['command_delta']=torch.from_numpy((2*(self.env.command_targets-self.env.commands)).astype(np.float32)[None])
        if self.envcfg.get('contact_load_observation',False):
            data['contact_load']=torch.from_numpy(self.env.contact_reader.features(self.env.d)[None])
        return TensorDict(data,batch_size=[1])

    def action(self,stochastic=False):
        obs=self.observations();o=obs['actor']
        with torch.inference_mode():
            action=self.policy(obs,stochastic_output=stochastic)
            if self.teacher is not None:
                from residual_policy import teacher_action,combine_actions
                base=teacher_action(self.teacher,o)
                if self.envcfg.get('residual_target_space',False):
                    self.env.teacher_base_action=base.cpu().numpy()[0].copy()
                action=combine_actions(base,action,self.envcfg['residual_scale'])
            return action.cpu().numpy()[0]
