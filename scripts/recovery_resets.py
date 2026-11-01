"""Opt-in native pre-reversal resets; policies still provide every action."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from evaluate_teleop import profile


class RecoveryResets:
    def __init__(self, env):
        self.env=env
        cfg=env.cfg
        assert cfg['physics_backend']=='native' and not cfg['residual_teacher']
        assert cfg['free_heading'] and not cfg['heading_observations'] and not cfg['command_path_limit']
        assert not any(cfg.get(k,0) for k in ('reversal_sequence_probability','reverse_command_probability','rightward_command_probability'))
        self.probability=cfg['recovery_reset_probability']
        assert 0 < self.probability <= 1 and cfg['recovery_prefix_seconds'] in (2.,4.)
        self.prefix_steps=round(cfg['recovery_prefix_seconds']/env.dt)
        path=Path(cfg['recovery_state_bank'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['recovery_state_bank_sha256']
        self.metadata=json.loads(path.read_text())
        meta=self.metadata
        bank_path=path.with_suffix('.npz')
        assert hashlib.sha256(bank_path.read_bytes()).hexdigest()==meta['bank_sha256']
        assert meta['mujoco_version']==mujoco.__version__ and meta['control_timestep_s']==env.dt
        assert meta['state_spec']==int(mujoco.mjtState.mjSTATE_INTEGRATION)
        assert meta['state_size']==mujoco.mj_stateSize(env.m,mujoco.mjtState.mjSTATE_INTEGRATION)
        root=Path(__file__).resolve().parents[1]
        assert hashlib.sha256(Path(meta['source_checkpoint']).read_bytes()).hexdigest()==meta['source_sha256']
        for filename,digest in meta['pose_hashes'].items():
            assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==digest
        with np.load(bank_path,allow_pickle=False) as bank:
            self.arrays={k:bank[k].copy() for k in bank.files}
        assert all(np.isfinite(v).all() and len(v)==meta['states'] for v in self.arrays.values())
        assert not self.arrays['reference_origin_xy'].any() and not self.arrays['reference_time'].any()
        assert self.arrays['observation'].shape==(meta['states'],124)
        valid=[i for i,e in enumerate(meta['entries']) if e['physics_timestep_s']==env.active_physics_timestep]
        assert valid
        self.valid=torch.tensor(valid,device=env.device,dtype=torch.long)
        self.tensors={k:torch.tensor(v,device=env.device,dtype=torch.float32)
                      for k,v in self.arrays.items() if k!='integration_state'}
        targets=[];lengths=[]
        for i,entry in enumerate(meta['entries']):
            assert entry['index']==i and entry['integration_state_max_error']==0
            remaining=round(entry['profile_seconds']/env.dt)-entry['snapshot_step']
            length=min(self.prefix_steps,remaining)
            assert length>0
            lengths.append(length)
            targets.append([profile(entry['profile'],(entry['snapshot_step']+step)*env.dt,
                                    meta['speed_mps'],entry['profile_seconds'])
                            if step<length else (0.,0.) for step in range(self.prefix_steps)])
        self.targets=torch.tensor(targets,device=env.device,dtype=torch.float32)
        self.bank_prefix_steps=torch.tensor(lengths,device=env.device,dtype=torch.long)
        self.episode_mask=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.prefix_step=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
        self.prefix_limit=torch.zeros_like(self.prefix_step)
        self.bank_index=torch.full_like(self.prefix_step,-1)
        self.total_restores=0

    def on_reset(self, ids):
        self.episode_mask[ids]=False
        self.prefix_step[ids]=-1
        self.prefix_limit[ids]=0
        self.bank_index[ids]=-1
        if self.env.speed_limit+1e-9 < self.metadata['speed_mps']:
            return
        eligible=ids[ids>=self.env.startup_env_count]
        if not len(eligible):
            return
        selected=eligible[torch.rand(len(eligible),device=self.env.device)<self.probability]
        if len(selected):
            indices=self.valid[torch.randint(len(self.valid),(len(selected),),device=self.env.device)]
            self.apply(selected,indices)

    def apply(self, ids, indices):
        """Install selected bank states after ordinary per-episode bookkeeping."""
        env=self.env
        assert len(ids)==len(indices) and bool((ids>=env.startup_env_count).all())
        assert bool((env.episode_length_buf[ids]==0).all())
        assert bool(torch.isin(indices,self.valid).all())
        env.sim.forward()
        env.sim.restore_integration_states(ids,self.arrays['integration_state'][indices.cpu().numpy()])
        for destination,source in [('actions','actions'),('commands','commands'),
                ('command_targets','command_targets'),('command_path_error','command_path_error'),
                ('finger_coverage','finger_coverage'),('reward_ball_velocity','measured_velocity')]:
            getattr(env,destination)[ids]=self.tensors[source][indices]
        self.episode_mask[ids]=True
        self.prefix_step[ids]=0
        self.prefix_limit[ids]=self.bank_prefix_steps[indices]
        self.bank_index[ids]=indices
        env.command_countdown[ids]=self.prefix_limit[ids]+1
        env.command_prepared[ids]=False
        self.total_restores+=len(ids)

    def prepare_commands(self, selected=None):
        env=self.env
        selected_mask=torch.ones_like(self.episode_mask) if selected is None else torch.zeros_like(self.episode_mask)
        if selected is not None:selected_mask[selected]=True
        released=((self.prefix_step>=0)&(self.prefix_step==self.prefix_limit)&selected_mask).nonzero().flatten()
        self.prefix_step[released]=-1
        env.command_countdown[released]=0
        active=self.prefix_step>=0
        ordinary=(~active&selected_mask).nonzero().flatten()
        env.command_countdown[ordinary]-=1
        env.sample_commands(((env.command_countdown<=0)&~active&selected_mask).nonzero().flatten())
        ids=(active&selected_mask).nonzero().flatten()
        if len(ids):
            target=self.targets[self.bank_index[ids],self.prefix_step[ids]]
            changed=(target-env.command_targets[ids]).norm(dim=-1)>1e-6
            env.command_path_error[ids[changed]]=0
            env.command_targets[ids]=target
            self.prefix_step[ids]+=1

    def cancel_prefixes(self):
        self.prefix_step[:]=-1

    def metrics(self):
        return dict(active_episodes=int(self.episode_mask.sum()),
                    active_prefixes=int((self.prefix_step>=0).sum()),
                    total_restores=self.total_restores)
