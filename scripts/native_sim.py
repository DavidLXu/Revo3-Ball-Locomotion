"""Native MuJoCo fine-tuning adapter with persistent per-world state and thread pool.

Only the physics backend differs from mjlab's Warp bridge. The same model,
observations, actions, rewards and full free-body dynamics are retained.
"""
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import mujoco

class NativeSimulation:
    def __init__(self,model,num_envs,device,decimation=5,threads=16,body_motion_ball_dof=None,contact_load_observation=False,body_acceleration_enabled=False):
        self.m=model;self.num_envs=num_envs;self.device=device
        self.ds=[mujoco.MjData(model) for _ in range(num_envs)]
        self.pool=ThreadPoolExecutor(max_workers=min(threads,num_envs))
        self.body_motion_ball_dof=body_motion_ball_dof
        self.body_acceleration_enabled=body_acceleration_enabled
        if body_acceleration_enabled:
            assert body_motion_ball_dof is not None
            from body_acceleration_cost import native_update
            self.acceleration_update=native_update
            self.filtered_body_velocity=np.zeros((num_envs,4))
            self.body_acceleration_array=np.zeros(num_envs)
            self.body_acceleration_cost=torch.zeros(num_envs,device=device)
        self.contact_reader=None
        if contact_load_observation:
            from contact_load import ContactLoadReader
            self.contact_reader=ContactLoadReader(model)
            self.contact_load_array=np.zeros((num_envs,15),dtype=np.float32)
            self.contact_load=torch.zeros((num_envs,15),device=device,dtype=torch.float32)
        if body_motion_ball_dof is not None:
            assert body_motion_ball_dof == model.joint('ball/free_ball').dofadr[0]
            assert model.joint('hand/floating_wrist').dofadr[0] == 0
            self.body_motion_cost_array=np.zeros(num_envs)
            self.body_motion_cost=torch.zeros(num_envs,device=device,dtype=torch.float32)
        self.decimation=decimation;self.pending=0;self.dirty=True
        self.reset_ids=set(range(num_envs))
        self.arrays={
            'qpos':np.repeat(model.qpos0[None],num_envs,axis=0),
            'qvel':np.zeros((num_envs,model.nv)),
            'ctrl':np.zeros((num_envs,model.nu)),
            'xfrc_applied':np.zeros((num_envs,model.nbody,6)),
            'xpos':np.zeros((num_envs,model.nbody,3)),
            'xmat':np.zeros((num_envs,model.nbody,9)),
            'sensordata':np.zeros((num_envs,model.nsensordata))}
        self.data=SimpleNamespace(**{k:torch.tensor(v,device=device,dtype=torch.float32) for k,v in self.arrays.items()})

    def reset(self,env_ids):
        ids=env_ids.cpu().numpy()
        for i in ids:mujoco.mj_resetData(self.m,self.ds[i])
        self.reset_ids.update(int(i) for i in ids)
        if self.body_acceleration_enabled:
            self.filtered_body_velocity[ids]=0
            self.body_acceleration_array[ids]=0
            self.body_acceleration_cost[env_ids]=0
        self.data.xfrc_applied[env_ids]=0
        if self.contact_reader is not None:
            self.contact_load[env_ids]=0
            self.contact_load_array[ids]=0
        if self.body_motion_ball_dof is not None:
            self.body_motion_cost[env_ids]=0
            self.body_motion_cost_array[ids]=0
        self.dirty=True

    def restore_integration_states(self,env_ids,states):
        """Restore selected clean worlds from native snapshots, retaining doubles.

        Call after an ordinary reset/forward has completed. Control tensors are
        float32, as in normal training; the next step applies those controls.
        Controller/filter memory is owned by the environment, not this adapter.
        """
        assert self.pending==0 and not self.dirty and not self.reset_ids
        ids=env_ids.detach().cpu().numpy()
        assert ids.ndim==1 and np.issubdtype(ids.dtype,np.integer)
        assert len(np.unique(ids))==len(ids) and np.all((ids>=0)&(ids<self.num_envs))
        values=np.asarray(states,dtype=np.float64)
        state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
        assert values.shape==(len(ids),mujoco.mj_stateSize(self.m,state_spec))
        assert np.isfinite(values).all()
        for i,value in zip(ids,values):
            d=self.ds[int(i)]
            mujoco.mj_resetData(self.m,d)
            mujoco.mj_setState(self.m,d,value,state_spec)
            assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
            mujoco.mj_forward(self.m,d)
            if self.contact_reader is not None:
                self.contact_load_array[i]=self.contact_reader.features(d)
            for name in self.arrays:
                self.arrays[name][i]=getattr(d,name)
        selected=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        for name in self.arrays:
            getattr(self.data,name)[selected]=torch.as_tensor(self.arrays[name][ids],device=self.device,dtype=torch.float32)
        if self.contact_reader is not None:
            self.contact_load[selected]=torch.as_tensor(self.contact_load_array[ids],device=self.device)

    def _run(self,nstep):
        for name in ['qpos','qvel','ctrl','xfrc_applied']:
            np.copyto(self.arrays[name],getattr(self.data,name).detach().cpu().numpy())
        def one(i):
            d=self.ds[i]
            # Preserve native double-precision integration between resets.
            if i in self.reset_ids:
                d.qpos[:]=self.arrays['qpos'][i]
                d.qvel[:]=self.arrays['qvel'][i]
            for name in ['ctrl','xfrc_applied']:
                getattr(d,name)[:]=self.arrays[name][i]
            if nstep and self.body_motion_ball_dof is not None:
                total=0.
                acceleration_total=0.
                for _ in range(nstep):
                    mujoco.mj_step(self.m,d)
                    angular=np.dot(d.qvel[3:6],d.qvel[3:6])
                    vertical=((d.qvel[2]-d.qvel[self.body_motion_ball_dof+2])/.08)**2
                    total+=np.minimum(angular,16.)+np.minimum(vertical,16.)
                    if self.body_acceleration_enabled:
                        velocity=np.r_[d.qvel[3:6],d.qvel[2]-d.qvel[self.body_motion_ball_dof+2]]
                        filtered,cost=self.acceleration_update(self.filtered_body_velocity[i],velocity,self.m.opt.timestep)
                        self.filtered_body_velocity[i]=filtered
                        acceleration_total+=cost
                self.body_motion_cost_array[i]=total/nstep
                if self.body_acceleration_enabled:
                    self.body_acceleration_array[i]=acceleration_total/nstep
            elif nstep:
                mujoco.mj_step(self.m,d,nstep=nstep)
            mujoco.mj_forward(self.m,d)
            if self.contact_reader is not None:
                self.contact_load_array[i]=self.contact_reader.features(d)
            for name in ['qpos','qvel','xpos','xmat','sensordata']:
                self.arrays[name][i]=getattr(d,name)
        list(self.pool.map(one,range(self.num_envs)))
        self.reset_ids.clear()
        for name in ['qpos','qvel','xpos','xmat','sensordata']:
            getattr(self.data,name).copy_(torch.from_numpy(self.arrays[name]))
        if nstep and self.body_motion_ball_dof is not None:
            self.body_motion_cost.copy_(torch.from_numpy(self.body_motion_cost_array))
        if nstep and self.body_acceleration_enabled:
            self.body_acceleration_cost.copy_(torch.from_numpy(self.body_acceleration_array))
        if self.contact_reader is not None:
            self.contact_load.copy_(torch.from_numpy(self.contact_load_array))
        self.dirty=False

    def step(self):
        self.pending+=1
        if self.pending==self.decimation:
            self._run(self.decimation);self.pending=0

    def forward(self):
        if self.dirty:self._run(0)
