"""MjLab GPU simulation with the task's substep cost and contact observations."""
import mujoco
import torch
import warp as wp
from mjlab.sim import Simulation
from contact_load import ContactLoadReader


@wp.kernel
def accumulate_motion(qvel: wp.array2d[float], ball_dof: int,
                      total: wp.array[float]):
    i = wp.tid()
    angular = qvel[i,3]*qvel[i,3]+qvel[i,4]*qvel[i,4]+qvel[i,5]*qvel[i,5]
    relative = (qvel[i,2]-qvel[i,ball_dof+2])/0.08
    total[i] += wp.min(angular,16.0)+wp.min(relative*relative,16.0)


@wp.kernel
def sum_contact_load(nacon: wp.array[int], geom: wp.array[wp.vec2i],
                     world: wp.array[int], address: wp.array2d[int],
                     force: wp.array2d[float], njmax: int, ball: int,
                     fingers: wp.array[int], sums: wp.array2d[float]):
    k = wp.tid()
    if k < nacon[0]:
        a, b = geom[k][0], geom[k][1]
        fa, fb = int(-1), int(-1)
        if a >= 0:
            fa = fingers[a]
        if b >= 0:
            fb = fingers[b]
        index = int(-1)
        if a == ball and fb >= 0:
            index = fb
        elif b == ball and fa >= 0:
            index = fa
        elif fa >= 0 and fb >= 0 and fa != fb:
            low, high = wp.min(fa,fb), wp.max(fa,fb)
            index = 5+low*(9-low)//2+high-low-1
        adr = address[k,0]
        if index >= 0 and adr >= 0 and adr < njmax:
            wp.atomic_add(sums,world[k],index,wp.max(0.0,force[world[k],adr]))


class MjlabBalanceSimulation(Simulation):
    def __init__(self, *args, decimation, body_motion_ball_dof=None,
                 contact_load_observation=False, body_acceleration_enabled=False, **kwargs):
        self.task_buffers_ready = False
        super().__init__(*args, **kwargs)
        self.torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        wp.synchronize_device(self.wp_device)
        self.decimation = decimation
        self.pending = 0
        self.body_motion_ball_dof = body_motion_ball_dof
        self.acceleration = None
        if body_acceleration_enabled:
            assert body_motion_ball_dof is not None
            from body_acceleration_cost import WarpBodyAcceleration
            self.acceleration = WarpBodyAcceleration(self.num_envs,self.device)
            self.body_acceleration_cost = self.acceleration.cost
        self.body_motion_cost = torch.zeros(self.num_envs,device=self.device)
        self.motion_sum = torch.zeros_like(self.body_motion_cost)
        self.motion_wp = wp.from_torch(self.motion_sum)
        self.contact_load = torch.zeros((self.num_envs,15),device=self.device)
        self.contact_sum = torch.zeros_like(self.contact_load)
        self.contact_sum_wp = wp.from_torch(self.contact_sum)
        self.contact_enabled = contact_load_observation
        if contact_load_observation:
            assert self._mj_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
            reader = ContactLoadReader(self._mj_model)
            self.ball_geom = reader.ball
            self.finger_wp = wp.array(reader.finger,dtype=wp.int32,device=self.wp_device)
        self.task_buffers_ready = True

    def reset(self,env_ids=None):
        with wp.ScopedStream(self.torch_stream):
            super().reset(env_ids)
        if self.task_buffers_ready:
            ids = slice(None) if env_ids is None else env_ids
            self.body_motion_cost[ids] = 0
            self.motion_sum[ids] = 0
            self.contact_load[ids] = 0
            if self.acceleration is not None:
                self.acceleration.reset(ids)

    def step(self):
        if self.pending == 0:
            self.motion_sum.zero_()
            if self.acceleration is not None:
                self.acceleration.begin()
        with wp.ScopedStream(self.torch_stream):
            super().step()
            if self.body_motion_ball_dof is not None:
                wp.launch(accumulate_motion,dim=self.num_envs,
                          inputs=[self.wp_data.qvel,self.body_motion_ball_dof,self.motion_wp],
                          device=self.wp_device)
            if self.acceleration is not None:
                self.acceleration.step(self.wp_data.qvel,self.body_motion_ball_dof,
                                       self._mj_model.opt.timestep,self.wp_device)
        self.pending += 1
        if self.pending == self.decimation:
            self.body_motion_cost.copy_(self.motion_sum/self.decimation)
            if self.acceleration is not None:
                self.acceleration.finish(self.decimation)
            self.pending = 0

    def forward(self):
        with wp.ScopedStream(self.torch_stream):
            super().forward()
            if self.task_buffers_ready and self.contact_enabled:
                self.contact_sum.zero_()
                d = self.wp_data
                wp.launch(sum_contact_load,dim=d.contact.geom.shape[0],
                          inputs=[d.nacon,d.contact.geom,d.contact.worldid,d.contact.efc_address,
                                  d.efc.force,d.njmax,self.ball_geom,self.finger_wp,self.contact_sum_wp],
                          device=self.wp_device)
                self.contact_load.copy_((self.contact_sum/20.).clamp(0.,4.))
