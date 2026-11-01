"""Opt-in compiled native batch adapter; production selection is unchanged."""
import ctypes
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import torch

from native_sim import NativeSimulation


def build_library(output):
    output = Path(output).absolute()
    package = Path(mujoco.__file__).parent
    library, = package.glob('libmujoco.so.*')
    source = Path(__file__).with_name('native_batch_step.c')
    command = ['gcc', '-O3', '-fPIC', '-shared', '-fopenmp', '-fno-fast-math',
               '-ffp-contract=off', '-I', str(package/'include'), str(source),
               str(library), '-Wl,-rpath,'+str(package), '-o', str(output)]
    subprocess.run(command, check=True)
    return command


class BatchedNativeSimulation(NativeSimulation):
    def __init__(self, *args, library, batch_threads=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool.shutdown()
        assert 1 <= batch_threads <= 128
        self.batch_threads = batch_threads
        self.library = ctypes.CDLL(str(Path(library).absolute()))
        self.library.batch_step_version.restype = ctypes.c_int
        assert self.library.batch_step_version() == mujoco.mj_version()
        self.batch = self.library.batch_step
        self.batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*5 + [ctypes.c_void_p]*11
        self.batch.restype = ctypes.c_int
        self.data_pointers = (ctypes.c_void_p*self.num_envs)(*[d._address for d in self.ds])
        self.reset_mask = np.zeros(self.num_envs, dtype=np.uint8)
        self.geom_fingers = (np.asarray(self.contact_reader.finger, dtype=np.int32)
                             if self.contact_reader is not None else None)

    def _run(self, nstep):
        for name in ['qpos', 'qvel', 'ctrl', 'xfrc_applied']:
            np.copyto(self.arrays[name], getattr(self.data, name).detach().cpu().numpy())
        self.reset_mask.fill(0)
        if self.reset_ids:
            self.reset_mask[list(self.reset_ids)] = 1
        def ptr(value):
            return None if value is None else value.ctypes.data
        bad = self.batch(self.m._address, self.data_pointers, self.num_envs, nstep,
            -1 if self.body_motion_ball_dof is None else self.body_motion_ball_dof,
            self.batch_threads, -1 if self.contact_reader is None else self.contact_reader.ball,
            ptr(self.geom_fingers), ptr(self.reset_mask),
            *[ptr(self.arrays[name]) for name in ('qpos','qvel','ctrl','xfrc_applied','xpos','xmat','sensordata')],
            ptr(getattr(self, 'body_motion_cost_array', None)),
            ptr(getattr(self, 'contact_load_array', None)))
        assert not bad, 'Nonfinite native contact load'
        self.reset_ids.clear()
        for name in ['qpos','qvel','xpos','xmat','sensordata']:
            getattr(self.data, name).copy_(torch.from_numpy(self.arrays[name]))
        if nstep and self.body_motion_ball_dof is not None:
            self.body_motion_cost.copy_(torch.from_numpy(self.body_motion_cost_array))
        if self.contact_reader is not None:
            self.contact_load.copy_(torch.from_numpy(self.contact_load_array))
        self.dirty = False
