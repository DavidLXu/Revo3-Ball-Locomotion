"""Resolve a requested physical GPU to its actual EGL device index."""
import ctypes
import os
import subprocess


def configure_render_gpu(prefer_idle=False):
    physical = os.environ.get('REVO_RENDER_PHYSICAL_GPU')
    if physical is None:
        # Existing milestone workers set both variables to an nvidia-smi index.
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible.isdecimal() and os.environ.get('MUJOCO_EGL_DEVICE_ID') == visible:
            physical = visible
        else:
            return None
    assert physical.isdecimal(), 'Expected an nvidia-smi physical GPU index'
    requested=int(physical)
    availability=None
    if prefer_idle:
        from evaluation_gpu import select_evaluation_gpu
        # Leave the last two cards out of opportunistic render placement.
        availability=select_evaluation_gpu(requested,allowed_indices=range(6))
        physical=str(availability['selected_gpu'])
    uuid = subprocess.check_output(['nvidia-smi', '-i', physical,
        '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).strip()
    assert uuid.startswith('GPU-') and '\n' not in uuid
    # EGL enumeration follows its own order. Restrict CUDA by UUID, then ask EGL
    # which device corresponds to the sole visible CUDA device, ordinal zero.
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    from mujoco.egl import egl_ext as egl
    from OpenGL.raw.EGL.NV.device_cuda import EGL_CUDA_DEVICE_NV
    query = ctypes.CFUNCTYPE(egl.EGLBoolean, egl.EGLDeviceEXT, egl.EGLint,
        ctypes.POINTER(ctypes.c_ssize_t))(egl.eglGetProcAddress('eglQueryDeviceAttribEXT'))
    matches = []
    for index, device in enumerate(egl.eglQueryDevicesEXT(max_devices=64)):
        value = ctypes.c_ssize_t(-1)
        valid = query(device, EGL_CUDA_DEVICE_NV, ctypes.byref(value))
        error = egl.eglGetError()
        if valid and error == egl.EGL_SUCCESS and value.value == 0:
            matches.append(index)
    assert len(matches) == 1, f'Cannot uniquely resolve physical GPU {physical}: {matches}'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(matches[0])
    return dict(physical_gpu=int(physical), requested_physical_gpu=requested,
                availability_at_launch=availability, gpu_uuid=uuid, egl_device_index=matches[0],
                method='UUID-restricted CUDA device queried through EGL_NV_device_cuda')
