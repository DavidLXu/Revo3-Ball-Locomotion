"""Shared physical model for policy training and independent native evaluation."""
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
KP=8.0
KV=.18
TORQUE_LIMIT=2.0
ACTION_SCALE=.45

def make_model(initial_pose='original', hand_inertia_mode='legacy'):
    if initial_pose not in ('original','opposed_thumb'):
        raise ValueError(f'Unknown initial pose: {initial_pose}')
    if hand_inertia_mode not in ('legacy', 'collision_only'):
        raise ValueError(f'Unknown hand inertia mode: {hand_inertia_mode}')
    directory=ROOT/'outputs' if initial_pose=='original' else ROOT/'outputs/opposed_thumb'
    scene_path=directory/'scene/scene.xml' if initial_pose=='original' else directory/'scene.xml'
    spec=mujoco.MjSpec.from_file(str(scene_path))
    for j in list(spec.joints):
        if j.type==mujoco.mjtJoint.mjJNT_HINGE:
            a=spec.add_actuator(name='pd_'+j.name,target=j.name,trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=KP,kv=KV)
            a.forcelimited=True
            a.forcerange=[-TORQUE_LIMIT,TORQUE_LIMIT]
    for f in ['thumb','index','middle','ring','little']:
        spec.add_sensor(name='contact_'+f,type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,objname='hand/right_'+f+'_DIP_link',
            reftype=mujoco.mjtObj.mjOBJ_BODY,refname='ball/ball',intprm=[1,1,1])
    for b in ['right_hand_base_link','right_plam_link','right_plam_touch_link','right_plam_rubber_link']:
        spec.add_sensor(name='contact_'+b,type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_BODY,objname='hand/'+b,
            reftype=mujoco.mjtObj.mjOBJ_BODY,refname='ball/ball',intprm=[1,1,1])
    spec.option.timestep=.002
    spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model=spec.compile()
    if hand_inertia_mode == 'collision_only':
        from inertia_variant import collision_only_hand_inertia
        collision_only_hand_inertia(model, initial_pose)
    q0=np.load(directory/'initial_qpos.npy')
    joints=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
    qa=model.jnt_qposadr[joints].copy()
    va=model.jnt_dofadr[joints].copy()
    model.key_qpos[0]=q0
    model.key_ctrl[0]=q0[qa]
    return model,q0,qa,va

if __name__=='__main__':
    m,q,qa,va=make_model();d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    print('nq/nv/nu',m.nq,m.nv,m.nu)
    print('contacts',[(m.geom(c.geom1).name,m.geom(c.geom2).name,c.pos.tolist()) for c in d.contact])
    for n in range(2500):
        d.ctrl[:]=q[qa];mujoco.mj_step(m,d)
        if n in [49,249,499,999,2499]:
            print('baseline',d.time,'hand',d.qpos[:3], 'ball',d.qpos[-7:-4], 'contacts',d.ncon, 'speed',np.linalg.norm(d.qvel[:6]),flush=True)


def configure_pd_derivative_gain(model, gain=KV):
    """Set the position controller damping without changing actuator limits or kp."""
    if not np.isfinite(gain) or not .1 <= gain <= .3:
        raise ValueError('PD derivative gain must be finite and between 0.1 and 0.3')
    assert model.nu == 21
    assert np.allclose(model.actuator_biasprm[:, 2], -KV)
    model.actuator_biasprm[:, 2] = -gain
