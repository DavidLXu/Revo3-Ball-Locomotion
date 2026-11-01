"""Clean 20-second random forward/left/back command sequence, left-front start with reversed orbit direction."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from render_gpu_selection import configure_render_gpu
RENDER_GPU_SELECTION=configure_render_gpu(prefer_idle=True)
import argparse,hashlib,json,time
from pathlib import Path
import imageio.v2 as imageio
import mujoco,numpy as np,torch
from policy_loader import load_controller

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build_sequence(seed,seconds,speed=.1):
    rng=np.random.default_rng(seed)
    allowed=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,0.]])
    # Randomized sequence excludes rightward commands ([0,-1]). Each segment
    # lasts 1.4–2.7 s and uses a smooth 0.65 s transition at the boundary.
    points=[];t=0.;previous=allowed[0]
    while t<seconds:
        choices=[i for i,v in enumerate(allowed) if not np.array_equal(v,previous)]
        target=allowed[int(rng.choice(choices))];duration=float(rng.uniform(1.4,2.7));points.append((t,min(t+duration,seconds),previous.copy(),target.copy()));previous=target;t+=duration
    return points

def command(t,points,speed=.1):
    for begin,end,old,new in points:
        if begin<=t<=end:
            u=np.clip((t-begin)/max(end-begin,1e-6),0.,1.)
            # Ease in/out between cardinal commands; never produces rightward
            # command components, only the permitted forward/left/back axes.
            h=u*u*(3.-2.*u);v=(1-h)*old+h*new
            return speed*v
    return np.zeros(2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261031);p.add_argument('--seconds',type=float,default=20.);p.add_argument('--physics-timestep',type=float,default=.0005);p.add_argument('--display-speed',type=float,default=.15);a=p.parse_args()
    assert a.seconds==20. and a.physics_timestep in (.001,.0005) and 0<a.display_speed<=.2 and not a.output.exists() and not a.output.with_suffix('.json').exists()
    ck=a.checkpoint.resolve();sha=digest(ck);torch.set_num_threads(1);c=load_controller(ck);env=c.env;env.m.opt.timestep=a.physics_timestep;env.envcfg['decimation']=round(.01/a.physics_timestep);assert env.dt==.01 and env.envcfg['target_speed']==.1
    source_target_speed=env.envcfg['target_speed'];env.envcfg['target_speed']=a.display_speed
    a.output.parent.mkdir(parents=True,exist_ok=True)
    points=build_sequence(a.seed,a.seconds,speed=a.display_speed);torch.manual_seed(a.seed);env.reset(a.seed,1.);renderer=mujoco.Renderer(env.m,height=720,width=1280);option=mujoco.MjvOption();option.geomgroup[3]=0;writer=imageio.get_writer(str(a.output),fps=50,codec='libx264',quality=8,macro_block_size=1)
    frames=0;rows=[];start=env.d.qpos[env.ballqa:env.ballqa+2].copy();follow=np.r_[start,.23]
    try:
        for step in range(round(a.seconds/.01)):
            target=command(step*.01,points,speed=a.display_speed);assert target[1]>=-1e-12
            env.set_command(*target);env.step(c.action());met=env.metrics();rows.append(met);assert not env.d.xfrc_applied.any() and not env.d.qfrc_applied.any()
            follow+=(1-np.exp(-.01/.45))*(np.r_[env.d.qpos[env.ballqa:env.ballqa+2],.23]-follow)
            if step%2==0:
                camera=mujoco.MjvCamera();camera.lookat[:]=follow;camera.distance=.95;camera.elevation=-24;camera.azimuth=-45-270*(step*.01/a.seconds)
                renderer.update_scene(env.d,camera=camera,scene_option=option);writer.append_data(renderer.render().copy());frames+=1
            if met['failed']:break
    finally:writer.close();renderer.close()
    completed=len(rows)==round(a.seconds/.01) and not rows[-1]['failed'];vsha=digest(a.output)
    result=dict(time_unix=time.time(),checkpoint=str(ck),checkpoint_sha256=sha,seed=int(a.seed),requested_seconds=float(a.seconds),physics_timestep_s=float(a.physics_timestep),control_timestep_s=.01,requested_speed_mps=float(a.display_speed),source_target_speed_mps=float(source_target_speed),camera_start_azimuth_degrees=-45.,camera_sweep_degrees=-270.,trajectory='random_forward_left_backward_only_leftrear_reverse_camera_270deg_large_targets',segments=[dict(begin_s=float(b),end_s=float(e),from_command=[float(x) for x in o],to_command=[float(x) for x in n]) for b,e,o,n in points],directions_used=[[int(x) for x in d] for d in sorted(set(tuple(int(y) for y in x[2]) for x in points))],rightward_command_excluded=True,camera_orbit_degrees=270.,frames=int(frames),fps=50,duration_s=float(frames/50),completed=bool(completed),survival_s=float(len(rows)*.01),failure_reason=str(rows[-1]['failure_reason']),video_sha256=vsha,render_gpu_selection=RENDER_GPU_SELECTION,overlay_text=False,training_modified=False,goal_achieved=False,scope='Clean visual render with no annotations; command sequence excludes rightward target. Actual speed and temporal smoothness require separate telemetry.')
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
