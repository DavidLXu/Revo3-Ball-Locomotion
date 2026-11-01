"""Independent native rollouts: cardinal directions, diagonal, stop, and reversals."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from policy_loader import load_controller as PolicyController

PROFILE_NAMES={'stop','forward','backward','left','right','diagonal',
    'diagonal_forward_right','diagonal_backward_left','diagonal_backward_right','switch','rapid_switch'}

def profile(name,t,speed,seconds):
    if name not in PROFILE_NAMES:raise ValueError(f'Unknown command profile: {name}')
    if name=='stop':return (0.,0.)
    if t<1.:return (0.,0.)
    if name=='rapid_switch':
        diagonal=speed/np.sqrt(2)
        sequence=[(speed,0.),(-speed,0.),(0.,0.),(0.,speed),(0.,-speed),(0.,0.),
                  (diagonal,diagonal),(-diagonal,-diagonal),(0.,0.),
                  (diagonal,-diagonal),(-diagonal,diagonal),(0.,0.)]
        return sequence[int((t-1.)/2.)%len(sequence)]
    if name=='forward':return (speed,0.)
    if name=='backward':return (-speed,0.)
    if name=='left':return (0.,speed)
    if name=='right':return (0.,-speed)
    if name=='diagonal':return (speed/np.sqrt(2),speed/np.sqrt(2))
    if name=='diagonal_forward_right':return (speed/np.sqrt(2),-speed/np.sqrt(2))
    if name=='diagonal_backward_left':return (-speed/np.sqrt(2),speed/np.sqrt(2))
    if name=='diagonal_backward_right':return (-speed/np.sqrt(2),-speed/np.sqrt(2))
    index=min(5,int((t-1)/max((seconds-1)/6,.1)))
    return [(speed,0.),(0.,0.),(-speed,0.),(0.,speed),(0.,-speed),(0.,0.)][index]

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--speed',type=float)
    p.add_argument('--seconds',type=float,default=15.);p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--seed',type=int,default=20000);p.add_argument('--profiles',default='stop,forward,backward,left,right,diagonal,switch')
    p.add_argument('--stochastic',action='store_true')
    p.add_argument('--physics-timestep',type=float)
    p.add_argument('--noise',type=float,default=1.);p.add_argument('--output');p.add_argument('--video',action='store_true')
    a=p.parse_args()
    names=a.profiles.split(',')
    if not names or any(name not in PROFILE_NAMES for name in names):
        p.error('Unknown profile; choose from '+','.join(sorted(PROFILE_NAMES)))
    torch.set_num_threads(1);c=PolicyController(a.checkpoint);env=c.env
    if a.physics_timestep is not None:
        assert a.physics_timestep>0
        decimation=round(env.dt/a.physics_timestep)
        assert decimation>=1 and abs(decimation*a.physics_timestep-env.dt)<1e-10
        env.m.opt.timestep=a.physics_timestep
        env.envcfg['decimation']=decimation
    speed=c.trained_speed_limit if a.speed is None else a.speed
    results=[];frames=[];renderer=None
    if a.video:
        import mujoco
        renderer=mujoco.Renderer(env.m,720,960);opt=mujoco.MjvOption();opt.geomgroup[3]=0
    for name in names:
        for ep in range(a.episodes):
            torch.manual_seed(a.seed+ep)
            env.reset(a.seed+ep,a.noise);rows=[]
            initial_hand=env.d.qpos[:2].copy();initial_ball=env.d.qpos[env.ballqa:env.ballqa+2].copy()
            for step in range(round(a.seconds/env.dt)):
                target=profile(name,step*env.dt,speed,a.seconds);env.set_command(*target)
                env.step(c.action(stochastic=a.stochastic));r=env.metrics();rows.append(r)
                if renderer and ep==0 and step%3==0:
                    cam=mujoco.MjvCamera();cam.azimuth=-45;cam.elevation=-25;cam.distance=.85
                    cam.lookat[:]=[r['ball_x_m'],r['ball_y_m'],.21]
                    renderer.update_scene(env.d,camera=cam,scene_option=opt)
                    from PIL import Image,ImageDraw,ImageFont
                    frame=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(frame)
                    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
                    except OSError:font=ImageFont.load_default()
                    draw.rectangle((0,0,960,82),fill=(12,22,35))
                    label=f'{c.checkpoint.parent.name}/{c.checkpoint.stem} | {name} | t={env.d.time:.2f}s'
                    velocity=r['measured_velocity_mps']
                    draw.text((16,10),label,font=font,fill='white')
                    draw.text((16,42),f'command ({target[0]:+.3f}, {target[1]:+.3f}) m/s | actual ({velocity[0]:+.3f}, {velocity[1]:+.3f}) m/s',font=font,fill='white')
                    frames.append(np.array(frame))
                if r['failed']:break
            steady=rows[min(200,len(rows)//2):]
            velocities=np.array([r['measured_velocity_mps'] for r in steady]);commands=np.array([r['command_mps'] for r in steady])
            # Displacement-derived windows distinguish sustained locomotion
            # from settling transients and contact-induced velocity oscillation.
            windows=[]
            window_steps=max(1,round(5./env.dt))
            for begin in range(0,len(rows),window_steps):
                end=min(begin+window_steps,len(rows));chunk=rows[begin:end]
                start_xy=initial_ball if begin==0 else np.array([rows[begin-1]['ball_x_m'],rows[begin-1]['ball_y_m']])
                end_xy=np.array([chunk[-1]['ball_x_m'],chunk[-1]['ball_y_m']])
                window_commands=np.array([r['command_mps'] for r in chunk])
                window_velocity=np.array([r['measured_velocity_mps'] for r in chunk])
                windows.append({'start_s':begin*env.dt,'end_s':end*env.dt,
                    'net_velocity_xy_mps':((end_xy-start_xy)/((end-begin)*env.dt)).tolist(),
                    'mean_command_xy_mps':window_commands.mean(axis=0).tolist(),
                    'mean_velocity_error_mps':float(np.linalg.norm(window_velocity-window_commands,axis=1).mean())})
            result={'profile':name,'seed':a.seed+ep,'requested_speed_mps':speed,'survival_s':len(rows)*env.dt,
              'completed':len(rows)==round(a.seconds/env.dt) and not rows[-1]['failed'],'failure_reason':rows[-1]['failure_reason'],
              'mean_velocity_error_mps':float(np.mean(np.linalg.norm(velocities-commands,axis=1))),
              'mean_actual_speed_mps':float(np.linalg.norm(velocities,axis=1).mean()),
              'mean_actual_velocity_mps':velocities.mean(axis=0).tolist(),
              'integrated_command_xy_m':(np.array([r['command_mps'] for r in rows]).sum(axis=0)*env.dt).tolist(),
              'hand_displacement_xy_m':(env.d.qpos[:2]-initial_hand).tolist(),
              'ball_net_average_velocity_xy_mps':((env.d.qpos[env.ballqa:env.ballqa+2]-initial_ball)/(len(rows)*env.dt)).tolist(),
              'ball_displacement_xy_m':[rows[-1]['ball_x_m'],rows[-1]['ball_y_m']],
              'palm_contact_fraction':float(np.mean([r['palm_contact'] for r in rows])),
              'mean_finger_contacts':float(np.mean([r['finger_contacts'] for r in rows]))}
            result['velocity_windows']=windows
            if name=='rapid_switch':
                # Preserve transition detail that five-second averages can hide.
                segments=[];begin=0
                for end in range(1,len(rows)+1):
                    if end<len(rows) and rows[end]['target_mps']==rows[begin]['target_mps']:continue
                    chunk=rows[begin:end]
                    errors=np.array([r['velocity_error_mps'] for r in chunk])
                    tail=errors[-max(1,round(.5/env.dt)):]
                    segments.append({'start_s':begin*env.dt,'end_s':end*env.dt,
                        'target_mps':chunk[0]['target_mps'],
                        'mean_velocity_error_mps':float(errors.mean()),
                        'final_half_second_error_mps':float(tail.mean()),
                        'complete_phase':not np.allclose(profile(name,end*env.dt,speed,a.seconds),
                            chunk[0]['target_mps'],rtol=0.,atol=1e-12)})
                    begin=end
                result['command_segments']=segments
            result['support']={'from_s':(len(rows)-len(steady))*env.dt,
                'finger_order':['thumb','index','middle','ring','little'],
                'distal_contact_fraction':np.mean([r['finger_mask'] for r in steady],axis=0).tolist(),
                'any_link_contact_fraction':np.mean(np.array([r['total_finger_normal_forces_N'] for r in steady])>1e-6,axis=0).tolist(),
                'load_bearing_contact_fraction':np.mean(np.array([r['total_finger_normal_forces_N'] for r in steady])>.15,axis=0).tolist(),
                'mean_total_normal_force_N':np.mean([r['total_finger_normal_forces_N'] for r in steady],axis=0).tolist(),
                'four_or_more_fraction':float(np.mean([r['finger_contacts']>=4 for r in steady])),
                'all_five_fraction':float(np.mean([r['finger_contacts']==5 for r in steady]))}
            results.append(result);print(json.dumps(result),flush=True)
    if renderer:renderer.close()
    report={'checkpoint':str(c.checkpoint),'training_speed_limit_mps':c.trained_speed_limit,'test_speed_mps':speed,
       'hand_inertia_mode':c.envcfg.get('hand_inertia_mode','legacy'),
       'hand_mass_kg':float(sum(env.m.body_mass[i] for i in range(env.m.nbody) if env.m.body(i).name.startswith('hand/'))),
       'seconds':a.seconds,'noise':a.noise,'engine':'native_mujoco','deterministic_policy':not a.stochastic,
       'physics_timestep_s':float(env.m.opt.timestep),'control_timestep_s':env.dt,
       'completion_rate':float(np.mean([r['completed'] for r in results])),
       'mean_velocity_error_mps':float(np.mean([r['mean_velocity_error_mps'] for r in results])),
       'episodes':results}
    path=Path(a.output) if a.output else c.checkpoint.with_name(c.checkpoint.stem+'_teleop_eval.json')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2))
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(str(path.with_suffix('.mp4')),frames,fps=1/(env.dt*3),codec='libx264',quality=8)
    print('REPORT',path,flush=True)
if __name__=='__main__':main()
