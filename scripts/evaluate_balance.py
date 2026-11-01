"""Independent native MuJoCo evaluation of a learned PPO policy; no GPU physics."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,json,time,csv
import numpy as np
import mujoco
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from scipy.spatial.transform import Rotation
from balance_model import make_model,ROOT,ACTION_SCALE,KV,configure_pd_derivative_gain

class NativeBalance:
    def __init__(self,envcfg=None):
        self.envcfg=envcfg or {};self.rolling=self.envcfg.get('task')=='roll'
        self.forward_speed=self.envcfg.get('forward_speed',.04)
        self.reference_origin_xy=np.zeros(2);self.reference_time=0.
        self.m,self.q0,self.qa,self.va=make_model(self.envcfg.get('initial_pose','original'),self.envcfg.get('hand_inertia_mode','legacy'))
        configure_pd_derivative_gain(self.m,self.envcfg.get('pd_derivative_gain',KV))
        self.m.opt.timestep=self.envcfg.get('physics_timestep',.002)
        self.d=mujoco.MjData(self.m)
        self.handbody=self.m.body('hand/right_hand_base_link').id
        self.ballbody=self.m.body('ball/ball').id
        self.ballqa=int(self.m.jnt_qposadr[self.m.joint('ball/free_ball').id]);self.ballva=int(self.m.jnt_dofadr[self.m.joint('ball/free_ball').id])
        self.bg=self.m.geom('ball/surface').id
        self.joint0=self.q0[self.qa];js=np.flatnonzero(self.m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        self.low=self.m.jnt_range[js,0];self.high=self.m.jnt_range[js,1]
        mujoco.mj_resetDataKeyframe(self.m,self.d,0);mujoco.mj_forward(self.m,self.d)
        self.rot0=self.d.xmat[self.handbody].reshape(3,3).copy()
        self.rel0=self.q0[:3]-self.q0[self.ballqa:self.ballqa+3]
        self.tipb=[];self.anchors=[]
        for f in ['thumb','index','middle','ring','little']:
            for c in self.d.contact:
                if self.bg not in c.geom:continue
                g=c.geom2 if c.geom1==self.bg else c.geom1
                if 'right_'+f+'_DIP' in self.m.geom(g).name:
                    b=self.m.geom_bodyid[g];self.tipb.append(b)
                    self.anchors.append(self.d.xmat[b].reshape(3,3).T@(c.pos-self.d.xpos[b]));break
        self.anchors=np.array(self.anchors);self.actions=np.zeros(21)

    def reset(self,seed,noise=1.):
        self.reference_origin_xy[:]=0;self.reference_time=0.
        rng=np.random.default_rng(seed);mujoco.mj_resetDataKeyframe(self.m,self.d,0)
        self.d.qpos[self.qa]+=rng.normal(0,.008*noise,21)
        self.d.qpos[:2]+=rng.normal(0,.0008*noise,2)
        dq=Rotation.from_rotvec(rng.normal(0,.006*noise,3))
        q=self.q0[3:7];rot=dq*Rotation.from_quat(q[[1,2,3,0]])
        self.d.qpos[3:7]=rot.as_quat()[[3,0,1,2]]
        self.d.ctrl[:]=self.joint0;self.actions[:]=0;mujoco.mj_forward(self.m,self.d)

    def features(self):
        q=self.d.qpos;v=self.d.qvel;r=self.d.xmat[self.handbody].reshape(3,3)@self.rot0.T
        rel=q[:3]-q[self.ballqa:self.ballqa+3]
        tips=self.d.xpos[self.tipb]+np.einsum('fij,fj->fi',self.d.xmat[self.tipb].reshape(5,3,3),self.anchors)
        tr=tips-q[None,self.ballqa:self.ballqa+3];gaps=np.linalg.norm(tr,axis=-1)-.12
        if self.envcfg.get('heading_observations',False):
            from heading_frame import heading_inverse_numpy
            inverse=heading_inverse_numpy(r)
            rel=inverse@rel;tr=tr@inverse.T;r=inverse@r
        return q,v,rel,r,gaps,tr

    def observation(self):
        q,v,rel,r,gaps,tr=self.features()
        ballxy=q[self.ballqa:self.ballqa+2].copy()
        handvel=v[:3].copy();ballvel=v[self.ballva:self.ballva+3].copy();ballang=v[self.ballva+3:self.ballva+6].copy()
        ballrot=self.d.xmat[self.ballbody].reshape(3,3)
        if self.envcfg.get('ball_angular_world',False):ballang=ballrot@ballang
        if self.rolling:
            ballxy-=self.reference_origin_xy
            ballxy[0]-=self.forward_speed*(self.d.time-self.reference_time)
            limit=self.envcfg.get('path_observation_limit',0.)
            if limit>0:ballxy[0]=np.clip(ballxy[0],-limit,limit)
        if self.rolling and self.envcfg.get('velocity_command_frame',False):
            handvel[0]-=self.forward_speed;ballvel[0]-=self.forward_speed
            if self.envcfg.get('ball_angular_world',False):ballang[1]-=self.forward_speed/.12
            else:ballang-=ballrot[1,:]*(self.forward_speed/.12)
        if self.envcfg.get('heading_observations',False):
            from heading_frame import heading_inverse_numpy
            inverse=heading_inverse_numpy(self.d.xmat[self.handbody].reshape(3,3)@self.rot0.T)
            handvel=inverse@handvel;ballvel=inverse@ballvel;ballang=inverse@ballang
        o=np.concatenate(((q[self.qa]-self.joint0)*2,v[self.va]*.08,(rel-self.rel0)*10,
                          r.ravel(),handvel*2,v[3:6]*.2,ballxy*5,
                          ballvel*2,ballang*.1,
                          gaps*30,tr.ravel()*5,self.actions,(self.d.sensordata[:5]>0).astype(float),
                          [float(np.any(self.d.sensordata[5:]>0))]))
        return np.clip(np.nan_to_num(o,nan=0.,posinf=20.,neginf=-20.),-20,20).astype(np.float32)

    def step(self,action,push=None):
        self.actions=.7*self.actions+.3*np.clip(action,-1,1)
        self.d.ctrl[:]=np.clip(self.joint0+ACTION_SCALE*self.actions,self.low,self.high)
        self.d.xfrc_applied[:]=0
        if push is not None:self.d.xfrc_applied[self.handbody,:3]=push
        for _ in range(self.envcfg.get('decimation',5)):mujoco.mj_step(self.m,self.d)
        mujoco.mj_forward(self.m,self.d)

    def metrics(self):
        q,v,rel,r,gaps,tr=self.features()
        tilt=float(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))
        if self.envcfg.get('free_heading',False):tilt=float(np.arccos(np.clip(r[2,2],-1,1)))
        if self.envcfg.get('free_heading',False) and not self.envcfg.get('heading_observations',False):
            from heading_frame import heading_inverse_numpy
            rel=heading_inverse_numpy(r)@rel
        drift=float(np.linalg.norm(q[self.ballqa:self.ballqa+2]))
        floor=any('floor' in [self.m.geom(c.geom1).name,self.m.geom(c.geom2).name] and
                  any(self.m.geom(g).name.startswith('hand/') for g in c.geom) for c in self.d.contact)
        path_error=q[self.ballqa:self.ballqa+2]-self.reference_origin_xy-[self.forward_speed*(self.d.time-self.reference_time),0]
        escaped=(np.linalg.norm(path_error)>.75 or abs(q[self.ballqa+1])>.5) if self.rolling else drift>.35
        physical_fall=not np.isfinite(q).all() or floor or rel[2]<.115 or rel[2]>.28 or tilt>np.arccos(.4) or np.linalg.norm(rel[:2]-self.rel0[:2])>.14
        forbidden_palm=self.envcfg.get('reject_palm_support',False) and bool(np.any(self.d.sensordata[5:]>0))
        failed=physical_fall or escaped or forbidden_palm
        failure_reason='fall' if physical_fall else ('palm_support' if forbidden_palm else ('tracking_limit' if escaped else 'none'))
        finger_forces=np.zeros(5)
        total_finger_forces=np.zeros(5)
        for ci,c in enumerate(self.d.contact):
            if self.bg not in c.geom:continue
            g=c.geom2 if c.geom1==self.bg else c.geom1
            name=self.m.geom(g).name
            for fi,f in enumerate(['thumb','index','middle','ring','little']):
                if 'right_'+f+'_' in name:
                    force=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,ci,force)
                    total_finger_forces[fi]+=max(0.,force[0])
                    if 'right_'+f+'_DIP' in name:finger_forces[fi]+=max(0.,force[0])
        world_ball_angular_velocity=self.d.xmat[self.ballbody].reshape(3,3)@v[self.ballva+3:self.ballva+6]
        return {'ball_x_m':float(q[self.ballqa]),'ball_y_m':float(q[self.ballqa+1]),
                'hand_x_m':float(q[0]),'ball_forward_speed_mps':float(v[self.ballva]),
                'rolling_slip_mps':float(abs(v[self.ballva]-.12*world_ball_angular_velocity[1])),
                'finger_normal_forces_N':finger_forces.tolist(),'total_finger_normal_forces_N':total_finger_forces.tolist(),
                'failed':bool(failed),'physical_fall':bool(physical_fall),'failure_reason':failure_reason,'tilt_deg':float(np.degrees(tilt)),'ball_drift_m':drift,
                'finger_contacts':int((self.d.sensordata[:5]>0).sum()),'finger_mask':(self.d.sensordata[:5]>0).astype(int).tolist(),'palm_contact':bool(np.any(self.d.sensordata[5:]>0)),
                'hand_floor_contact':floor,'relative_height_m':float(rel[2]),'hand_speed_mps':float(np.linalg.norm(v[:3]))}

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seconds',type=float,default=10.);p.add_argument('--noise',type=float,default=1.)
    p.add_argument('--push',type=float,default=0.);p.add_argument('--video',action='store_true');p.add_argument('--output')
    p.add_argument('--trace',action='store_true',help='Save the first rollout at100Hz for motion/support analysis.')
    p.add_argument('--stochastic',action='store_true')
    p.add_argument('--seed',type=int,default=10000)
    args=p.parse_args();torch.set_num_threads(1)
    ck=Path(args.checkpoint);cfg=json.loads((ck.parent/'runner_config.json').read_text())
    envcfg=json.loads((ck.parent/'env_config.json').read_text())
    env=NativeBalance(envcfg);dummy=torch.tensor(env.observation()[None])
    ob=TensorDict({'actor':dummy,'critic':dummy},batch_size=[1])
    acfg=dict(cfg['actor']);acfg.pop('class_name',None)
    policy=MLPModel(ob,cfg['obs_groups'],'actor',21,**acfg)
    saved=torch.load(ck,map_location='cpu',weights_only=False);policy.load_state_dict(saved['actor_state_dict']);policy.eval()
    result=[];frames=[];renderer=None
    artifact_stem=ck.stem+(f'_seed{args.seed}' if args.seed!=10000 else '')+('_sampled' if args.stochastic else '')
    if args.video:renderer=mujoco.Renderer(env.m,720,960)
    for ep in range(args.episodes):
        torch.manual_seed(args.seed+ep)
        env.reset(args.seed+ep,args.noise);metrics=[];initial_hand_x=float(env.d.qpos[0])
        for step in range(round(args.seconds/.01)):
            inp=torch.from_numpy(env.observation()[None])
            with torch.inference_mode():action=policy(TensorDict({'actor':inp,'critic':inp},batch_size=[1]),stochastic_output=args.stochastic).numpy()[0]
            t=step*.01;push=None
            for k,axis in enumerate([0,1,0]):
                if 5+7*k<=t<5.1+7*k:
                    push=np.zeros(3);push[axis]=args.push*(-1 if k==2 else 1)
            env.step(action,push);met=env.metrics();metrics.append(met)
            if renderer is not None and ep==0 and step%3==0:
                cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-22;cam.distance=.85;cam.lookat[:]=[env.d.qpos[env.ballqa] if env.rolling else 0,0,.205]
                opt=mujoco.MjvOption();opt.geomgroup[3]=0;renderer.update_scene(env.d,cam,scene_option=opt);frames.append(renderer.render().copy())
            if met['failed']:break
        after=metrics[min(200,len(metrics)//2):]
        row={'seed':args.seed+ep,'survival_s':len(metrics)*.01,'fell':metrics[-1]['physical_fall'],
             'task_failed':metrics[-1]['failed'],'failure_reason':metrics[-1]['failure_reason'],
             'all_five_fraction':float(np.mean([x['finger_contacts']==5 for x in after])),
             'four_or_more_fraction':float(np.mean([x['finger_contacts']>=4 for x in after])),
             'mean_finger_contacts':float(np.mean([x['finger_contacts'] for x in after])),
             'all_five_loaded_fraction':float(np.mean([all(f>.05 for f in x['finger_normal_forces_N']) for x in after])),
             'per_finger_mean_normal_force_N':dict(zip(['thumb','index','middle','ring','little'],np.mean([x['finger_normal_forces_N'] for x in after],axis=0).tolist())),
             'per_finger_mean_total_normal_force_N':dict(zip(['thumb','index','middle','ring','little'],np.mean([x['total_finger_normal_forces_N'] for x in after],axis=0).tolist())),
             'finger_force_scope':'Legacy per_finger_mean_normal_force_N includes DIP and distal pads; total includes every phalanx of each finger.',
             'per_finger_contact_fraction':dict(zip(['thumb','index','middle','ring','little'],np.mean([x['finger_mask'] for x in after],axis=0).tolist())),
             'min_relative_height_m':min(x['relative_height_m'] for x in metrics),
             'palm_contact_fraction':float(np.mean([x['palm_contact'] for x in after])),
             'max_tilt_deg':max(x['tilt_deg'] for x in metrics),'max_ball_drift_m':max(x['ball_drift_m'] for x in metrics),
             'final_hand_speed_mps':metrics[-1]['hand_speed_mps']}
        row.update({'forward_distance_m':metrics[-1]['ball_x_m'],
            'hand_forward_distance_m':metrics[-1]['hand_x_m']-initial_hand_x,
            'mean_forward_speed_mps':float(np.mean([x['ball_forward_speed_mps'] for x in after])),
            'max_lateral_deviation_m':max(abs(x['ball_y_m']) for x in metrics),
            'mean_rolling_slip_mps':float(np.mean([x['rolling_slip_mps'] for x in after]))})
        row['rolling_success']=bool(not row['task_failed'] and row['forward_distance_m']>=.7*env.forward_speed*args.seconds
            and row['hand_forward_distance_m']>=.7*env.forward_speed*args.seconds-.1
            and row['four_or_more_fraction']>=.8 and min(row['per_finger_contact_fraction'].values())>=.25
            and row['palm_contact_fraction']==0)
        result.append(row);print(json.dumps(row),flush=True)
        if args.trace and ep==0:
            fields=['ball_x_m','ball_y_m','hand_x_m','ball_forward_speed_mps','rolling_slip_mps','tilt_deg','finger_contacts','palm_contact','failed','physical_fall','failure_reason']
            dest=ck.parent/(artifact_stem+'_trace.csv')
            with dest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['time_s']+fields);w.writeheader()
                for i,met in enumerate(metrics):w.writerow({'time_s':(i+1)*.01,**{k:met[k] for k in fields}})
    if renderer is not None:
        renderer.close()
        import mediapy as media
        import imageio_ffmpeg
        from PIL import Image
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        for idx,label in [(0,'start'),(len(frames)//2,'middle'),(-1,'end')]:
            Image.fromarray(frames[idx]).save(ck.parent/(artifact_stem+'_'+label+'.png'))
        media.write_video(str(ck.parent/(artifact_stem+'_native.mp4')),frames,fps=1/.03)
    report={'checkpoint':str(ck),'engine':'native MuJoCo (independent of training GPU dynamics)',
            'policy_mode':'sampled' if args.stochastic else 'mean',
            'initial_seed':args.seed,
            'task':envcfg.get('task','balance'),'commanded_forward_speed_mps':env.forward_speed if env.rolling else 0,
            'duration_s':args.seconds,'noise_multiplier':args.noise,'push_force_N':args.push,
            'push_duration_s':.1,'training':False,'episodes':result,
            'survival_success_rate':float(np.mean([not x['task_failed'] for x in result])),
            'mean_survival_s':float(np.mean([x['survival_s'] for x in result])),
            'five_finger_success_rate':float(np.mean([not x['task_failed'] and x['all_five_fraction']>=.8 and x['palm_contact_fraction']==0 for x in result]))}
    if env.rolling:
        report.update({'rolling_success_rate':float(np.mean([x['rolling_success'] for x in result])),
            'mean_forward_distance_m':float(np.mean([x['forward_distance_m'] for x in result]))})
    dest=Path(args.output) if args.output else ck.parent/(artifact_stem+'_native_eval.json')
    dest.write_text(json.dumps(report,indent=2));print('SAVED',dest,flush=True)

if __name__=='__main__':main()
