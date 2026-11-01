"""Local-only web keyboard controller; simulation runs on the remote GPU host."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,io,json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch
import mujoco
from PIL import Image
from policy_loader import load_controller as PolicyController
from balance_model import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--port',type=int,default=8765);p.add_argument('--follow-latest',action='store_true')
    p.add_argument('--command-audit',type=Path,help='Optional JSONL log for isolated keyboard transport verification.')
    a=p.parse_args()
    audit=a.command_audit.open('x',buffering=1) if a.command_audit else None
    torch.set_num_threads(1);c=PolicyController(a.checkpoint);env=c.env;env.reset(noise=0.)
    lock=threading.Lock();shared={'target':[0.,0.],'last_command':0.,'reset':False,'paused':True,'toggle':False,'frame':b'', 'state':{}}
    def simulate():
        renderer=mujoco.Renderer(env.m,720,960);opt=mujoco.MjvOption();opt.geomgroup[3]=0
        cam=mujoco.MjvCamera();cam.azimuth=-45;cam.elevation=-25;cam.distance=.85
        frame_idx=0;failed=False;deadline=time.monotonic()
        try:
            while True:
                with lock:
                    target=shared['target'][:] if time.monotonic()-shared['last_command']<.75 else [0.,0.]
                    reset=shared['reset'];shared['reset']=False
                    if shared['toggle']:shared['paused']=not shared['paused'];shared['toggle']=False
                    paused=shared['paused']
                if reset:
                    if a.follow_latest:
                        try:
                            progress=json.loads((c.checkpoint.parent/'progress.json').read_text())
                            candidate=c.checkpoint.parent/Path(progress['checkpoint']).name
                            if candidate!=c.checkpoint:
                                saved=torch.load(candidate,map_location='cpu',weights_only=False)
                                c.policy.load_state_dict(saved['actor_state_dict']);c.checkpoint=candidate
                                c.trained_speed_limit=(saved.get('infos') or {}).get('teleop_curriculum',{}).get('speed_limit',0.)
                        except (OSError,ValueError,KeyError):pass
                    env.reset(noise=0.);failed=False
                    with lock:shared['paused']=True
                    paused=True
                env.set_command(*target)
                if not failed and not paused:env.step(c.action())
                met=env.metrics();failed=met['failed'];met.update(time_s=float(env.d.time),checkpoint=f'{c.checkpoint.parent.name}/{c.checkpoint.name}',trained_speed_limit_mps=c.trained_speed_limit,paused=paused,
                    target_speed_mps=env.envcfg['target_speed'],
                    follow_latest=a.follow_latest,physics_timestep_s=float(env.m.opt.timestep),control_timestep_s=env.dt,
                    load_bearing_fingers=int(sum(f>.15 for f in met['total_finger_normal_forces_N'])))
                if frame_idx%7==0:
                    cam.lookat[:]=[env.d.qpos[env.ballqa],env.d.qpos[env.ballqa+1],.21]
                    renderer.update_scene(env.d,camera=cam,scene_option=opt)
                    b=io.BytesIO();Image.fromarray(renderer.render()).save(b,format='JPEG',quality=85)
                    with lock:shared['frame']=b.getvalue()
                with lock:shared['state']=met
                frame_idx+=1;deadline+=env.dt;delay=deadline-time.monotonic()
                if delay>0:time.sleep(delay)
                elif delay<-.2:deadline=time.monotonic()
        except Exception as e:
            import traceback;traceback.print_exc()
            with lock:shared['state']={'error':str(e)}
        finally:renderer.close()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,data,ctype='application/json',status=200):
            self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            if self.path=='/':return self.respond((ROOT/'web/teleop.html').read_bytes(),'text/html; charset=utf-8')
            if self.path.startswith('/frame'):
                with lock:data=shared['frame']
                return self.respond(data,'image/jpeg')
            if self.path=='/state':
                with lock:data=json.dumps(shared['state']).encode()
                return self.respond(data)
            self.respond(b'{}',status=404)
        def do_POST(self):
            if self.path=='/toggle':
                with lock:shared['toggle']=True
                return self.respond(b'{}')
            if self.path=='/reset':
                with lock:shared.update(reset=True,target=[0.,0.])
                return self.respond(b'{}')
            if self.path=='/command':
                try:
                    size=int(self.headers.get('Content-Length',0));assert 0<size<1024
                    body=json.loads(self.rfile.read(size));target=np.array([float(body['vx']),float(body['vy'])]);assert np.isfinite(target).all()
                    target*=min(1.,env.envcfg['target_speed']/max(np.linalg.norm(target),1e-9))
                    with lock:
                        shared.update(target=target.tolist(),last_command=time.monotonic())
                        if audit:audit.write(json.dumps({'time_unix':time.time(),'target_mps':target.tolist()})+'\n')
                    return self.respond(b'{}')
                except (ValueError,KeyError,AssertionError,TypeError):return self.respond(b'{}',status=400)
            self.respond(b'{}',status=404)
    threading.Thread(target=simulate,daemon=True).start()
    print(f'TELEOP http://127.0.0.1:{a.port} checkpoint={a.checkpoint}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
