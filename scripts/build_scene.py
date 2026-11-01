"""Build an untrained, floating-base Revo3 right hand standing on a free ball."""
from pathlib import Path
import json, math, os, copy
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / 'assets/brainco-description/revo3_system/mjcf/revo3_right.xml'
RADIUS = .12
FINGERS = ['thumb', 'index', 'middle', 'ring', 'little']

def hand_spec():
    spec = mujoco.MjSpec.from_file(str(SOURCE))
    spec.body('right_hand_base_link').add_freejoint(name='floating_wrist')
    spec.option.cone=mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio=1
    # The converted palm collision hull overlaps adjacent knuckle housings.
    # Exclude only these mechanical neighbors, retaining all finger-finger contacts.
    for link in ['thumb_CMR','thumb_MCP','index_MCP','middle_MCP','ring_MCP','little_MCP']:
        for palm in ['right_hand_base_link','right_plam_link','right_plam_touch_link','right_plam_rubber_link']:
            spec.add_exclude(bodyname1=palm,bodyname2='right_'+link+'_link')
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.damping[:] = .04
            j.armature = .00005
    for g in spec.geoms:
        if g.group == 3:
            g.friction = [1.1, .005, .0002]
            g.condim = 4
    return spec

def ball_spec():
    s = mujoco.MjSpec()
    b = s.worldbody.add_body(name='ball')
    b.add_freejoint(name='free_ball')
    b.add_geom(name='surface', type=mujoco.mjtGeom.mjGEOM_SPHERE,
               size=[RADIUS,0,0], mass=.62, rgba=[1,.32,.035,1],
               friction=[1,.005,.0002], condim=4)
    # Three thin great-circle ribbons are visual only; collision remains one sphere.
    for axis in range(3):
        vs, fs = [], []
        for i in range(160):
            t = 2*np.pi*i/160
            for sign in [-1,1]:
                v = np.array([np.cos(t)*np.cos(.011), np.sin(t)*np.cos(.011), sign*np.sin(.011)])*(RADIUS+.0001)
                vs.extend(np.roll(v,axis))
            a=2*i; n=2*((i+1)%160)
            fs.extend([a,n,a+1,a+1,n,n+1])
        s.add_mesh(name=f'seam{axis}', uservert=vs, userface=fs)
        b.add_geom(name=f'seam{axis}', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'seam{axis}',
                   contype=0, conaffinity=0, group=2, mass=0, rgba=[.035,.026,.018,1])
    return s

def decorate(s):
    s.option.timestep = .002
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.gravity = [0,0,-9.81]
    s.visual.global_.offwidth = 1600
    s.visual.global_.offheight = 1400
    s.visual.headlight.ambient = [.3,.3,.3]
    s.add_texture(name='tiles', type=mujoco.mjtTexture.mjTEXTURE_2D,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                  width=512,height=512,rgb1=[.13,.20,.26],rgb2=[.19,.28,.35])
    mat=s.add_material(name='floor_mat',texrepeat=[6,6],texuniform=True,reflectance=.06)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]='tiles'
    s.visual.quality.numslices=80
    s.visual.quality.numstacks=40
    s.add_texture(name='sky',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,width=512,height=3072,rgb1=[.10,.17,.24],rgb2=[.30,.40,.50])
    s.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[3,3,.02],
                         material='floor_mat',friction=[1,.005,.0001],condim=4)
    s.worldbody.add_light(name='key',pos=[-.5,-.6,1.5],dir=[.3,.4,-1],diffuse=[.8,.8,.8],castshadow=True)
    s.worldbody.add_light(name='fill',pos=[.7,.4,1.0],dir=[-.5,-.3,-1],diffuse=[.5,.5,.5],castshadow=False)


def build():
    cfg=SceneCfg(num_envs=1,env_spacing=1,entities={
        'hand':EntityCfg(spec_fn=hand_spec),
        'ball':EntityCfg(spec_fn=ball_spec,init_state=EntityCfg.InitialStateCfg(pos=(0,0,RADIUS)))},spec_fn=decorate)
    scene=Scene(cfg,device='cuda:0')
    m=scene.compile(); d=mujoco.MjData(m)
    joints=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
    qa=np.array([m.jnt_qposadr[j] for j in joints])
    jnames=[m.joint(j).name for j in joints]
    handadr=m.jnt_qposadr[m.joint('hand/floating_wrist').id]
    balladr=m.jnt_qposadr[m.joint('ball/free_ball').id]
    bg=m.geom('ball/surface').id
    distal=[[g for g in range(m.ngeom) if m.geom(g).name.startswith('hand/right_'+f+'_DIP') and m.geom_group[g]==3] for f in FINGERS]
    other=[g for g in range(m.ngeom) if m.geom(g).name.startswith('hand/') and m.geom_group[g]==3 and g not in sum(distal,[])]
    tipbody=[m.body('hand/right_'+f+'_tip_Link').id for f in FINGERS]
    handbody=m.body('hand/right_hand_base_link').id
    seed=[]
    for n in jnames:
        typ=n.split('_')[-2]
        seed.append({'MPR':0.,'MCP':.7,'PIP':.55,'DIP':.3,'CMP':1.2,'CMR':.5}[typ])
    seed=np.array(seed)
    x0=np.r_[[-.09,0,.325,0],seed]
    lo=np.r_[[-.16,-.05,.27,-.35],m.jnt_range[joints,0]+.001]
    hi=np.r_[[.01,.05,.41,.35],m.jnt_range[joints,1]-.001]
    def pose(x):
        d.qpos[handadr:handadr+3]=x[:3]
        quat=Rotation.from_euler('y',np.pi/2+x[3]).as_quat()
        d.qpos[handadr+3:handadr+7]=quat[[3,0,1,2]]
        d.qpos[qa]=x[4:]
        d.qpos[balladr:balladr+7]=[0,0,RADIUS,1,0,0,0]
        mujoco.mj_forward(m,d)
    def distances(gs):
        return np.array([mujoco.mj_geomDistance(m,d,g,bg,.5,None) for g in gs])
    def residual(x):
        pose(x)
        dd=np.array([min(distances(gs)) for gs in distal])
        od=distances(other)
        tips=d.xpos[tipbody]
        com=d.subtree_com[handbody]
        return np.r_[800*(dd+.00008), 1000*np.minimum(od-.001,0),
                     200*np.minimum(tips[:,2]-.205,0),
                     20*com[:2], .06*(x[4:]-seed), .2*x[3]]
    result=least_squares(residual,x0,bounds=(lo,hi),max_nfev=220,ftol=1e-8,xtol=1e-8,gtol=1e-7)
    pose(result.x)
    print('OPT',result.cost,result.nfev,flush=True)
    print('FINGERS',dict(zip(FINGERS,[float(min(distances(gs))) for gs in distal])),flush=True)
    print('ROOT',result.x[:4],flush=True)
    # Save a native keyframe and entity initialization for exact reproduction.
    (OUT/'scene').mkdir(exist_ok=True)
    xml_path=OUT/'scene/scene.xml'
    xml_path.write_text(scene.spec.to_xml())
    tree=ET.parse(xml_path); xml=tree.getroot()
    for asset in list(xml.find('asset')):
        if asset.get('file'):
            src=Path(asset.get('file'))
            if not src.is_absolute(): src=(SOURCE.parent/src).resolve()
            asset.set('file',os.path.relpath(src,xml_path.parent))
    old=xml.find('keyframe')
    if old is not None: xml.remove(old)
    keys=ET.SubElement(xml,'keyframe')
    ET.SubElement(keys,'key',name='initial_five_finger_support',qpos=' '.join(map(str,d.qpos)))
    ET.indent(tree); tree.write(xml_path,encoding='unicode')
    contacts=[]
    for c in d.contact:
        contacts.append({'geom1':m.geom(c.geom1).name,'geom2':m.geom(c.geom2).name,'distance_m':float(c.dist)})
    report={'trained':False,'simulation_steps':0,'source_commit':'f197d9d720b1b106a3298e2438ff07e055bf15ec',
            'ball_radius_m':RADIUS,'ball_mass_kg':.62,
            'hand_root_pos':d.qpos[handadr:handadr+3].tolist(),
            'hand_root_quat_wxyz':d.qpos[handadr+3:handadr+7].tolist(),
            'joint_positions_rad':dict(zip(jnames,d.qpos[qa].tolist())),
            'fingertip_ball_gap_m':dict(zip(FINGERS,[float(min(distances(gs))) for gs in distal])),
            'other_ball_min_gap_m':float(min(distances(other))),
            'hand_com_world':d.subtree_com[handbody].tolist(),'contacts':contacts,
            'nq':m.nq,'nv':m.nv,'num_hand_joints':len(joints)}
    (OUT/'initial_state.json').write_text(json.dumps(report,indent=2))
    np.save(OUT/'initial_qpos.npy',d.qpos)
    render(m,d)
    return scene,m,d

def render(m,d):
    renderer=mujoco.Renderer(m,height=1400,width=1600)
    opt=mujoco.MjvOption(); opt.geomgroup[3]=0
    for name,az,el,dist,target in [
        ('initial_oblique',135,-22,.79,[0,0,.205]),
        ('initial_front',180,-25,.75,[0,0,.205]),
        ('initial_opposite',-45,-28,.74,[0,0,.205]),
        ('initial_top',90,-75,.65,[0,0,.21])]:
        cam=mujoco.MjvCamera(); cam.azimuth=az; cam.elevation=el; cam.distance=dist; cam.lookat[:]=target
        renderer.update_scene(d,camera=cam,scene_option=opt)
        Image.fromarray(renderer.render()).save(OUT/(name+'.png'))
    renderer.close()

if __name__=='__main__':
    build()
