"""Continue a fresh-training lineage, with explicit backend migration and full Adam restore."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, copy, hashlib, json, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from balance_model import ROOT
from teleop_env import TeleopCfg
from fixed_physics_teleop import FixedPhysicsTeleopEnv
from history_context import HISTORY_CONFIG
from history_lifecycle import HistoryTeleopEnv, pack_frame
from train_smooth_from_scratch import ScratchRunner, write_json


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def lineage_samples(checkpoint, visiting=None):
    """Count transitions across changed batch sizes, verifying every parent link."""
    checkpoint=Path(checkpoint).resolve()
    visiting=set() if visiting is None else visiting
    assert checkpoint not in visiting, 'Checkpoint lineage cycle'
    visiting.add(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    info=saved['infos'];cfg=json.loads((checkpoint.parent/'env_config.json').read_text())
    steps=info['teleop_curriculum']['global_steps']
    if info.get('parent_checkpoint'):
        parent=Path(info['parent_checkpoint'])
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==info['parent_sha256']
        previous=torch.load(parent,map_location='cpu',weights_only=False)
        parent_steps=previous['infos']['teleop_curriculum']['global_steps']
        assert steps>=parent_steps
        total=lineage_samples(parent,visiting)+(steps-parent_steps)*cfg['num_envs']
    else:
        assert info['fresh_training'] and info['pretrained_weights_loaded'] is False
        total=steps*cfg['num_envs']
    if 'lineage_environment_steps' in info:
        assert total==info['lineage_environment_steps']
    return total


class BranchRunner(ScratchRunner):
    def save(self,path,infos=None):
        additional=(self.env.global_steps-self.source_global_steps)*self.env.num_envs
        self.branch_metadata.update(lineage_environment_steps=self.source_environment_steps+additional,
            curriculum_recent=list(self.env.recent))
        if hasattr(self,'command_exposure'):
            exposure=self.command_exposure.report()
            assert exposure['world_control_steps']==additional
            self.branch_metadata['command_exposure']=exposure
        super().save(path,infos)
        p=Path(path).parent/'progress.json';state=json.loads(p.read_text())
        state.update(branch_environment_steps=additional,
            source_environment_steps=self.source_environment_steps,
            environment_steps=self.source_environment_steps+additional,
            parent_checkpoint=self.branch_metadata['parent_checkpoint'],
            velocity_error_huber_weight=self.env.cfg['velocity_error_huber_weight'])
        if hasattr(self,'command_exposure'):state['command_exposure']=exposure
        write_json(p,state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--huber-weight',type=float,choices=[0.,4.,8.,12.,16.],required=True)
    p.add_argument('--continue-lineage',action='store_true',
                   help='Continue a fresh-lineage checkpoint with exactly its saved reward configuration.')
    p.add_argument('--revise-tracking-reward',action='store_true',
                   help='Explicit bounded continuation changing only velocity Huber weight; requires --continue-lineage.')
    p.add_argument('--practice-env-fraction',type=float,
                   help='Explicit bounded training-distribution experiment; retain reward and physical task.')
    p.add_argument('--target-speed-practice',action='store_true',
                   help='Use only the final target speed in existing practice worlds; retain ordinary curriculum and rewards.')
    p.add_argument('--num-envs',type=int,default=256);p.add_argument('--iterations',type=int,default=31)
    p.add_argument('--seed',type=int,default=122);p.add_argument('--device',default='cuda:0')
    p.add_argument('--native-batch-proof',type=Path,
                   help='Verified native batch execution manifest; changes execution only.')
    p.add_argument('--physics-backend',choices=['native','warp'],
                   help='Explicit backend migration; retain model, controller, observations and reward.')
    p.add_argument('--teacher-rollin',type=Path,
                   help='Training-only moving starts from physical teacher rollouts outside PPO.')
    p.add_argument('--teacher-rollin-sha256')
    p.add_argument('--rollin-block-updates',type=int,default=4)
    a=p.parse_args();assert a.num_envs>=32 and a.num_envs%16==0
    assert 1<=a.iterations<=(501 if a.continue_lineage else 101)
    if a.practice_env_fraction is not None:
        assert a.continue_lineage and 0<a.practice_env_fraction<=.75 and a.iterations<=101
    if a.target_speed_practice:
        assert a.continue_lineage and a.iterations<=201
        assert a.practice_env_fraction is None and not a.revise_tracking_reward
        assert a.teacher_rollin is None and a.physics_backend is None and a.native_batch_proof is None
    if a.revise_tracking_reward:
        assert a.continue_lineage and a.iterations<=101 and a.practice_env_fraction is None
    if a.teacher_rollin is not None:
        assert a.continue_lineage and a.teacher_rollin_sha256 and a.iterations<=32
        assert 1<=a.rollin_block_updates<=32 and a.native_batch_proof is None
        assert not a.revise_tracking_reward and a.practice_env_fraction is None
    else:assert a.teacher_rollin_sha256 is None
    assert Path(a.run).name==a.run and a.run not in ('.','..')
    source=a.source.resolve();assert hashlib.sha256(source.read_bytes()).hexdigest()==a.source_sha256
    saved=torch.load(source,map_location='cpu',weights_only=False)
    assert saved['infos']['fresh_training'] and not saved['infos'].get('ground_teacher_loaded',False)
    if not a.continue_lineage:assert saved['infos']['pretrained_weights_loaded'] is False
    cfg=json.loads((source.parent/'env_config.json').read_text());rcfg=json.loads((source.parent/'runner_config.json').read_text())
    assert cfg.pop('history_observation')==HISTORY_CONFIG
    source_huber_weight=cfg.get('velocity_error_huber_weight',0.)
    if a.revise_tracking_reward:
        assert source_huber_weight < a.huber_weight <= 16.
        assert a.physics_backend in (None,cfg['physics_backend'])
    elif a.continue_lineage:assert a.huber_weight==source_huber_weight
    assert (cfg['body_motion_weight'],cfg['joint_target_rate_weight'],cfg['joint_target_acceleration_weight'])==(1.5,.02,.00002)
    assert cfg['target_speed']==.1 and cfg['initial_pose']=='opposed_thumb'
    assert cfg['action_filter_alpha']==.3 and cfg['pd_derivative_gain']==.18 and not cfg['push_force']
    run=ROOT/'runs'/a.run;run.mkdir(exist_ok=False)
    curriculum=saved['infos']['teleop_curriculum'];source_steps=curriculum['global_steps']
    source_samples=lineage_samples(source)
    cfg.update(num_envs=a.num_envs,seed=a.seed,velocity_error_huber_weight=a.huber_weight)
    source_practice_fraction=cfg.get('practice_env_fraction',0.)
    source_practice_speeds=list(cfg.get('practice_speeds',[]))
    if a.practice_env_fraction is not None:
        assert cfg.get('practice_speeds') and cfg['target_speed']==.1
        cfg['practice_env_fraction']=a.practice_env_fraction
    if a.target_speed_practice:
        assert cfg['physics_backend']=='warp' and cfg['practice_env_fraction']>0
        assert source_practice_speeds and cfg['target_speed']==.1
        cfg['practice_speeds']=[cfg['target_speed']]
    source_backend=cfg['physics_backend']
    if a.physics_backend is not None:cfg['physics_backend']=a.physics_backend
    if a.teacher_rollin is not None:assert cfg['physics_backend']==source_backend=='warp'
    rcfg.update(seed=a.seed,run_name=a.run,max_iterations=a.iterations,save_interval=5,upload_model=False)
    assert len(saved['optimizer_state_dict']['param_groups'])==1
    source_lr=saved['optimizer_state_dict']['param_groups'][0]['lr']
    rcfg['algorithm']['learning_rate']=source_lr
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    for name in ('pose_initial_qpos.npy','pose_initial_state.json','pose_scene.xml','numeric_precision.json'):
        shutil.copy2(source.parent/name,run/name)
    snapshot=run/'source';snapshot.mkdir()
    for f in (ROOT/'scripts').glob('*.py'):shutil.copy2(f,snapshot/f.name)
    native_execution=None
    if a.native_batch_proof is not None:
        assert cfg['physics_backend']=='native'
        proof=json.loads(a.native_batch_proof.read_text())
        assert proof['passed'] and proof['full_environment_worlds']==1024
        assert proof['failure_reset_parity'] and proof['timeout_reset_parity']
        for item in proof['files']:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
        native_library=run/'native_batch.so'
        shutil.copy2(proof['extension_path'],native_library)
        shutil.copy2(a.native_batch_proof,run/'native_batch_proof.json')
        shutil.copy2(ROOT/'scripts/native_batch_step.c',snapshot/'native_batch_step.c')
        native_execution=dict(kind='compiled_native_batch',threads=proof['threads'],
            library_sha256=hashlib.sha256(native_library.read_bytes()).hexdigest(),
            physics_backend_unchanged=True,proof=str(run/'native_batch_proof.json'))
        write_json(run/'native_execution.json',native_execution)
    write_json(run/'env_config.json',{**cfg,'history_observation':HISTORY_CONFIG})
    write_json(run/'runner_config.json',rcfg)
    write_json(run/'launch_args.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    if native_execution is None:
        env=HistoryTeleopEnv(FixedPhysicsTeleopEnv(TeleopCfg(**cfg),a.device))
    else:
        import native_sim
        from native_batch_sim import BatchedNativeSimulation
        original_sim=native_sim.NativeSimulation
        try:
            native_sim.NativeSimulation=lambda *args,**kw:BatchedNativeSimulation(
                *args,library=native_library,batch_threads=native_execution['threads'],**kw)
            env=HistoryTeleopEnv(FixedPhysicsTeleopEnv(TeleopCfg(**cfg),a.device))
        finally:
            native_sim.NativeSimulation=original_sim
    try:
        runner=BranchRunner(env,copy.deepcopy(rcfg),str(run),device=a.device)
        runner.load(str(source),map_location=a.device)
        runner.alg.learning_rate=source_lr
        assert equal_tree(saved['actor_state_dict'],runner.alg.get_policy().state_dict())
        assert equal_tree(saved['critic_state_dict'],runner.alg.critic.state_dict())
        assert equal_tree(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict())
        runner.current_learning_iteration=saved['iter']+1
        env.speed_limit=curriculum['speed_limit'];env.global_steps=source_steps
        env.curriculum_records=copy.deepcopy(curriculum['records']);env.last_promotion_step=curriculum['last_promotion_step']
        restored_recent=saved['infos'].get('curriculum_recent',[])
        env.recent.extend(restored_recent)
        ids=torch.arange(env.num_envs,device=env.device);env.sample_commands(ids)
        env.history.reset(ids,pack_frame(env.get_observations()))
        runner.source_global_steps=source_steps;runner.source_environment_steps=source_samples
        runner.branch_metadata=dict(fresh_training=True,pretrained_weights_loaded=True,
            initialization='continue_fresh_training_lineage',parent_checkpoint=str(source),
            parent_sha256=a.source_sha256,ground_teacher_loaded=False,
            physical_worlds_reinitialized=True,
            reward_revision='smooth_scratch_velocity_huber_v1' if a.huber_weight else 'smooth_from_scratch_v1',
            velocity_error_huber_weight=a.huber_weight)
        runner.branch_metadata.update(source_physics_backend=source_backend,
            physics_backend=cfg['physics_backend'],physics_backend_changed=source_backend!=cfg['physics_backend'])
        runner.branch_metadata['tracking_reward_revision']=dict(
            explicitly_requested=a.revise_tracking_reward,parameter='velocity_error_huber_weight',
            source=source_huber_weight,destination=a.huber_weight,
            changed=source_huber_weight!=a.huber_weight)
        runner.branch_metadata['training_distribution_revision']=dict(
            parameter='practice_env_fraction',source=source_practice_fraction,
            destination=cfg.get('practice_env_fraction',0.),
            changed=source_practice_fraction!=cfg.get('practice_env_fraction',0.))
        if a.target_speed_practice:
            from command_exposure import CommandExposure
            runner.command_exposure=CommandExposure(env)
            runner.branch_metadata['practice_speed_revision']=dict(
                source=source_practice_speeds,destination=cfg['practice_speeds'],
                original_pose_training=True,teacher_rollin_used=False,
                rewards_and_controller_unchanged=True)
        if native_execution is not None:
            runner.branch_metadata['native_execution']=native_execution
        runner.save(run/'branch_initial.pt')
        initial_obs=env.get_observations()
        observation_digest=hashlib.sha256(b''.join(initial_obs[k].detach().cpu().numpy().tobytes()
            for k in sorted(initial_obs.keys()))).hexdigest()
        state_digest=hashlib.sha256(b''.join(g.sim.data.qpos.cpu().numpy().tobytes()
            for g in env.groups)).hexdigest()
        write_json(run/'initialization_audit.json',dict(actor_exact=True,critic_exact=True,adam_exact=True,
            source_learning_rate_restored=source_lr,source_sha256=a.source_sha256,
            source_iteration=saved['iter'],next_update_iteration=runner.current_learning_iteration,
            paired_seed=a.seed,reward_parameter='velocity_error_huber_weight',
            source_physics_backend=source_backend,physics_backend=cfg['physics_backend'],
            physics_backend_changed=source_backend!=cfg['physics_backend'],
            goal_speed_mps=.1,physical_parameters_unchanged=True,
            source_environment_samples=source_samples,
            restored_recent_episode_records=len(restored_recent),
            tracking_reward_revision=runner.branch_metadata['tracking_reward_revision'],
            training_distribution_revision=runner.branch_metadata['training_distribution_revision'],
            practice_worlds=sum(g.practice_env_count for g in env.groups),
            initial_command_target_counts={str(speed):int(torch.isclose(
                torch.cat([g.command_targets.norm(dim=-1) for g in env.groups]),
                torch.tensor(speed,device=env.device),atol=1e-6,rtol=0.).sum())
                for speed in (0.,.025,.05,.075,.1)},
            physical_worlds_reinitialized_at_original_pose=True,
            initial_observation_sha256=observation_digest,initial_qpos_sha256=state_digest,
            native_execution=native_execution))
        write_json(run/'process.json',dict(pid=os.getpid(),start_time=time.time(),num_envs=a.num_envs))
        if a.teacher_rollin is None:
            runner.learn(a.iterations,init_at_random_ep_len=False)
        else:
            from teacher_rollin_curriculum import TeacherRollinCurriculum,learn_with_rollin
            curriculum=TeacherRollinCurriculum(env,a.teacher_rollin,a.teacher_rollin_sha256,
                                              run,a.rollin_block_updates)
            write_json(run/'rollin_curriculum.json',curriculum.metadata)
            learn_with_rollin(runner,curriculum,a.iterations,a.rollin_block_updates)
        write_json(run/'finished.json',dict(iteration=runner.current_learning_iteration,time_unix=time.time()))
    finally:
        for g in env.groups:
            if hasattr(g.sim,'pool'):g.sim.pool.shutdown()

if __name__=='__main__':main()
