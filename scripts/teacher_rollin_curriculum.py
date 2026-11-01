"""Generate moving training starts through real GPU rollouts outside PPO.

This changes training reset distribution only. Deployment remains the ordinary
history policy; independent evaluations always begin at the original pose.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from policy_loader import load_controller


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


class TeacherRollinCurriculum:
    def __init__(self,env,checkpoint,sha256,run,block_updates):
        checkpoint=Path(checkpoint).resolve()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==sha256
        controller=load_controller(checkpoint)
        self.teacher=controller.policy.to(env.device).eval()
        for parameter in self.teacher.parameters():parameter.requires_grad_(False)
        for key in ('initial_pose','hand_inertia_mode','action_scale','action_filter_alpha',
                    'pd_derivative_gain','free_heading','heading_observations','command_path_limit',
                    'command_target_observation','contact_load_observation','history_observation',
                    'filtered_velocity_observations','velocity_reward_tau'):
            assert controller.envcfg[key]==env.cfg[key],key
        assert env.cfg['physics_backend']=='warp' and env.dt==.01
        assert not env.cfg['push_force'] and not env.cfg['recovery_reset_probability']
        assert env.cfg['reset_velocity']==0
        assert env.cfg['target_speed']==.1 and env.cfg['task']=='teleop'
        for group in env.groups:
            assert not hasattr(group,'teacher_start_mask')
            group.teacher_start_mask=torch.zeros(group.num_envs,dtype=torch.bool,device=env.device)
        self.frozen={k:v.clone() for k,v in self.teacher.state_dict().items()}
        self.run=Path(run);self.records=[]
        self.metadata=dict(teacher_checkpoint=str(checkpoint),teacher_sha256=sha256,
            prefix_seconds=2.,prefix_stop_seconds=1.,fraction_of_main_worlds=.5,
            minimum_aligned_speed_mps=.01,assisted_command_hold_seconds=20.,
            block_updates=block_updates,teacher_environment_transitions=0,
            teacher_steps_in_PPO=0,student_only_inference=True,
            original_pose_evaluation_required=True,goal_achieved=False)

    # PPO advances the simulator under inference_mode; some environment
    # buffers are therefore inference tensors after its first rollout.
    @torch.inference_mode()
    def prepare(self,runner):
        env=runner.env
        assert runner.alg.storage.step==0,'Finish PPO returns/update before changing reset distribution'
        before=copy.deepcopy(runner.alg.save())
        learner_steps=env.global_steps
        recent=list(env.recent)
        auto=env.base.cfg['auto_curriculum'];env.base.cfg['auto_curriculum']=False
        counters=[];log_values=[]
        for group in env.groups:
            assert not group.external_command
            counters.append({k:getattr(group,k).clone() for k in
                ('practice_requested_steps','practice_applied_steps','practice_terminations')})
            log_values.append({k:v for k,v in vars(group).items()
                               if (k.startswith('last_') or k=='completed') and isinstance(v,(int,float))})
        base_logs={k:v for k,v in vars(env.base).items() if k.startswith('last_') and isinstance(v,(int,float))}
        all_ids=torch.arange(env.num_envs,device=env.device)
        env.reset(all_ids)
        directions=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],device=env.device)
        local_directions=[];candidates=[]
        offset=0
        for group in env.groups:
            group.external_command=True
            group.command_targets.zero_();group.command_prepared.fill_(False)
            ids=torch.arange(group.num_envs,device=env.device)
            local_directions.append(directions[ids%4])
            main=ids[group.startup_env_count:]
            chosen=main[torch.randperm(len(main),device=env.device)[:len(main)//2]]
            candidates.append(chosen+offset);offset+=group.num_envs
        candidates=torch.cat(candidates)
        from history_lifecycle import pack_frame
        env.history.reset(all_ids,pack_frame(env.get_observations()))
        failed=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        for step in range(200):
            for group,direction in zip(env.groups,local_directions):
                group.command_targets.copy_(direction*(0. if step<100 else .1))
                group.command_prepared.fill_(False)
            actions=self.teacher(env.get_observations(),stochastic_output=False)
            assert torch.isfinite(actions).all()
            _,_,done,extras=env.step(actions)
            failed |= done.bool()
            assert int(extras['log']['Balance/nonfinite_world_count'])==0
        velocity=torch.cat([g.reward_ball_velocity for g in env.groups])
        direction=torch.cat(local_directions)
        aligned=(velocity*direction).sum(-1)
        eligible=candidates[(~failed[candidates]) & (aligned[candidates]>.01)]
        assert len(eligible)>0,'No usable moving starts: do not silently run an unchanged standing trial'
        pure_mask=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);pure_mask[eligible]=False
        pure_ids=all_ids[pure_mask]
        historical=env.history.frames[eligible].clone();cursor=env.history.cursor
        physical=[];offset=0
        for group in env.groups:
            ids=eligible[(eligible>=offset)&(eligible<offset+group.num_envs)]-offset
            physical.append((ids,{k:getattr(group.sim.data,k)[ids].clone() for k in ('qpos','qvel','ctrl')},
                             {k:getattr(group,k)[ids].clone() for k in ('actions','previous_target_rate','reward_ball_velocity')}))
            offset+=group.num_envs
        # Ordinary resets restore all controller/history buffers for pure starts.
        # Moving starts retain their physically integrated state and history.
        env.reset(pure_ids)
        past_slots=[i for i in range(env.history.capacity) if i!=cursor]
        assert torch.equal(env.history.frames[eligible][:,past_slots],historical[:,past_slots])
        reports=[];offset=0
        for index,(group,(ids,states,buffers)) in enumerate(zip(env.groups,physical)):
            for k,v in states.items():assert torch.equal(getattr(group.sim.data,k)[ids],v),k
            for k,v in buffers.items():assert torch.equal(getattr(group,k)[ids],v),k
            local_pure=(pure_ids[(pure_ids>=offset)&(pure_ids<offset+group.num_envs)]-offset)
            assert not group.sim.data.qvel[local_pure].any() and not group.actions[local_pure].any()
            assert not group.reward_ball_velocity[local_pure].any()
            assert not group.sim.data.xfrc_applied.any() and not group.sim.data.qfrc_applied.any()
            # A new student episode starts here; teacher rewards and durations
            # never enter episode statistics, curriculum gates or PPO storage.
            for key in ('episode_length_buf','returns','tracking_sum','command_error_sum'):
                getattr(group,key).zero_()
            group.command_countdown[ids]=2000
            group.teacher_start_mask.zero_();group.teacher_start_mask[ids]=True
            group.external_command=False;group.command_prepared.fill_(False)
            for key,value in counters[index].items():getattr(group,key).copy_(value)
            for key,value in log_values[index].items():setattr(group,key,value)
            reports.append(dict(physics_timestep_s=group.active_physics_timestep,
                original_pose_worlds=len(local_pure),assisted_worlds=len(ids),
                per_direction=[dict(direction=d,worlds=int((ids%4==d).sum()),
                    mean_aligned_velocity_mps=float(aligned[ids[ids%4==d]+offset].mean()) if (ids%4==d).any() else None)
                    for d in range(4)]))
            offset+=group.num_envs
        env.global_steps=learner_steps
        env.recent.clear();env.recent.extend(recent)
        env.base.cfg['auto_curriculum']=auto
        for key,value in base_logs.items():setattr(env.base,key,value)
        assert runner.alg.storage.step==0 and equal_tree(before,runner.alg.save())
        assert all(torch.equal(value,self.teacher.state_dict()[key]) for key,value in self.frozen.items())
        # Forced block boundaries are truncations after a finished PPO update,
        # not fabricated successful episodes in the logging buffers.
        for key in ('cur_reward_sum','cur_episode_length'):
            getattr(runner.logger,key).zero_()
        self.metadata['teacher_environment_transitions']+=200*env.num_envs
        record=dict(block=len(self.records),time_unix=time.time(),
            next_student_update=runner.current_learning_iteration,
            teacher_control_steps=200,learner_control_steps_before=learner_steps,
            learner_control_steps_after=env.global_steps,candidate_worlds=len(candidates),
            assisted_worlds=len(eligible),original_pose_worlds=len(pure_ids),
            prefix_terminated_worlds=int(failed.sum()),groups=reports,
            actor_critic_optimizer_unchanged=True,teacher_unchanged=True,
            PPO_storage_empty=True,live_history_preserved=True,pure_starts_zero_velocity=True,
            assisted_starts_excluded_from_curriculum=True,
            external_forces=False,goal_achieved=False)
        self.records.append(record)
        from train_smooth_from_scratch import write_json
        write_json(self.run/'rollin_audit.json',dict(metadata=self.metadata,blocks=self.records))
        print('TEACHER_ROLLIN',json.dumps(record),flush=True)
        return env.get_observations()


def learn_with_rollin(runner,curriculum,iterations,block_updates):
    remaining=iterations
    start=runner.current_learning_iteration
    while remaining:
        curriculum.prepare(runner)
        runner.branch_metadata['teacher_rollin_curriculum']=dict(curriculum.metadata)
        count=min(remaining,block_updates)
        steps=runner.env.global_steps
        runner.learn(count,init_at_random_ep_len=False)
        assert runner.env.global_steps-steps==count*runner.cfg['num_steps_per_env']
        assert runner.alg.storage.step==0
        # TensorBoard's stop_logging_writer does not close SummaryWriter.
        # Close it explicitly before the next learn call creates another.
        if runner.logger.writer is not None:runner.logger.writer.close()
        remaining-=count
        if remaining:runner.current_learning_iteration+=1
    assert runner.current_learning_iteration==start+iterations-1
