"""Training-only command counters at the reward boundary, before automatic resets.

These count actual control intervals, not independent resampling draws. They do
not alter commands, rewards, physical state, observations or policy inputs.
"""
import types

import torch


class CommandExposure:
    def __init__(self, env):
        self.env=env
        self.counts=[]
        self.masks=[]
        for index,group in enumerate(env.groups):
            assert not hasattr(group,'teacher_start_mask'), 'Teacher roll-in requires separate exclusion accounting'
            self.counts.append(torch.zeros((3,2,4),dtype=torch.long,device=env.device))
            ids=torch.arange(group.num_envs,device=env.device)
            self.masks.append((ids<group.startup_env_count,
                               (ids>=group.startup_env_count)&~group.practice_env_mask,
                               group.practice_env_mask))
            assert torch.stack(self.masks[-1]).sum(0).eq(1).all()
            original=group.command_reward

            def capture(this,q,v,found,previous_ball_xy,index=index,original=original):
                # Commands have been sampled and slew-limited for the interval;
                # terminal physical samples have not yet reset either command.
                for kind,values in enumerate((this.command_targets,this.commands)):
                    speed=values.norm(dim=-1)
                    category=torch.where(speed<1e-6,0,
                        torch.where((speed-this.cfg['target_speed']).abs()<1e-6,2,
                            torch.where(speed<this.cfg['target_speed'],1,3)))
                    for population,mask in enumerate(self.masks[index]):
                        self.counts[index][population,kind]+=torch.bincount(category[mask],minlength=4)
                return original(q,v,found,previous_ball_xy)

            group.command_reward=types.MethodType(capture,group)

    def report(self):
        rows=[]
        for group,counts,masks in zip(self.env.groups,self.counts,self.masks):
            populations={}
            for index,name in enumerate(('startup','ordinary','practice')):
                a,b=counts[index].cpu().tolist()
                assert sum(a)==sum(b)
                populations[name]=dict(worlds=int(masks[index].sum()),world_control_steps=sum(a),
                    requested=dict(zip(('stop','below_target','at_target','above_target'),a)),
                    applied=dict(zip(('stop','below_target','at_target','above_target'),b)))
            rows.append(dict(physics_timestep_s=group.active_physics_timestep,populations=populations))
        total=sum(p['world_control_steps'] for r in rows for p in r['populations'].values())
        requested=sum(p['requested']['at_target'] for r in rows for p in r['populations'].values())
        applied=sum(p['applied']['at_target'] for r in rows for p in r['populations'].values())
        return dict(world_control_steps=total,requested_at_target_steps=requested,
                    applied_at_target_steps=applied,
                    requested_at_target_fraction=requested/total if total else None,
                    applied_at_target_fraction=applied/total if total else None,
                    control_timestep_s=self.env.dt,groups=rows,
                    scope='Current process only; command exposure is not actual ball speed or policy qualification.')
