"""Optional passive control-boundary snapshots for numerical training faults.

Records available arrays and actor weights, not a complete resumable solver state.
The original reward, termination and reset handling remain responsible for faults.
"""
import hashlib
import json
from pathlib import Path
import time
import types

import torch


class TrainingFailureCapture:
    def __init__(self, env, output, policy, max_events=4):
        self.env = env
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.policy = policy
        self.max_events = max_events
        assert max_events > 0
        self.events = []
        self.steps = 0
        self.before = []
        self.observations = None
        self.original_step = env.step
        for index, group in enumerate(env.groups):
            original = group.command_reward

            def observe(this, q, v, found, previous, original=original, index=index):
                reward = original(q, v, found, previous)
                if len(self.events) < self.max_events:
                    bad = (~torch.isfinite(q).all(-1) | ~torch.isfinite(v).all(-1)
                           | ~torch.isfinite(reward))
                    if bool(bad.any()):
                        ids = bad.nonzero().flatten()
                        offset = sum(g.num_envs for g in env.groups[:index])
                        global_ids = ids + offset
                        event = dict(index=len(self.events), time_unix=time.time(),
                            captured_control_step=self.steps, group=index,
                            group_global_step=this.global_steps,
                            physics_timestep_s=this.active_physics_timestep,
                            local_world_ids=ids.cpu().tolist(),
                            global_world_ids=global_ids.cpu().tolist(),
                            before_qv_finite=all(bool(torch.isfinite(self.before[index][key][ids]).all())
                                                for key in ('qpos', 'qvel')),
                            input_actions_finite=bool(torch.isfinite(self.before[index]['input_actions'][ids]).all()),
                            bad_q_worlds=int((~torch.isfinite(q[ids]).all(-1)).sum()),
                            bad_v_worlds=int((~torch.isfinite(v[ids]).all(-1)).sum()),
                            bad_command_reward_worlds=int((~torch.isfinite(reward[ids])).sum()))
                        path = self.output / f'event_{len(self.events)}.pt'
                        torch.save(dict(event=event, env_config=dict(this.cfg),
                            before={k:value[ids].detach().cpu() for k,value in self.before[index].items()},
                            policy_observations={k:value[global_ids].detach().cpu()
                                                 for k,value in self.observations.items()},
                            actor_state_dict={k:value.detach().cpu() for k,value in self.policy.state_dict().items()},
                            after_q=q[ids].detach().cpu(), after_v=v[ids].detach().cpu(),
                            after_command_reward=reward[ids].detach().cpu(),
                            after_ctrl=this.sim.data.ctrl[ids].detach().cpu(),
                            after_sensordata=this.sim.data.sensordata[ids].detach().cpu()), path)
                        event.update(snapshot=str(path), snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        self.events.append(event)
                        self.publish()
                return reward

            group.command_reward = types.MethodType(observe, group)

        def step(this, actions):
            if len(self.events) < self.max_events:
                self.observations = {k:v.clone() for k,v in this.get_observations().items()}
                self.before = []
                offset = 0
                for group in this.groups:
                    fields = {}
                    for name in ('qpos', 'qvel', 'ctrl', 'qacc', 'qacc_warmstart', 'act', 'sensordata'):
                        value = getattr(group.sim.data, name, None)
                        cloned = value.clone() if value is not None and hasattr(value, 'clone') else None
                        if isinstance(cloned, torch.Tensor) and cloned.ndim and cloned.shape[0] == group.num_envs:
                            fields[name] = cloned
                    assert {'qpos', 'qvel', 'ctrl'} <= fields.keys()
                    fields.update(input_actions=actions[offset:offset+group.num_envs].clone(),
                        filtered_actions=group.actions.clone(), commands=group.commands.clone(),
                        command_targets=group.command_targets.clone(),
                        previous_target_rate=group.previous_target_rate.clone(),
                        target_rate_valid=group.target_rate_valid.clone(),
                        measured_velocity=group.reward_ball_velocity.clone())
                    self.before.append(fields)
                    offset += group.num_envs
                assert offset == this.num_envs
            result = self.original_step(actions)
            self.steps += 1
            if self.steps % 128 == 0:
                self.publish()
            return result

        # HistoryTeleopEnv forwards normal assignments to its base; replacing
        # base.step with a wrapper of history.step would recurse indefinitely.
        object.__setattr__(env, 'step', types.MethodType(step, env))
        self.publish()

    def publish(self):
        data = dict(time_unix=time.time(), captured_control_steps=self.steps,
            max_events=self.max_events, events=self.events,
            scope='Passive numerical q/v/command-reward snapshots at control boundaries; '
                  'normal failure/reset handling unchanged. Not full solver-state replay. '
                  'No-event outcome does not certify numerical safety.')
        path = self.output / 'status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(path)
