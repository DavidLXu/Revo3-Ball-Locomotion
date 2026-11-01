"""Optional control-step history adapters for training and native inference."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from history_context import HISTORY_CONFIG, HistoryContextModel
from temporal_observation_buffer import TemporalObservationBuffer
from teleop_control import NativeTeleop, PolicyController


def pack_frame(obs):
    frame = torch.cat((obs['actor'], obs['command_delta'], obs['contact_load']), -1)
    assert frame.shape[-1] == HISTORY_CONFIG['feature_count']
    return frame


class _Delegated:
    # Forward assignments too: runner curriculum restoration and randomized
    # episode lengths must update the physical environment, not shadow it.
    _local_names = frozenset(('base', 'history', 'cfg'))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __setattr__(self, name, value):
        if name in self._local_names:
            object.__setattr__(self, name, value)
        else:
            setattr(self.base, name, value)

    def _initialize_history(self, obs):
        frame = pack_frame(obs)
        self.history = TemporalObservationBuffer(len(frame), frame.shape[-1],
            past_samples=HISTORY_CONFIG['past_samples'], stride=HISTORY_CONFIG['stride'],
            device=frame.device, dtype=frame.dtype)
        self.history.reset(torch.arange(len(frame), device=frame.device), frame)

    def _attach(self, obs):
        obs = obs.clone(False)
        obs['past_history'] = self.history.read()[:, :-1].flatten(1)
        return obs


class HistoryTeleopEnv(_Delegated):
    """Wrap a fixed-physics batch without changing its reset/command schedule."""
    def __init__(self, base):
        assert base.cfg['contact_load_observation']
        assert base.cfg['command_target_observation']
        assert abs(base.dt - .01) < 1e-12
        assert not base.cfg.get('recovery_reset_probability', 0)
        self.base = base
        self.cfg = copy.deepcopy(base.cfg)
        self.cfg['history_observation'] = dict(HISTORY_CONFIG)
        self._initialize_history(base.get_observations())

    def get_observations(self):
        obs = self.base.get_observations()
        self.history.replace_current(pack_frame(obs))
        return self._attach(obs)

    def step(self, actions):
        # Record any command change made since the previous observation. The
        # base command preparation is idempotent within a control interval.
        self.get_observations()
        obs, reward, done, extras = self.base.step(actions)
        frame = pack_frame(obs)
        self.history.advance(frame)
        ids = done.nonzero().flatten()
        self.history.reset(ids, frame[ids])
        return self._attach(obs), reward, done, extras

    def reset(self, ids):
        self.base.reset(ids)
        for group in self.base.groups:
            group.sim.forward()
        obs = self.base.get_observations()
        frame = pack_frame(obs)
        self.history.replace_current(frame)
        self.history.reset(ids, frame[ids])


class NativeHistoryTeleop(_Delegated):
    """Native evaluation/keyboard lifecycle; reads never advance history."""
    def __init__(self, base):
        assert base.envcfg['contact_load_observation']
        assert base.envcfg['command_target_observation']
        assert abs(base.dt - .01) < 1e-12
        self.base = base
        self._initialize_history(self._raw_observations())

    def _raw_observations(self):
        state = torch.from_numpy(self.base.observation()[None])
        return TensorDict(dict(actor=state, critic=state,
            command_delta=torch.from_numpy((2 * (self.base.command_targets -
                self.base.commands)).astype(np.float32)[None]),
            contact_load=torch.from_numpy(self.base.contact_reader.features(self.base.d)[None])),
            batch_size=[1])

    def observations(self):
        obs = self._raw_observations()
        self.history.replace_current(pack_frame(obs))
        return self._attach(obs)

    def reset(self, *args, **kwargs):
        result = self.base.reset(*args, **kwargs)
        self.history.reset(torch.tensor([0]), pack_frame(self._raw_observations()))
        return result

    def step(self, *args, **kwargs):
        self.observations()
        result = self.base.step(*args, **kwargs)
        self.history.advance(pack_frame(self._raw_observations()))
        return result


class HistoryPolicyController(PolicyController):
    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        cfg = json.loads((self.checkpoint.parent / 'runner_config.json').read_text())
        self.envcfg = json.loads((self.checkpoint.parent / 'env_config.json').read_text())
        assert self.envcfg.get('history_observation') == HISTORY_CONFIG
        assert cfg['actor']['class_name'] == 'history_context:HistoryContextModel'
        assert not self.envcfg.get('residual_teacher')
        self.saved = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        infos = self.saved.get('infos') or {}
        assert infos.get('history_observation') == HISTORY_CONFIG
        assert infos.get('hand_inertia_mode') == self.envcfg['hand_inertia_mode']
        assert infos.get('contact_load_observation') is True
        from body_acceleration_cost import validate_metadata as validate_acceleration
        validate_acceleration(self.envcfg, infos.get('physical_smooth_reward'))
        from action_bounds_cost import validate_metadata as validate_action_bounds
        validate_action_bounds(self.envcfg, infos.get('physical_smooth_reward'))
        if infos.get('practice_stop_adaptation') is not None:
            from practice_stop_adaptation import validate_environment as validate_stop
            validate_stop(self.envcfg, infos['practice_stop_adaptation'])
        if infos.get('physics_integration_adaptation') is not None:
            from physics_integration_adaptation import validate_environment
            validate_environment(self.envcfg, infos['physics_integration_adaptation'])
        self.env = NativeHistoryTeleop(NativeTeleop(self.envcfg))
        acfg = copy.deepcopy(cfg['actor']); acfg.pop('class_name')
        self.policy = HistoryContextModel(self.observations(), cfg['obs_groups'], 'actor', 21, **acfg)
        self.policy.load_state_dict(self.saved['actor_state_dict']); self.policy.eval()
        self.teacher = None
        self.trained_speed_limit = infos.get('teleop_curriculum', {}).get('speed_limit', 0.)

    def observations(self):
        return self.env.observations()
