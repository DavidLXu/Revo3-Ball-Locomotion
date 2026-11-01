"""Staged history projection; not selected by the current training/inference code."""
import copy

import torch
from torch.nn import functional as F
from rsl_rl.models.mlp_model import unpad_trajectories

from contact_context import ContactContextModel, split_contact_optimizer


HISTORY_CONFIG = dict(past_samples=7, stride=2, feature_count=141,
                      encoding='normalized_difference_from_current')


class HistoryContextModel(ContactContextModel):
    """Retain current observations and add seven past frames through zero weights.

    Each frame is [actor state(124), command delta(2), contact loads(15)].
    History must be provided by an explicit lifecycle adapter; model calls are
    stateless. The existing normalizer scales the state differences and is not
    updated from historical frames. Contact/command scales remain unchanged.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_projection = torch.nn.Parameter(
            self.mlp[0].weight.new_zeros(self.mlp[0].out_features, 7 * 141))

    def history_latent(self, obs, current_state):
        history = obs['past_history']
        if history.shape[-1] != 7 * 141:
            raise ValueError('History must contain seven flattened 141-feature frames')
        history = history.reshape(*history.shape[:-1], 7, 141)
        normalized = self.obs_normalizer(history[..., :124])
        current_extra = torch.cat((obs['command_delta'], obs['contact_load']), -1)
        delta = torch.cat((normalized - current_state.unsqueeze(-2),
                           history[..., 124:] - current_extra.unsqueeze(-2)), -1)
        return delta.flatten(-2)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        value = self.mlp[0](latent) + F.linear(obs['command_delta'], self.command_projection)
        value = value + F.linear(obs['contact_load'], self.contact_projection)
        value = value + F.linear(self.history_latent(obs, latent), self.history_projection)
        for index in range(1, len(self.mlp)):
            value = self.mlp[index](value)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(value)
                return self.distribution.sample()
            return self.distribution.deterministic_output(value)
        return value


def split_history_optimizer(algorithm):
    """Preserve all three existing groups and append the history parameters."""
    split_contact_optimizer(algorithm)
    parameters = [algorithm.get_policy().history_projection,
                  algorithm.critic.history_projection]
    optimizer = algorithm.optimizer
    assert len(optimizer.param_groups) == 3 and not optimizer.state
    group = optimizer.param_groups[0]
    group['params'] = [p for p in group['params'] if all(p is not b for b in parameters)]
    optimizer.add_param_group({'params': parameters, 'lr': group['lr']})


def migrate_checkpoint(saved, learning_rate):
    """Add neutral history weights and empty Adam slots; change no old state."""
    if learning_rate <= 0:
        raise ValueError('History learning rate must be positive')
    result = copy.deepcopy(saved)
    for role in ('actor', 'critic'):
        state = result[role + '_state_dict']
        assert 'history_projection' not in state
        assert state['mlp.0.weight'].shape == (256, 124)
        assert state['contact_projection'].shape == (256, 15)
        state['history_projection'] = state['mlp.0.weight'].new_zeros(256, 7 * 141)
    optimizer = result['optimizer_state_dict']
    assert [g['params'] for g in optimizer['param_groups']] == [list(range(17)), [17, 18], [19, 20]]
    assert not any(i in optimizer['state'] for i in (21, 22))
    group = {k: copy.deepcopy(v) for k, v in optimizer['param_groups'][-1].items() if k != 'params'}
    group.update(params=[21, 22], lr=learning_rate)
    optimizer['param_groups'].append(group)
    result.setdefault('infos', {})['history_observation'] = dict(HISTORY_CONFIG)
    return result
