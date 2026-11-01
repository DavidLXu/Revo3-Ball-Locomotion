"""Requested-command context without changing the pretrained input projection."""
import copy

import torch
from torch.nn import functional as F
from rsl_rl.models import MLPModel
from rsl_rl.models.mlp_model import unpad_trajectories


class CommandContextModel(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.obs_dim == 124 and not self.is_recurrent
        # No random initialization: preserve the legacy RNG stream as well.
        self.command_projection = torch.nn.Parameter(self.mlp[0].weight.new_zeros(
            self.mlp[0].out_features, 2))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        value = self.mlp[0](latent) + F.linear(obs['command_delta'], self.command_projection)
        for index in range(1, len(self.mlp)):
            value = self.mlp[index](value)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(value)
                return self.distribution.sample()
            return self.distribution.deterministic_output(value)
        return value

    def as_jit(self, *args, **kwargs):
        raise NotImplementedError('Command context needs a two-input export wrapper')

    def as_onnx(self, *args, **kwargs):
        raise NotImplementedError('Command context needs a two-input export wrapper')


def split_command_optimizer(algorithm):
    """Keep legacy Adam parameter order and append both new projections."""
    models = (algorithm.get_policy(), algorithm.critic)
    branches = [m.command_projection for m in models]
    optimizer = algorithm.optimizer
    assert len(optimizer.param_groups) == 1 and not optimizer.state
    group = optimizer.param_groups[0]
    group['params'] = [p for p in group['params'] if all(p is not b for b in branches)]
    optimizer.add_param_group({'params': branches, 'lr': group['lr']})


def migrate_checkpoint(saved, branch_learning_rate):
    """Explicit one-time legacy migration; all old weights and Adam slots stay intact."""
    result = copy.deepcopy(saved)
    for role in ('actor', 'critic'):
        state = result[f'{role}_state_dict']
        assert 'command_projection' not in state and state['mlp.0.weight'].shape[1] == 124
        state['command_projection'] = state['mlp.0.weight'].new_zeros(256, 2)
    optimizer = result['optimizer_state_dict']
    assert len(optimizer['param_groups']) == 1
    base = optimizer['param_groups'][0]
    next_id = max(base['params']) + 1
    branch = {k: copy.deepcopy(v) for k, v in base.items() if k != 'params'}
    branch.update(params=[next_id, next_id + 1], lr=branch_learning_rate)
    optimizer['param_groups'].append(branch)
    infos = result.setdefault('infos', {})
    assert not infos.get('actor_frozen_for_critic_warmup', False)
    infos.update(command_branch_only=True, actor_normalization_frozen=True)
    return result
