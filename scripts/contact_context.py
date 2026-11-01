"""Staged load-feedback projection with exactly unchanged legacy inputs."""
import copy

import torch
from torch.nn import functional as F
from rsl_rl.models.mlp_model import unpad_trajectories

from command_context import CommandContextModel,split_command_optimizer


class ContactContextModel(CommandContextModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.contact_projection=torch.nn.Parameter(self.mlp[0].weight.new_zeros(self.mlp[0].out_features,15))

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None else obs
        latent=self.get_latent(obs,masks,hidden_state)
        value=self.mlp[0](latent)+F.linear(obs['command_delta'],self.command_projection)
        value=value+F.linear(obs['contact_load'],self.contact_projection)
        for index in range(1,len(self.mlp)):value=self.mlp[index](value)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(value)
                return self.distribution.sample()
            return self.distribution.deterministic_output(value)
        return value


def split_contact_optimizer(algorithm):
    split_command_optimizer(algorithm)
    models=(algorithm.get_policy(),algorithm.critic)
    parameters=[m.contact_projection for m in models]
    opt=algorithm.optimizer
    assert len(opt.param_groups)==2 and not opt.state
    group=opt.param_groups[0]
    group['params']=[p for p in group['params'] if all(p is not b for b in parameters)]
    opt.add_param_group({'params':parameters,'lr':group['lr']})


def migrate_checkpoint(saved,learning_rate):
    result=copy.deepcopy(saved)
    for role in ('actor','critic'):
        state=result[f'{role}_state_dict']
        assert 'contact_projection' not in state and state['mlp.0.weight'].shape==(256,124)
        assert state['command_projection'].shape==(256,2)
        state['contact_projection']=state['mlp.0.weight'].new_zeros(256,15)
    optimizer=result['optimizer_state_dict']
    assert [g['params'] for g in optimizer['param_groups']]==[list(range(17)),[17,18]]
    new_group={k:copy.deepcopy(v) for k,v in optimizer['param_groups'][1].items() if k!='params'}
    new_group.update(params=[19,20],lr=learning_rate)
    optimizer['param_groups'].append(new_group)
    result.setdefault('infos',{}).update(contact_load_observation=True,contact_branch_only=True,
        command_branch_only=False,command_backbone_trainable=False)
    return result
