"""Frozen balance teacher plus bounded joint-only residual for locomotion."""
from pathlib import Path
import json
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from balance_model import ROOT

def load_balance_teacher(checkpoint,device):
    path=Path(checkpoint)
    if not path.is_absolute():path=ROOT/path
    cfg=json.loads((path.parent/'runner_config.json').read_text())
    envcfg=json.loads((path.parent/'env_config.json').read_text())
    assert envcfg['initial_pose']=='opposed_thumb' and envcfg['task']=='teleop'
    dummy=torch.zeros((1,117),device=device)
    ob=TensorDict({'actor':dummy,'critic':dummy},batch_size=[1],device=device)
    acfg=dict(cfg['actor']);acfg.pop('class_name',None)
    policy=MLPModel(ob,cfg['obs_groups'],'actor',21,**acfg).to(device)
    saved=torch.load(path,map_location=device,weights_only=False)
    policy.load_state_dict(saved['actor_state_dict']);policy.eval();policy.requires_grad_(False)
    return policy

def teacher_action(policy,observations):
    # The balance teacher was trained with zero commands. Never inject new
    # command values into its frozen normalization statistics.
    obs=observations[:,:117].clone();obs[:,115:117]=0;obs[:,60:62]=0
    td=TensorDict({'actor':obs,'critic':obs},batch_size=[len(obs)],device=obs.device)
    with torch.no_grad():return policy(td,stochastic_output=False).detach()

def combine_actions(base,residual,scale):
    return (base.clamp(-1,1)+scale*residual.clamp(-1,1)).clamp(-1,1)

def residual_joint_target(joint0,previous_actions,filtered_actions,teacher_actions,low,high,action_scale):
    """Apply this step's filtered correction after limiting the teacher target.

    Accepts NumPy arrays or Torch tensors. The latent action filter is unchanged.
    With residual=0, filtered_actions equals filtered_teacher, reproducing the
    original teacher target exactly. Joint limits and torque limits are retained.
    """
    filtered_teacher=.7*previous_actions+.3*teacher_actions.clip(-1,1)
    teacher_target=(joint0+action_scale*filtered_teacher).clip(low,high)
    correction=action_scale*(filtered_actions-filtered_teacher)
    return (teacher_target+correction).clip(low,high)
