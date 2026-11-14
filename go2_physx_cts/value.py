"""AS2 MLP value model, with real-event heads only in the two event variants."""
import torch
from torch import nn
from torch.nn import functional as F
from . import state as st
from .networks import RunningNorm, mlp
from .precision import network_autocast
from .task import TASK_CONTEXT_DIM, GOAL, ORIGIN, ROUTE, ROUTE_YAW


class Critic(nn.Module):
    def __init__(self, hidden=(512, 256, 128), field=None, context_dim=TASK_CONTEXT_DIM,
                 compile=False, mixed_precision=True, variant='simple', task_count=None):
        super().__init__()
        self.variant = variant
        self.task_count = (1 if variant == 'simple' else 5) if task_count is None else int(task_count)
        if self.task_count != (1 if variant == 'simple' else 5):
            raise ValueError('Critic task count contradicts experiment variant')
        self.mixed_precision = mixed_precision
        self.context_dim = 0 if variant == 'simple' else TASK_CONTEXT_DIM
        self.physical_dim = st.PRIV_DIM+self.context_dim+(0 if variant == 'simple' else 14)
        self.feature_dim = self.physical_dim+(2 if variant == 'simple' else 3)
        self.norm = RunningNorm(self.physical_dim+2)
        self.net = mlp(self.physical_dim+2, hidden, 1 if variant=='simple' else self.task_count*2, nn.ELU)

    @staticmethod
    def _tags(priv, branch):
        tag = priv.new_zeros((len(priv), 2))
        if isinstance(branch, tuple):
            tag[:branch[0], 0] = 1.; tag[branch[0]:, 1] = 1.
        elif torch.is_tensor(branch):
            tag.scatter_(1, branch.reshape(-1, 1).long(), 1.)
        else:
            tag[:, int(branch)] = 1.
        return tag

    def features(self, priv, branch, task_ids=None, context=None, pose=None, contacts=None):
        tag = self._tags(priv, branch)
        if self.variant == 'simple':
            return torch.cat((priv, tag), -1)
        if context is None or pose is None or contacts is None or task_ids is None:
            raise ValueError('Event value needs task memory, physical pose and contacts')
        if context.shape[-1] != TASK_CONTEXT_DIM or contacts.shape[-1] != 14:
            raise ValueError('Event value context/contact shape mismatch')
        canonical = context.clone()
        yaw = st.yaw_from_pose(pose)
        canonical[:, GOAL] = st.yaw_inverse(context[:, GOAL]-pose[:, :3], yaw)
        displacement = torch.cat((pose[:, :2]-context[:, ORIGIN], pose.new_zeros(len(pose), 1)), -1)
        canonical[:, ORIGIN] = st.yaw_inverse(displacement, yaw)[:, :2]
        route = torch.cat((context[:, ROUTE]-pose[:, :2], pose.new_zeros(len(pose), 1)), -1)
        canonical[:, ROUTE] = st.yaw_inverse(route, yaw)[:, :2]
        canonical[:, ROUTE_YAW] = torch.atan2(torch.sin(context[:, ROUTE_YAW]-yaw), torch.cos(context[:, ROUTE_YAW]-yaw))
        return torch.cat((priv, canonical, contacts, task_ids.reshape(-1, 1).to(priv.dtype), tag), -1)

    def event_features(self, priv_after, branch, task_ids, pre_context, after_pose, next_contacts):
        return self.features(priv_after, branch, task_ids, pre_context, after_pose, next_contacts)

    def norm_input(self, x):
        return x if self.variant=='simple' else torch.cat((x[..., :self.physical_dim], x[..., -2:]), -1)

    @torch.no_grad()
    def update_stats(self, x):
        self.norm.update(self.norm_input(x))

    def values(self, x):
        shape = x.shape[:-1]; flat = x.reshape(-1, self.feature_dim)
        with network_autocast(flat, self.mixed_precision):
            raw = self.net(self.norm(self.norm_input(flat)))
        raw = raw.float()
        if self.variant=='simple':
            return torch.stack((raw[:, 0], torch.zeros_like(raw[:, 0])), -1).reshape(*shape, 2)
        tasks = flat[:, -3].long()
        selected = raw.reshape(-1, self.task_count, 2)[torch.arange(len(flat), device=x.device), tasks]
        return selected.reshape(*shape, 2)

    def forward(self, x, kind='continuous'):
        if kind not in ('continuous', 'event'):
            raise ValueError('Unknown value channel')
        return self.values(x)[..., 0 if kind=='continuous' else 1]

    def forward_event(self, x):
        return self.forward(x, kind='event')

    def loss(self, x, targets, kind='continuous', delta=1.):
        prediction = self.forward(x, kind=kind)
        per = 2*F.huber_loss(prediction.float(), targets.float(), reduction='none', delta=delta)
        tags = x[..., -2:]
        return .5*sum((per*tags[..., b]).sum()/tags[..., b].sum().clamp_min(1) for b in (0, 1))


@torch.no_grad()
def event_td_target(real_event_reward, bootstrap_mask, next_event_value, gamma=.99):
    return real_event_reward+gamma*bootstrap_mask.to(real_event_reward.dtype)*next_event_value
