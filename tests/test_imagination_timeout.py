"""CPU analytical time-limit return/target and imagined-state boundary tests.

Run: python tests/test_imagination_timeout.py
"""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from go2_physx_cts.imagine import Imagination,time_limit_return,time_limit_lambda_targets
from go2_physx_cts.commands import OfficialCommands,EPISODE_ELAPSED
from go2_physx_cts.networks import Critic
from go2_physx_cts import state as st


def run():
    checks=[]
    reward=torch.tensor([[1.,2.,100.]],requires_grad=True)
    done=torch.zeros_like(reward);timeout=torch.tensor([[False,True,False]])
    values=torch.tensor([[0.,10.,20.,30.]],requires_grad=True)
    result,surv,immediate,boot=time_limit_return(reward,done,timeout,values,.9)
    torch.testing.assert_close(result,torch.tensor([19.]))
    torch.testing.assert_close(surv,torch.tensor([[1.,1.,0.,0.]]))
    torch.testing.assert_close(immediate,torch.tensor([2.8]))
    result.sum().backward()
    torch.testing.assert_close(reward.grad,torch.tensor([[1.,.9,0.]]))
    torch.testing.assert_close(values.grad,torch.tensor([[0.,0.,.81,0.]]))
    checks.append('timeout_keeps_terminal_reward_and_differentiable_Vnext_not_Htail')
    target=time_limit_lambda_targets(reward,done,timeout,values,.9,.95)
    torch.testing.assert_close(target[:,:2],torch.tensor([[18.55,20.]]))
    assert not target.requires_grad
    # Values/rewards after timeout cannot affect any valid target or objective.
    changed_r=reward.detach().clone();changed_r[:,2]=123456.
    changed_v=values.detach().clone();changed_v[:,3]=-987654.
    changed=time_limit_lambda_targets(changed_r,done,timeout,changed_v,.9,.95)
    torch.testing.assert_close(changed[:,:2],target[:,:2])
    checks.append('lambda_recursion_ends_at_timeout_and_detaches_targets')
    fall=torch.tensor([[0.,1.,0.]])
    obj,_,_,boot=time_limit_return(reward,fall,timeout,values,.9)
    torch.testing.assert_close(obj,torch.tensor([2.8]));torch.testing.assert_close(boot,torch.zeros(1))
    fall_target=time_limit_lambda_targets(reward,fall,timeout,values,.9,.95)
    torch.testing.assert_close(fall_target[:,1],torch.tensor([2.]))
    checks.append('fall_at_time_limit_does_not_bootstrap')
    soft=torch.tensor([[.1,.25,.2]]);no_timeout=torch.zeros_like(timeout)
    obj,surv,_,_=time_limit_return(reward,soft,no_timeout,values,.9)
    expected=1.+.9*.9*2.+.9**2*.9*.75*100.+.9**3*.9*.75*.8*30.
    torch.testing.assert_close(obj,torch.tensor([expected]))
    checks.append('without_timeout_preserves_original_soft_survival_objective')

    cfg=SimpleNamespace(dt=.02,episode_seconds=20.,compile=False,horizon=3,gamma=.99,lam=.95,stage='flat')
    sampler=OfficialCommands(cfg.dt,cfg.episode_seconds)
    initial=torch.zeros(3,70);initial[:,0]=.3;initial[:,3]=-1.
    pose=torch.zeros(3,3);cmd=sampler.initial(pose[:,:2],pose[:,:2],0,random=torch.full((3,8),.9))
    cmd[:,EPISODE_ELAPSED]=torch.tensor([19.98,19.96,0.])
    seed={'s':initial,'pose':pose,'a_prev':torch.zeros(3,12),'obs':torch.zeros(3,450),
          'cmd_state':cmd,'xi':torch.zeros(3,st.CONTEXT_DIM)}
    class TinyPolicy(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.2))
        def forward(self,hist,priv,branch):return self.weight.expand(len(hist),12)
    class TinyWM:
        def __call__(self,s,a,a_prev,xi,relief):
            return torch.cat((s[:,:1]+a[:,:1]*.01,s[:,1:]),-1),torch.zeros(len(s),3)
    class TinyCritic:
        features=staticmethod(Critic.features)
        def __call__(self,features):return features[...,0]
    policy=TinyPolicy();critic=TinyCritic();imagination=Imagination(cfg,None,None,sampler)
    def reward_only_height(old,new,*args,**kwargs):return new[:,0],{},torch.zeros(len(new))
    with patch('go2_physx_cts.imagine.reward',reward_only_height):
        objective,data=imagination.rollout(TinyWM(),policy,critic,seed,0,iteration=5000)
    assert data['critic_valid'].tolist()==[[True,False,False],[True,True,False],[True,True,True]]
    assert data['timeout'].tolist()==[[True,False,False],[False,True,False],[False,False,False]]
    torch.testing.assert_close(data['critic_in'][0,:,0],torch.tensor([.3,.302,.302,.302]))
    torch.testing.assert_close(data['rewards'][0],torch.tensor([.302,0.,0.]))
    torch.testing.assert_close(objective[0],torch.tensor(.302*(1+.99)))
    objective.sum().backward();assert torch.isfinite(policy.weight.grad) and policy.weight.grad.abs()>0
    assert all(bool(torch.isfinite(value).all()) for value in data.values())
    targets=imagination.targets(critic,data)
    torch.testing.assert_close(targets[0,0],torch.tensor(.302*(1+.99)))
    checks.append('actual_rollout_20s_prefix_masks_freezes_state_and_preserves_actor_gradient')
    # A critic fit using the emitted validity mask excludes synthetic post-limit
    # rows, even though fixed-size computation emits finite placeholder targets.
    predictions=torch.zeros_like(targets,requires_grad=True)
    loss=(predictions[data['critic_valid']]-targets[data['critic_valid']]).square().mean()
    loss.backward();assert bool((predictions.grad[~data['critic_valid']]==0).all())
    checks.append('critic_valid_excludes_post_timeout_fit_samples')
    print({'status':'PASS','checks':checks,'kind':'CPU_analytical_and_mock_rollout_not_robot_capability'})


if __name__=='__main__':run()
