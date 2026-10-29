import unittest,math,tempfile
from types import SimpleNamespace
import torch
from go2_physx_cts import state as st
from go2_physx_cts.buffer import Replay,Archive,rarity,mixed_windows
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.networks import RunningNorm,Policy,Critic
from go2_physx_cts.world_model import WorldModel
from go2_physx_cts.task import RobotConsts,reward
from go2_physx_cts.imagine import Imagination
from go2_physx_cts.learner import frozen
from go2_physx_cts.learner import branch_loss,provider_assignment
from go2_physx_cts.commands import OfficialCommands
from go2_physx_cts.driver import official_default

torch.set_num_threads(2)
class Mechanisms(unittest.TestCase):
 def test_normalization(self):
  n=RunningNorm(3);x=torch.randn(1000,3)*2+5;n.update(x[:400]);n.update(x[400:])
  self.assertTrue(torch.allclose(n.mean,x.mean(0),atol=1e-5))
  self.assertTrue(torch.allclose(n.var,x.var(0,unbiased=False),atol=1e-5))
 def test_history_and_frame(self):
  hist=torch.zeros(2,450)
  for i in range(12):hist=st.push_history(hist,torch.full((2,45),float(i)))
  self.assertTrue(torch.equal(st.single_from_history(hist),torch.full((2,45),11.)))
  self.assertEqual(hist[0,0].item(),2.)
 def test_body_velocity(self):
  s=torch.zeros(1,70);p=.4;r=.3
  s[:,1:4]=torch.tensor([math.sin(p),-math.cos(p)*math.sin(r),-math.cos(p)*math.cos(r)])
  s[:,4:7]=torch.tensor([math.cos(p),0,-math.sin(p)])
  self.assertTrue(torch.allclose(st.velocity_body(s),torch.tensor([[1.,0,0]]),atol=1e-6))
 def test_replay_boundary_and_context(self):
  b=Replay(4,16,'cpu')
  for t in range(40):
   v={k:torch.zeros(4,x.shape[-1],dtype=x.dtype) for k,x in b.data.items()}
   v['s'][:]=t;v['xi'][:]=t//4;v['episode'][:]=t//4;v['done'][:]=t%4==3;v['source'][:,0]=torch.arange(4)
   b.add(**v)
  w=b.windows(200,8,torch.full((4,5),-9.))
  self.assertTrue(torch.equal(w['xi'][:,0],torch.floor(w['s'][:,0,0]/4)))
  for j in range(8):
   mask=w['valid'][:,j].bool()
   self.assertTrue(torch.all(torch.floor(w['s'][mask,j+1,0]/4)==torch.floor(w['s'][mask,0,0]/4)))
  seed=b.seeds(50,None,source=1);self.assertTrue((seed['source']==1).all())
  source=b.seeds(1000,None)['source'];self.assertEqual(set(source.flatten().tolist()),{0,1,2,3})
 def test_actor_gradient_ownership(self):
  cfg=TrainCfg(device='cpu',compile=False,actor_hidden=(32,32),critic_hidden=(32,32),wm_hidden=(32,32));cfg.horizon=3
  policy=Policy(cfg.actor_hidden);critic=Critic(cfg.critic_hidden);wm=WorldModel(cfg.wm_hidden)
  c=RobotConsts(torch.zeros(12),torch.ones(12)*.25,torch.ones(12)*25,torch.ones(12)*.5,torch.ones(12)*24,torch.tensor([[-3.,3.]]*12))
  im=Imagination(cfg,c);s=torch.zeros(8,70);s[:,0]=.3;s[:,3]=-1
  seed={'s':s,'pose':torch.zeros(8,3),'a_prev':torch.zeros(8,12),'obs':torch.randn(8,450),'command':torch.ones(8,3)*.3,'timer':torch.ones(8,1)*5,'xi':torch.zeros(8,5)}
  seed['cmd_state']=OfficialCommands().initial(seed['pose'][:,:2],seed['pose'][:,:2])
  with frozen(wm,critic,policy.student_encoder):
   obj,data=im.rollout(wm,policy,critic,seed,0);(-obj.mean()).backward()
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.teacher_encoder.parameters()))
  self.assertTrue(all(p.grad is None for p in wm.parameters()))
  self.assertTrue(all(p.grad is None for p in critic.parameters()))
  self.assertTrue(all(p.grad is None for p in policy.student_encoder.parameters()))
  policy.zero_grad(set_to_none=True)
  with frozen(wm,critic,policy.student_encoder):
   obj,data=im.rollout(wm,policy,critic,seed,1);(-obj.mean()).backward()
  self.assertTrue(all(p.grad is None for p in policy.teacher_encoder.parameters()))
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.actor.parameters()))
  self.assertTrue(all(p.grad is None for p in policy.student_encoder.parameters()))
  self.assertTrue(torch.isfinite(im.targets(critic,data)).all())
 def test_unbounded_action_and_mapping(self):
  p=Policy((16,));last=p.actor
  # Verify there is no saturating action head; final projection is linear.
  self.assertFalse(any(isinstance(m,torch.nn.Tanh) for m in p.actor.modules()))
  default=official_default('cpu');a=torch.randn(32,12)*4;our=torch.tensor([.1,.9,-1.8]*4);scale=torch.tensor([.33,.33,.28]*4)
  converted=(default+.25*a-our)/scale
  self.assertTrue(torch.allclose(our+scale*converted,default+.25*a,atol=1e-6))
 def test_fused_cts_batch_equivalence(self):
  p=Policy((32,32));hist=torch.randn(16,450);priv=torch.randn(16,st.PRIV_DIM)
  a=p(hist,priv,(12,4));b=torch.cat((p(hist[:12],priv[:12],0),p(hist[12:],priv[12:],1)))
  self.assertTrue(torch.allclose(a,b,atol=1e-6))
  ca=Critic.features(priv,(12,4));cb=torch.cat((Critic.features(priv[:12],0),Critic.features(priv[12:],1)))
  self.assertTrue(torch.equal(ca,cb))
  ga=torch.autograd.grad(a.square().mean(),tuple(p.parameters()),allow_unused=True)
  gb=torch.autograd.grad(b.square().mean(),tuple(p.parameters()),allow_unused=True)
  for x,y in zip(ga,gb):
   if x is not None:self.assertTrue(torch.allclose(x,y,atol=1e-6,rtol=1e-4))
 def test_branch_loss_microbatch_gradient_equivalence(self):
  torch.manual_seed(15);x=torch.randn(16,4);w=torch.randn(4,requires_grad=True);sizes=(12,4)
  expected=branch_loss((x@w).square(),sizes,sizes)
  ga,=torch.autograd.grad(expected,w)
  for chunk in (1,3,5,12):
   w.grad=None
   for branch,(start,end) in enumerate(((0,12),(12,16))):
    for lo in range(start,end,chunk):branch_loss((x[lo:min(end,lo+chunk)]@w).square(),branch,sizes).backward()
   self.assertTrue(torch.allclose(w.grad,ga,atol=1e-6,rtol=1e-6))
  half=-.5*((x[:12]@w).square().mean()+(x[12:]@w).square().mean())
  self.assertTrue(torch.allclose(expected,2*half))
 def test_providers_cover_every_terrain_column(self):
  src=provider_assignment(4096,.1,.1,.75,'cpu')
  self.assertEqual(torch.bincount(src.long()).tolist(),[2458,820,409,409])
  for col in torch.tensor_split(src,20):self.assertEqual(set(col.tolist()),{0,1,2,3})

if __name__=='__main__':unittest.main(verbosity=2)
