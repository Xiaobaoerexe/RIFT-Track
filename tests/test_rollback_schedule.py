import contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.schedules import LearningRateController,collapse_trend
from go2_physx_cts.learner import Learner

class RollbackRules(unittest.TestCase):
 def rows(self):return [dict(falls=f,xy_rmse=e,reward=r) for f,e,r in [(6,.2,.05),(7,.3,.04),(8,.4,.03)]]
 def test_three_joint_strict_trends(self):
  self.assertTrue(collapse_trend(self.rows()));self.assertFalse(collapse_trend(self.rows()[:2]))
  for key,value in [('falls',5.),('xy_rmse',.2),('reward',.05)]:
   rows=self.rows();rows[1][key]=value;self.assertFalse(collapse_trend(rows))
 def test_nonfinite_rejected(self):
  rows=self.rows();rows[1]['reward']=float('nan');self.assertFalse(collapse_trend(rows))
 def test_ramp_exact_half_and_original_endpoint(self):
  s=LearningRateController(TrainCfg());before=s.rates(2000);start=s.ramp_start;s.rollback(2000)
  self.assertEqual(s.ramp_start,start)
  for key in ('actor','teacher','critic'):self.assertAlmostEqual(s.rates(2000)[key],before[key]/2)
  self.assertAlmostEqual(s.rates(2500)['actor'],2e-4)
  self.assertAlmostEqual(s.rates(2500)['teacher'],2e-4)
  self.assertAlmostEqual(s.rates(2500)['critic'],2e-4)
 def test_repeated_ramp_keeps_original_target(self):
  s=LearningRateController(TrainCfg());s.rollback(2000);before=s.rates(2200);s.rollback(2200)
  self.assertAlmostEqual(s.rates(2200)['actor'],before['actor']/2)
  self.assertAlmostEqual(s.rates(2500)['actor'],2e-4)
 def test_plateau_and_feedback_recovery(self):
  s=LearningRateController(TrainCfg());s.rollback(8000)
  self.assertAlmostEqual(s.rates(8999)['actor'],1e-4)
  s.last_window=8500
  good={k:{'status':'good'} for k in ('wm','critic','actor','tracking')}
  for it in (9500,10500,11500):s.feedback(it,good)
  self.assertAlmostEqual(s.rates(11501)['actor'],1.3e-4)
 def test_exact_half_at_feedback_lower_bound(self):
  s=LearningRateController(TrainCfg());s.current['actor']=4e-6;s.rollback(8000)
  self.assertEqual(s.rates(8001)['actor'],2e-6)
 def test_state_roundtrip_and_legacy_rejection(self):
  a=LearningRateController(TrainCfg());a.rollback(2000);a.rates(2100)
  b=LearningRateController(TrainCfg());b.load_state_dict(copy.deepcopy(a.state_dict()))
  self.assertEqual(a.rates(2300),b.rates(2300));self.assertEqual(a.rates(2500),b.rates(2500))
  old=a.state_dict();old.pop('version')
  with self.assertRaisesRegex(ValueError,'Legacy'):b.load_state_dict(old)
 def log_stub(self,folder):
  l=Learner.__new__(Learner);l.cfg=TrainCfg();l.provider_counts=[2765,922,409]
  l.actor_frozen_until=0;l.counts={};l.start_time=0;l.archive=SimpleNamespace(size=0,statistics=lambda n:{})
  l.buf=SimpleNamespace(total_collections=1000);l.nonfinite=0;l.env=SimpleNamespace(terrain_level_statistics={})
  l.output=Path(folder);l.collapse_history=[];l.collapse_strikes=0;l.last_good=Path('previous.pt')
  l.schedule=LearningRateController(l.cfg);return l
 def test_actual_log_requires_third_complete_window(self):
  with tempfile.TemporaryDirectory() as folder:
   l=self.log_stub(folder);calls=[];l.rollback=lambda:calls.append(l.it) or True
   for it,row in zip((8050,8100,8150),self.rows()):
    l.it=it;l.metrics=[dict(height=.3,falls=row['falls'],falls_all=row['falls'],xy_rmse=row['xy_rmse'],yaw_rmse=.1,reward=row['reward'])]*50
    with contextlib.redirect_stdout(io.StringIO()):l.log({})
   self.assertEqual(calls,[8150])
   records=[json.loads(x) for x in (l.output/'training.jsonl').read_text().splitlines()]
   self.assertEqual([r['rolled_back'] for r in records],[False,False,True])
 def test_partial_window_cannot_trigger(self):
  with tempfile.TemporaryDirectory() as folder:
   l=self.log_stub(folder);l.rollback=lambda:self.fail('Partial window triggered rollback')
   for it,row,n in zip((8050,8100,8150),self.rows(),(50,49,50)):
    l.it=it;l.metrics=[dict(height=.3,falls=row['falls'],falls_all=row['falls'],xy_rmse=row['xy_rmse'],yaw_rmse=.1,reward=row['reward'])]*n
    with contextlib.redirect_stdout(io.StringIO()):l.log({})
   self.assertEqual(len(l.collapse_history),1)
 def test_weights_only_no_freeze_optimizer_moments_preserved(self):
  with tempfile.TemporaryDirectory() as folder:
   l=Learner.__new__(Learner);l.cfg=TrainCfg();l.device='cpu';l.it=2000;l.teacher_stable=False
   l.policy=torch.nn.ModuleDict({name:torch.nn.Linear(1,1) for name in ('actor','teacher','student')})
   l.critic=torch.nn.Linear(1,1);l.wm=torch.nn.Linear(1,1)
   l.opt_actor=torch.optim.Adam([dict(params=l.policy['actor'].parameters(),name='actor'),dict(params=l.policy['teacher'].parameters(),name='teacher')])
   l.opt_critic=torch.optim.Adam(l.critic.parameters());l.opt_wm=torch.optim.Adam(l.wm.parameters())
   l.opt_cts=torch.optim.Adam(l.policy['student'].parameters())
   for opt in (l.opt_actor,l.opt_critic,l.opt_cts):
    opt.zero_grad();sum(p.sum() for g in opt.param_groups for p in g['params']).backward();opt.step()
   before=[copy.deepcopy(opt.state_dict()['state']) for opt in (l.opt_actor,l.opt_critic,l.opt_cts)]
   l.schedule=LearningRateController(l.cfg);l.actor_frozen_until=0;l.collapse_history=[{}]
   l.feedback=SimpleNamespace(reset=lambda:None)
   l.real_hidden=torch.ones(2,256);l.camera_episode=torch.ones(2,dtype=torch.long)
   l.last_good=Path(folder)/'old.pt';torch.save({'policy':l.policy.state_dict(),'critic':l.critic.state_dict()},l.last_good)
   expected=copy.deepcopy(l.policy.state_dict());opts=[l.opt_actor,l.opt_critic,l.opt_cts]
   with torch.no_grad():l.policy['actor'].weight.add_(1.)
   self.assertTrue(l.rollback());self.assertEqual(opts,[l.opt_actor,l.opt_critic,l.opt_cts])
   self.assertEqual(l.schedule.ramp_start,1500);self.assertEqual(l.actor_frozen_until,0)
   for key,value in expected.items():self.assertTrue(torch.equal(l.policy.state_dict()[key],value))
   for opt,saved in zip(opts,before):
    for p,state in opt.state_dict()['state'].items():
     for key,value in state.items():self.assertTrue(torch.equal(value,saved[p][key]) if torch.is_tensor(value) else value==saved[p][key])
   self.assertEqual(l.real_hidden.count_nonzero().item(),0)
 def test_adaptation_clock_and_student_gate(self):
  s=LearningRateController(TrainCfg())
  self.assertFalse(s.window_due(2500));self.assertTrue(s.window_due(3500))
  self.assertTrue(s.observe_due(2700));self.assertFalse(s.window_due(80000))
  self.assertEqual(s.rates(1499)['cts'],0.)
  self.assertEqual(s.rates(1500)['cts'],3e-4)
 def test_legacy_launch_threshold_rejected(self):
  with self.assertRaisesRegex(ValueError,'threshold5'):TrainCfg(collapse_falls=3.).validate(False)

if __name__=='__main__':unittest.main()
