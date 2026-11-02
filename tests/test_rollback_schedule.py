import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.schedules import LearningRateController,collapse_trend
from go2_physx_cts.learner import Learner

class RollbackRules(unittest.TestCase):
 def rows(self):
  return [dict(falls=f,xy_rmse=e,reward=r) for f,e,r in [(6,.2,.05),(7,.3,.04),(8,.4,.03)]]
 def test_three_joint_strict_trends(self):
  rows=self.rows();self.assertTrue(collapse_trend(rows));self.assertFalse(collapse_trend(rows[:2]))
  for key,value in [('falls',5.),('xy_rmse',.2),('reward',.05)]:
   rows=self.rows();rows[1][key]=value;self.assertFalse(collapse_trend(rows))
 def test_nonfinite_rejected(self):
  rows=self.rows();rows[1]['reward']=float('nan');self.assertFalse(collapse_trend(rows))
 def test_ramp_halves_without_reset_and_recovers_original_endpoint(self):
  s=LearningRateController(TrainCfg());before=s.rates(2000);start=s.ramp_start;s.rollback(2000)
  self.assertEqual(s.ramp_start,start)
  self.assertAlmostEqual(s.rates(2000)['actor'],before['actor']/2)
  self.assertAlmostEqual(s.rates(2000)['critic'],before['critic']/2)
  self.assertAlmostEqual(s.rates(2500)['actor'],2e-4)
  self.assertAlmostEqual(s.rates(2500)['critic'],2e-4)
 def test_repeated_ramp_rollback_retains_original_target(self):
  s=LearningRateController(TrainCfg());s.rollback(2000);before=s.rates(2200);s.rollback(2200)
  self.assertAlmostEqual(s.rates(2200)['actor'],before['actor']/2)
  self.assertAlmostEqual(s.rates(2500)['actor'],2e-4)
 def test_plateau_stays_halved_and_candidates_use_it(self):
  s=LearningRateController(TrainCfg());s.rollback(8000)
  self.assertAlmostEqual(s.rates(8999)['actor'],1e-4)
  self.assertAlmostEqual(s.rates(12000)['actor'],1e-4)

  for actual,expected in zip(s._candidates('actor'),[1e-4/1.3,1e-4,1.3e-4]):self.assertAlmostEqual(actual,expected)
 def test_rollback_halves_even_at_trial_lower_bound(self):
  s=LearningRateController(TrainCfg());s.current['actor']=4e-6;s.rollback(8000)
  self.assertEqual(s.rates(8001)['actor'],2e-6)
 def test_legacy_checkpoint_cannot_postpone_original_ramp(self):
  old=LearningRateController(TrainCfg()).state_dict()
  old.pop('original_ramp_semantics');old['ramp_start']=7950;old['current']['actor']=4e-6
  s=LearningRateController(TrainCfg());s.load_state_dict(old)
  self.assertEqual(s.rates(8000)['actor_ramp'],1.)
  self.assertEqual(s.rates(8000)['actor'],4e-6)
 def test_recovery_checkpoint_roundtrip(self):
  a=LearningRateController(TrainCfg());a.rollback(2000);a.rates(2100)
  b=LearningRateController(TrainCfg());b.load_state_dict(a.state_dict())
  self.assertEqual(a.rates(2300),b.rates(2300))
  self.assertEqual(a.rates(2500),b.rates(2500))
 def test_actual_log_only_triggers_on_third_window(self):
  with tempfile.TemporaryDirectory() as folder:
   l=Learner.__new__(Learner);l.cfg=TrainCfg(collapse_falls=5.)
   l.provider_counts=[2458,820,409,409];l.actor_frozen_until=1500;l.counts={}
   l.start_time=0;l.archive=SimpleNamespace(size=0);l.nonfinite=0
   l.env=SimpleNamespace(terrain_level_statistics={});l.output=Path(folder)
   l.collapse_history=[];l.collapse_strikes=0;l.last_good=Path('previous.pt')
   l.schedule=LearningRateController(l.cfg);calls=[];l.rollback=lambda:calls.append(l.it) or True
   for it,row in zip((8050,8100,8150),self.rows()):
    l.it=it;l.metrics=[dict(height=.3,falls=row['falls'],falls_all=row['falls'],xy_rmse=row['xy_rmse'],yaw_rmse=.1,reward=row['reward'])]*50
    with contextlib.redirect_stdout(io.StringIO()):l.log({})
   self.assertEqual(calls,[8150])
   records=[json.loads(x) for x in (l.output/'training.jsonl').read_text().splitlines()]
   self.assertEqual([r['rolled_back'] for r in records],[False,False,True])
 def test_rollback_preserves_optimizer_and_original_clock(self):
  with tempfile.TemporaryDirectory() as folder:
   l=Learner.__new__(Learner);l.cfg=TrainCfg();l.device='cpu';l.it=2000
   l.policy=torch.nn.Linear(1,1);l.critic=torch.nn.Linear(1,1);l.wm=torch.nn.Linear(1,1)
   l.opt_actor=torch.optim.Adam(l.policy.parameters());l.opt_critic=torch.optim.Adam(l.critic.parameters())
   l.opt_wm=torch.optim.Adam(l.wm.parameters());l.opt_cts=torch.optim.Adam(l.policy.parameters())
   l.schedule=LearningRateController(l.cfg);l.lr_scale=1.;l.ramp_start=1000;l.collapse_strikes=1;l.collapse_history=[{}]
   l.last_good=Path(folder)/'old.pt'
   torch.save({'policy':l.policy.state_dict(),'critic':l.critic.state_dict()},l.last_good)
   opts=[l.opt_actor,l.opt_critic,l.opt_cts];expected=l.policy.weight.detach().clone()
   with torch.no_grad():l.policy.weight.add_(1.)
   self.assertTrue(l.rollback());self.assertTrue(torch.equal(l.policy.weight,expected))
   self.assertEqual(opts,[l.opt_actor,l.opt_critic,l.opt_cts]);self.assertEqual(l.ramp_start,1000)
if __name__=='__main__':unittest.main()
