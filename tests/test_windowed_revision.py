import unittest,math
import torch
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.schedules import LearningRateController
from go2_physx_cts.terrain_allocation import redistribute,TerrainAllocator
from go2_physx_cts.networks import Policy
from go2_physx_cts.learner import frozen,critic_branch_loss
from go2_physx_cts import state as st
torch.set_num_threads(2)
class Revision(unittest.TestCase):
    def add(self,c,end,reward=1.,err=.2,falls=.3,context='fixed',wm=1.,critic=.1,cts=.1,stable=False):
        for it in range(end-950,end+1,50):
            g={'reward':reward,'xy_rmse':err,'falls':falls}
            c.observe(dict(iteration=it,reward=reward,xy_rmse=err,falls=falls,context=context,
                wm_error=wm,critic_error=critic,cts_error=cts,teacher_drift=.005,cts_stable=stable,
                groups={'0/0/0':g,'1/0/0':g}))
        return c.adjust(end)
    def test_full_windows_cooldown_recovery_no_ramp_reset(self):
        c=LearningRateController(TrainCfg())
        self.add(c,6000);self.add(c,7000,.8,.3,.6)
        r=self.add(c,8000,.6,.5,1.)
        self.assertAlmostEqual(c.current['actor'],2e-4/1.3)
        self.assertEqual(c.ramp_start,1500);self.assertEqual(c.rates(8001)['actor_ramp'],1.)
        self.add(c,9000,.61,.49,.99);self.add(c,10000,.62,.48,.98)
        self.assertAlmostEqual(c.current['actor'],2e-4)
        self.assertFalse(hasattr(c,'rollback'));self.assertFalse(hasattr(c,'actor_trial'))
    def test_context_change_cannot_reduce(self):
        c=LearningRateController(TrainCfg());self.add(c,6000)
        self.add(c,7000,.1,1,3,context='harder');self.add(c,8000,.08,2,5,context='reallocated')
        self.assertEqual(c.current['actor'],2e-4)
    def test_wm_student_plateau_and_restore(self):
        c=LearningRateController(TrainCfg())
        for i in (6000,7000,8000):self.add(c,i,wm=10**(i/1000-6),cts=10**(i/1000-6))
        self.assertEqual(c.current['wm'],2e-4);self.assertEqual(c.current['cts'],3e-4)
        d=LearningRateController(TrainCfg());d.load_state_dict(c.state_dict())
        self.assertEqual(c.state_dict(),d.state_dict())
    def test_critic_cap(self):
        c=LearningRateController(TrainCfg());self.add(c,6000)
        self.add(c,7000,critic=.2);self.add(c,8000,critic=.4)
        self.assertAlmostEqual(c.current['critic'],2e-4/1.3)
        self.assertLessEqual(c.current['critic'],2e-4)
    def test_allocation_conservation_and_matching(self):
        base=[410]*9+[406];c=base[:]
        for _ in range(20):
            n=redistribute(c,[0,1],[2,3,4],base)
            self.assertEqual(sum(n),4096);self.assertTrue(all(200<=a<=2*b for a,b in zip(n,base)))
            self.assertEqual(n[5:],c[5:]);c=n
        a=TerrainAllocator(base)
        report={str(i):dict(valid=True,cohort_id='same',rate=.3 if i<2 else .9) for i in range(10)}
        self.assertFalse(a.observe(20000,report,base)['changed'])
        self.assertTrue(a.observe(25000,report,base)['changed'])
        for r in report.values():r['cohort_id']='different'
        self.assertFalse(a.observe(30000,report,base)['changed'])
    def test_critic_equal_branch_loss(self):
        def loss(nt,ns):
            tags=torch.zeros(nt+ns,2);tags[:nt,0]=1;tags[nt:,1]=1
            err=torch.cat((torch.ones(nt),torch.ones(ns)*2)).requires_grad_()
            value=critic_branch_loss(err,torch.zeros_like(err),tags,1.)
            value.backward()
            self.assertAlmostEqual(float(err.grad[:nt].sum()),1.,places=6)
            self.assertAlmostEqual(float(err.grad[nt:].sum()),1.,places=6)
            return float(value)
        self.assertAlmostEqual(loss(12,4),loss(4,12));self.assertAlmostEqual(loss(12,4),2.)
    def test_action_distill_keeps_actor_frozen_but_student_connected(self):
        p=Policy((32,32));hist=torch.randn(12,450);priv=torch.randn(12,st.PRIV_DIM)
        with frozen(p.actor,p.teacher_encoder):
            with torch.no_grad():target=p(hist,priv,0)
            pred=p(hist,None,1);(pred-target).square().mean().backward()
        self.assertTrue(all(v.grad is None for v in p.actor.parameters()))
        self.assertTrue(all(v.grad is None for v in p.teacher_encoder.parameters()))
        self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for v in p.student_encoder.parameters()))

    def test_student_stability_decay_after_cohort_refresh(self):
        c=LearningRateController(TrainCfg())
        self.add(c,40000,context='old',stable=True)
        self.add(c,41000,context='new',stable=True)
        self.assertEqual(c.current['cts'],3e-4)
        self.add(c,42000,context='new',stable=True)
        self.assertAlmostEqual(c.current['cts'],3e-4/1.3)
        self.add(c,43000,context='new',stable=True)
        self.assertAlmostEqual(c.current['cts'],3e-4/1.3)
    def test_wm_training_windows_exclude_holdout(self):
        from go2_physx_cts.buffer import Replay
        b=Replay(8,20,'cpu');b.size=20;b.ptr=0
        for i in range(8):b.data['s'][i,:,0]=i
        b.wm_holdout_envs=torch.tensor([1,5])
        train=b.windows(512,8,None);valid=b.windows(128,8,None,env_ids=b.wm_holdout_envs)
        self.assertFalse(bool(torch.isin(train['s'][:,0,0],b.wm_holdout_envs.float()).any()))
        self.assertTrue(bool(torch.isin(valid['s'][:,0,0],b.wm_holdout_envs.float()).all()))

if __name__=='__main__':unittest.main()
