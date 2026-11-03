"""Collection source ownership and episode reset regressions; no simulator."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.learner import Learner
from go2_physx_cts.schedules import ProviderSchedule
from go2_physx_cts.driver import MoeCtsDriver


class CollectionProviders(unittest.TestCase):
    def learner(self,n=40):
        l=Learner.__new__(Learner);l.cfg=TrainCfg();l.device='cpu';l.it=0
        l.providers=ProviderSchedule(l.cfg)
        l.source=torch.zeros(n,dtype=torch.uint8)
        l.real_hidden=torch.ones(n,256);l.camera_episode=torch.full((n,),-1)
        l.before={'task_ids':torch.zeros(n,dtype=torch.long)}
        l.env=SimpleNamespace(iteration=0,task_assignment_version=0,snapshot=lambda:l.before)
        l.refresh_providers()
        return l

    def test_cutoff_and_no_reentry(self):
        p=ProviderSchedule(TrainCfg())
        self.assertGreater(p.fraction(0,7999),0)
        for it in (8000,8500,120000):
            self.assertTrue(all(p.fraction(t,it)==0 for t in range(10)))
        restored=ProviderSchedule(TrainCfg());restored.load_state_dict(p.state_dict())
        self.assertEqual(restored.fraction(0,8500),0)

    def test_skipped_boundary_refreshes(self):
        l=self.learner();self.assertTrue((l.source==2).any())
        l.it=l.env.iteration=8500
        l.before['task_ids']=torch.arange(10).repeat_interleave(4)
        l.sync_providers(l.before)
        self.assertFalse((l.source==2).any())
        self.assertEqual(l._provider_task_version,l.env.task_assignment_version)

    def test_task_change_without_phase_change(self):
        l=self.learner();l.it=l.env.iteration=5001;l.sync_providers(l.before)
        l.before['task_ids'].fill_(5);l.it=l.env.iteration=5002
        l.env.task_assignment_version+=1
        l.sync_providers(l.before)
        self.assertFalse((l.source==2).any())

    def test_phase_change_and_cutoff_without_task_change(self):
        l=self.learner();l.it=l.env.iteration=2501;l.sync_providers(l.before)
        self.assertEqual(l._provider_phase,1)
        l.it=8500;l.sync_providers(l.before)
        self.assertFalse((l.source==2).any())

    def test_unchanged_assignment_needs_no_device_comparison(self):
        l=self.learner();l.it=l.env.iteration=8500;l.sync_providers(l.before)
        l.refresh_providers=Mock(wraps=l.refresh_providers)
        with patch('torch.equal',side_effect=AssertionError('device comparison')):
            for it in (8501,9000,10000):
                l.it=l.env.iteration=it;l.sync_providers(l.before)
        l.refresh_providers.assert_not_called()
        l.env.task_assignment_version+=1;l.sync_providers(l.before)
        l.refresh_providers.assert_called_once()

    def test_held_iteration_refreshes_only_once(self):
        l=self.learner();l.it=l.env.iteration=4000
        l.refresh_providers=Mock(wraps=l.refresh_providers)
        for _ in range(4):l.sync_providers(l.before)
        l.refresh_providers.assert_called_once()

    def test_episode_reset_clears_official_history(self):
        l=self.learner(3);l.source.zero_();l.camera_episode=torch.tensor([7,8,9])
        l.depth_pose=torch.zeros(3,7);l.depth_scene_ids=torch.zeros(3,dtype=torch.long)
        l.depth_frame_index=torch.zeros(3,dtype=torch.long);l.real_depth=torch.zeros(3,1,48,64)
        l.driver=MoeCtsDriver.__new__(MoeCtsDriver)
        l.driver.hist=torch.ones(3,450);l.driver.last=torch.ones(3,12)
        before=dict(frame_index=torch.ones(3,dtype=torch.long),episode=torch.tensor([7,10,9]),
                    pose=torch.zeros(3,7),scene_ids=torch.zeros(3,dtype=torch.long))
        l.camera_inputs(before)
        self.assertEqual(float(l.driver.hist[1].sum()),0.)
        self.assertEqual(float(l.driver.last[1].sum()),0.)
        self.assertTrue((l.driver.hist[[0,2]]==1).all())
        l.it=8500;l.driver=Mock();l.camera_inputs(before)
        l.driver.reset.assert_not_called()

    def test_no_official_inference_after_cutoff(self):
        l=self.learner(4);l.cfg.num_envs=4;l.it=l.env.iteration=8500
        l.before.update(s=torch.zeros(4,70),a_prev=torch.zeros(4,12),command=torch.zeros(4,3),
                        xi=torch.zeros(4,5),relief=torch.zeros(4,77),obs=torch.zeros(4,450),
                        pose=torch.tensor([[0.,0.,.3,0.,0.,0.,1.]]).repeat(4,1))
        l.env.field=None;l.env.step=Mock(side_effect=RuntimeError('test stops before physics'))
        l.camera_inputs=Mock(return_value=torch.zeros(4,1,48,64))
        l.policy=SimpleNamespace(step=lambda hist,*a,**kw:(torch.zeros(len(hist),12),kw['hidden'],None,None))
        l.driver=Mock();l.ou=torch.zeros(4,12);l.index=torch.arange(4)
        with self.assertRaisesRegex(RuntimeError,'test stops before physics'):l.collect()
        l.driver.act.assert_not_called()
        self.assertFalse((l.source==2).any())


if __name__=='__main__':unittest.main()
