"""Real/imagined recovery deadlines, reward events and timeout bootstrap."""
import unittest
from unittest.mock import patch
import torch
from go2_physx_cts import task
from go2_physx_cts.imagine import time_limit_return,time_limit_lambda_targets


class EpisodeClock(unittest.TestCase):
    def test_exact_ticks_and_legacy_drift(self):
        elapsed=torch.zeros(1)
        for step in range(1,1001):
            elapsed=task.advance_episode_time(elapsed)
            self.assertTrue(torch.equal(elapsed,torch.tensor([step])*task.DT))
            self.assertEqual(bool(task.episode_time_reached(elapsed,8.)),step>=400)
        legacy=torch.zeros(1)
        for _ in range(399):legacy+=task.DT
        self.assertEqual(float(task.advance_episode_time(legacy)),8.)

    def test_non_grid_deadline_not_early(self):
        self.assertFalse(bool(task.episode_time_reached(torch.tensor(8.),8.005)))
        self.assertTrue(bool(task.episode_time_reached(torch.tensor(8.02),8.005)))

    def reward(self,step,soft=False,completed=False,grad=False):
        s=torch.zeros(1,70);s[:,0]=.38;s[:,3]=-1. if completed else 1.;s[:,4]=.2
        s.requires_grad_(grad)
        pose=torch.tensor([[0.,0.,.38,0.,0.,0.,1.]])
        ctx=task.initial_task_context(pose,episode_limit=8.,recovery_deadline=8.)
        ctx[:,task.ELAPSED]=step*task.DT;ctx[:,task.COMPLETED]=float(completed)
        contacts=torch.zeros(1,14)
        if completed:contacts[:,:4]=1.
        c=task.RobotConsts(torch.zeros(12),torch.ones(12),torch.ones(12),torch.ones(12),
                          torch.ones(12),torch.tensor([[-2.,2.]]*12))
        a=torch.zeros(1,12);budget={'tasks':[{'success_bonus':10.,'progress_budget':0.} for _ in range(10)],'failure_penalty':18.}
        with patch.object(task,'_BUDGET',budget):
            out=task.task_reward(s,s,a,a,torch.tensor([[1.,0.,0.]]),c,
                task_id=torch.tensor([6]),terrain_id=torch.tensor([7]),context=ctx,pose0=pose,pose=pose,
                telemetry={'contact_prob':contacts,'applied_torque':a,'foot_velocity':torch.zeros(1,4,3)},
                previous2=a,soft=soft,height_above_crop=torch.tensor([.38]))
        return out,s

    def test_recovery_failure_on_400th_step_hard_and_soft(self):
        for soft in (False,True):
            before,_=self.reward(398,soft)
            self.assertEqual(float(before[3]),0.)
            self.assertFalse(bool(task.episode_time_reached(before[2][:,task.ELAPSED],8.)))
            at,_=self.reward(399,soft)
            self.assertEqual(float(at[2][:,task.ELAPSED]),8.)
            self.assertEqual(float(at[3]),1.)
            self.assertEqual(float(at[1]['event_failure']),-18.)
            self.assertTrue(bool(task.episode_time_reached(at[2][:,task.ELAPSED],8.)))

    def test_completed_recovery_keeps_pure_timeout(self):
        for soft in (False,True):
            at,_=self.reward(399,soft,completed=True)
            self.assertEqual(float(at[3]),0.)
            self.assertEqual(float(at[1]['event_failure']),0.)
            self.assertTrue(bool(task.episode_time_reached(at[2][:,task.ELAPSED],8.)))

    def test_failure_suppresses_timeout_bootstrap(self):
        rewards=torch.tensor([[-18.],[2.]])
        done=torch.tensor([[1.],[0.]])
        timeout=torch.ones(2,1,dtype=torch.bool);values=torch.tensor([[0.,100.],[0.,100.]])
        result=time_limit_return(rewards,done,timeout,values,.99)
        target=time_limit_lambda_targets(rewards,done,timeout,values,.99,.95)
        torch.testing.assert_close(result[0],torch.tensor([-18.,101.]))
        torch.testing.assert_close(target[:,0],result[0])

    def test_physical_reward_gradients_remain(self):
        out,s=self.reward(100,soft=True,completed=True,grad=True)
        out[0].sum().backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(float(s.grad[:,4].abs().sum()),0.)

    def test_clock_compiles_fullgraph(self):
        fn=torch.compile(lambda t:task.episode_time_reached(task.advance_episode_time(t),8.),
                         backend='eager',fullgraph=True)
        self.assertEqual(fn(torch.tensor([7.96,7.98])).tolist(),[False,True])


if __name__=='__main__':unittest.main()
