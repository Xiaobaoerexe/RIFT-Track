"""Integer tick deadlines and AS2 versus finite-task value bootstraps."""
import unittest
import torch
from go2_physx_cts import task
from go2_physx_cts.imagine import time_limit_return,time_limit_lambda_targets


class EpisodeClock(unittest.TestCase):
    def test_exact_ticks_and_legacy_drift(self):
        elapsed=torch.zeros(1)
        for step in range(1,1001):
            elapsed=task.advance_episode_time(elapsed)
            torch.testing.assert_close(elapsed,torch.tensor([step])*task.DT,rtol=0,atol=0)
            self.assertEqual(bool(task.episode_time_reached(elapsed,8.)),step>=400)
        legacy=torch.zeros(1)
        for _ in range(399):legacy+=task.DT
        self.assertEqual(float(task.advance_episode_time(legacy)),8.)

    def test_non_grid_deadline_not_early(self):
        self.assertFalse(bool(task.episode_time_reached(torch.tensor(8.),8.005)))
        self.assertTrue(bool(task.episode_time_reached(torch.tensor(8.02),8.005)))

    def test_failure_and_finite_deadline_remove_tail(self):
        rewards=torch.tensor([[-5.],[2.],[2.]])
        done=torch.tensor([[1.],[0.],[0.]])
        timeout=torch.ones(3,1,dtype=torch.bool)
        # Caller supplies zero finite-task tail, ordinary AS2 truncations bootstrap.
        values=torch.tensor([[0.,100.],[0.,100.],[0.,0.]])
        result=time_limit_return(rewards,done,timeout,values,.99)
        targets=time_limit_lambda_targets(rewards,done,timeout,values,.99,.95)
        torch.testing.assert_close(result[0],torch.tensor([-5.,101.,2.]))
        torch.testing.assert_close(targets[:,0],result[0])

    def test_clock_compiles_fullgraph(self):
        fn=torch.compile(lambda t:task.episode_time_reached(task.advance_episode_time(t),8.),
                         backend='eager',fullgraph=True)
        self.assertEqual(fn(torch.tensor([7.96,7.98])).tolist(),[False,True])


if __name__=='__main__':unittest.main()
