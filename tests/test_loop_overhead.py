"""Execution-only optimizations preserve log reductions and save semantics."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from go2_physx_cts.learner import Learner
from go2_physx_cts.physx_env import materialize_metrics


class LoopOverhead(unittest.TestCase):
    def test_deferred_metrics_are_exact_per_step(self):
        torch.manual_seed(17)
        pending=[];expected=[]
        for step in range(50):
            n=40;s=torch.randn(n,70);command=torch.randn(n,3)
            velocity=torch.randn(n,3);reward=torch.randn(n)
            tasks=torch.randint(10,(n,));source=torch.randint(3,(n,))
            own=source<2;terminated=torch.rand(n)<.2
            scalars=(s[own,0].mean(),(velocity[own,:2]-command[own,:2]).square().mean().sqrt(),
                     (s[own,9]-command[own,2]).square().mean().sqrt(),reward[own].mean(),
                     (terminated&own).sum(),terminated.sum())
            counts=torch.bincount(tasks[own],minlength=10).float()
            bad=torch.bincount(tasks[own&terminated],minlength=10).float()
            rates=bad/counts.clamp_min(1)*60/.02
            pending.append({'_packed':torch.cat((torch.stack([v.double() for v in scalars]),rates.double(),counts.double())),
                            'source_counts':torch.bincount(source,minlength=3)})
            row=dict(zip(('height','xy_rmse','yaw_rmse','reward'),map(float,scalars[:4])))
            row.update(falls=int(scalars[4]),falls_all=int(scalars[5]),
                       task_metrics={str(t):{'falls_per_env_min':float(rates[t]),'own_envs':int(counts[t])}
                                     for t in range(10) if counts[t]>0},
                       source_counts=torch.bincount(source,minlength=3).tolist())
            expected.append(row)
        actual=materialize_metrics(pending)
        self.assertEqual(actual,expected)
        # Log and collapse history consume the same per-step means, not pooled RMSE.
        for key in ('height','xy_rmse','yaw_rmse','reward','falls','falls_all'):
            self.assertEqual(sum(m[key] for m in actual)/50,sum(m[key] for m in expected)/50)
        self.assertEqual(materialize_metrics([]),[])

    def runner(self,start,end,save_interval=5):
        l=Learner.__new__(Learner);l.it=start
        l.cfg=SimpleNamespace(iterations=end,wm_window=8,warmup_iterations=1,policy_start=100,
            cts_burnin=16,cts_sequence_steps=64,log_interval=5,save_interval=save_interval,
            evaluation_interval=5,evaluation_python='python')
        l.env=SimpleNamespace(iteration=start,allocation_status=lambda:{})
        l.buf=SimpleNamespace(size=0);l.schedule=SimpleNamespace(student_start=None,teacher_stable=False,check_student=Mock())
        l.feedback=SimpleNamespace(finish=lambda:None);l.learning_rates=lambda:{}
        l.collect=Mock();l.improve_policy=Mock(return_value={});l.log=Mock()
        l.save=Mock(side_effect=lambda:f'model_{l.it}.pt');l.evaluate_feedback=Mock()
        l.last_good=None;l.collapse_strikes=0
        return l

    def test_save_reused_for_evaluation_and_timing_sampled(self):
        l=self.runner(1,6);l.run()
        self.assertEqual(l.save.call_count,2)  # 5 and final6
        l.evaluate_feedback.assert_called_once_with('model_5.pt')
        self.assertEqual([c.kwargs['detailed_timing'] for c in l.improve_policy.call_args_list],
                         [False,False,False,True,True])

    def test_evaluation_without_periodic_save_still_saves(self):
        l=self.runner(4,6,save_interval=10);l.run()
        self.assertEqual(l.save.call_count,2)
        l.evaluate_feedback.assert_called_once_with('model_5.pt')

    def test_post_evaluation_schedule_save_retained(self):
        l=self.runner(4,5)
        l.schedule.check_student=lambda it:setattr(l.schedule,'student_start',it)
        l.run();self.assertEqual(l.save.call_count,2)


if __name__=='__main__':unittest.main()
