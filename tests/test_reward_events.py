"""AS2 motion parity, hard-event accounting, and afterstate-value contracts."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch
from go2_physx_cts import task
from go2_physx_cts.calibration import build_budget
from go2_physx_cts.value import Critic, event_td_target


def budget():
    return build_budget([{'net_return':50.,'qualified_complete':True}],
                        {'policy_sha256':'unit-test-fixture','robot':'as2'})


def fixture(n=4,stable_seconds=.64):
    s = torch.zeros(n,70);s[:,0]=.3;s[:,3]=-1
    pose = torch.zeros(n,7);pose[:,2]=.3;pose[:,6]=1
    a = torch.zeros(n,12)
    constants = task.RobotConsts(torch.zeros(12),torch.ones(12),torch.ones(12),
        torch.ones(12),torch.ones(12),torch.tensor([[-2.,2.]]*12))
    telemetry = {'applied_torque':a.clone(),'joint_acceleration':a.clone(),
        'foot_velocity':torch.zeros(n,4,3),'log_contact_force':torch.zeros(n,8),
        'contact_prob':torch.cat((torch.ones(n,4),torch.zeros(n,10)),-1)}
    goal=pose.clone()[:,:3];goal[:,2]=0
    ctx=task.initial_task_context(pose,goal,episode_limit=8.,recovery_deadline=8.,stable_seconds=stable_seconds)
    ctx[:,task.ORIGIN]=torch.tensor([-1.,0.])
    return s,pose,a,constants,telemetry,ctx


def call(s,pose,a,constants,telemetry,ctx,*,task_id=3,terrain=10,soft=False,command=None):
    n=len(s)
    command=s.new_tensor([[.5,0,0]]).expand(n,-1) if command is None else command
    return task.task_reward(s,s,a,a,command,constants,task_id=torch.full((n,),task_id),
        terrain_id=torch.full((n,),terrain),context=ctx,pose0=pose,pose=pose,
        telemetry=telemetry,previous2=a,height_above_crop=s[:,0],soft=soft,variant='events')


class MotionParity(unittest.TestCase):
    def test_fifteen_terms_and_gradients_against_pinned_as2(self):
        root=Path(__file__).resolve().parents[2]/'research/upstream/as2_baseline_2061be6/go2_physx_cts'
        if not root.exists():
            self.skipTest('Pinned source parity is performed in the local audit workspace')
        spec=importlib.util.spec_from_file_location('_as2_reference',root/'__init__.py',submodule_search_locations=[str(root)])
        package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
        baseline=__import__('_as2_reference.task',fromlist=['reward'])
        torch.manual_seed(5)
        s,pose,a,c,telemetry,ctx=fixture(12)
        s[:,4:]=torch.randn_like(s[:,4:])*.2
        s.requires_grad_(True)
        a=torch.randn_like(a).requires_grad_(True)
        p=torch.randn_like(a);p2=torch.randn_like(a)
        command=torch.randn(12,3);command[0]=0;command[1]=torch.tensor([4.,0,0])
        telemetry['applied_torque']=torch.randn(12,12)*8
        telemetry['joint_acceleration']=torch.randn(12,12)*100
        telemetry['foot_velocity']=torch.randn(12,4,3)
        telemetry['log_contact_force']=torch.rand(12,8)*4
        aux=torch.cat((telemetry['applied_torque'],telemetry['joint_acceleration'],
            telemetry['foot_velocity'][...,:2].flatten(1),telemetry['log_contact_force']),-1)
        on_flat=torch.arange(12)%2==0
        for iteration in (0,1500,3000,40000):
            for soft in (False,True):
                raw,weighted,_=task.motion_terms(s,s,a,p,command,c,pose=pose,telemetry=telemetry,
                    previous2=p2,on_flat=on_flat,iteration=iteration,height_above_crop=s[:,0],soft_contacts=soft)
                actual=sum(weighted.values())
                expected,terms,_=baseline.reward(s,s,a,p,command,c,soft=soft,iteration=iteration,
                    height_above_crop=s[:,0],base_height_target=.30,aux=aux,previous_previous=p2,on_flat=on_flat)
                self.assertEqual(set(raw),set(terms))
                for key in raw:torch.testing.assert_close(raw[key],terms[key])
                torch.testing.assert_close(actual,expected)
                ga=torch.autograd.grad(actual.sum(),(s,a),retain_graph=True)
                gb=torch.autograd.grad(expected.sum(),(s,a),retain_graph=True)
                for x,y in zip(ga,gb):torch.testing.assert_close(x,y)


class HardEvents(unittest.TestCase):
    def setUp(self):budget()

    def test_gap_stability_bonus_once_and_continued_motion(self):
        s,p,a,c,t,ctx=fixture(1)
        for step in range(1,61):
            p[:,0]=step*.05
            out=call(s,p,a,c,t,ctx);ctx=out[2]
            self.assertEqual(float(out[4]),float(step==32))
            self.assertEqual(bool(ctx[:,task.COMPLETED]),step>=32)
            self.assertFalse(bool(out[3]))
            torch.testing.assert_close(out[0],out[1]['continuous_reward']+out[1]['event_reward'])
            torch.testing.assert_close(out[0],out[1]['positive']+out[1]['negative'])
        self.assertGreater(float(p[:,0]),float(ctx[:,task.TARGET_RADIUS]))

    def test_stability_break_restarts_counter(self):
        s,p,a,c,t,ctx=fixture(1)
        for _ in range(20):ctx=call(s,p,a,c,t,ctx)[2]
        t['contact_prob'][:,:4]=0
        ctx=call(s,p,a,c,t,ctx)[2]
        self.assertEqual(float(ctx[:,task.STABLE]),0.)
        t['contact_prob'][:,:4]=1
        for _ in range(31):
            out=call(s,p,a,c,t,ctx);ctx=out[2];self.assertFalse(bool(out[4]))
        self.assertTrue(bool(call(s,p,a,c,t,ctx)[4]))

    def test_ups_downs_recovery_sixteen_steps(self):
        for tid,terrain in ((3,11),(3,12),(4,7)):
            s,p,a,c,t,ctx=fixture(1,stable_seconds=.32)
            for i in range(16):
                out=call(s,p,a,c,t,ctx,task_id=tid,terrain=terrain);ctx=out[2]
                self.assertEqual(bool(out[4]),i==15)

    def test_failure_wins_over_success(self):
        s,p,a,c,t,ctx=fixture(1);ctx[:,task.STABLE]=31*task.DT;t['contact_prob'][:,5]=1
        out=call(s,p,a,c,t,ctx)
        self.assertTrue(bool(out[3]));self.assertFalse(bool(out[4]))
        self.assertEqual(float(out[1]['event_failure']),-5.)

    def test_recovery_does_not_fail_while_lying_until_deadline(self):
        s,p,a,c,t,ctx=fixture(1);s[:,3]=1;t['contact_prob'][:,4]=1
        before=call(s,p,a,c,t,ctx,task_id=4,terrain=7)
        self.assertFalse(bool(before[3]))
        ctx[:,task.ELAPSED]=7.98
        out=call(s,p,a,c,t,ctx,task_id=4,terrain=7)
        self.assertTrue(bool(out[3]));self.assertEqual(float(out[1]['bootstrap_mask']),0.)

    def test_imagination_never_manufactures_event_label(self):
        s,p,a,c,t,ctx=fixture(1);ctx[:,task.STABLE]=31*task.DT
        s.requires_grad_(True)
        out=call(s,p,a,c,t,ctx,soft=True)
        self.assertEqual(float(out[1]['event_reward']),0.)
        self.assertFalse(bool(out[4]));torch.testing.assert_close(out[2][:,task.STABLE],ctx[:,task.STABLE])
        out[0].sum().backward();self.assertTrue(torch.isfinite(s.grad).all())

    def test_real_recovery_positive_bound(self):
        s,p,a,c,t,ctx=fixture(1,stable_seconds=.32);ctx[:,task.STABLE]=15*task.DT
        s[:,4]=.5
        out=call(s,p,a,c,t,ctx,task_id=4,terrain=7)
        base=out[1]['positive']-out[1]['event_success']
        self.assertLessEqual(float(base),3.*task.DT+1e-7)

    def test_mode_correction_preserves_real_total(self):
        # Independent pre-refactor formula: low flat posture cost while pending,
        # no pending velocity reward, and shaping ceases on the completion step.
        for completed in (False,True):
            s,p,a,c,t,ctx=fixture(4);s[:,10:22]=.2;s[:,4]=.5
            ctx[:,task.COMPLETED]=float(completed)
            command=torch.tensor([[0.,0,0],[.5,0,0],[.5,.2,0],[.5,0,.2]])
            result=call(s,p,a,c,t,ctx,task_id=4,terrain=7,command=command)
            _,old,_=task.motion_terms(s,s,a,a,command,c,pose=p,telemetry=t,
                previous2=a,on_flat=torch.full((4,),completed),height_above_crop=s[:,0])
            expected=sum(old.values())
            if not completed:
                expected=expected-old['undesired_contacts']-old['track_xy']-old['track_yaw']
                expected=expected+1.5*task.DT-.1*task.DT
            torch.testing.assert_close(result[0],expected)
            # Both physical-state-identical continuous values are completion independent.
            alternate=ctx.clone();alternate[:,task.COMPLETED]=1-float(completed)
            other=call(s,p,a,c,t,alternate,task_id=4,terrain=7,command=command)
            torch.testing.assert_close(result[1]['continuous_reward'],other[1]['continuous_reward'])

    def test_timeout_bootstrap_differs_for_event_task(self):
        ids=torch.arange(5);zeros=torch.zeros(5);timeout=torch.ones(5,dtype=torch.bool)
        self.assertEqual(task.finite_horizon_mask(ids,zeros,timeout,'events').tolist(),[1.,1.,1.,0.,0.])
        self.assertEqual(task.finite_horizon_mask(ids,zeros,timeout,'simple').tolist(),[1.]*5)

    def test_old_go2_calibration_rejected(self):
        with self.assertRaises(ValueError):task.configure_reward_budget({'schema':'depth_reward_budget_v1','tasks':[{}]*10})


class EventValue(unittest.TestCase):
    def test_same_physics_different_stability_requirements_are_observable(self):
        budget()
        s,p,a,c,t,ctx=fixture(2,stable_seconds=torch.tensor([.64,.32]))
        ctx[:,task.STABLE]=15*task.DT
        # The reward reads the context threshold, even if geometric terrain IDs
        # are identical after reaching a flat landing region.
        out=call(s,p,a,c,t,ctx,task_id=3,terrain=11)
        self.assertEqual(out[4].tolist(),[0.,1.])
        model=Critic(hidden=(16,),variant='events',mixed_precision=False)
        x=model.event_features(torch.zeros(2,167),0,torch.tensor([3,3]),ctx,p,t['contact_prob'])
        difference=(x[0]-x[1]).nonzero().flatten().tolist()
        self.assertEqual(difference,[167+task.REQUIRED_STABLE_SECONDS])
        self.assertEqual(model.feature_dim,205)

    def test_dimensions_and_no_policy_latent(self):
        self.assertEqual(Critic(mixed_precision=False).feature_dim,169)
        self.assertEqual(Critic(variant='events',mixed_precision=False).feature_dim,205)

    def test_afterstate_preserves_pre_event_context_and_pose_gradient(self):
        s,p,a,c,t,ctx=fixture(2);p.requires_grad_(True)
        model=Critic(hidden=(16,),variant='events',mixed_precision=False)
        priv=torch.randn(2,167,requires_grad=True)
        x=model.event_features(priv,0,torch.tensor([3,4]),ctx,p,t['contact_prob'])
        torch.testing.assert_close(x[:,167+task.COMPLETED],ctx[:,task.COMPLETED])
        model.forward_event(x).sum().backward()
        self.assertGreater(float(p.grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_td_is_real_reward_plus_one_bootstrap(self):
        r=torch.tensor([21.,-5.,0.],requires_grad=True)
        y=event_td_target(r,torch.tensor([1.,0.,1.]),torch.tensor([2.,100.,4.]))
        torch.testing.assert_close(y,torch.tensor([22.98,-5.,3.96]))
        self.assertFalse(y.requires_grad)


if __name__=='__main__':unittest.main()
