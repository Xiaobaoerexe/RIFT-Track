"""Semantic regression checks for held-depth WM supervision and gradient owners."""
import unittest
from types import SimpleNamespace
import torch
from go2_physx_cts import state as st
from go2_physx_cts.latent import camera_context,prepare_latent_targets
from go2_physx_cts.networks import Policy,PlainActor,Student
from go2_physx_cts.world_model import WorldModel


def poses(n):
    p=torch.zeros(n,7);p[:,2]=.4;p[:,6]=1
    return p


class FakeDepth:
    def __init__(self):
        self.field=SimpleNamespace(scene_ids=lambda p:torch.zeros(len(p),dtype=torch.long))
        self.reads=[]
    def captured(self,pose,scene,frame,episode,trajectory):
        self.reads.append((pose.clone(),frame.clone(),episode.clone(),trajectory.clone()))
        # Pose/frame-dependent labels expose incorrect reset and timing keys.
        value=(pose[:,0]+frame.float()*.01+1).reshape(-1,1,1,1)
        return value.expand(-1,1,48,64).clone()
    def paired(self,pose,scene,frame,episode,trajectory):
        image=self.captured(pose,scene,frame,episode,trajectory)
        return image,image+.05


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.proj=torch.nn.Linear(1,128)
    def forward(self,image):return self.proj(image.mean((1,2,3))[:,None])


def window(batch=3,steps=3):
    p=poses(batch)[:,None].expand(-1,steps,-1).clone()
    s=st.reconcile_state(torch.zeros(batch,70),p[:,0])[:,None].expand(-1,steps,-1).clone()
    frames=torch.arange(8,8+steps)[None].expand(batch,-1).clone()
    w={'s':s,'s_next':s.clone(),'pose':p,'pose_next':p.clone(),
       'a':torch.zeros(batch,steps,12),'a_prev':torch.zeros(batch,steps,12),
       'xi':torch.zeros(batch,steps,5),'valid':torch.ones(batch,steps),
       'applied_torque':torch.zeros(batch,steps,12),'foot_velocity':torch.zeros(batch,steps,12),
       'joint_acceleration':torch.zeros(batch,steps,12),'log_contact_force':torch.zeros(batch,steps,8),
       'contact_prob':torch.zeros(batch,steps,14),'depth_pose':p.clone(),
       'frame_index':frames[...,None],'episode':torch.full((batch,steps,1),12),
       'depth_frame_index':torch.zeros(batch,steps,1,dtype=torch.long),
       'depth_scene_ids':torch.zeros(batch,steps,1,dtype=torch.long),
       'trajectory_ids':torch.arange(batch)}
    w['pose_next'][:,:,0]=torch.arange(batch)[:,None]*.1
    return w


class VisualLatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_plain_actor_and_teacher_do_not_use_task_labels(self):
        policy=Policy(hidden=(32,16),experts=1,mixed_precision=False)
        self.assertIsInstance(policy.actor,PlainActor)
        self.assertFalse(any('router' in n for n,_ in policy.actor.named_parameters()))
        priv=torch.randn(2,st.PRIV_DIM);height=torch.randn(2,2,16,16)
        a=policy.teacher_latent(priv,height,torch.tensor([0,1]))
        b=policy.teacher_latent(priv,height,torch.tensor([9,4]))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        depth_ids={id(p) for p in policy.student_encoder.depth_encoder.parameters()}
        cts_ids={id(p) for p in policy.student_encoder.distillation_parameters()}
        self.assertFalse(depth_ids&cts_ids)
        self.assertEqual(depth_ids|cts_ids,{id(p) for p in policy.student_encoder.parameters()})

    def test_held_latent_refresh_and_action_gradient(self):
        wm=WorldModel(hidden=(32,32),mixed_precision=False)
        # At cold init the source-style zero delta head intentionally predicts
        # persistence. A nonzero learned head must carry input gradients.
        with torch.no_grad():wm.net[-1].weight[-128:].normal_(std=.01)
        wm.requires_grad_(False)
        p=poses(2);s=st.reconcile_state(torch.zeros(2,70),p)
        a=torch.zeros(2,12,requires_grad=True);z=torch.randn(2,128,requires_grad=True)
        args=(s,a,torch.zeros_like(a),torch.zeros(2,5),torch.randn(2,2,9,9),p)
        _,_,telemetry,nz=wm.predict(*args,z,torch.tensor([8,9]),p)
        torch.testing.assert_close(nz[0],z[0],rtol=0,atol=0)
        self.assertEqual(telemetry['reward_aux'].shape,(2,40))
        grad=torch.autograd.grad(nz[1].square().mean(),a)[0]
        self.assertGreater(grad[1].abs().sum().item(),0)
        self.assertEqual(grad[0].abs().sum().item(),0)
        self.assertEqual(wm.predict(*args)[3],None)
        self.assertTrue(all(p.grad is None for p in wm.parameters()))

    def test_real_targets_only_refresh_and_before_reset(self):
        w=window();w['valid'][1,1:]=0
        depth=FakeDepth();encoder=TinyEncoder()
        target=prepare_latent_targets(w,depth,encoder,reconstruction_images=2)
        self.assertFalse(target.initial.requires_grad)
        self.assertFalse(target.targets.requires_grad)
        self.assertEqual(target.refresh.tolist(),[[False,True,False],[False,False,False],[False,True,False]])
        refreshed=depth.reads[1]
        self.assertEqual(refreshed[1].tolist(),[10,10])
        self.assertEqual(refreshed[2].tolist(),[12,12])
        torch.testing.assert_close(refreshed[0],w['pose_next'][[0,2],1])

    def test_only_reconstruction_updates_shared_cnn(self):
        w=window();encoder=TinyEncoder();depth=FakeDepth()
        target=prepare_latent_targets(w,depth,encoder,reconstruction_images=2)
        wm=WorldModel(hidden=(32,32),mixed_precision=False)
        loss,terms=wm.loss(w,return_terms=True,encoder=encoder,latent_targets=target)
        self.assertTrue(torch.isfinite(loss))
        (terms['state']+terms['latent']).backward(retain_graph=True)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        terms['reconstruction'].backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in encoder.parameters()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None))

    def test_frozen_student_retains_latent_input_gradient(self):
        student=Student(mixed_precision=False).requires_grad_(False)
        z=torch.randn(2,128,requires_grad=True)
        fused,_,hidden=student.forward_features(torch.randn(2,450),torch.randn(2,45),z,torch.zeros(2,256))
        grad=torch.autograd.grad(fused[:,0].sum()+hidden[:,0].sum(),z)[0]
        self.assertGreater(grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in student.parameters()))

    def test_camera_context_full_rotation_has_finite_gradient(self):
        p=poses(2).requires_grad_();context=camera_context(p,torch.tensor([0,8]),poses(2))
        self.assertEqual(context.shape,(2,7))
        g=torch.autograd.grad(context.sum(),p)[0]
        self.assertTrue(torch.isfinite(g).all())


if __name__=='__main__':unittest.main()
