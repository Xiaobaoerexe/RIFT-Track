"""Temporary integration checks: real schema, root Q, recurrent actor gradient."""
import unittest
from types import SimpleNamespace
import torch
from go2_physx_cts import state as st
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.networks import Policy
from go2_physx_cts.world_model import WorldModel
from go2_physx_cts.value import Critic
from go2_physx_cts.buffer import Replay
from go2_physx_cts.commands import OfficialCommands
from go2_physx_cts.task import RobotConsts,initial_task_context,configure_reward_budget
from go2_physx_cts.robots import as2_motor_values
from go2_physx_cts.imagine import Imagination
from go2_physx_cts.learner import frozen,branch_loss

class FakeDepth:
    def captured(self,pose,scene,frame,episode,trajectory,**kwargs):
        return pose.new_full((len(pose),1,48,64),2.)

class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.cfg=TrainCfg(device='cpu',compile=False,num_envs=8,buffer_len=80,imag_batch=8,wm_batch=8,cts_burnin=0)
        self.pose=torch.zeros(8,7);self.pose[:,2]=.4;self.pose[:,6]=1
        inertia,kp,kd,effort,scale=as2_motor_values('cpu')
        q=torch.tensor([-.1,.9,-1.8,.1,.9,-1.8,-.1,.9,-1.8,.1,.9,-1.8])
        self.c=RobotConsts(q,scale,kp,kd,effort,torch.tensor([[-3.,3.]]*12))
        self.s=st.reconcile_state(torch.zeros(8,70),self.pose,None,q)
        self.commands=OfficialCommands(.02,20.,variant='simple')
        cs=self.commands.initial(self.pose[:,:2],self.pose[:,:2],task_id=torch.zeros(8,dtype=torch.long),terrain_id=torch.full((8,),7))
        command=self.commands.command(cs);a=torch.zeros(8,12)
        obs=torch.zeros(8,450);obs=st.push_history(obs,st.frame(self.s,a,command,noise=False))
        self.seed={'s':self.s,'pose':self.pose,'a_prev':a,'a_prev2':a.clone(),'obs':obs,'cmd_state':cs,'command':command,
            'xi':torch.zeros(8,5),'task_context':initial_task_context(self.pose),'task_ids':torch.zeros(8,1,dtype=torch.long),
            'terrain_ids':torch.full((8,1),7),'frame_index':torch.zeros(8,1,dtype=torch.long),
            'depth_pose':self.pose,'depth_scene_ids':torch.zeros(8,1,dtype=torch.long),'depth_frame_index':torch.zeros(8,1,dtype=torch.long),
            'episode':torch.zeros(8,1,dtype=torch.long),'trajectory_ids':torch.arange(8),'contact_prob':torch.zeros(8,14)}
        self.policy=Policy((32,32),experts=1,mixed_precision=False)
        self.wm=WorldModel((32,32),mixed_precision=False)
        self.wm.update_stats(self.s,a,a,torch.zeros(8,5),None,self.s,torch.zeros(8,6),
            {'applied_torque':a,'foot_velocity':a,'joint_acceleration':a,'log_contact_force':a[:,:8]},pose=self.pose)

    def test_teacher_actor_gradient_and_critic_targets(self):
        critic=Critic((32,32),mixed_precision=False)
        imag=Imagination(self.cfg,self.c,None,self.commands,FakeDepth())
        with frozen(self.wm,critic,self.policy.student_encoder):
            objective,data=imag.rollout(self.wm,self.policy,critic,self.seed,0,iteration=2000)
            self.assertTrue(torch.isfinite(objective).all())
            (-objective.mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.policy.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        targets=imag.targets(critic,data)
        self.assertEqual(targets.shape,(8,32));self.assertTrue(torch.isfinite(targets).all())

    def test_event_variant_root_value_is_once(self):
        budget={'schema':'as2_depth_event_budget_v1','robot':'as2','reference_net_max':50.,'positive_cap':45.,'failure_penalty':5.,
            'source':{'policy_sha256':'test'},'tasks':[{'success_bonus':21. if i>=3 else 0.,'episode_seconds':8. if i>=3 else 20.,'base_positive_rate_bound':3.,'unfinished_rate':.1 if i>=3 else 0.} for i in range(5)]}
        configure_reward_budget(budget);self.cfg.variant='events';self.cfg.task_count=5
        critic=Critic((32,32),variant='events',mixed_precision=False)
        for p in critic.parameters():p.data.zero_()
        critic.net[-1].bias.data[1::2]=3.
        imag=Imagination(self.cfg,self.c,None,self.commands,FakeDepth())
        with torch.no_grad():objective,data=imag.rollout(self.wm,self.policy,critic,self.seed,0,iteration=2000)
        torch.testing.assert_close(data['root_event'],torch.full((8,),3.))
        torch.testing.assert_close(objective,data['immediate']+data['bootstrap']+3.)
        configure_reward_budget(None)

    def test_mixed_student_recurrence_preserves_actor_gradient(self):
        cfg=self.cfg;cfg.cts_burnin=16
        prefix={k:v[6:,None].expand(2,16,*v.shape[1:]).clone() for k,v in self.seed.items()
                if k in ('obs','depth_pose','depth_scene_ids','depth_frame_index','episode')}
        prefix['trajectory_ids']=torch.arange(6,8)
        prefix['reset']=torch.zeros(2,16,dtype=torch.bool);prefix['reset'][:,0]=True
        prefix['valid']=torch.ones(2,16,dtype=torch.bool)
        self.seed['student_prefix']=prefix
        depth=FakeDepth();depth.images=SimpleNamespace(keys=lambda pose,scene,frame,episode,trajectory: list(range(len(pose))))
        critic=Critic((32,32),mixed_precision=False)
        imag=Imagination(cfg,self.c,None,self.commands,depth)
        with frozen(self.wm,critic,self.policy.student_encoder):
            obj,data=imag.rollout(self.wm,self.policy,critic,self.seed,(6,2),iteration=2000)
            branch_loss(obj,(6,2),(6,2)).backward()
        self.assertTrue(torch.isfinite(obj).all())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.policy.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.policy.student_encoder.parameters()))

if __name__=='__main__':unittest.main()
