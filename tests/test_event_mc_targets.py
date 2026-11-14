"""The diagnostic's MC labels use only real finite episode event rewards."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
import torch

spec=importlib.util.spec_from_file_location('event_value_tool',Path(__file__).resolve().parents[1]/'tools/validate_event_value.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


class EventMC(unittest.TestCase):
    def test_reset_readback_exports_actual_named_pose_and_joints(self):
        pose=torch.zeros(2,7);pose[:,6]=1
        data=SimpleNamespace(root_link_pose_w=pose,joint_pos=torch.full((2,12),.2),
            joint_vel=torch.zeros(2,12),body_link_pose_w=pose[:,None,:].expand(2,3,7))
        env=SimpleNamespace(robot=SimpleNamespace(data=data,joint_names=[str(i) for i in range(12)],body_names=['a','b','c']),
            recovery_kind=torch.tensor([0,2]),task_ids=torch.tensor([4,4]),terrain_ids=torch.tensor([7,7]),
            origins=torch.zeros(2,3),task_context=torch.zeros(2,21),command=torch.zeros(2,3),
            consts=SimpleNamespace(soft_limits=torch.tensor([[-1.,1.]]*12)))
        with tempfile.TemporaryDirectory() as folder:
            report=tool.save_reset_evidence(env,[('natural_recovery',4,7,False)],torch.zeros(2,dtype=torch.long),
                torch.tensor([0,1]),torch.tensor([False,True]),folder)
            loaded=torch.load(Path(folder)/'recovery_reset_states.pt',weights_only=False)
            torch.testing.assert_close(loaded['root_pose_w'],pose)
            torch.testing.assert_close(loaded['joint_position'],data.joint_pos)
            self.assertEqual(report['samples'],2)
            self.assertTrue(report['joint_limits_respected'])
            self.assertFalse(report['collision_overlap_tested'])

    def test_no_cross_reset_reward_leak(self):
        reward=torch.tensor([[1.,5.,1000.],[-1.,-2.,-5.]])
        done=torch.tensor([[False,True,False],[False,False,True]])
        valid=torch.tensor([[True,True,False],[True,True,True]])
        actual=tool.discounted_event_returns(reward,done,valid,.9)
        expected=torch.tensor([[5.5,5.,0.],[-6.85,-6.5,-5.]])
        torch.testing.assert_close(actual,expected)

    def test_bonus_occurs_once_then_continuation_corrections(self):
        reward=torch.tensor([[0.,20.,-.2,-.2]])
        done=torch.tensor([[False,False,False,True]])
        actual=tool.discounted_event_returns(reward,done,torch.ones_like(done),1.)
        torch.testing.assert_close(actual,torch.tensor([[19.6,19.6,-.4,-.2]]))


if __name__=='__main__':unittest.main()
