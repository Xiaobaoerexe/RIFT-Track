"""CPU contract checks; simulator readback is additionally required before launch."""
import ast
import math
import pathlib
import types
import unittest
import xml.etree.ElementTree as ET
import torch
from go2_physx_cts import state as st
from go2_physx_cts.commands import OfficialCommands,VelocityGrid,TERRAIN_NAMES,TASK_TERRAINS,task_assignments,CMD_STATE_DIM,COMPLETED,TASK_ID,TERRAIN_ID,GRID_BIN,COMMAND,TIMER
from go2_physx_cts.robots import as2_motor_values,profile

ROOT=pathlib.Path(__file__).resolve().parents[1]

def isolated_function(filename,name,namespace):
    tree=ast.parse((ROOT/'go2_physx_cts'/filename).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),namespace)
    return namespace[name]

class PhysicsContracts(unittest.TestCase):
    def test_as2_profile_and_motors(self):
        p=profile('as2');inertia,kp,kd,effort,scale=as2_motor_values('cpu')
        self.assertEqual((p.initial_height,p.left_hip,p.reward_height),(.4,-.1,.30))
        for tensor,expected in ((inertia,[.026,.026,.038]),(kp,[40,40,60]),(kd,[2,2,3]),(effort,[50,50,75])):
            torch.testing.assert_close(tensor,torch.tensor(expected*4,dtype=torch.float32))
        torch.testing.assert_close(scale,torch.full((12,),.3125))
        self.assertEqual((st.STATE_DIM,st.POSE_DIM,st.POSE_DELTA_DIM,st.AUX_DIM),(70,7,6,40))

    def test_fk_matches_official_urdf_and_is_differentiable(self):
        tree=ET.parse(ROOT/'assets/as2/official/as2_description/urdf/as2.urdf').getroot()
        joints={j.get('name'):j for j in tree.findall('joint')}
        q=(torch.rand(8,12,dtype=torch.float64)-.5).requires_grad_()
        predicted=st.forward_kinematics(q)
        expected=[[],[],[]]
        for li,leg in enumerate(st.LEGS):
            pos=torch.zeros(8,3,dtype=q.dtype);rotation=torch.eye(3,dtype=q.dtype).expand(8,3,3)
            for ji,part in enumerate(('hip','thigh','calf','foot')):
                joint=joints[f'{leg}_{part}_joint'];offset=q.new_tensor([float(x) for x in joint.find('origin').get('xyz').split()])
                pos=pos+torch.einsum('bij,j->bi',rotation,offset)
                if ji:expected[ji-1].append(pos)
                if ji<3:
                    a=q[:,li*3+ji];c,s=a.cos(),a.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
                    r=torch.stack((o,z,z,z,c,-s,z,s,c),-1).reshape(-1,3,3) if ji==0 else torch.stack((c,z,s,z,o,z,-s,z,c),-1).reshape(-1,3,3)
                    rotation=rotation@r
        for actual,wanted in zip(predicted,expected):torch.testing.assert_close(actual,torch.stack(wanted,1),atol=1e-12,rtol=1e-12)
        predicted[-1].square().sum().backward();self.assertTrue(torch.isfinite(q.grad).all());self.assertGreater(float(q.grad.abs().max()),0.)

    def test_source_terrain_upgrade_with_special_exceptions(self):
        f=isolated_function('terrain_scene.py','terrain_level_decisions',{'torch':torch})
        d=torch.tensor([4.1,3.9,.9,4.,.1]);cmd=torch.tensor([[1.,0.]]*5)
        up,down=f(d,cmd,5.,.1,9.,.5)
        self.assertEqual(up.tolist(),[True,False,False,False,False])
        self.assertEqual(down.tolist(),[False,False,True,False,True])
        up,down=f(d,cmd,5.,.1,9.,.5,success=torch.tensor([False,True,False,False,False]),task_ids=torch.tensor([3,3,3,4,4]))
        self.assertEqual(up.tolist(),[False,True,False,False,False]);self.assertEqual(down.tolist(),[True,False,True,False,False])

    def test_no_avoidance_terrain_and_equal_task_counts(self):
        self.assertEqual(len(TERRAIN_NAMES),13);self.assertNotIn('indoor_obstacles',TERRAIN_NAMES)
        self.assertEqual(TASK_TERRAINS[3],(10,11,12));self.assertEqual(TASK_TERRAINS[4],(7,))
        for variant in ('events','moe'):
            self.assertEqual(torch.bincount(task_assignments(4096,5000,'cpu',variant)).tolist(),[820,819,819,819,819])
        self.assertEqual(int(task_assignments(4096,120000,'cpu','simple').max()),0)
        source=(ROOT/'go2_physx_cts/terrain_scene.py').read_text(encoding='utf-8')
        self.assertNotIn('indoor_obstacles',source)

class Commands(unittest.TestCase):
    def fixture(self,variant,n=64,task=0,terrain=7,iteration=35000):
        cmd=OfficialCommands(variant=variant)
        tasks=torch.full((n,),task,dtype=torch.long);terrains=torch.full_like(tasks,terrain)
        s=cmd.initial(torch.zeros(n,2),torch.zeros(n,2),iteration,task_id=tasks,terrain_id=terrains)
        return cmd,s

    def test_simple_preserves_source_sampler_and_speed_cap(self):
        cmd,s=self.fixture('simple',n=1024)
        self.assertEqual(s.shape,(1024,CMD_STATE_DIM));self.assertIsNone(cmd.grid)
        self.assertLessEqual(float(s[:,COMMAND][:,0].max()),8.00001)
        self.assertGreater(float(s[:,COMMAND][:,0].max()),7.9)
        self.assertGreaterEqual(float(s[:,COMMAND][:,0].min()),-4.50001)
        xy=torch.ones(1024,2,requires_grad=True);out=cmd.advance(s,xy,35000)
        out[:,12].sum().backward();self.assertTrue(torch.isfinite(xy.grad).all());self.assertGreater(float(xy.grad.abs().sum()),0.)

    def test_special_completion_restores_three_dimensional_commands(self):
        cmd,s=self.fixture('events',n=1024,task=3,terrain=10)
        self.assertTrue((s[:,COMMAND][:,1:]==0).all());self.assertTrue((s[:,COMMAND][:,0]>=.2).all())
        s=cmd.advance(s,torch.zeros(1024,2),35000,completed=torch.ones(1024))
        self.assertTrue((s[:,COMPLETED]==1).all());self.assertGreater(float(s[:,COMMAND][:,1:].abs().max()),.5)
        s=cmd.reset(s,torch.ones(1024,dtype=torch.bool),torch.zeros(1024,2),torch.zeros(1024,2),35000)
        self.assertTrue((s[:,COMPLETED]==0).all());self.assertTrue((s[:,COMMAND][:,1:]==0).all())

    def test_rapid_turn_resampling_and_pure_forward_fraction(self):
        cmd,s=self.fixture('events',n=512,task=2)
        torch.testing.assert_close(s[:,TIMER],torch.ones(512))
        pure=(s[:,COMMAND][:,1:]==0).all(-1).float().mean()
        self.assertGreaterEqual(float(pure),.3)

    def test_grid_uses_only_own_data_separates_terrain_and_roundtrips(self):
        grid=VelocityGrid();tasks=torch.tensor([0,0,4,0]);terrains=torch.tensor([7,1,7,7]);bins=grid.bin_indices(torch.tensor([[.5,0.,0.]]*4))
        grid.record(tasks,terrains,bins,torch.tensor([1.,0.,1.,0.]),torch.tensor([True,True,True,False]))
        self.assertEqual(float(grid.count.sum()),2.)
        result=grid.update(6000)
        self.assertGreater(float(grid.extent[0,7,0]),float(grid.extent[0,1,0]))
        self.assertEqual(result['0']['terrains']['flat']['samples'],1)
        clone=VelocityGrid();clone.load_state_dict(grid.state_dict())
        torch.testing.assert_close(clone.extent,grid.extent);torch.testing.assert_close(clone.weights,grid.weights)

if __name__=='__main__':unittest.main()
