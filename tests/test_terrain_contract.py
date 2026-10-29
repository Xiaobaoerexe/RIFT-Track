"""CPU source/math contracts; actual PhysX and ray collision tests are separate."""
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from go2_physx_cts.physx_env import PhysXEnv,motor_values

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'go2_physx_cts/terrain_scene.py'
tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
selected=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and
          node.name in ('TerrainPositionLookup','terrain_level_decisions','height_grid_axis')]
scope={'torch':torch,'np':np}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),scope)
Lookup=scope['TerrainPositionLookup'];decisions=scope['terrain_level_decisions']

class TerrainContracts(unittest.TestCase):
    def lookup(self):
        names=('wave','slope_up','slope_down','rough_slope','stairs_up','stairs_down','obstacles','flat','wheels','small_squares')
        proportions=(.05,.1,.1,.05,.25,.1,.1,.15,.05,.05)
        cfg=SimpleNamespace(sub_terrains={name:SimpleNamespace(proportion=p) for name,p in zip(names,proportions)},
            num_rows=10,num_cols=20,size=(9.,9.),border_width=25.,sub_terrain_border_width=.5,seed=17)
        return Lookup(cfg,'cpu')

    def test_source_allocation_and_seams(self):
        lookup=self.lookup()
        self.assertEqual(lookup.metadata['column_counts'],[1,2,2,1,5,2,2,3,1,1])
        self.assertEqual(lookup.metadata['raw_proportion_sum'],1.)
        positions=torch.tensor([[-45.,-90.],[-45.,-81.],[44.999,89.999],[45.,0.],[0.,90.],[-45.001,0.],[float('nan'),0.]])
        self.assertEqual(lookup.query(positions).tolist(),[0,1,9,-1,-1,-1,-1])
        self.assertEqual(lookup(positions).tolist(),lookup.query(positions).tolist())

    def test_distance_progression_strict_and_exclusive(self):
        maximum=torch.tensor([4.,4.001,0.,1.,5.])
        accumulated=torch.tensor([[2.,0.],[0.,0.],[0.,0.],[2.,0.],[100.,0.]])
        up,down=decisions(maximum,accumulated,5.,.1)
        self.assertEqual(up.tolist(),[False,True,False,False,True])
        self.assertEqual(down.tolist(),[True,False,False,True,False])

    def test_large_heightfield_grid_has_no_float32_step_drift(self):
        axis=scope['height_grid_axis'](-115.,115.,.05)
        expected=-115.+np.arange(4601,dtype=np.float64)*.05
        self.assertEqual(len(axis),4601)
        self.assertLess(float(np.abs(axis-expected).max()),1e-5)
        self.assertEqual(float(axis[-1]),115.)

    def test_custom_geometry_rng_is_reproducible_and_restored(self):
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CapturedTerrain')
        class Importer:
            def __init__(self,cfg):self.samples=np.random.uniform(size=16)
        namespace={'np':np,'TerrainImporter':Importer}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(seed=17))
        np.random.seed(101);state=np.random.get_state();first=namespace['CapturedTerrain'](cfg)
        after=np.random.uniform(size=8);np.random.set_state(state);expected=np.random.uniform(size=8)
        self.assertTrue(np.array_equal(after,expected))
        np.random.seed(202);second=namespace['CapturedTerrain'](cfg)
        self.assertTrue(np.array_equal(first.samples,second.samples))

    def test_pinned_heightfield_geometry_copy(self):
        original=Path('C:/WINDOWS/Temp/go2_rl_lab_source_6c95d9fe/go2_rl_lab-6c95d9feef4814c25eb6536e61bb38e1d1d718d4/source/go2_rl_lab/go2_rl_lab/tasks/mixed/terrain.py')
        if not original.exists():self.skipTest('Pinned checkout absent; hashes are in terrain_source.json')
        copied=ROOT/'go2_physx_cts/official_terrain_heightfield.py'
        self.assertEqual(ast.dump(ast.parse(original.read_text(encoding='utf-8'))),
                         ast.dump(ast.parse(copied.read_text(encoding='utf-8'))))

    def test_mesh_generation_body_source_exact(self):
        original=Path('C:/WINDOWS/Temp/go2_rl_lab_source_6c95d9fe/go2_rl_lab-6c95d9feef4814c25eb6536e61bb38e1d1d718d4/source/go2_rl_lab/go2_rl_lab/tasks/mixed/terrain_mesh.py')
        if not original.exists():self.skipTest('Pinned checkout absent')
        copied=ast.parse((ROOT/'go2_physx_cts/official_terrain_mesh.py').read_text(encoding='utf-8'))
        source=ast.parse(original.read_text(encoding='utf-8'))
        source_nodes={node.name:ast.dump(node) for node in source.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        for node in copied.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                self.assertEqual(ast.dump(node),source_nodes[node.name],node.name)

    def test_checkpoint_control_constants_preserved(self):
        inertia,kp,kd,effort,scale=motor_values('cpu')
        expected=torch.tensor([.000111842*(6.33*(1.5 if j==2 else 1))**2 for _ in range(4) for j in range(3)])
        self.assertTrue(torch.equal(inertia,expected))
        self.assertTrue(torch.allclose(kp,expected*(2*math.pi*10)**2))
        self.assertTrue(torch.allclose(kd,4*expected*(2*math.pi*10)))
        self.assertTrue(torch.allclose(scale,.25*effort/kp))
        text=(ROOT/'go2_physx_cts/physx_env.py').read_text(encoding='utf-8')
        from go2_physx_cts.robots import profile
        self.assertEqual(profile('go2').left_hip,.1)
        self.assertEqual(profile('go2').initial_height,.29)
        self.assertIn("'.*L_hip_joint':self.profile.left_hip",text)
        self.assertIn("'.*_thigh_joint':.9",text)
        self.assertIn("'.*_calf_joint':-1.8",text)

    def test_payload_at_reset_actual_mass_context_no_mid_episode(self):
        env=PhysXEnv.__new__(PhysXEnv);env.device='cpu';env.steps=torch.zeros(3,dtype=torch.long)
        env.xi=torch.zeros(3,5);env._base_id=0
        env._nominal_masses=torch.tensor([[7.,1.],[7.,1.],[7.,1.]])
        live=env._nominal_masses.clone();env.actual_base_mass_kg=live[:,0].clone()
        def set_masses_index(*,masses,body_ids,env_ids):live[env_ids[:,None],torch.tensor(body_ids)[None,:]]=masses
        env.robot=SimpleNamespace(data=SimpleNamespace(body_mass=live),set_masses_index=set_masses_index)
        env.set_payload(torch.arange(3),torch.tensor([0.,1.,2.]))
        self.assertEqual(live[:,0].tolist(),[7.,8.,9.]);self.assertEqual(live[:,1].tolist(),[1.,1.,1.])
        self.assertEqual(env.xi[:,4].tolist(),[0.,1.,2.])
        self.assertTrue(torch.equal(env.actual_base_mass_kg,live[:,0]))
        env.steps[1]=1
        with self.assertRaises(ValueError):env.set_payload(torch.tensor([1]),0.)

    def test_command_override_is_explicit_state(self):
        env=PhysXEnv.__new__(PhysXEnv);env.num_envs=3;env.cmd_state=torch.zeros(3,17)
        env.command_override=torch.tensor([[1.,0.,0.],[0.,.3,0.],[0.,0.,.5]])
        env._command_views()
        self.assertTrue(torch.equal(env.command,env.command_override))
        self.assertTrue(torch.equal(env.cmd_state[:,:3],env.command_override))

    def test_true_ray_crop_fullscan_fallback_math(self):
        ray_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CollisionHeightRaycaster')
        method=next(n for n in ray_class.body if isinstance(n,ast.FunctionDef) and n.name=='height_crop')
        namespace={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(SOURCE),'exec'),namespace)
        counts=[]
        def cast(starts):
            counts.append(starts.shape[1]);heights=starts[...,0]*.1+starts[...,1]*.2
            valid=torch.ones_like(heights,dtype=torch.bool);valid[1]=False
            return heights,valid
        fake=SimpleNamespace(cast=cast)
        pos=torch.tensor([[2.,3.,1.],[4.,5.,2.]])
        height,valid=namespace['height_crop'](fake,pos,torch.tensor([.3,.8]))
        self.assertEqual(counts,[187]);self.assertAlmostEqual(height[0].item(),.2,places=5)
        self.assertAlmostEqual(height[1].item(),-19.,places=5)
        self.assertEqual(valid.tolist(),[True,False])

if __name__=='__main__':unittest.main()
